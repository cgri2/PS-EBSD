import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pandas as pd
import h5py
import os
import shutil
from orix.io import plugins
from orix.crystal_map import Phase, CrystalMap
from orix.quaternion import Rotation
from orix import sampling, plot, io
from orix.vector import Vector3d
import hyperspy.api as hs  
import kikuchipy as kp
import sys
sys.path.append("/cluster/work/mandm/cgriesbach/EBSDindexing")
import EBSD_extra_functions as xfn
from EBSD_refine_geometry import optimize_geometry_and_orientations

# -------------------- Set filepaths ---------------------------------------------------------------
pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
mp_path = os.environ.get("MP_PATH", "")
energy_kV = float(os.environ.get("ENERGY_KV", "25"))
r = int(os.environ["RADIUS"])

# -------------------- Load data -------------------------------------------------------------------
# Load patterns
xpat = kp.load(os.path.join(pname, f'{mapname}_PP_NPA{r}.h5'), lazy=True)
Ny, Nx, py, px = xpat.data.shape
# Load crystal map
xmap = plugins.ang.file_reader(os.path.join(pname, f'{mapname}.ang'))
# Load detector
det = kp.detectors.EBSDDetector.load(os.path.join(pname, f'{mapname}_Detector.txt'))
print('Initial detector:', det, 'Std_pc:', det.pcx.std(), det.pcy.std(), det.pcz.std())
# Load the master pattern
with h5py.File(mp_path,'r') as f:
    lower_hemisphere = f["Data/Master/Dynamical/Lower"][()]
    upper_hemisphere = f["Data/Master/Dynamical/Upper"][()]
south_signal = hs.signals.Signal2D(lower_hemisphere)
north_signal = hs.signals.Signal2D(upper_hemisphere)
mp = kp.signals.EBSDMasterPattern([north_signal, south_signal], hemisphere='both',) # Create the EBSDMasterPattern signal, explicitly setting hemispheres
mp.hemispheres = {"north", "south"} # Assign hemispheres
mp.projection = 'stereographic' #Assign projection
mp_l = mp.as_lambert() # Convert the master pattern to the square Lambert projection
mp_l.phase = xmap.phases[0]
# Select subset of data for geometry refinement
xpatS, detS, xmapS, pc_indices = xfn.EBSD_subset(xpat, det, xmap, n_points=100)
Ny_c, Nx_c, _, _ = xpatS.data.shape
oris = xmapS.rotations.reshape(Ny_c, Nx_c)
exp_patterns = xpatS.data
print(exp_patterns.shape)
print(detS)
print(detS.navigation_shape)
print(oris.shape)

# Refine geometry iteratively using DIC approach
PS_rotations=Rotation.from_axes_angles(((1, 0, 0),(1, 0, 0),(1, 0, 0),(0, 1, 0),(0, 1, 0)), (180, 90, -90, 90, -90), degrees=True)
det_best, xmap_best, log = optimize_geometry_and_orientations(
    xpat=xpatS,                 # kikuchipy.Patterns, 4D (Ny,Nx,H,W)
    xmap=xmapS,               # your orientation map (provides .orientations)
    det0=detS,                 # initial detector
    master_pattern=mp_l,       # same one you already use
    energy=energy_kV,
    keys=['pcx','pcy','pcz','sample_tilt','azimuthal','tilt'],
    steps=[0.01, 0.01, 0.02, 1.0, 0.1, 0.1],
    binning=4,
    nrows=8, ncols=8,
    out_idx=(1, 1),            # (j,i) in nav grid for plots/GIFs
    pname=os.path.join(pname,f'{mapname}_GeomRefine'),
    max_iters=10,
    tol_delta_ncc=5e-3,
    jacobian_method="fingerprint",  # or "pcc"
    # orientation-refine options
    method="LN_NELDERMEAD",
    trust_region=(3,3,3),
    pseudo_symmetry_ops=PS_rotations,
    rtol=1e-4,
)
print("best NCC:", max(r["ncc_after"] for r in log))

print(det_best)
print('dtheta:')
print('sample tilt:', det_best.sample_tilt - detS.sample_tilt)
print('azimuthal:', det_best.azimuthal - detS.azimuthal)
print('pcx:', det_best.pc_average[0] - detS.pc_average[0], 'pcy:', det_best.pc_average[1] - detS.pc_average[1], 'pcz:', det_best.pc_average[2] - detS.pc_average[2])

#Create new detector
det_final=det_best.extrapolate_pc(
        pc_indices=pc_indices,
        navigation_shape=det.navigation_shape,
        step_sizes=(xmap.dy, xmap.dx)
        )
print(det_final)
det_final.save(filename=os.path.join(pname,f"{mapname}_CalibratedDetector.txt"))

#move out file to outdir
job_name = os.getenv("SLURM_JOB_NAME", "unknown_job")
job_id = os.getenv("SLURM_JOB_ID", "noid")
out_file = f"logs/{job_name}_{job_id}.out"

# Move the log after the script finishes
if os.path.exists(out_file):
    shutil.move(out_file, os.path.join(f'{mapname}_GeomRefine', "job.out"))
