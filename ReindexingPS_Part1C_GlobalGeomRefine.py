import os
from orix.quaternion import Rotation
import kikuchipy as kp
import pipeline_io as pio
import EBSD_extra_functions as xfn
from EBSD_refine_geometry import optimize_geometry_and_orientations

# -------------------- Set filepaths ---------------------------------------------------------------
config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    raise RuntimeError(
        "CONFIG_PATH is not set. Submit with: "
        "submit_pipeline.sh --config /path/to/PS-EBSD_config.toml"
    )

config = pio.load_config(config_path)
paths = pio.resolve_pipeline_paths(config, config_path)

pname = paths["pname"]
mapname = paths["mapname"]
mp_path = paths["mp_path"]

energy_kV = float(config["global"]["energy_kV"])
overwrite_h5 = pio.get_overwrite_h5(config)

cfg1B = config.get("part1B", {})
r = int(cfg1B.get("radius", 7))

cfg1C = config.get("part1C", {})

h5_in = pio.h5_path_for_stage(config, config_path, "Part1C_input")
#h5_out = pio.h5_path_for_stage(config, config_path, "Part1C_output")

# -------------------- Load data -------------------------------------------------------------------
ebsd = kp.load(h5_in, lazy=True)
xpat = ebsd
xmap = ebsd.xmap
det = ebsd.detector
Ny, Nx, py, px = xpat.data.shape
print("Loaded current pipeline H5:", h5_in)
print("Pattern shape:", xpat.data.shape)
print("Initial detector:", det)
print("Std_pc:", det.pcx.std(), det.pcy.std(), det.pcz.std())
print("xmap shape:", xmap.shape)
# Load the master pattern
mp = xfn.load_oxford_mp(mp_path,xmap=xmap)
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
PS_rotations = pio.get_ps_rotations(config)

geom_outdir = os.path.join(pname, f"{mapname}_GeomRefine")

det_best, xmap_best, log = optimize_geometry_and_orientations(
    xpat=xpatS,
    xmap=xmapS,
    det0=detS,
    master_pattern=mp,
    energy=energy_kV,
    keys=cfg1C.get("keys",["pcx", "pcy", "pcz", "sample_tilt", "azimuthal", "tilt"]),
    steps=cfg1C.get("steps",[0.01, 0.01, 0.02, 1.0, 0.1, 0.1]),
    binning=int(cfg1C.get("binning", 4)),
    nrows=int(cfg1C.get("nrows", 8)),
    ncols=int(cfg1C.get("ncols", 8)),
    out_idx=tuple(cfg1C.get("out_idx", [1, 1])),
    pname=geom_outdir,
    max_iters=int(cfg1C.get("max_iters", 10)),
    tol_delta_ncc=float(cfg1C.get("tol_delta_ncc", 5e-3)),
    jacobian_method=cfg1C.get("jacobian_method", "fingerprint"),
    # orientation-refine options
    method=cfg1C.get("method", "LN_NELDERMEAD"),
    trust_region=tuple(cfg1C.get("trust_region", [3, 3, 3])),
    pseudo_symmetry_ops=PS_rotations,
    rtol=float(cfg1C.get("rtol", 1e-4)),
)

print("best NCC:", max(r["ncc_after"] for r in log))

print(det_best)
print('dtheta:')
print('sample tilt:', det_best.sample_tilt - detS.sample_tilt)
print('azimuthal:', det_best.azimuthal - detS.azimuthal)
print('tilt:', det_best.tilt - detS.tilt)
print('twist:', det_best.twist - detS.twist)
print('pcx:', det_best.pc_average[0] - detS.pc_average[0], 'pcy:', det_best.pc_average[1] - detS.pc_average[1], 'pcz:', det_best.pc_average[2] - detS.pc_average[2])

#Create new detector
det_final=det_best.extrapolate_pc(
        pc_indices=pc_indices,
        navigation_shape=det.navigation_shape,
        step_sizes=(xmap.dy, xmap.dx)
        )
print(det_final)

#Save the calibrated detector to the h5 file and txt file
det_final.save(filename=os.path.join(pname,f"{mapname}_CalibratedDetector.txt"))
