import kikuchipy as kp
import numpy as np
import os
import h5py
import hyperspy.api as hs
from orix.io import plugins
from orix.crystal_map import Phase, CrystalMap, PhaseList
from orix.quaternion import Rotation
import EBSD_extra_functions as xfn

pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
mp_path = os.environ["MP_PATH"]

# Load the EBSD map from h5 file
xpat = kp.load(os.path.join(pname,f"{mapname}.h5"),lazy=True) #or can load up1/up2 file directly but check kp version for compatibility
xmap = plugins.ang.file_reader(os.path.join(pname,f'{mapname}.ang')) #map
Ny, Nx, py, px = ebsd.data.shape

# Crop patterns to a square (edges are typically noisy and square patterns are better for geometry refinement step)
xpat, crop_info = xfn.crop_ebsd_to_square(ebsd, mask='circular', p=1.0)
py_c, px_c = xpat.data.shape[-2:]
x0, x1 = crop_info["left"], crop_info["right"]
y0, y1 = crop_info["top"], crop_info["bottom"]

#Define detector **replace values with those for your scan/system
det = kp.detectors.EBSDDetector(
        shape=(py,px),
        pc=(0.5, 0.5, 0.5),
        convention='edax',
        sample_tilt=70,
        tilt=10,
        azimuthal=-2,
        px_size=66.67,
        binning=1,
        )
det = xfn.crop_detector(det, (py_c, px_c), (y0,y1,x0,x1))
iy0 = Ny // 2
ix0 = Nx // 2
det = det.extrapolate_pc(
        pc_indices=[iy0, ix0],
        navigation_shape=xmap.shape,
        step_sizes=(xmap.dy, xmap.dx),
        )

#Package into one ebsd signal
sig_shape = (py_c, px_c)
dtype = xpat.data.dtype
xpat.static_background = np.zeros(sig_shape, dtype=dtype)

ebsd = kp.signals.EBSD(
    xpat,
    xmap=xmap,
    detector=det,
    static_background=xpat.static_background,
)
ebsd.set_scan_calibration(step_x=xmap.dx, step_y=xmap.dy)

# Save EBSD dataset to h5 file
ebsd.compute(show_progressbar=True)
ebsd.save(os.path.join(pname,f"{mapname}.h5"), overwrite=True)

# Load master pattern
mp = xfn.load_oxford_mp(mp_path)
mp.phase = ebsd.xmap.phases[0]

#plot an example patterns
example_dir = os.path.join(pname, "example_patterns")
os.makedirs(example_dir, exist_ok=True)
rng = np.random.default_rng()  # optionally use np.random.default_rng(0) for reproducibility
valid_k = np.where(ebsd.xmap.phase_id != -1)[0] # Valid points: avoid unindexed pixels if phase_id uses -1 for not indexed
# Randomly choose 3 unique indexed points
k_examples = rng.choice(valid_k, size=3, replace=False)

for n, k in enumerate(k_examples, start=1):

    # Convert flat row-major index back to map row/column
    i, j = np.unravel_index(k, shape=(Ny, Nx), order="C")

    print(f"\nExample {n}")
    print(f"i={i}, j={j}, k={k}")
    print("x:", ebsd.xmap.x[k], "y:", ebsd.xmap.y[k])

    # Select point-specific detector
    det1 = ebsd.detector.deepcopy()

    if np.ndim(det1.pc) == 3:
        det1.pc = ebsd.detector.pc[i, j]
    print(det1)

    # Select point-specific orientation
    rot1 = ebsd.xmap.rotations[k]
    print("Euler angles, kikuchipy frame, deg:")
    print(rot1.to_euler(degrees=True))

    # Simulate one pattern
    sim = mp.get_patterns(
        rotations=rot1,
        detector=det1,
        energy=15,
        compute=True,
    )
    # Get experimental and simulated arrays
    exp_pat = ebsd.inav[j, i].data
    sim_pat = sim.data[0] if sim.data.ndim == 3 else sim.data
    # Compute if EBSD signal is lazy
    if hasattr(exp_pat, "compute"):
        exp_pat = exp_pat.compute()

    if hasattr(sim_pat, "compute"):
        sim_pat = sim_pat.compute()

    gif_path = os.path.join(
        example_dir,
        f"{mapname}_ExpSimFlash_random{n}_Id{k}_i{i}_j{j}.gif",
    )

    xfn.make_flash_gif(
        exp_pat,
        sim_pat,
        gif_path,
        duration=0.5,
        n_flashes=2,
    )

    print("Saved:", gif_path)

