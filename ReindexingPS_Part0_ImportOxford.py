import kikuchipy as kp
import numpy as np
import os
import h5py
import hyperspy.api as hs
from orix.io import plugins
from orix.crystal_map import Phase, CrystalMap, PhaseList
from orix.quaternion import Rotation
import EBSD_extra_functions as xfn
import time
start_time = time.time()

pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
mp_path = os.environ["MP_PATH"]

map_num = 1

# Load the EBSD map from h5oina file
ebsd = kp.load(os.path.join(pname, f"{mapname}.h5oina"),lazy=True)
Ny, Nx, py, px = ebsd.data.shape

# Crop patterns to a square (edges are typically noisy and square patterns are better for geometry refinement step)
ebsd, crop_info = xfn.crop_ebsd_to_square(ebsd)
py_c, px_c = ebsd.data.shape[-2:]
x0, x1 = crop_info["left"], crop_info["right"]
y0, y1 = crop_info["top"], crop_info["bottom"]

# Load the detector from h5oina file
with h5py.File(os.path.join(pname, f"{mapname}.h5oina"), 'r') as hf:
    det_ori = hf[f'/{map_num}/EBSD/Header/Detector Orientation Euler'][()]
det_ori_rad = np.asarray(det_ori, dtype=float).squeeze().reshape(3,)
det_ori_deg = np.rad2deg(det_ori_rad)
ebsd.detector.px_size = 13.2315*1024/py
ebsd.detector.tilt = det_ori_deg[1] - 90
ebsd.detector.azimuthal = det_ori_deg[0]
ebsd.detector.twist = det_ori_deg[2]
print(ebsd.detector)

# Load the crystal map
with h5py.File(os.path.join(pname, f"{mapname}.h5oina"), 'r') as hf:
    euler_raw = hf[f'/{map_num}/EBSD/Data/Euler'][()]
    x_raw = hf[f'/{map_num}/EBSD/Data/X'][()]
    y_raw = hf[f'/{map_num}/EBSD/Data/Y'][()]
    phaseID_raw = hf[f'/{map_num}/EBSD/Data/Phase'][()]
    phaseName_raw = hf[f'/{map_num}/EBSD/Header/Phases/1/Phase Name'][()]
    phaseSG_raw = hf[f'/{map_num}/EBSD/Header/Phases/1/Space Group'][()]

phaseName = phaseName_raw.item().decode('utf-8')
space_group = int(np.asarray(phaseSG_raw).squeeze())
euler = np.asarray(euler_raw, dtype=float)
x = np.asarray(x_raw, dtype=float).reshape(-1)
y = np.asarray(y_raw, dtype=float).reshape(-1)
phase_id = np.asarray(phaseID_raw, dtype=int).reshape(-1)
n = euler.shape[0]

# Convert Oxford sample-frame convention to kikuchipy/EDAX convention.
rot_ox = Rotation.from_euler(euler, degrees=False)
R_oxford_to_kp = Rotation.from_axes_angles([0, 0, 1], -np.pi / 2)
rot_kp = rot_ox * R_oxford_to_kp

# Build phase
phase_id[phase_id == 0] = -1 # Oxford/H5OINA: 0 = not indexed =>  orix: -1 = not indexed
unique_phase_ids = np.unique(phase_id)
indexed_ids = unique_phase_ids[unique_phase_ids >= 0]
if indexed_ids.size == 0:
    raise ValueError(
        f"No indexed phase IDs found. Unique phase IDs are: {unique_phase_ids}"
    )
if indexed_ids.size != 1:
    raise ValueError(
        "This simplified loader expects one indexed phase plus optional -1 "
        f"not-indexed pixels, but found phase IDs: {unique_phase_ids}"
    )
pid = int(indexed_ids[0])
phase = Phase(
    name=phaseName,
    space_group=space_group,)
phase_list = PhaseList(
    phases=[phase],
    ids=[pid],
)

# Build the crystal map
ebsd.xmap = CrystalMap(
    rotations=rot_kp,
    phase_id=phase_id,
    x=x,
    y=y,
    phase_list=phase_list,
    )
ckpt1 = time.time() - start_time
print(f"Time to finish building EBSD signal: {ckpt1:.2f} seconds")

# Save EBSD dataset to h5 file
#ebsd.compute(show_progressbar=True)
ebsd.save(os.path.join(pname,f"{mapname}.h5"), overwrite=True)
ckpt2 = time.time() - ckpt1
print(f"Time to finish saving h5: {ckpt2:.2f} seconds")

# Load master pattern
mp = xfn.load_oxford_mp(mp_path, xmap=ebsd.xmap)
#mp.phase = ebsd.xmap.phases[1]

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
    print("Detector pc, Oxford convention:", det1.pc_oxford())

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
