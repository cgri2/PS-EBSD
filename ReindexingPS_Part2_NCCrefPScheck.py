import numpy as np
import os
from orix.io import plugins
from orix.quaternion import Rotation
from orix.crystal_map import CrystalMap
from orix import io
from pathlib import Path
import hyperspy.api as hs
import kikuchipy as kp
import h5py
from dask.distributed import Client, wait
from dask_mpi import initialize
from collections import Counter
import time
import EBSD_extra_functions as xfn
start = time.time()

# -------------------------- Initialize: filepaths and start dask jobs ---------------------------------------
#read common environment variables from input file or define directly here 
# ** review other variables and inputs in script and change as needed **
pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
mp_path = os.environ.get("MP_PATH", "")
energy_kV = float(os.environ.get("ENERGY_KV", "25"))
r = int(os.environ["RADIUS"])

outdir = os.path.join(pname, f"{mapname}_NPA{r}_Refine-1step")
os.makedirs(outdir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Start dask jobs
num_workers = int(os.environ.get("SLURM_NTASKS", os.cpu_count())) - 2
mem = (1024 * 1024 * int(os.environ["SLURM_MEM_PER_CPU"])
       if os.environ.get("SLURM_MEM_PER_CPU") else "auto")
dask_tmp = Path(pname) / "dask-temp" / "dask-mpi-workers"
dask_tmp.mkdir(parents=True, exist_ok=True)
initialize(nthreads=1, memory_limit=mem, local_directory=str(dask_tmp))
client = Client()
print("dashboard:", client.dashboard_link)
print("Scheduler address:", client.scheduler.address)

# ------------------------- Load data and prepare for reindexing ----------------------------------------------
# Load crystal map (from hough data)
xmap = plugins.ang.file_reader(os.path.join(pname, f"{mapname}.ang"))
# Load patterns
xpat = kp.load(os.path.join(pname, f"{mapname}_PP_NPA{r}.h5"), lazy=True)
Ny, Nx, py, px = xpat.data.shape
xpat.set_scan_calibration(step_x=xmap.dx, step_y=xmap.dy)
# Circular signal mask
#signal_mask = xfn.make_circular_signal_mask(py, px)
# Load master pattern
mp = load_oxford_mp(mp_path)
mp.phase = xmap.phases[0]
# Load calibrated detector
det_xmap = kp.detectors.EBSDDetector.load(os.path.join(pname, f"{mapname}_CalibratedDetector.txt"))
print("Loaded detector and master pattern")
# Rechunk patterns
cy, cx, nyc, nxc, total = xfn.choose_nav_chunks(Ny, Nx, target=5*max(1, num_workers), min_chunk=24)
print(f"Chosen chunk sizes: {cy}×{cx}  (≈ {nyc}×{nxc} = {total} nav chunks)")
xpat.data = xpat.data.rechunk((cy, cx, -1, -1)).persist()
wait(xpat.data)
client.rebalance(xpat.data)
owners = Counter(w for ws in client.who_has(xpat.data).values() for w in ws)
print("Chunks per worker:", dict(owners))
t1 = time.time()
print(f"Time to finish setup: {t1 - start:.2f} s")

# ----------------------- Refine orientations within each of 6 PS "bubbles" ----------------------------------
# PS ops
PS_rotations = Rotation.from_axes_angles(
    ((1,0,0),(1,0,0),(1,0,0),(0,1,0),(0,1,0)),
    (180,90,-90,90,-90),
    degrees=True
)

# Build 6 seeds: V0 = original xmap, V1..V5 = PS_applied
seed_maps = [xmap] + [xfn.xmap_PS(xmap, PS_rotations[k]) for k in range(5)]
refined_maps = []
tn = t1

# Prepare h5 file
final_h5 = os.path.join(outdir, f"{mapname}_CIwcc_data.h5")
f = h5py.File(final_h5, "w")
CI_stack_ds = f.create_dataset(
    "CI_stack", shape=(Ny, Nx, 6), dtype="f4",
    chunks=(min(Ny,128), min(Nx,128), 1), compression="gzip"
)
for vi in range(6):
    g = f.create_group(f"V{vi}")
f.create_group("Final")

for vi, seed in enumerate(seed_maps):
    print(f"--- Refining variant V{vi} ---")
    # IMPORTANT: refine without pseudo_symmetry switching so each pass sticks to its seed
    ref_results = xpat.refine_orientation(
        xmap=seed,
        detector=det_xmap,
        master_pattern=mp,
        pseudo_symmetry_ops=None,
        energy=energy_kV,
        #signal_mask=signal_mask,
        method="minimize",
        method_kwargs={"method": "Powell", "options": {"maxiter": 300}},
        trust_region=[2, 2, 2],
        rtol=1e-4,
        compute=False,
        rechunk=False,
    )
    xmap_ref = kp.indexing.compute_refine_orientation_results(
        results=ref_results,
        xmap=seed,
        master_pattern=mp,
        pseudo_symmetry_checked=False,
    )
    refined_maps.append(xmap_ref)

    # Save ANG for each refined map
    io.save(os.path.join(outdir, f"{mapname}_Refine1step_V{vi}.ang"), xmap_ref, overwrite=True)
    tn1 = time.time()
    print(f"Finished refinement of variant V{vi} in: {tn1 - tn:.2f} s. Saved map.")
    
    # --- Calculate CI wcc values ---
    CI_map, eta_map, Xi_e_map, Xi_t_map, Xi_eN_map = xfn.compute_wcc_map(
        xpat=xpat,
        xmap=xmap_ref,
        detector=det_xmap,
        master_pattern=mp,
        #signal_mask=signal_mask,
        energy_kV=energy_kV,
        stripe_x=40, stripe_y=40
        )
    # Save CI metrics for this variant
    g = f[f"V{vi}"]
    if "CI_map" not in g:
        g.create_dataset("CI_map", data=CI_map.astype("f4"), compression="gzip")
        g.create_dataset("eta_map", data=eta_map.astype("f4"), compression="gzip")
        g.create_dataset("Xi_e_map", data=Xi_e_map.astype("f4"), compression="gzip")
        g.create_dataset("Xi_t_map", data=Xi_t_map.astype("f4"), compression="gzip")   # (Ny, Nx, 6)
        g.create_dataset("Xi_eN_map", data=Xi_eN_map.astype("f4"), compression="gzip") # (Ny, Nx, 6)
    else:
        g["CI_map"][...]  = CI_map.astype("f4")
        g["eta_map"][...] = eta_map.astype("f4")
        g["Xi_e_map"][...] = Xi_e_map.astype("f4")
        g["Xi_t_map"][...] = Xi_t_map.astype("f4")
        g["Xi_eN_map"][...] = Xi_eN_map.astype("f4")

    CI_stack_ds[:, :, vi] = CI_map.astype("f4")
    tn2 = time.time()
    tn = tn2
    print(f"Saved metrics. Finished CI computation step in {tn2 - tn1:.2f} s.")

# --------- Select best variant per pixel using CI_wcc ---------
print('Selecting best variant per pixel...')
CI_stack = CI_stack_ds[...]
best_idx = np.argmax(CI_stack, axis=-1).astype(np.uint8)  # (Ny, Nx)
ci_sorted = np.sort(CI_stack, axis=-1)
cross_variant_margin = (ci_sorted[..., -1] - ci_sorted[..., -2]).astype("f4")

# Allocate final selections
Xi_t_sel  = np.empty((Ny, Nx, 6), dtype="f4")
Xi_eN_sel = np.empty((Ny, Nx, 6), dtype="f4")
Xi_e_sel  = np.empty((Ny, Nx),     dtype="f4")
CI_sel    = np.empty((Ny, Nx),     dtype="f4")

# Fill by reading one variant at a time (low peak RAM)
for vi in range(6):
    mask = (best_idx == vi)
    g = f[f"V{vi}"]
    # Load just the needed arrays for this variant
    Xi_t_v  = g["Xi_t_map"][...]   # (Ny, Nx, 6)
    Xi_eN_v = g["Xi_eN_map"][...]  # (Ny, Nx, 6)
    Xi_e_v  = g["Xi_e_map"][...]   # (Ny, Nx)
    CI_v    = g["CI_map"][...]     # (Ny, Nx)

    # Scatter into selected arrays using masks
    Xi_e_sel[mask] = Xi_e_v[mask]
    CI_sel[mask]   = CI_v[mask]
    # For the 3D ones, broadcast mask along last axis
    m3 = np.repeat(mask[:, :, None], 6, axis=2)
    Xi_t_sel[m3]  = Xi_t_v[m3]
    Xi_eN_sel[m3] = Xi_eN_v[m3]

# Save final group
gfin = f["Final"]
gfin.create_dataset("best_idx", data=best_idx, compression="gzip")
gfin.create_dataset("cross_variant_margin", data=cross_variant_margin, compression="gzip")
gfin.create_dataset("Xi_t", data=Xi_t_sel, compression="gzip")
gfin.create_dataset("Xi_eN", data=Xi_eN_sel, compression="gzip")
gfin.create_dataset("Xi_e", data=Xi_e_sel, compression="gzip")
gfin.create_dataset("CI_sel", data=CI_sel, compression="gzip")

# --- Collect best rotations and CI scores for the final crystal map ---
N = Ny * Nx
variant = best_idx.reshape(-1)  # per-pixel chosen variant in [0..5], length N

# Start from a copy and fill in per-variant
rots = refined_maps[0].rotations
for v in range(len(refined_maps)):   # 6 variants
    m = (variant == v)
    if np.any(m):
        rots[m] = refined_maps[v].rotations[m]

# (Optional) sanity checks
assert rots.size == N
# You can also check a few random pixels if you want
i = np.random.randint(N); assert rots[i] == refined_maps[variant[i]].rotations[i]

# Build final map
xmap_final = CrystalMap(
    rotations=rots,
    phase_id=xmap_ref.phase_id,
    x=xmap_ref.x,
    y=xmap_ref.y,
    phase_list=xmap_ref.phases,
    prop=xmap_ref.prop,
    scan_unit=xmap_ref.scan_unit,
    is_in_data=np.ones(xmap_ref.x.size, dtype=bool),
)

io.save(os.path.join(outdir, f"{mapname}_FinalSelected.ang"), xmap_final, overwrite=True)

f.close()

tf = time.time()
print(f"All done. Total time: {tf - start:.2f} s. Final outputs saved in: {outdir}")

