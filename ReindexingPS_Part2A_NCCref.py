import sys
import numpy as np
import os
from orix import io
from pathlib import Path
import kikuchipy as kp
import h5py
import dask
from dask.distributed import Client, wait
from dask_mpi import initialize
from collections import Counter
import time
import EBSD_extra_functions as xfn
import pipeline_io as pio

start = time.time()

# ---- Read environment variables ----
config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    raise RuntimeError("CONFIG_PATH is not set.")

vi_str = os.environ.get("VARIANT_IDX")
if vi_str is None:
    raise RuntimeError("VARIANT_IDX is not set.")
vi = int(vi_str)

config = pio.load_config(config_path)
paths = pio.resolve_pipeline_paths(config, config_path)

pname     = paths["pname"]
mapname   = paths["mapname"]
mp_path   = paths["mp_path"]
energy_kV = float(config["global"]["energy_kV"])
cfg1B     = config.get("part1B", {})
r         = int(cfg1B.get("radius", 7))
cfg2      = config.get("part2", {})

h5_in = pio.h5_path_for_stage(config, config_path, "Part2_input")
outdir_suffix = cfg2.get("outdir_suffix", f"NPA{r}_Refine-1step")
outdir = os.path.join(pname, f"{mapname}_{outdir_suffix}")
os.makedirs(outdir, exist_ok=True)

# ---- Early exit if this variant already completed ----
# Both outputs must exist: the .ang (refinement) and the per-variant .h5 (CI/WCC metrics).
ang_out = os.path.join(outdir, f"{mapname}_Refine1step_V{vi}.ang")
h5_out  = os.path.join(outdir, f"{mapname}_CIwcc_V{vi}.h5")
if os.path.exists(ang_out) and os.path.exists(h5_out):
    print(f"V{vi} already complete ({ang_out}). Skipping.")
    sys.exit(0)

skip_refinement = os.path.exists(ang_out) and not os.path.exists(h5_out)
if skip_refinement:
    print(f"V{vi}: .ang exists but no .h5 — skipping refinement, recomputing CI/WCC only.")

# ---- Initialize Dask/MPI ----
num_workers = int(os.environ.get("SLURM_NTASKS", os.cpu_count())) - 2
mem = (1024 * 1024 * int(os.environ["SLURM_MEM_PER_CPU"])
       if os.environ.get("SLURM_MEM_PER_CPU") else "auto")
dask_tmp = Path(pname) / "dask-temp" / f"dask-mpi-V{vi}"
dask_tmp.mkdir(parents=True, exist_ok=True)
# Increase TCP/heartbeat timeouts before initializing the cluster.
# Workers run a GIL-bound Python loop that can legitimately block the event loop
# for 60-120 s per chunk; without these settings the scheduler kills healthy workers.
# Also force task-based rechunking to avoid the P2P shuffle path, whose in-flight
# state goes inconsistent (P2PConsistencyError) once any worker is evicted this way.
dask.config.set({
    "distributed.scheduler.worker-ttl": None,        # disable heartbeat-based killing
    "distributed.comm.timeouts.tcp": "600s",
    "distributed.comm.timeouts.connect": "120s",
    "distributed.worker.memory.target": 0.70,        # spill to disk above 70 %
    "distributed.worker.memory.spill": 0.80,         # hard spill above 80 %
    "distributed.worker.memory.pause": 0.90,         # pause tasks above 90 %
    "distributed.worker.memory.terminate": 0.98,     # terminate only at 98 %
    "array.rechunk.method": "tasks",                 # avoid P2P shuffle for rechunk
})
initialize(nthreads=1, memory_limit=mem, local_directory=str(dask_tmp))
client = Client()
print("dashboard:", client.dashboard_link)
print("Scheduler address:", client.scheduler.address)

# ---- Load patterns, detector, master pattern ----
xpat = kp.load(h5_in, lazy=True)
xmap = xpat.xmap
Ny, Nx, py, px = xpat.data.shape
xpat.set_scan_calibration(step_x=xmap.dx, step_y=xmap.dy)

det_path = os.path.join(pname, f"{mapname}_CalibratedDetector.txt")
if os.path.exists(det_path):
    det_xmap = kp.detectors.EBSDDetector.load(det_path)
    print(f"Loaded calibrated detector from: {det_path}")
else:
    det_xmap = xpat.detector
    print(f"WARNING: Calibrated detector file not found: {det_path}\n"
          "Using detector stored in the H5 file instead.")

mp = xfn.load_oxford_mp(mp_path, xmap=xmap)
print("Loaded detector and master pattern")

cy, cx, nyc, nxc, total = xfn.choose_nav_chunks(Ny, Nx, target=5*max(1, num_workers), min_chunk=24)
print(f"Chosen chunk sizes: {cy}×{cx}  (≈ {nyc}×{nxc} = {total} nav chunks)")
xpat.data = xpat.data.rechunk((cy, cx, -1, -1)).persist()
wait(xpat.data)
client.rebalance(xpat.data)
owners = Counter(w for ws in client.who_has(xpat.data).values() for w in ws)
print("Chunks per worker:", dict(owners))
t1 = time.time()
print(f"Time to finish setup: {t1 - start:.2f} s")

# ---- Refine orientations or load existing results ----
if not skip_refinement:
    PS_rotations = pio.get_ps_rotations(config)
    seed = xmap if vi == 0 else xfn.xmap_PS(xmap, PS_rotations[vi - 1])

    print(f"--- Refining variant V{vi} ---")
    print("Refinement information:")
    print(f"  Method: {cfg2.get('minimize_method', 'Powell')} (local) from SciPy")
    print(f"  Trust region (+/-): {cfg2.get('trust_region', [2, 2, 2])}")
    print(f"  Keyword arguments passed to method: {{'method': '{cfg2.get('minimize_method', 'Powell')}', 'options': {{'maxiter': {int(cfg2.get('maxiter', 300))}}}}}")
    print(f"Refining {Ny * Nx} orientation(s):")

    ref_results = xpat.refine_orientation(
        xmap=seed,
        detector=det_xmap,
        master_pattern=mp,
        pseudo_symmetry_ops=None,
        energy=energy_kV,
        method=cfg2.get("method", "minimize"),
        method_kwargs={
            "method": cfg2.get("minimize_method", "Powell"),
            "options": {"maxiter": int(cfg2.get("maxiter", 300))},
        },
        trust_region=cfg2.get("trust_region", [2, 2, 2]),
        rtol=float(cfg2.get("rtol", 1e-4)),
        compute=False,
        rechunk=False,
    )
    xmap_ref = kp.indexing.compute_refine_orientation_results(
        results=ref_results,
        xmap=seed,
        master_pattern=mp,
        pseudo_symmetry_checked=False,
    )

    io.save(ang_out, xmap_ref, overwrite=True)
    t2 = time.time()
    print(f"Finished refinement of variant V{vi} in: {t2 - t1:.2f} s. Saved map.")
else:
    print(f"Loading existing refinement results from {ang_out}")
    xmap_ref = io.load(ang_out)
    t2 = time.time()

# ---- Compute CI/WCC metrics ----
CI_map, eta_map, Xi_e_map, Xi_t_map, Xi_eN_map = xfn.compute_wcc_map(
    xpat=xpat,
    xmap=xmap_ref,
    detector=det_xmap,
    master_pattern=mp,
    energy_kV=energy_kV,
    stripe_x=40, stripe_y=40,
)

# ---- Save per-variant HDF5 ----
h5_out = os.path.join(outdir, f"{mapname}_CIwcc_V{vi}.h5")
with h5py.File(h5_out, "w") as f:
    g = f.create_group(f"V{vi}")
    g.create_dataset("CI_map",    data=CI_map.astype("f4"),    compression="gzip")
    g.create_dataset("eta_map",   data=eta_map.astype("f4"),   compression="gzip")
    g.create_dataset("Xi_e_map",  data=Xi_e_map.astype("f4"),  compression="gzip")
    g.create_dataset("Xi_t_map",  data=Xi_t_map.astype("f4"),  compression="gzip")
    g.create_dataset("Xi_eN_map", data=Xi_eN_map.astype("f4"), compression="gzip")

t3 = time.time()
print(f"Saved metrics to {h5_out}. CI step: {t3 - t2:.2f} s.")
print(f"All done for V{vi}. Total time: {t3 - start:.2f} s.")
