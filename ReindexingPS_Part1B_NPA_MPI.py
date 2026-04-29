import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from orix import io
from orix.io import plugins
import hyperspy.api as hs  
import kikuchipy as kp
import h5py
import dask.array as da
from pathlib import Path
from tqdm import tqdm
from dask.distributed import Client, progress, wait
from dask_mpi import initialize
import time
start_time=time.time()

#Filepaths
#read common environment variables from input file or define directly here
pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
r=int(os.environ["RADIUS"]) #radius for NPA
bl=20 #lower bound of search window
bu=20 #upper bound of search window
sig_z=5 #num std dev

# ** review other variables and inputs in script and change as needed **

# ─── 1) Start your Dask‐MPI cluster ────────────────────────────────────────────
num_workers = int(os.environ.get("SLURM_NTASKS", os.cpu_count())) - 2 #reads ntasks from job script and reserves 2 tasks for other roles
#set memory based on the per-cpu allocation by the job script
mem = (
    1024 * 1024 * int(os.environ["SLURM_MEM_PER_CPU"])
    if os.environ.get("SLURM_MEM_PER_CPU")
    else "auto"
    )
#Lanch the dask cluster
dask_tmp = Path(pname) / "dask-temp" / "dask-mpi-workers"
dask_tmp.mkdir(parents=True, exist_ok=True)
initialize(nthreads=1, memory_limit=mem, local_directory=str(dask_tmp))
client = Client()

# ─── 2) Load & chunk patterns ─────────────────────────────────────────────────
#Load data from an h5 file (warning: loading from up2 does not work with MPI/dask)
xpat=kp.load(os.path.join(pname,f"{mapname}_PP.h5"),lazy=True)
Ny, Nx, py, px = xpat.data.shape
data_type=xpat.data.dtype
#Pad in navigation dimensions to preserve edges
xpat_pad = da.pad(xpat.data, pad_width=((r, r), (r, r), (0, 0), (0, 0)), mode='edge')
#Auto-compute chunk size from workers and map dimensions and rechunk
chunk_nav = max(1, int(np.ceil(Ny / np.sqrt(num_workers))))
xpat_pad = xpat_pad.rechunk((chunk_nav, chunk_nav, -1, -1))
#distribute patterns to workers
xpat_pad = xpat_pad.persist()
client.rebalance(xpat_pad)
print("Rechunked over workers:",client.who_has(xpat_pad))

# ─── 3) Define a per‐chunk NPA function ──────────────────────────────────────
def select_random_targets(Ny, Nx, K=5, seed=42):
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, Ny, size=K)
    xs = rng.integers(0, Nx, size=K)
    return list(zip(ys.tolist(), xs.tolist()))

def first_jump_cutoff(ncc, lookback=10, lookahead=5, z=5.0, ncc_min=0.05):
    """
    Detect the first statistically significant jump in sorted NCC values.

    lookback:   number of diffs before the candidate to include
    lookahead:  number of diffs after the candidate to include
    z:          threshold in MAD units for declaring a jump
    ncc_min:    minimum NCC allowed for the jump point (to avoid low-NCC tails)
    """
    order = np.argsort(ncc)[::-1]
    ns = ncc[order]
    diffs = np.concatenate([[0.0], ns[:-1] - ns[1:]])

    keep = None
    for i in range(1, len(diffs) - lookahead):
        # Build a two-sided local window around i
        start = max(1, i - lookback)
        end = min(len(diffs), i + lookahead)
        window = np.concatenate([diffs[start:i], diffs[i+1:end]])
        if window.size < 4:
            continue
        m = np.median(window)
        mad = np.median(np.abs(window - m))
        sigma = 1.4826 * mad + 1e-12

        # if the current diff is far above the local baseline
        if diffs[i] > m + z * sigma and ns[i] > ncc_min:
            keep = i
            break

    if keep is None:
        keep = int(np.argmax(diffs))

    return keep, order, ns, diffs


def _npa_block(block, r, targets_padded=None, pname=".", block_info=None):
    """
    targets_padded: a set of (gy, gx) *padded* global nav indices where we want a diagnostic plot
    pname: directory to save figures
    block_info: provided by Dask map_overlap
    """
    B0r, B1r, py, px = block.shape
    B0, B1 = B0r - 2*r, B1r - 2*r
    out = np.empty((B0, B1, py, px), dtype=np.float32)

    # global position (in the *padded* nav array) of this chunk
    # block_info[0]["array-location"] -> ((y0,y1), (x0,x1), (py0,py1), (px0,px1))
    loc = block_info[0]["array-location"]
    start_y = loc[0][0]
    start_x = loc[1][0]

    # Precompute Manhattan‐radius offsets
    offsets = [(di, dj)
               for di in range(-r, r + 1)
               for dj in range(-r, r + 1)
               if 0 < abs(di) + abs(dj) <= r]

    os.makedirs(pname, exist_ok=True)

    for i in range(r, r + B0):
        for j in range(r, r + B1):
            A = block[i, j]
            A_dev = A - A.mean()
            normA = np.linalg.norm(A_dev)

            neigh = np.stack([block[i+di, j+dj] for di, dj in offsets], axis=0)
            neigh_dev = neigh - neigh.mean(axis=(1, 2), keepdims=True)
            normsN = np.linalg.norm(neigh_dev.reshape(neigh_dev.shape[0], -1), axis=1)

            numer = np.tensordot(neigh_dev, A_dev, axes=([1, 2], [0, 1]))
            denom = normA * normsN
            ncc = np.zeros_like(denom)
            np.divide(numer, denom, out=ncc, where=denom > 0)

            # --- first-jump detection (two-sided window version) ---
            keep, order, ns, diffs = first_jump_cutoff(
                ncc, lookback=bl, lookahead=bu, z=sig_z, ncc_min=0.05
                )
            # --- diagnostic plotting at selected global (padded) targets ---
            if targets_padded:
                gy = start_y + i        # global *padded* y
                gx = start_x + j        # global *padded* x
                if (gy, gx) in targets_padded:
                    fig, ax1 = plt.subplots(figsize=(6, 4))

                    # --- Left axis: NCC values ---
                    ax1.plot(np.arange(ns.size), ns, marker='o', color='tab:blue', label='NCC')
                    ax1.axvline(keep - 0.5, linestyle='--', color='k', label='First jump')
                    ax1.set_xlabel("Neighbor rank (sorted by NCC)")
                    ax1.set_ylabel("NCC", color='tab:blue')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')

                    # --- Right axis: NCC differences ---
                    ax2 = ax1.twinx()
                    ax2.plot(np.arange(diffs.size), diffs, marker='s', color='tab:orange', alpha=0.6, label='ΔNCC')
                    ax2.set_ylabel("ΔNCC", color='tab:orange')
                    ax2.tick_params(axis='y', labelcolor='tab:orange')

                    # --- Title and legend ---
                    ax1.set_title(f"Ordered NCC and ΔNCC @ (y={gy-r}, x={gx-r}) unpadded")

                    # Combine legends from both axes
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

                    fig.tight_layout()
                    fig.savefig(os.path.join(pname, f"npa_ncc_diag_y{gy-r}_x{gx-r}.png"), dpi=300)
                    plt.close(fig)
            """
            # --- diagnostic plotting at selected global (padded) targets ---
            if targets_padded:
                gy = start_y + i        # global *padded* y
                gx = start_x + j        # global *padded* x
                if (gy, gx) in targets_padded:
                    fig = plt.figure(figsize=(6, 4))
                    ax = fig.add_subplot(111)
                    ax.plot(np.arange(ns.size), ns, marker='o')
                    ax.axvline(keep - 0.5, linestyle='--', label='First jump')
                    ax.set_xlabel("Neighbor rank (sorted by NCC)")
                    ax.set_ylabel("NCC")
                    ax.set_title(f"Ordered NCC @ (y={gy-r}, x={gx-r}) unpadded")
                    ax.legend()
                    fig.tight_layout()
                    # Save with *unpadded* coordinates in filename for easy lookup
                    fig.savefig(os.path.join(pname, f"npa_ncc_diag_y{gy-r}_x{gx-r}.png"), dpi=300)
                    plt.close(fig)
            """
            # --- averaging ---
            if keep == 0:
                out[i-r, j-r] = A
            else:
                top = order[:keep]
                w = ncc[top]
                contrib = np.tensordot(w, neigh[top], axes=([0], [0]))
                out[i-r, j-r] = (A + contrib) / (1.0 + w.sum())

    return out

# ─── 4) Apply via map_overlap ─────────────────────────────────────────────────
depth = {0: r, 1: r, 2: 0, 3: 0}
targets_unpadded = select_random_targets(Ny, Nx, K=8, seed=123)
targets_padded = {(r + y, r + x) for (y, x) in targets_unpadded}

outdir = os.path.join(pname,f"{mapname}_NPA{r}_plots")  # where to save pngs

pat_npa = xpat_pad.map_overlap(
    lambda block, block_info=None: _npa_block(
        block, r, targets_padded=targets_padded, pname=outdir, block_info=block_info
    ),
    depth=depth,
    boundary=0,
    dtype=np.float32,
    trim=False
)

with tqdm(total=1, desc="Dask NPA computation") as pbar:
    pat_npa = client.persist(pat_npa)
    wait(pat_npa)
    pbar.update()

#remove padding
pat_npa = pat_npa[r:-r, r:-r]   #crop padded patterns to return to original Ny x Nx map
pat_float = pat_npa.compute()   
vmin, vmax = pat_float.min(), pat_float.max()   #compute intensity range
#normalize patterns and convert back to original data type
norm = (pat_float - vmin) / (vmax - vmin)
norm = np.clip(norm, 0, 1)
if data_type == "uint8":
    scaled = norm * 255
    xpat_NPA = scaled.astype(np.uint8)
elif data_type == "uint16":
    scaled = norm * 65535
    xpat_NPA=scaled.astype(np.uint16)
else:
    raise ValueError("data_type must be 'uint8' or 'uint16'")
print(xpat_NPA.shape)
print('NPA complete')

ckpt1 = time.time() - start_time
print(f"Time to finish NPA: {ckpt1:.2f} seconds")

# ─── 5) Plot example patterns  ────────────────────────────────────────────────
p0 = xpat.inav[r,r].data.compute()
p1 = xpat_NPA[r,r]
print("p1 min:", p1.min(), "max:", p1.max(), "dtype:", p1.dtype)
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(p0, cmap="gray")
ax[0].set_title("Original Pattern")
ax[1].imshow(p1, cmap="gray", vmin=p0.min(), vmax=p0.max())
_ = ax[1].set_title("NPA applied")
plt.savefig(os.path.join(outdir,f"{mapname}_NPA{r}.png"),dpi=300)

# ─── 6) Save patterns to h5 file  ────────────────────────────────────────────────
#copy h5 file of original patterns (including all metadata)
Opath = os.path.join(pname, f"{mapname}_PP.h5")
Npath = os.path.join(pname, f"{mapname}_PP_NPA{r}.h5")
shutil.copyfile(Opath,Npath)

#Configure patterns for saving
pat_N =  xpat_NPA.reshape(Ny * Nx, py, px)

#open copied file and replace pattern data with npa patterns
with h5py.File(Npath, "r+") as f:
    pattern_path = "Scan 1/EBSD/Data/patterns"
    # Double check path, data shape and type
    if pattern_path not in f:
        raise ValueError(f"{pattern_path} not found in file.")
    pat_O = f[pattern_path]
    print("Original pattern shape:", pat_O.shape, "; New pattern shape:", pat_N.shape)
    print("Original pattern dtype:", pat_O.dtype, "; New pattern dtype:", pat_N.dtype)
    print("Original pattern intensity range:", pat_O[0,0].min(), ",", pat_O[0,0].max())
    print("New pattern intensity range:", pat_N[0,0].min(), ",", pat_N[0,0].max())
    if pat_N.shape != pat_O.shape:
        raise ValueError(f"Shape mismatch: {pat_N.shape} != {pat_O.shape}")
    if pat_N.dtype != pat_O.dtype:
        raise ValueError(f"dtype mismatch: {pat_N.dtype} != {pat_O.dtype}")

    # Overwrite pattern data
    pat_O[...] = pat_N

print(f"Saved NPA patterns")
ckpt2 = time.time() - start_time
print(f"Time to finish NPA and saving: {ckpt2:.2f} seconds")

