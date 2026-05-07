import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import kikuchipy as kp
import h5py
import dask.array as da
from pathlib import Path
from tqdm import tqdm
from dask.distributed import Client, wait
from dask_mpi import initialize
import time
start_time=time.time()

#Filepaths
#read common environment variables from input file or define directly here
pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
r=int(os.environ["RADIUS"]) #radius for NPA
#Define kwargs for jump detection
lookback = int(os.environ.get("NPA_LOOKBACK", 20))
lookahead = int(os.environ.get("NPA_LOOKAHEAD", 20))
z = float(os.environ.get("NPA_Z", 5.0))
ncc_min = float(os.environ.get("NPA_NCC_MIN", 0.05))
w_pre = int(os.environ.get("NPA_W_PRE", 10))
w_post = int(os.environ.get("NPA_W_POST", 10))
step_z = float(os.environ.get("NPA_STEP_Z", 3.0))
step_abs_min = float(os.environ.get("NPA_STEP_ABS_MIN", 0.005))
max_keep_env = os.environ.get("NPA_MAX_KEEP", "80")
max_keep = None if max_keep_env.lower() == "none" else int(max_keep_env)

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

def first_jump_cutoff(
    ncc,
    lookback=20,
    lookahead=20,
    z=5.0,
    ncc_min=0.05,
    w_pre=10,
    w_post=10,
    step_z=3.0,
    step_abs_min=0.005,
):
    """
    Detect the first statistically significant jump in sorted NCC values, with an
    additional validation that the NCC curve shows a sustained level shift after
    the jump (robust median-before/after test).

    Parameters
    ----------
    lookback / lookahead:
        Window sizes (in diffs) used to estimate the local baseline and MAD scale
        for the diff-outlier test.
    z:
        Threshold in MAD units for declaring a candidate diff jump.
    ncc_min:
        Minimum NCC allowed at the jump point (to avoid low-NCC tails).
    w_pre / w_post:
        Window sizes (in NCC points) used to compare median NCC levels before vs
        after the candidate jump.
    step_z:
        Required sustained step size, expressed in robust MAD units of the *pre*
        NCC window.
    step_abs_min:
        Optional absolute minimum sustained step (leave 0.0 for scale-adaptive).
    Returns
    -------
    keep : int
        Number of neighbors to keep (rank index in sorted NCC list).
    order : ndarray
        Indices that sort NCC descending.
    ns : ndarray
        Sorted NCC values.
    diffs : ndarray
        First differences of sorted NCC (with diffs[0]=0).
    """
    order = np.argsort(ncc)[::-1]
    ns_full = ncc[order]
    # drop NCC=1 plateau (edge case) from detection
    p = 0
    idx = np.flatnonzero(ns_full != 1.0)
    if idx.size == 0:
        diffs_full = np.concatenate([[0.0], ns_full[:-1] - ns_full[1:]])
        return 0, order, ns_full, diffs_full
    p = int(idx[0])
    ns = ns_full[p:]
    diffs = np.concatenate([[0.0], ns[:-1] - ns[1:]])

    def robust_sigma(x):
        m = np.median(x)
        mad = np.median(np.abs(x - m))
        return 1.4826 * mad + 1e-12
    keep_det = None

    for i in range(1, len(diffs) - lookahead):
        # A) diff outlier test (local baseline)
        start = max(1, i - lookback)
        end = min(len(diffs), i + lookahead)
        window = np.concatenate([diffs[start:i], diffs[i + 1:end]])
        if window.size < 4:
            continue

        m = np.median(window)
        sigma_d = robust_sigma(window)

        if not (diffs[i] > m + z * sigma_d and ns[i] > ncc_min):
            continue

        # B) sustained NCC level shift
        pre = ns[max(0, i - w_pre):i]
        post = ns[i:min(len(ns), i + w_post)]
        if pre.size < max(3, w_pre // 2) or post.size < max(3, w_post // 2):
            continue

        step = np.median(pre) - np.median(post)
        sigma_pre = robust_sigma(pre)

        if (step >= step_abs_min) and (step >= step_z * sigma_pre):
            keep_det = i
            break

    if keep_det is None:
        keep_det = int(np.argmax(diffs))

    # convert keep back to full indexing
    keep_full = p + keep_det

    # return full ns/diffs for plotting consistency
    diffs_full = np.concatenate([[0.0], ns_full[:-1] - ns_full[1:]])
    return keep_full, order, ns_full, diffs_full

def _npa_block(block, r, jump_kwargs, max_keep=None, targets_padded=None, pname=".", block_info=None):
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

    # --- print parameters once per worker process ---
    global _printed_first_jump_params
    if not globals().get("_printed_first_jump_params", False):
        print("\n===== first_jump_cutoff parameters (from driver) =====", flush=True)
        for k in sorted(jump_kwargs):
            print(f"{k:15s} = {jump_kwargs[k]!r}", flush=True)
        print(f"{'max_keep':15s} = {max_keep!r}", flush=True)
        print("=====================================================\n", flush=True)
        _printed_first_jump_params = True
    
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
            
            # --- first-jump detection ---
            keep, order, ns, diffs = first_jump_cutoff(ncc, **jump_kwargs)
            if max_keep is not None:
                keep = min(keep, max_keep)
            
            # --- diagnostic plotting at selected global (padded) targets ---
            if targets_padded:
                gy = start_y + i        # global *padded* y
                gx = start_x + j        # global *padded* x
                if (gy, gx) in targets_padded:
                    # TWO-SUBPLOT DIAGNOSTIC FIGURE (A and B made explicit)
                    fig, (ax_top, ax_bot) = plt.subplots(
                        nrows=2, ncols=1, figsize=(7, 7), sharex=True,
                        gridspec_kw={"height_ratios": [2, 2]}
                    )
                    x = np.arange(ns.size)
                    # --- TOP: NCC curve + cutoff ---
                    ax_top.plot(x, ns, marker='o', color='tab:blue', label='NCC')
                    ax_top.axvline(keep - 0.5, linestyle='--', color='0.3', label='Cutoff (keep)')
                    ax_top.set_ylabel("NCC")
                    #ax_top.set_ylim(0.0, 1.0)
                    ax_top.legend(loc="best")
                    ax_top.set_title(f"Diagnostics @ (y={gy-r}, x={gx-r}) unpadded")
                    # --- Helpers ---
                    lb   = jump_kwargs.get("lookback", 10)
                    la   = jump_kwargs.get("lookahead", 5)
                    zthr = jump_kwargs.get("z", 5.0)
                    wpre        = jump_kwargs.get("w_pre", 8)
                    wpost       = jump_kwargs.get("w_post", 8)
                    step_z      = jump_kwargs.get("step_z", 3.0)
                    step_abs_min = jump_kwargs.get("step_abs_min", 0.0)
                    def robust_sigma(arr):
                        m = np.median(arr)
                        mad = np.median(np.abs(arr - m))
                        return 1.4826 * mad + 1e-12
                    # --- BOTTOM (A): local robust z-score of diffs, exactly like test A uses ---
                    A_COLOR = "tab:purple"
                    B_COLOR = "tab:orange"
                    local_z = np.full_like(diffs, np.nan, dtype=float)
                    for ii in range(1, len(diffs) - la):
                        start = max(1, ii - lb)
                        end = min(len(diffs), ii + la)
                        w = np.concatenate([diffs[start:ii], diffs[ii+1:end]])
                        if w.size < 4:
                            continue
                        m = np.median(w)
                        sigma = robust_sigma(w)
                        local_z[ii] = (diffs[ii] - m) / sigma
                    ax_bot.plot(x, local_z, marker='s', alpha=0.6, color=A_COLOR, label="A: ΔNCC (local robust z)")
                    ax_bot.axhline(zthr, linestyle=':', color=A_COLOR, alpha=0.7, label=f"A threshold (z={zthr:g})")
                    ax_bot.set_ylabel("A: local robust z")
                    ax_bot.tick_params(axis='y', colors=A_COLOR)
                    ax_bot.yaxis.label.set_color(A_COLOR)
                    ax_bot.set_ylim(-1, 10)  # fixed for comparability (adjust to taste)

                    # --- BOTTOM-RIGHT (B): sustained median step + its threshold ---
                    ax_bot2 = ax_bot.twinx()

                    step_vals = np.full_like(ns, np.nan, dtype=float)
                    b_thr = np.full_like(ns, np.nan, dtype=float)
                    b_ok = np.zeros_like(ns, dtype=bool)
                    ncc_min_used = jump_kwargs.get("ncc_min", 0.05)
                    for ii in range(1, len(ns)):
                        pre = ns[max(0, ii - wpre):ii]
                        post = ns[ii:min(len(ns), ii + wpost)]
                        if pre.size < max(3, wpre // 2) or post.size < max(3, wpost // 2):
                            continue

                        step = np.median(pre) - np.median(post)
                        sigma_pre = robust_sigma(pre)
                        thr = max(step_abs_min, step_z * sigma_pre)

                        step_vals[ii] = step
                        b_thr[ii] = thr
                        # Note: B only matters where ns[ii] > ncc_min is plausible (optional)
                        b_ok[ii] = (step >= thr) and (ns[ii] > ncc_min_used)

                    ax_bot2.plot(x, step_vals, alpha=0.9, color=B_COLOR, label="B: median(pre)−median(post)")
                    ax_bot2.plot(x, b_thr, linestyle='--', alpha=0.95, color=B_COLOR, label="B threshold")

                    idxB = np.where(b_ok)[0]
                    if idxB.size:
                        ax_bot2.scatter(idxB, step_vals[idxB], marker="*", s=60, alpha=0.9, color=B_COLOR,  label="B fulfilled")

                    ax_bot2.set_ylabel("B: step")
                    ax_bot2.tick_params(axis='y', colors=B_COLOR)
                    ax_bot2.yaxis.label.set_color(B_COLOR)
                    # Give B a readable range (robust upper bound)
                    bmax = np.nanmax(np.concatenate([step_vals[np.isfinite(step_vals)], b_thr[np.isfinite(b_thr)]])) if np.any(np.isfinite(step_vals)) else 0.02
                    ax_bot2.set_ylim(0.0, max(0.02, 1.5 * bmax))

                    # --- Shared x and cutoff marker ---
                    ax_bot.axvline(keep - 0.5, linestyle='--', color='k', alpha=0.6)

                    ax_bot.set_xlabel("Neighbor rank (sorted by NCC)")

                    # --- Combine legends from bottom axes ---
                    l1, lab1 = ax_bot.get_legend_handles_labels()
                    l2, lab2 = ax_bot2.get_legend_handles_labels()
                    ax_bot.legend(l1 + l2, lab1 + lab2, loc="best")

                    fig.tight_layout()
                    fig.savefig(os.path.join(pname, f"npa_ncc_diag_y{gy-r}_x{gx-r}.png"), dpi=300)
                    plt.close(fig)
                    # -------------------------------------------------------------------------

            # --- averaging ---
            # Exclude only exactly repeated patterns from averaging.
            idx = np.flatnonzero(ns != 1.0)
            p = int(idx[0]) if idx.size else keep
            if keep <= p:
                avg_pat = A.astype(np.float32, copy=False)
            else:
                start_rank = min(p, keep)
                top = order[start_rank:keep]
                w = ncc[top]
                contrib = np.tensordot(w, neigh[top], axes=([0], [0]))
                avg_pat = (A + contrib) / (1.0 + w.sum())
            
            out[i-r, j-r] = avg_pat

            # --- diagnostic pattern plotting (original vs averaged) ---
            if targets_padded:
                gy = start_y + i  # global *padded* y
                gx = start_x + j  # global *padded* x
                if (gy, gx) in targets_padded:
                    fig2, axp = plt.subplots(ncols=2, figsize=(10, 5))
                    p0 = A
                    p1 = avg_pat
                    vmin = float(p0.min())
                    vmax = float(p0.max())
                    axp[0].imshow(p0, cmap="gray")
                    axp[0].set_title("Original Pattern")
                    axp[0].axis("off")
                    axp[1].imshow(p1, cmap="gray", vmin=vmin, vmax=vmax)
                    axp[1].set_title("NPA applied")
                    axp[1].axis("off")
                    fig2.suptitle(f"Patterns @ (y={gy-r}, x={gx-r}) unpadded")
                    fig2.tight_layout()
                    fig2.savefig(
                        os.path.join(pname, f"npa_pat_diag_y{gy-r}_x{gx-r}.png"),
                        dpi=300
                    )
                    plt.close(fig2)

    return out

# ─── 4) Apply via map_overlap ─────────────────────────────────────────────────
depth = {0: r, 1: r, 2: 0, 3: 0}
targets_unpadded = select_random_targets(Ny, Nx, K=8, seed=123)
targets_padded = {(r + y, r + x) for (y, x) in targets_unpadded}

outdir = os.path.join(pname,f"{mapname}_NPA{r}_plots")  # where to save pngs

# Build kwargs on the driver from your top-of-script variables
jump_kwargs = dict(
    lookback=lookback,
    lookahead=lookahead,
    z=z,
    ncc_min=ncc_min,
    w_pre=w_pre,
    w_post=w_post,
    step_z=step_z,
    step_abs_min=step_abs_min,   # <- your 0.01 will actually propagate
)
max_keep = globals().get("max_keep", None)

pat_npa = xpat_pad.map_overlap(
    lambda block, block_info=None: _npa_block(
        block, 
        r,
        jump_kwargs=jump_kwargs,
        max_keep=max_keep,
        targets_padded=targets_padded, 
        pname=outdir, 
        block_info=block_info
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

# ─── 5) Save patterns to h5 file  ────────────────────────────────────────────────
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

