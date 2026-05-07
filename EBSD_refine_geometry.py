from __future__ import annotations
import numpy as np
import cv2
from skimage.transform import downscale_local_mean
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Any, List
from scipy.linalg import lstsq
import os
from matplotlib import cm
import time
from EBSD_extra_functions import make_flash_gif

# ---------- Utilities ---------------------------------------------------

def simulate_fn(master_pattern, oris, det, energy: float) -> np.ndarray:
    """Simulate patterns; returns either (N,H,W) or (Ny,Nx,H,W) depending on `oris`."""
    sims = master_pattern.get_patterns(
        rotations=oris,
        detector=det,
        energy=energy,
        dtype_out=np.float32,
        compute=True,
    )
    return sims.data.astype(np.float32)

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def bin_image(img: np.ndarray, binning: Optional[int]) -> np.ndarray:
    """Local-mean binning (edge-padded)."""
    if binning is None or binning == 1:
        return img.astype(np.float32)
    H, W = img.shape
    ph = (-H) % binning
    pw = (-W) % binning
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    return downscale_local_mean(img, (binning, binning)).astype(np.float32)

def prebin_stack(stack: np.ndarray, binning: Optional[int]) -> np.ndarray:
    Ny, Nx, py, px = stack.shape
    out = np.empty((Ny, Nx,
                    (py + (-py) % (binning or 1)) // (binning or 1),
                    (px + (-px) % (binning or 1)) // (binning or 1)),
                   dtype=np.float32)
    for j in range(Ny):
        for i in range(Nx):
            out[j, i] = bin_image(stack[j, i], binning)
    return out

def norm255(img: np.ndarray) -> np.uint8:
    z = img - img.min()
    rng = z.max() if z.max() > 0 else 1.0
    return (255.0 * z / rng).astype(np.uint8)

# ---------- Shape helpers ------------------------------------------------

def _nav_shape_from_xpat(xpat) -> Tuple[int,int,int,int]:
    """Return (Ny,Nx,H,W) given kikuchipy.Patterns or ndarray."""
    data = xpat.data if hasattr(xpat, "data") else np.asarray(xpat)
    if data.ndim != 4:
        raise ValueError(f"xpat must be 4D (Ny,Nx,H,W); got {data.shape}")
    return data.shape  # Ny, Nx, H, W

def _stack_to_NHW(x) -> np.ndarray:
    """Accept (Ny,Nx,H,W) or (N,H,W) and return (N,H,W)."""
    data = x.data if hasattr(x, "data") else np.asarray(x)
    data = np.asarray(data, np.float32)
    if data.ndim == 4:
        Ny, Nx, H, W = data.shape
        return data.reshape(Ny*Nx, H, W)
    elif data.ndim == 3:
        return data
    else:
        raise ValueError(f"Expected 3D or 4D patterns, got shape {data.shape}")

def _N_to_nav(x_NHW: np.ndarray, nav: Tuple[int,int]) -> np.ndarray:
    """Reshape (N,H,W) to (Ny,Nx,H,W) with row-major flatten ordering."""
    Ny, Nx = nav
    N, H, W = x_NHW.shape
    if N != Ny*Nx:
        raise ValueError(f"Cannot reshape N={N} to (Ny,Nx)=({Ny},{Nx})")
    return x_NHW.reshape(Ny, Nx, H, W)

def _select_by_idx(arr: np.ndarray, idx: Optional[Sequence[int]]):
    if idx is None:
        return arr
    return arr[idx]

# ---------- Create ROIs -------------------------------------------

def subdivide_pattern(pat: np.ndarray, nrows=5, ncols=5):
    if pat.ndim != 2:
        raise ValueError(f"Expected 2D pattern, got {pat.shape}")
    h, w = pat.shape
    dh, dw = h // nrows, w // ncols
    rois, boxes = [], []
    for i in range(nrows):
        for j in range(ncols):
            r0, r1 = i*dh, (i+1)*dh
            c0, c1 = j*dw, (j+1)*dw
            rois.append(pat[r0:r1, c0:c1])
            boxes.append((r0, r1, c0, c1))
    return rois, boxes

def roi_centers(boxes):
    ys = np.array([(r0+r1)/2 for (r0,r1,_,_) in boxes], float)
    xs = np.array([(c0+c1)/2 for (_,_,c0,c1) in boxes], float)
    return ys, xs

# ----------- Quiver plots ----------------------------------------------

def plot_quiver_with_grid(ref2d, dx, dy, nrows, ncols,
                          title, scale, outpath) -> None:
    _, boxes = subdivide_pattern(ref2d, nrows, ncols)
    centers = np.column_stack(roi_centers(boxes))
    mag = float(np.sqrt(dy*dy + dx*dx).mean())
    plt.figure(figsize=(5.8,4.8))
    ax = plt.gca()
    ax.imshow(ref2d, origin='upper')
    ax.quiver(centers[:,1], centers[:,0], dx*scale, dy*scale,
              angles='xy', scale_units='xy', scale=1, color='k', pivot='mid')
    ax.set_title(f"{title}\nmean|Δ| = {mag:.3f} px")
    # draw ROI grid
    h, w = ref2d.shape
    dh, dw = h//nrows, w//ncols
    for i in range(1, nrows): ax.axhline(i*dh, color='w', ls='--', lw=0.7, alpha=0.5)
    for j in range(1, ncols): ax.axvline(j*dw, color='w', ls='--', lw=0.7, alpha=0.5)
    plt.tight_layout()
    if outpath:
        _ensure_dir(outpath)
        plt.savefig(outpath, dpi=200)
    plt.show(); plt.close()

# ---------- PCC (SIM–SIM) on equal ROIs --------------------------------

def local_displacements(a2d, b2d, up=15, nrows=5, ncols=5):
    a_rois, boxes = subdivide_pattern(a2d, nrows, ncols)
    b_rois, _     = subdivide_pattern(b2d, nrows, ncols)
    disps = []
    for A, B in zip(a_rois, b_rois):
        shift, error, _ = phase_cross_correlation(A, B, upsample_factor=up)
        disps.append(shift)
    return np.asarray(disps, np.float32), boxes

def averaged_field_pcc(ref4d, mov4d, nrows=5, ncols=5, upsample=15):
    nroi = nrows*ncols
    Ny, Nx, _, _ = ref4d.shape
    acc_dy = np.zeros(nroi, float)
    acc_dx = np.zeros(nroi, float)
    cnt = 0
    for iy in range(Ny):
        for ix in range(Nx):
            disp, _ = local_displacements(mov4d[iy, ix], ref4d[iy, ix],
                                          up=upsample, nrows=nrows, ncols=ncols)
            acc_dy += disp[:, 0]; acc_dx += disp[:, 1]; cnt += 1
    dy = acc_dy/max(cnt,1); dx = acc_dx/max(cnt,1)
    return dy, dx

# ---------- Sparse LK with denser detection -----------------------------

@dataclass
class SparseParams:
    # detection
    max_corners: int = 4000
    quality: float = 0.002
    min_dist: int = 1
    block_size: int = 7
    use_harris: bool = False
    harris_k: float = 0.04
    # preproc
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int,int] = (8,8)
    blur_sigma: float = 0.8
    # LK
    lk_win: int = 25
    lk_levels: int = 5
    lk_eps: float = 1e-3
    lk_count: int = 30
    # misc
    dedup_cell: int = 1
    use_median: bool = False
    # ROI quota
    min_per_roi: int = 6

SPARSE_DEFAULTS = SparseParams()

def _prep_for_detection(img8: np.uint8, p: SparseParams) -> np.uint8:
    out = img8
    if p.clahe:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=p.clahe_grid)
        out = clahe.apply(out)
    if p.blur_sigma and p.blur_sigma > 0:
        out = cv2.GaussianBlur(out, ksize=(0,0), sigmaX=p.blur_sigma, sigmaY=p.blur_sigma)
    return out

def _detect_with_roi_quota(A8: np.uint8, nrows: int, ncols: int, p: SparseParams) -> np.ndarray:
    H, W = A8.shape
    dh, dw = H // nrows, W // ncols
    pts = []
    for i in range(nrows):
        for j in range(ncols):
            y0, y1 = i*dh, (i+1)*dh
            x0, x1 = j*dw, (j+1)*dw
            roi = A8[y0:y1, x0:x1]
            q = p.quality
            got = None
            for _ in range(5):
                c = cv2.goodFeaturesToTrack(
                    roi,
                    maxCorners=max(p.min_per_roi, 1),
                    qualityLevel=q,
                    minDistance=max(p.min_dist, 1),
                    blockSize=p.block_size,
                    useHarrisDetector=p.use_harris,
                    k=p.harris_k if p.use_harris else 0.04)
                if c is not None and len(c) >= p.min_per_roi:
                    got = c
                    break
                q *= 0.5
            if got is None:
                continue
            g = got.reshape(-1,2)
            g[:,0] += x0; g[:,1] += y0
            pts.append(g)
    return np.vstack(pts).astype(np.float32) if pts else np.empty((0,2), np.float32)

def sparse_field_on_unbinned_grid(
    ref_un: np.ndarray, mov_un: np.ndarray,
    nrows: int, ncols: int,
    binning: Optional[int] = None,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2,      # allow up to N-ROI hops for large rotations
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse LK on (optionally) binned images; features detected on EXP (ref).
    Tracks are kept iff endpoints stay within the unbinned ROI rectangle and hop ≤ max_hops.
    Returns centers (N,2)[y,x], vecs (N,2)[dy,dx] REF->MOV, counts per ROI.
    """
    # ROI grid from UNBINNED
    _, boxes = subdivide_pattern(ref_un, nrows, ncols)
    ys, xs = roi_centers(boxes)
    centers = np.column_stack([ys, xs]).astype(np.float32)
    dh = boxes[0][1] - boxes[0][0]
    dw = boxes[0][3] - boxes[0][2]
    Hc, Wc = dh*nrows, dw*ncols

    # binning
    s = 1 if (binning is None or binning == 1) else int(binning)
    A = bin_image(ref_un, binning)
    B = bin_image(mov_un, binning)
    A8, B8 = norm255(A), norm255(B)

    # preprocess + detect (ROI-quota first, then fallback global if empty)
    A8p = _prep_for_detection(A8, params)
    P0b = _detect_with_roi_quota(A8p, nrows, ncols, params)
    if P0b.size == 0:
        c1 = cv2.goodFeaturesToTrack(
            A8p, maxCorners=params.max_corners, qualityLevel=params.quality,
            minDistance=params.min_dist, blockSize=params.block_size,
            useHarrisDetector=params.use_harris, k=params.harris_k if params.use_harris else 0.04)
        c2 = cv2.goodFeaturesToTrack(
            255 - A8p, maxCorners=params.max_corners, qualityLevel=params.quality,
            minDistance=params.min_dist, blockSize=params.block_size,
            useHarrisDetector=params.use_harris, k=params.harris_k if params.use_harris else 0.04)
        if c1 is None and c2 is None:
            return centers, np.full((nrows*ncols,2), np.nan, np.float32), np.zeros(nrows*ncols, int)
        P0b = np.vstack([x.reshape(-1,2) for x in (c1, c2) if x is not None]).astype(np.float32)

    # de-duplicate
    seen, keep = set(), []
    for x, y in P0b:
        xi, yi = int(round(x)), int(round(y))
        key = (yi // max(params.dedup_cell,1), xi // max(params.dedup_cell,1))
        if key in seen: 
            continue
        seen.add(key); keep.append([x, y])
    if not keep:
        return centers, np.full((nrows*ncols,2), np.nan, np.float32), np.zeros(nrows*ncols, int)
    P0b = np.array(keep, np.float32).reshape(-1,1,2)

    # LK tracking (A->B)
    lk_params = dict(winSize=(params.lk_win, params.lk_win), maxLevel=params.lk_levels,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               params.lk_count, params.lk_eps))
    P1b, st, err = cv2.calcOpticalFlowPyrLK(A8, B8, P0b, None, **lk_params)
    good = (st.reshape(-1) == 1)
    if not np.any(good):
        return centers, np.full((nrows*ncols,2), np.nan, np.float32), np.zeros(nrows*ncols, int)

    P0b = P0b.reshape(-1,2)[good]
    P1b = P1b.reshape(-1,2)[good]

    # map to UNBINNED
    P0u = P0b * s
    P1u = P1b * s
    dxu = P1u[:,0] - P0u[:,0]
    dyu = P1u[:,1] - P0u[:,1]

    # keep tracks whose endpoints are inside the ROI rectangle
    def inside(P):
        x, y = P[:,0], P[:,1]
        return (x >= 0) & (x < Wc) & (y >= 0) & (y < Hc)

    both_in = inside(P0u) & inside(P1u)
    if not np.any(both_in):
        return centers, np.full((nrows*ncols,2), np.nan, np.float32), np.zeros(nrows*ncols, int)

    P0u = P0u[both_in]; P1u = P1u[both_in]
    dxu = dxu[both_in]; dyu = dyu[both_in]

    gx0 = (P0u[:,0] // dw).astype(int); gy0 = (P0u[:,1] // dh).astype(int)
    gx1 = (P1u[:,0] // dw).astype(int); gy1 = (P1u[:,1] // dh).astype(int)

    # hop filter (Chebyshev distance in ROI grid)
    dgy = np.abs(gy1 - gy0); dgx = np.abs(gx1 - gx0)
    hop_ok = (np.maximum(dgy, dgx) <= max_hops)
    if not np.any(hop_ok):
        return centers, np.full((nrows*ncols,2), np.nan, np.float32), np.zeros(nrows*ncols, int)

    P0u = P0u[hop_ok]; P1u = P1u[hop_ok]
    dxu = dxu[hop_ok]; dyu = dyu[hop_ok]
    rid0 = (gy0*ncols + gx0)[hop_ok]
    rid1 = (gy1*ncols + gx1)[hop_ok]

    # accumulate into every ROI touched by the endpoints
    N = nrows*ncols
    sums   = np.zeros((N, 2), np.float64)
    counts = np.zeros(N, int)
    for k in range(P0u.shape[0]):
        if rid0[k] == rid1[k]:
            rids = (int(rid0[k]),)
        else:
            rids = (int(rid0[k]), int(rid1[k]))
        for rid in rids:
            if 0 <= rid < N:
                sums[rid, 0] += dyu[k]
                sums[rid, 1] += dxu[k]
                counts[rid]  += 1

    vecs = np.full((N, 2), np.nan, np.float32)
    ok = counts > 0
    vecs[ok, 0] = (sums[ok, 0] / counts[ok]).astype(np.float32)
    vecs[ok, 1] = (sums[ok, 1] / counts[ok]).astype(np.float32)
    return centers, vecs, counts

def avg_field_sparseFeatures(
    stack_ref_un: np.ndarray, stack_mov_un: np.ndarray,
    nrows: int, ncols: int,
    binning: Optional[int] = None,
    noise_mov_sigma: Optional[float] = None,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2
):
    """Compute average sparse LK field per ROI over a nav grid."""
    # --- Coerce to plain NumPy 4D arrays
    ref4d = stack_ref_un.data if hasattr(stack_ref_un, "data") else np.asarray(stack_ref_un)
    mov4d = stack_mov_un.data if hasattr(stack_mov_un, "data") else np.asarray(stack_mov_un)
    ref4d = np.asarray(ref4d, np.float32)
    mov4d = np.asarray(mov4d, np.float32)
    if ref4d.ndim != 4 or mov4d.ndim != 4:
        raise ValueError(f"avg_field_sparseFeatures expects 4D inputs; got {ref4d.shape=} {mov4d.shape=}")

    Ny, Nx, _, _ = ref4d.shape
    centers = None
    sums = None
    counts_tot = None

    for iy in range(Ny):
        for ix in range(Nx):
            ref2d = ref4d[iy, ix]
            mov2d = mov4d[iy, ix]
            if noise_mov_sigma is not None and noise_mov_sigma > 0:
                rng = np.random.default_rng(0)
                mov2d = mov2d + rng.normal(0, noise_mov_sigma, size=mov2d.shape).astype(np.float32)

            c, v, cnt = sparse_field_on_unbinned_grid(
                ref2d, mov2d, nrows, ncols, binning=binning,
                params=params, max_hops=max_hops
            )
            if centers is None:
                centers = c
                sums = np.zeros_like(v, dtype=np.float64)
                counts_tot = np.zeros(v.shape[0], int)

            ok = ~np.isnan(v).any(axis=1)
            sums[ok]       += v[ok]
            counts_tot[ok] += cnt[ok]

    mean_vec = np.full_like(sums, np.nan, np.float32)
    ok = counts_tot > 0
    mean_vec[ok] = (sums[ok] / counts_tot[ok,None]).astype(np.float32)
    return centers, mean_vec, counts_tot

def expsim_fingerprint(
    xpat: np.ndarray, spat: np.ndarray, binning: Optional[int],
    nrows: int, ncols: int,
    noise_sim_sigma: Optional[float] = 0.05,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2
):
    """Return SIM->EXP average displacement field using sparse features."""
    # Force 4D arrays
    x4d = xpat.data if hasattr(xpat, "data") else np.asarray(xpat)
    s4d = spat.data if hasattr(spat, "data") else np.asarray(spat)
    x4d = np.asarray(x4d, np.float32)
    s4d = np.asarray(s4d, np.float32)
    if x4d.ndim != 4 or s4d.ndim != 4:
        raise ValueError(f"expsim_fingerprint expects 4D stacks; got {x4d.shape=} {s4d.shape=}")

    centers, v_exp2sim, counts = avg_field_sparseFeatures(
        stack_ref_un=x4d, stack_mov_un=s4d,
        nrows=nrows, ncols=ncols,
        binning=binning, noise_mov_sigma=noise_sim_sigma,
        params=params, max_hops=max_hops
    )
    v_sim2exp = -v_exp2sim
    return centers, v_sim2exp, counts

# ---------- Vector helpers & pair-plot -----------------------------------

def field_to_vector(vecs: np.ndarray) -> np.ndarray:
    v = np.empty(2*vecs.shape[0], dtype=np.float64)
    v[0::2] = vecs[:,0]
    v[1::2] = vecs[:,1]
    return v

def plot_pair_matches_side_by_side(
    ref_un: np.ndarray,
    mov_un: np.ndarray,
    nrows: int, ncols: int,
    binning: Optional[int] = None,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2,
    show_rejected: bool = False,
    outpath: Optional[str] = None,
    title: str = "Matched features (same color = same pair)"
) -> Tuple[int, int]:
    _, boxes = subdivide_pattern(ref_un, nrows, ncols)
    dh = boxes[0][1] - boxes[0][0]
    dw = boxes[0][3] - boxes[0][2]
    Hc, Wc = dh*nrows, dw*ncols

    s = 1 if (binning is None or binning == 1) else int(binning)
    A = bin_image(ref_un, binning)
    B = bin_image(mov_un, binning)
    Hb, Wb = A.shape
    A8, B8 = norm255(A), norm255(B)

    A8p = _prep_for_detection(A8, params)
    # global detect for visualization (keeps colors stable)
    c1 = cv2.goodFeaturesToTrack(
        A8p, maxCorners=params.max_corners, qualityLevel=params.quality,
        minDistance=params.min_dist, blockSize=params.block_size,
        useHarrisDetector=params.use_harris, k=params.harris_k if params.use_harris else 0.04)
    c2 = cv2.goodFeaturesToTrack(
        255 - A8p, maxCorners=params.max_corners, qualityLevel=params.quality,
        minDistance=params.min_dist, blockSize=params.block_size,
        useHarrisDetector=params.use_harris, k=params.harris_k if params.use_harris else 0.04)
    if c1 is None and c2 is None:
        print("[diag] no corners detected"); return 0, 0
    P0b = np.vstack([x.reshape(-1,2) for x in (c1, c2) if x is not None]).astype(np.float32)

    # dedup
    seen, keep = set(), []
    for idx, (x, y) in enumerate(P0b):
        xi, yi = int(round(x)), int(round(y))
        key = (yi // max(params.dedup_cell,1), xi // max(params.dedup_cell,1))
        if key in seen: continue
        seen.add(key); keep.append(idx)
    if not keep:
        return 0, 0
    P0b = P0b[keep].reshape(-1,1,2)

    # LK
    lk_params = dict(winSize=(params.lk_win, params.lk_win), maxLevel=params.lk_levels,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,
                               params.lk_count, params.lk_eps))
    P1b, st, err = cv2.calcOpticalFlowPyrLK(A8, B8, P0b, None, **lk_params)
    good = (st.reshape(-1) == 1)
    if not np.any(good):
        print("[diag] no successful tracks"); return 0, 0
    P0b = P0b.reshape(-1,2)[good]
    P1b = P1b.reshape(-1,2)[good]

    # filter in UNBINNED coords + hop limit
    P0u = P0b * s; P1u = P1b * s
    def inside(P):
        x, y = P[:,0], P[:,1]
        return (x >= 0) & (x < Wc) & (y >= 0) & (y < Hc)
    both_in = inside(P0u) & inside(P1u)
    if not np.any(both_in):
        print("[diag] all tracks outside grid"); return 0, int(good.sum())

    P0b, P1b = P0b[both_in], P1b[both_in]
    P0u, P1u = P0u[both_in], P1u[both_in]

    gx0 = (P0u[:,0] // dw).astype(int); gy0 = (P0u[:,1] // dh).astype(int)
    gx1 = (P1u[:,0] // dw).astype(int); gy1 = (P1u[:,1] // dh).astype(int)
    dgy = np.abs(gy1 - gy0); dgx = np.abs(gx1 - gx0)
    hop_ok = (np.maximum(dgy, dgx) <= max_hops)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axL.imshow(A, cmap='gray', origin='upper'); axL.set_title("EXP (binned)")
    axR.imshow(B, cmap='gray', origin='upper'); axR.set_title("SIM (binned)")
    for ax in (axL, axR):
        for i in range(1, nrows): ax.axhline((i*dh)/s, color='w', ls='--', lw=0.8, alpha=0.6)
        for j in range(1, ncols): ax.axvline((j*dw)/s, color='w', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlim(0, Wb); ax.set_ylim(Hb, 0)

    idx_acc = np.where(hop_ok)[0]
    idx_rej = np.where(~hop_ok)[0]
    n_acc = len(idx_acc)
    cmap = cm.get_cmap('nipy_spectral', max(n_acc, 1))

    for j, k in enumerate(idx_acc):
        color = cmap(j)
        axL.plot(P0b[k,0], P0b[k,1], 'o', ms=4, mec='k', mfc=color, alpha=0.95)
        axR.plot(P1b[k,0], P1b[k,1], 'o', ms=4, mec='k', mfc=color, alpha=0.95)

    if show_rejected and len(idx_rej) > 0:
        axL.plot(P0b[idx_rej,0], P0b[idx_rej,1], 'o', ms=3, mfc='none', mec='0.7', alpha=0.6)
        axR.plot(P1b[idx_rej,0], P1b[idx_rej,1], 'o', ms=3, mfc='none', mec='0.7', alpha=0.6)

    fig.suptitle(f"{title}\naccepted: {n_acc} / {P0b.shape[0]} (bin={s}, hops ≤ {max_hops})")
    if outpath:
        _ensure_dir(outpath); fig.savefig(outpath, dpi=200)
    plt.show(); plt.close(fig)
    return n_acc, int(P0b.shape[0])

# ---------- Detector helpers (generic) -----------------------------------

def apply_det_update(
    det,
    keys: Sequence[str],
    dtheta: Sequence[float],
    xmap=None,
    update_pc: bool = True,
):
    """
    Update detector params in-place.

    If update_pc=True and any tilt-like parameter changes, refresh the per-point
    PC field using kikuchipy's extrapolate_pc based on the spatial coordinates in xmap.
    If update_pc=False, only the scalar detector geometry parameters are changed and
    the existing PC field is left untouched.
    """
    # Update numeric detector attributes
    for k, delta in zip(keys, dtheta):
        if hasattr(det, k):
            setattr(det, k, getattr(det, k) + float(delta))
        else:
            try:
                det[k] = det[k] + float(delta)
            except Exception as e:
                raise AttributeError(f"Don't know how to set detector key '{k}'") from e

    if not update_pc:
        return det

    tilt_keys = {"sample_tilt", "azimuthal", "tilt", "twist"}
    if xmap is None or not any(k in tilt_keys for k in keys):
        return det

    if not hasattr(xmap, "x") or not hasattr(xmap, "y"):
        raise ValueError("xmap must provide x and y coordinates for PC extrapolation.")
    if not hasattr(xmap, "dx") or not hasattr(xmap, "dy"):
        raise ValueError("xmap must provide dx and dy for PC extrapolation.")

    x = np.asarray(xmap.x, dtype=float).ravel()
    y = np.asarray(xmap.y, dtype=float).ravel()

    x0 = np.min(x)
    y0 = np.min(y)

    ix = np.rint((x - x0) / float(xmap.dx)).astype(int)
    iy = np.rint((y - y0) / float(xmap.dy)).astype(int)
    pc_indices = np.vstack([iy, ix])

    nav_shape = getattr(det, "navigation_shape", None)
    if nav_shape is None:
        # fall back to xmap shape if needed
        if hasattr(xmap, "shape") and len(xmap.shape) > 0:
            nav_shape = tuple(xmap.shape)
        else:
            nav_shape = (x.size,)

    det_new = det.extrapolate_pc(
        pc_indices=pc_indices,
        navigation_shape=nav_shape,
        step_sizes=(float(xmap.dy), float(xmap.dx)),
    )

    # copy refreshed PCs back into the existing detector object
    if hasattr(det, "pc") and hasattr(det_new, "pc"):
        det.pc = det_new.pc

    for attr in ("pcx", "pcy", "pcz", "pc_all", "pcx_all", "pcy_all", "pcz_all"):
        if hasattr(det, attr) and hasattr(det_new, attr):
            setattr(det, attr, getattr(det_new, attr))

    return det

# ---------- Parameter analysis (SIM–SIM; PCC) ----------------------------

def parameter_analysis(
    oris, master_pattern, energy, det0, pname: str,
    keys: Sequence[str], steps: Sequence[float],
    nrows: int, ncols: int,
    out_idx: Tuple[int,int], SavePlots: bool = True,
    arrow_scale: float = 20.0,
    xmap=None,
    update_pc=False,
):
    sim0 = simulate_fn(master_pattern, oris, det0, energy)
    if sim0.ndim == 3:
        raise ValueError("parameter_analysis expects 4D sim (Ny,Nx,H,W); got (N,H,W). Provide grid-shaped orientations or reshape.")
    j0, i0 = out_idx
    im_disp = sim0[j0,i0]

    _, boxes = subdivide_pattern(im_disp, nrows, ncols)
    centers = np.column_stack(roi_centers(boxes)).astype(np.float32)

    Jcols = []
    for k, step in zip(keys, steps):
        det1 = det0.deepcopy()
        apply_det_update(det1, [k], [step], xmap=xmap, update_pc=update_pc)
        sim1 = simulate_fn(master_pattern, oris, det1, energy)

        dy, dx = averaged_field_pcc(sim0, sim1, nrows, ncols, upsample=15)
        vecs = np.column_stack((np.ravel(dy), np.ravel(dx)))
        sens_vec = field_to_vector(vecs) / float(step)
        Jcols.append(sens_vec)

        if SavePlots:
            plot_quiver_with_grid(im_disp, dx, dy, nrows, ncols,
                                  f"Sensitivity: {k} (step={step})",
                                  arrow_scale, os.path.join(pname, f"sensitivity_{k}.png"))
            make_flash_gif(sim0[j0,i0], sim1[j0,i0],
                           outpath=os.path.join(pname, f"sim-sim_{k}.gif"),
                           duration=0.5, n_flashes=4)

    J = np.column_stack(Jcols)

    mask = ~np.any(~np.isfinite(J), axis=1)
    Jm = J[mask]
    norms = np.linalg.norm(Jm, axis=0) + 1e-12
    coupling = (Jm / norms).T @ (Jm / norms)

    plt.figure(figsize=(4.5,4))
    plt.imshow(coupling, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.yticks(range(len(keys)), keys)
    plt.title("Parameter coupling (cosine)")
    plt.colorbar()
    _ensure_dir(os.path.join(pname, "dummy.png"))
    plt.savefig(os.path.join(pname, "coupling_matrix.png"), dpi=200, bbox_inches="tight")
    plt.show(); plt.close()

    return centers, J, coupling

# ---------- One-step geometry refinement (fingerprint J) -----------------

def one_step_geometry_refinement(
    xpat, master_pattern, energy, oris_grid, det0, pname: str,
    keys: Sequence[str], steps: Sequence[float],
    binning: Optional[int], nrows: int, ncols: int,
    out_idx: Tuple[int,int], SavePlots: bool = True,
    damping: float = 1e-3,
    noise_sim_sigma: Optional[float] = 0.05,
    arrow_scale: float = 20.0,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2,
    jacobian_method: str = "fingerprint",   # "fingerprint" or "pcc"
    xmap=None,
    update_pc=False,
) -> Dict[str, object]:
    """
    One-step refinement using chosen Jacobian:
      - jacobian_method="fingerprint": build J via EXPSIM fingerprint (sparse LK, central difference)
      - jacobian_method="pcc":        build J via SIM–SIM PCC (finite difference)
    Residual b is always SIM(det) → EXP via sparse features.
    """
    _ensure_dir(os.path.join(pname, "x"))
    # coerce experimental stack to 4D numpy
    xpat_4d = xpat.data if hasattr(xpat, "data") else np.asarray(xpat)
    xpat_4d = np.asarray(xpat_4d, np.float32)
    if xpat_4d.ndim != 4:
        raise ValueError("xpat must be 4D (Ny,Nx,H,W)")

    # --- Build Jacobian and residual
    if jacobian_method.lower() in ("fingerprint", "fp", "expsim", "sparse"):
        # base sim at det0
        spat0 = simulate_fn(master_pattern, oris_grid, det0, energy)
        if spat0.ndim == 3:
            Ny, Nx, _, _ = xpat_4d.shape
            spat0 = _N_to_nav(_stack_to_NHW(spat0), (Ny, Nx))

        # Residual b from SIM(det0) -> EXP
        centers, v_before, roi_counts = expsim_fingerprint(
            xpat_4d, spat0, binning, nrows, ncols, noise_sim_sigma,
            params=params, max_hops=max_hops
        )
        b_full = field_to_vector(v_before)

        # Jacobian columns by central differences (resimulate ± step)
        j0, i0 = out_idx
        bg = xpat_4d[j0, i0]
        Jcols = []
        for k, step in zip(keys, steps):
            det_p = det0.deepcopy(); apply_det_update(det_p, [k], [step], xmap=xmap, update_pc=update_pc)
            spat_p = simulate_fn(master_pattern, oris_grid, det_p, energy)
            if spat_p.ndim == 3:
                Ny, Nx, _, _ = xpat_4d.shape
                spat_p = _N_to_nav(_stack_to_NHW(spat_p), (Ny, Nx))

            det_m = det0.deepcopy(); apply_det_update(det_m, [k], [-step], xmap=xmap, update_pc=update_pc)
            spat_m = simulate_fn(master_pattern, oris_grid, det_m, energy)
            if spat_m.ndim == 3:
                Ny, Nx, _, _ = xpat_4d.shape
                spat_m = _N_to_nav(_stack_to_NHW(spat_m), (Ny, Nx))

            _, v_p, _ = expsim_fingerprint(xpat_4d, spat_p, binning, nrows, ncols, noise_sim_sigma,
                                           params=params, max_hops=max_hops)
            _, v_m, _ = expsim_fingerprint(xpat_4d, spat_m, binning, nrows, ncols, noise_sim_sigma,
                                           params=params, max_hops=max_hops)
            col = (field_to_vector(v_p) - field_to_vector(v_m)) / (2.0 * float(step))
            Jcols.append(col)

            if SavePlots:
                disp_for_plot = (v_p - v_m) * 0.5  # show finite-step diff
                plot_quiver_with_grid(
                    bg, disp_for_plot[:,1], disp_for_plot[:,0], nrows, ncols,
                    f"Sensitivity (fingerprint): {k} (±{step})",
                    arrow_scale, os.path.join(pname, f"sensitivity_fingerprint_{k}.png")
                )

        J = np.column_stack(Jcols)
        # Column coupling (diagnostic)
        mask = ~np.any(~np.isfinite(J), axis=1)
        Jm = J[mask]
        norms = np.linalg.norm(Jm, axis=0) + 1e-12
        coupling = (Jm / norms).T @ (Jm / norms)
        solve_sign = -1.0  # rhs = -J^T b  (SIM->EXP convention)

    elif jacobian_method.lower() in ("pcc", "sim-sim"):
        # PCC uses SIM–SIM to build J; needs 4D sim at det0
        spat0 = simulate_fn(master_pattern, oris_grid, det0, energy)
        if spat0.ndim == 3:
            Ny, Nx, _, _ = xpat_4d.shape
            spat0 = _N_to_nav(_stack_to_NHW(spat0), (Ny, Nx))

        centers, J, coupling = parameter_analysis(
            oris_grid, master_pattern, energy, det0, pname, keys, steps, nrows, ncols, out_idx,
            SavePlots=True, arrow_scale=arrow_scale,  xmap=xmap
        )
        # Residual b from SIM(det0) -> EXP
        _, v_before, roi_counts = expsim_fingerprint(
            xpat_4d, spat0, binning, nrows, ncols, noise_sim_sigma,
            params=params, max_hops=max_hops
        )
        b_full = field_to_vector(v_before)
        solve_sign = +1.0  # rhs = +J^T b
    else:
        raise ValueError("jacobian_method must be 'fingerprint' or 'pcc'")

    # --- Row weights from ROI track counts (clipped)
    if np.any(roi_counts > 0):
        c90 = np.percentile(roi_counts[roi_counts > 0], 90)
    else:
        c90 = 1.0
    w_cnt = np.minimum(roi_counts.astype(float), c90) / max(c90, 1.0)
    Wsqrt = np.repeat(np.sqrt(np.maximum(w_cnt, 1e-6)), 2)  # (2*Nroi,)

    # --- Mask rows with NaNs/zeros
    finite_J_rows = ~np.any(~np.isfinite(J), axis=1)
    finite_b_rows =  np.isfinite(b_full)
    good_rows = finite_J_rows & finite_b_rows & (Wsqrt > 0)

    if good_rows.sum() < len(keys):
        print(f"[warn] Only {good_rows.sum()} finite/weighted rows for {len(keys)} parameters.")

    Jp, bp, Wp = J[good_rows], b_full[good_rows], Wsqrt[good_rows]

    # --- Weighted, column-normalized LM solve
    Jw = Jp * Wp[:, None]; bw = bp * Wp
    col_norms = np.linalg.norm(Jw, axis=0)
    S_inv = 1.0 / np.maximum(col_norms, 1e-12)
    Jws = Jw * S_inv[None, :]
    JTJ = Jws.T @ Jws
    lam = damping * (np.trace(JTJ) / max(len(keys), 1) if JTJ.size else damping)
    A = JTJ + lam * np.eye(len(keys))
    rhs = solve_sign * (Jws.T @ bw)
    dtheta_scaled, *_ = lstsq(A, rhs)
    dtheta = dtheta_scaled * S_inv

    # --- Diagnostics
    print(f"[solve/{jacobian_method}] column norms (after row weights):")
    for k, cn in zip(keys, col_norms):
        print(f"  {k:>12s}: {cn:.4e}")
    svals = np.linalg.svd(Jw, compute_uv=False)
    if svals.size:
        cond = (svals[0] / max(svals[-1], 1e-12))
        print(f"[solve/{jacobian_method}] SVD cond ≈ {cond:.3e}, λ = {lam:.3e}")

    # --- Update detector and compute before/after fields
    det1 = det0.deepcopy()
    apply_det_update(det1, keys, dtheta, xmap=xmap, update_pc=update_pc)

    # Residual fields (before/after) for plots
    _, v_before, _ = expsim_fingerprint(xpat_4d, spat0, binning, nrows, ncols,
                                        noise_sim_sigma, params=params, max_hops=max_hops)
    spat1 = simulate_fn(master_pattern, oris_grid, det1, energy)
    if spat1.ndim == 3:
        Ny, Nx, _, _ = xpat_4d.shape
        spat1 = _N_to_nav(_stack_to_NHW(spat1), (Ny, Nx))
    _, v_after,  _ = expsim_fingerprint(xpat_4d, spat1, binning, nrows, ncols,
                                        noise_sim_sigma, params=params, max_hops=max_hops)

    # --- Plots and GIFs
    j0, i0 = out_idx
    im_disp = xpat_4d[j0, i0]
    plot_quiver_with_grid(im_disp, v_before[:,1], v_before[:,0], nrows, ncols,
                          "Avg displacement (before)", arrow_scale,
                          os.path.join(pname,"avg_field_before.png"))
    plot_quiver_with_grid(im_disp, v_after[:,1], v_after[:,0], nrows, ncols,
                          "Avg displacement (after)", arrow_scale,
                          os.path.join(pname,"avg_field_after.png"))
    plot_pair_matches_side_by_side(xpat_4d[j0,i0], spat0[j0,i0], nrows, ncols, 
            binning, params, max_hops, 
            outpath=os.path.join(pname,"feature_pairs.png"))

    # GIFs comparing EXP to SIM (before/after)
    make_flash_gif(im_disp, spat0[j0,i0], os.path.join(pname,"exp-vs-sim_before.gif"),
                   duration=1.0, n_flashes=4)
    make_flash_gif(im_disp, spat1[j0,i0], os.path.join(pname,"exp-vs-sim_after.gif"),
                   duration=1.0, n_flashes=4)

    mag_before = float(np.nanmean(np.hypot(v_before[:,1], v_before[:,0])))
    mag_after  = float(np.nanmean(np.hypot(v_after[:,1],  v_after[:,0])))

    return dict(
        centers=centers,
        J=J, coupling=coupling,
        v_before=v_before, v_after=v_after,
        dtheta=np.array(dtheta),
        keys=list(keys),
        mag_before=mag_before,
        mag_after=mag_after,
        det_refined=det1,
        good_rows=int(good_rows.sum()),
        total_rows=int(J.shape[0])
    )

# ======================= SIMPLE ITERATIVE OPTIMIZER =======================

def _ncc_batch(a: np.ndarray, b: np.ndarray) -> float:
    """Average normalized cross-correlation over batch (N,H,W)."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a - a.mean(axis=(1,2), keepdims=True)
    b = b - b.mean(axis=(1,2), keepdims=True)
    denom = (np.linalg.norm(a, axis=(1,2)) * np.linalg.norm(b, axis=(1,2)) + 1e-12)
    ncc_each = (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).sum(axis=1) / denom
    return float(np.mean(ncc_each))

def optimize_geometry_and_orientations(
    xpat,                      # kikuchipy.Patterns (experimental) – 4D (Ny,Nx,H,W)
    xmap,                      # orientation map (e.g., xmap_s); must provide .orientations
    det0,                      # initial detector
    master_pattern,            # for simulation
    keys: Sequence[str],       # detector parameter names to refine
    steps: Sequence[float],    # finite-difference steps for geometry sensitivity
    energy: float = 25.0,
    binning: Optional[int] = None,
    nrows: int = 8,
    ncols: int = 8,
    out_idx: Tuple[int,int] = (0,0),
    pname: str = "iter_refine",
    batch_indices: Optional[Sequence[int]] = None,  # on flattened (row-major) order
    max_iters: int = 8,
    tol_delta_ncc: float = 5e-4,
    jacobian_method: str = "fingerprint",
    SavePlots: bool = True,
    update_pc: bool = True, #choose whether to update detector pc values based on xmap positions when detector tilts change
    # orientation refinement kwargs:
    method: str = "LN_NELDERMEAD",
    trust_region: Sequence[float] = (3,3,3),
    pseudo_symmetry_ops: Optional[Any] = None,
    rtol: float = 1e-4,
    # fingerprint/expsim kwargs
    noise_sim_sigma: Optional[float] = 0.05,
    params: SparseParams = SPARSE_DEFAULTS,
    max_hops: int = 2,
    # solver/damping
    damping: float = 1e-3,
    verbose: bool = True,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """
    Iterate: propose a geometry update (det_g), then ALWAYS run orientation refinement
    under that geometry, and compute NCC. Accept the step only if the post-orientation
    NCC improves over the previous run. Otherwise reject the geometry (and orientations).
    Also saves a GIF of EXP vs SIM after the orientation refinement each iteration.
    """
    # Shapes & data
    Ny, Nx, H, W = _nav_shape_from_xpat(xpat)
    exp_4d = xpat.data if hasattr(xpat, "data") else np.asarray(xpat, np.float32)
    exp_NHW = _stack_to_NHW(exp_4d)
    j0, i0 = out_idx  # for GIFs

    # Orientations grid: use exactly the nav shape of xpat
    oris_full = xmap.orientations
    try:
        oris_grid = oris_full.reshape(Ny, Nx)
    except Exception:
        oris_grid = oris_full

    # initial state
    det = det0
    best_det = det0
    best_xmap = xmap

    # NCC with master pattern (simulate full, then subset/flatten for NCC)
    spat0 = simulate_fn(master_pattern, oris_grid, det, energy)
    if spat0.ndim == 3:
        spat0 = _N_to_nav(_stack_to_NHW(spat0), (Ny, Nx))
    sim_NHW = _stack_to_NHW(spat0)
    sim_batch = _select_by_idx(sim_NHW, batch_indices)
    exp_batch = _select_by_idx(exp_NHW, batch_indices)

    ncc0 = _ncc_batch(exp_batch, sim_batch)
    best_ncc = prev_ncc = ncc0
    if verbose:
        print(f"[init] NCC = {ncc0:.6f}")

    log: List[Dict[str, Any]] = []

    for it in range(1, max_iters+1):
        it_rec: Dict[str, Any] = dict(iter=it, ncc_before=prev_ncc)
        t0 = time.time()
        improved = False

        # 1) ---- Geometry proposal (det_g) ----
        if update_pc is True:
            xmap_push=xmap
        else:
            xmap_push=None

        result = one_step_geometry_refinement(
            xpat=exp_4d,
            master_pattern=master_pattern,
            energy=energy,
            oris_grid=oris_grid,
            det0=det,
            pname=os.path.join(pname, f"iter{it:02d}"),
            keys=keys, steps=steps,
            binning=binning, nrows=nrows, ncols=ncols,
            out_idx=(j0, i0), SavePlots=SavePlots,
            damping=damping,
            noise_sim_sigma=noise_sim_sigma,
            params=params, max_hops=max_hops,
            jacobian_method=jacobian_method,
            xmap=xmap_push,   # pass xmap so PCs auto-refresh if tilts changed
        )
        det_g = result.get("det_refined", det)
        # --- Geometry-step diagnostics from one_step_geometry_refinement
        good_rows = result.get("good_rows", None)
        total_rows = result.get("total_rows", None)
        mag_before = result.get("mag_before", None)
        mag_after  = result.get("mag_after", None)

        it_rec.update(
            geom_good_rows=good_rows,
            geom_total_rows=total_rows,
            geom_good_frac=(float(good_rows) / float(total_rows)) if (good_rows is not None and total_rows) else None,
            geom_mag_before=mag_before,
            geom_mag_after=mag_after,
        )
        # --- Normalize displacement magnitudes by detector width (in the same pixel space)
        W_eff = float(W) / float(binning) if (binning not in (None, 0, 1)) else float(W)
        it_rec["geom_width_px_eff"] = W_eff
        it_rec["geom_mag_before_fracW"] = (float(mag_before) / W_eff) if (mag_before is not None) else None
        it_rec["geom_mag_after_fracW"]  = (float(mag_after)  / W_eff) if (mag_after is not None) else None

        # (Optional) purely diagnostic NCC after geometry only (no acceptance)
        spat_g_only = simulate_fn(master_pattern, oris_grid, det_g, energy)
        if spat_g_only.ndim == 3:
            spat_g_only = _N_to_nav(_stack_to_NHW(spat_g_only), (Ny, Nx))
        sim_g_only = _select_by_idx(_stack_to_NHW(spat_g_only), batch_indices)
        ncc_geom_only = _ncc_batch(exp_batch, sim_g_only)
        it_rec.update(geom_ncc_pre_orient=ncc_geom_only)

        # 2) ---- Orientation refinement under candidate geometry (det_g) ----
        xmap_ref = xpat.refine_orientation(
            xmap=xmap,
            detector=det_g,
            master_pattern=master_pattern,
            energy=energy,
            method=method,
            trust_region=list(trust_region),
            pseudo_symmetry_ops=pseudo_symmetry_ops,
            rtol=rtol,
        )
        oris_o = xmap_ref.orientations
        try:
            oris_grid_o = oris_o.reshape(Ny, Nx)
        except Exception:
            oris_grid_o = oris_o

        # 3) ---- Evaluate NCC AFTER orientation refinement (decision metric) ----
        spat_after = simulate_fn(master_pattern, oris_grid_o, det_g, energy)
        if spat_after.ndim == 3:
            spat_after = _N_to_nav(_stack_to_NHW(spat_after), (Ny, Nx))
        sim_after = _select_by_idx(_stack_to_NHW(spat_after), batch_indices)
        ncc_after = _ncc_batch(exp_batch, sim_after)

        # --- NEW: GIF after orientation refinement
        try:
            if SavePlots:
                im_exp = exp_4d[j0, i0]
                make_flash_gif(im_exp, spat_after[j0, i0],
                               os.path.join(pname, f"iter{it:02d}", "exp-vs-sim_after_orient.gif"),
                               duration=0.5, n_flashes=4)
        except Exception as e:
            print(f"[warn] could not save exp-vs-sim_after_orient.gif: {e}")

        it_rec.update(
            orient_ncc=ncc_after,
            orient_scores_mean=(getattr(xmap_ref, "scores", None).mean() if hasattr(xmap_ref, "scores") else None)
        )

        # 4) ---- Accept/reject the entire step based on post-orientation NCC ----
        if ncc_after > prev_ncc + 1e-12:
            # accept geometry + orientations
            det = det_g
            xmap = xmap_ref
            oris_grid = oris_grid_o
            prev_ncc = ncc_after
            improved = True
            if verbose:
                print(f"[iter {it:02d}] ACCEPT: NCC {ncc_after:.6f} (Δ={ncc_after-best_ncc:.6f} vs best)")
        else:
            # reject geometry (and orientations computed under det_g)
            if verbose:
                print(f"[iter {it:02d}] REJECT: NCC {ncc_after:.6f} (<= {prev_ncc:.6f})")

        if prev_ncc > best_ncc:
            best_ncc = prev_ncc; best_det = det; best_xmap = xmap

        it_rec.update(ncc_after=prev_ncc, improved=improved, elapsed_s=time.time()-t0)
        log.append(it_rec)

        # stopping rules
        if not improved:
            if verbose: print(f"[iter {it:02d}] no improvement; stopping.")
            break
        if it_rec["ncc_after"] - it_rec["ncc_before"] < tol_delta_ncc:
            if verbose: print(f"[iter {it:02d}] improvement {it_rec['ncc_after']-it_rec['ncc_before']:.3e} < tol {tol_delta_ncc:.3e}; stopping.")
            break

    if verbose:
        print(f"[done] best NCC = {best_ncc:.6f}")

    return best_det, best_xmap, log
