import math
import numpy as np
import imageio.v3 as iio
import kikuchipy as kp
from orix.crystal_map import CrystalMap
import h5py
from kikuchipy.signals.util._master_pattern import _project_single_pattern_from_master_pattern
from kikuchipy.indexing._refinement._refinement import _get_master_pattern_data
import EBSD_extra_functions_numba as xfn_nb

def make_flash_gif(pattern1, pattern2, outpath, duration=0.4, n_flashes=2):
    """
    Create a simple 'flashing' GIF alternating between two patterns.

    Parameters
    ----------
    pattern1, pattern2 : ndarray
        2D arrays (grayscale patterns) of the same shape.
    outpath : str or pathlib.Path
        Path to the output GIF file.
    duration : float, optional
        Duration of each frame in seconds (default 0.4).
    n_flashes : int, optional
        Number of flash cycles (pattern1 -> pattern2) to include (default 2).

    Returns
    -------
    outpath : pathlib.Path
        Path to the saved GIF.
    """

    # normalize 0..1
    def _norm01(a):
        a = np.asarray(a, dtype=np.float32)
        amin, amax = np.min(a), np.max(a)
        if amax > amin:
            return (a - amin) / (amax - amin)
        else:
            return np.zeros_like(a, dtype=np.float32)

    p1 = _norm01(pattern1)
    p2 = _norm01(pattern2)

    # convert to 8-bit grayscale
    p1_u8 = (p1 * 255).astype(np.uint8)
    p2_u8 = (p2 * 255).astype(np.uint8)

    frames = []
    for _ in range(int(n_flashes) if n_flashes else 1):
        frames.extend([p1_u8, p2_u8])

    # Convert seconds -> milliseconds and write as GIF
    dur_ms = int(round(float(duration) * 1000))
    if not str(outpath).lower().endswith(".gif"):
        outpath = str(outpath) + ".gif"
    iio.imwrite(outpath, frames, loop=0, duration=dur_ms, format="GIF")
    return outpath

def make_circular_signal_mask(h, w, radius_px=None, invert=True):
    """
    Returns a boolean mask of shape (h, w).
    True = masked (excluded) pixels, False = included.
    If radius_px is None, uses a circle that fits the frame.
    """
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y, x = np.ogrid[:h, :w]
    r = radius_px if radius_px is not None else min(h, w) / 2.0
    inside = (x - cx) ** 2 + (y - cy) ** 2 <= (r ** 2)
    return ~inside if invert else inside

def EBSD_subset(xpat, det, xmap, n_points=None, indices=None, indices_shape=None):
    """
    Subsample EBSD patterns, detector PCs, and CrystalMap consistently.

    Parameters
    ----------
    xpat : kp.signals.EBSD
        EBSD pattern dataset with shape (Ny, Nx, py, px)
    det : kp.detectors.EBSDDetector
        Detector with per-point or global PCs
    xmap : CrystalMap
        Crystal map corresponding to xpat
    n_points : int, optional
        Approximate total number of points for a regular rectangular subgrid.
        Uses one common stride multiplier m so sampled points lie on the
        original grid with step sizes (m*dy, m*dx).
    indices : array-like of int, optional
        Explicit flat navigation indices in the FULL/original map
    indices_shape : tuple[int, int], optional
        Optional output navigation shape for explicit indices

    Returns
    -------
    xpatS : kp.signals.EBSD
    detS : kp.detectors.EBSDDetector
    xmapS : CrystalMap
    pc_indices : ndarray of shape (2, K)
        Full-map pixel coordinates of the subset points, in (row, col) order.
        This can be used directly in detS.extrapolate_pc(...).
    """

    if (n_points is None) == (indices is None):
        raise ValueError("Provide exactly one of n_points or indices.")

    if xpat.data.ndim != 4:
        raise ValueError("xpat.data must have shape (Ny, Nx, py, px).")

    Ny, Nx, py, px = xpat.data.shape
    N = Ny * Nx

    if xmap.size != N:
        raise ValueError(f"xmap size ({xmap.size}) != xpat nav size ({N}).")

    def _choose_regular_grid_from_total_points(Ny, Nx, n_points_total):
        if n_points_total < 1:
            raise ValueError("n_points must be >= 1.")

        best = None
        for m in range(1, max(Ny, Nx) + 1):
            ys = np.arange(0, Ny, m, dtype=int)
            xs = np.arange(0, Nx, m, dtype=int)
            n = len(ys) * len(xs)
            err = abs(n - n_points_total)

            # Preference:
            # 1) smallest error
            # 2) prefer n >= requested over n < requested
            # 3) smaller m
            candidate = (err, n < n_points_total, m, ys, xs)

            if best is None or candidate[:3] < best[:3]:
                best = candidate

            if n == n_points_total:
                break

        _, _, m_best, ys_best, xs_best = best
        return ys_best, xs_best, m_best

    def _subset_detector_nav(arr, idx_arr, Ny, Nx, nav_shape):
        a = np.asarray(arr)

        if a.ndim == 0:
            sel = np.full(np.prod(nav_shape), a)

        elif a.ndim == 1:
            if a.size == 1:
                sel = np.full(np.prod(nav_shape), a.item())
            elif a.size == Ny * Nx:
                sel = a[idx_arr]
            else:
                raise ValueError(
                    f"1D detector array length {a.size} is invalid; expected 1 or {Ny*Nx}."
                )

        elif a.ndim == 2:
            if a.shape != (Ny, Nx):
                raise ValueError(
                    f"2D detector array shape {a.shape} is invalid; expected {(Ny, Nx)}."
                )
            iy = idx_arr // Nx
            ix = idx_arr % Nx
            sel = a[iy, ix]

        else:
            raise ValueError("Detector PC arrays must be scalar, 1D, or 2D.")

        return sel.reshape(nav_shape, order="C")

    # -------------------------
    # Case A: regular subgrid from total n_points
    # -------------------------
    if n_points is not None:
        ys, xs, m = _choose_regular_grid_from_total_points(Ny, Nx, int(n_points))

        iy_grid, ix_grid = np.meshgrid(ys, xs, indexing="ij")
        idx_arr = (iy_grid * Nx + ix_grid).ravel(order="C")
        nav_shape = (len(ys), len(xs))

        x_full = np.asarray(xmap.x).reshape(Ny, Nx)
        y_full = np.asarray(xmap.y).reshape(Ny, Nx)
        x_sub = x_full[np.ix_(ys, xs)]
        y_sub = y_full[np.ix_(ys, xs)]

    # -------------------------
    # Case B: explicit indices
    # -------------------------
    else:
        idx_arr = np.asarray(indices, dtype=int).ravel()

        if idx_arr.size == 0:
            raise ValueError("indices is empty.")
        if np.any(idx_arr < 0) or np.any(idx_arr >= N):
            raise IndexError("indices contain out-of-bounds values.")

        if indices_shape is not None:
            if np.prod(indices_shape) != idx_arr.size:
                raise ValueError("indices_shape does not match number of indices.")
            nav_shape = tuple(indices_shape)
        else:
            nav_shape = (idx_arr.size,)

        if len(nav_shape) == 2:
            # Artificial coordinates so CrystalMap.shape matches nav_shape
            ys_fake = np.arange(nav_shape[0], dtype=float) * xmap.dy
            xs_fake = np.arange(nav_shape[1], dtype=float) * xmap.dx
            x_grid_fake, y_grid_fake = np.meshgrid(xs_fake, ys_fake, indexing="xy")
            x_sub = x_grid_fake
            y_sub = y_grid_fake
        else:
            # 1D subset: original coordinates are fine
            x_sub = np.asarray(xmap.x)[idx_arr].reshape(nav_shape, order="C")
            y_sub = np.asarray(xmap.y)[idx_arr].reshape(nav_shape, order="C")
    
    # -------------------------
    # pc_indices for extrapolate_pc
    # -------------------------
    # These are FULL-map coordinates of the subset PCs, in (row, col) order
    iy_full = idx_arr // Nx
    ix_full = idx_arr % Nx
    pc_indices = np.vstack([iy_full, ix_full])   # shape (2, K)

    # -------------------------
    # Subset xpat
    # -------------------------
    x1d = xpat.deepcopy()
    x1d.unfold_navigation_space()
    data1d = x1d.data
    data_sel = data1d[idx_arr]
    if hasattr(data_sel, "compute"):
        data_sel = data_sel.compute()
    data_sel = np.asarray(data_sel)
    if len(nav_shape) == 2:
        data_sel = data_sel.reshape(*nav_shape, py, px, order="C")
    xpatS = kp.signals.EBSD(data_sel)

    # -------------------------
    # Subset detector
    # -------------------------
    pcx = _subset_detector_nav(det.pcx, idx_arr, Ny, Nx, nav_shape)
    pcy = _subset_detector_nav(det.pcy, idx_arr, Ny, Nx, nav_shape)
    pcz = _subset_detector_nav(det.pcz, idx_arr, Ny, Nx, nav_shape)
    pc = np.stack([pcx, pcy, pcz], axis=-1)

    detS = kp.detectors.EBSDDetector(
        pc=pc,
        shape=det.shape,
        sample_tilt=det.sample_tilt,
        azimuthal=det.azimuthal,
        tilt=det.tilt,
        binning=det.binning,
        px_size=det.px_size,
    )

    # -------------------------
    # Subset xmap
    # -------------------------
    xmapS = CrystalMap(
        rotations=xmap.rotations[idx_arr],
        phase_id=xmap.phase_id[idx_arr],
        x=x_sub.ravel(order="C"),
        y=y_sub.ravel(order="C"),
        phase_list=xmap.phases,
        scan_unit=xmap.scan_unit,
    )

    return xpatS, detS, xmapS, pc_indices


def xmap_update(xmap_old, xmap_new, nav_mask):
    """
    Update xmap with new orientations and CI values after partial/full refinement.

    PARAMETERS
    ----------
    xmap_old: original xmap with shape (Ny, Nx)
    xmap_new: refined xmap with shape (NyR, NxR) or 1D of length (# refined pixels)
    nav_mask: boolean mask with shape (Ny, Nx); True = masked-out (NOT refined), False = refined

    RETURNS
    -------
    xmap_updated: updated xmap with shape (Ny, Nx); orientations and CI values for refined pixels are updated.
    """
    # Ensure boolean 1D mask of refined pixels
    nav_mask = np.asarray(nav_mask, dtype=bool)
    idx_refined = (~nav_mask).ravel()

    # Flatten old data (views) and make writable copies
    rot_old = np.array(xmap_old.rotations).reshape(-1).copy()
    ci_old  = np.array(xmap_old.scores).reshape(-1).copy()

    # Flatten new data
    rot_new = np.array(xmap_new.rotations).reshape(-1)
    ci_new  = np.array(xmap_new.scores).reshape(-1)

    # Sanity check: numbers must match
    if rot_new.shape[0] != idx_refined.sum() or ci_new.shape[0] != idx_refined.sum():
        raise ValueError(
            f"Number of refined pixels ({idx_refined.sum()}) does not match "
            f"length of new rotations ({rot_new.shape[0]}) / scores ({ci_new.shape[0]})."
        )

    # Update only refined pixels
    rot_old[idx_refined] = rot_new
    ci_old[idx_refined]  = ci_new

    # Build new CrystalMap (reuse metadata from old map)
    xmap_updated = CrystalMap(
        rotations=rot_old,
        phase_id=xmap_old.phase_id,
        x=xmap_old.x,
        y=xmap_old.y,
        phase_list=xmap_old.phases,
        scan_unit=xmap_old.scan_unit,
    )
    xmap_updated.scores = ci_old
    return xmap_updated

def xmap_PS(xmap_old, PS_operation):
    """
    create a new xmap with orientations rotated by a pseudosymmetry operation
    
    PARAMETERS
    ----------
    xmap_old: original xmap
    PS_operation: rotation to transform to another pseudosymmetry variant

    RETURNS
    -------
    xmap_PSvar: new xmap with orientations updated
    """
    xmap_PSvar =  CrystalMap(
        rotations=PS_operation*xmap_old.rotations,
        phase_id=xmap_old.phase_id,
        x=xmap_old.x,
        y=xmap_old.y,
        phase_list=xmap_old.phases,
        scan_unit=xmap_old.scan_unit,
    )
    return xmap_PSvar

def crop_detector(det, new_shape, corners):
    """
    crops detector, revising pc values

    PARAMETERS
    ----------
    det: detector
    new_shape: (py, px)
    corners: (x0, x1, y0, y1)
    
    OUTPUT
    ------
    new cropped detector
    """
    det_cropped = kp.detectors.EBSDDetector(
            shape=new_shape,
            pc=det.crop(corners).pc_average,
            sample_tilt=det.sample_tilt,
            tilt=det.tilt,
            azimuthal=det.azimuthal,
            px_size=det.px_size,
            binning=det.binning,
            )
    return det_cropped    

def choose_nav_chunks(ny, nx, target=100, min_chunk=16, max_chunk=None):
    """
    Pick navigation chunk sizes (cy, cx) so the number of nav chunks
    ~ target, while keeping chunks reasonably sized.
    """
    # ideal counts of chunks along each axis
    r = ny / nx if nx else 1.0
    ny_chunks = max(1, int(round(math.sqrt(target * r))))
    nx_chunks = max(1, int(math.ceil(target / ny_chunks)))

    # convert to chunk sizes
    cy = int(math.ceil(ny / ny_chunks))
    cx = int(math.ceil(nx / nx_chunks))

    # clamp chunk sizes
    if min_chunk:
        cy = max(cy, int(min_chunk))
        cx = max(cx, int(min_chunk))
    if max_chunk:
        cy = min(cy, int(max_chunk))
        cx = min(cx, int(max_chunk))

    # resulting chunk counts
    nyc = int(math.ceil(ny / cy))
    nxc = int(math.ceil(nx / cx))
    total = nyc * nxc
    return cy, cx, nyc, nxc, total

def make_binned_detector(det, factor):
    """
    Build a detector matching patterns downsampled by `factor`.
    Keeps PC in Bruker fractions; adjusts shape and binning.
    """
    return kp.detectors.EBSDDetector(
        shape=(det.nrows // factor, det.ncols // factor),
        px_size=det.px_size,                  # unbinned pixel size
        binning=det.binning * factor,         # total binning
        tilt=det.tilt,
        azimuthal=det.azimuthal,
        sample_tilt=det.sample_tilt,
        pc=det.pc,                            # Bruker fractions
        convention="bruker",
    )

def save_metrics_h5(
    out_h5_path,
    ci_scalar, eta_map, xi_e_map, xi_t_map, xi_eN_map,
    dataset_prefix="WCC"
):
    """Save CI/eta-like metrics as HDF5 datasets alongside map.
    Shapes:
      ci_scalar  (ny,nx)
      eta_map  (ny,nx)
      xi_e_map   (ny,nx)
      xi_t_map   (ny,nx,K+1)
      xi_eN_map  (ny,nx,K+1)
    """
    with h5py.File(out_h5_path, "a") as f:
        grp = f.require_group(dataset_prefix)
        for name, arr in [
            ("CI_wcc", ci_scalar),
            ("eta", eta_map),
            ("Xi_e", xi_e_map),
            ("Xi_t", xi_t_map),
            ("Xi_eN", xi_eN_map),
        ]:
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=arr, compression="gzip")
    return out_h5_path

# ======================= CI_wcc caluclations ================================
import numpy as np
from kikuchipy.signals.util._master_pattern import _project_single_pattern_from_master_pattern

DTYPE = np.float64
SUM_DTYPE = np.float64

# precompute the 5 PS variant quaternions once (same as before)
_axes  = np.array([[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0]], dtype=DTYPE)
_angles= np.array([180.0, 90.0, -90.0, 90.0, -90.0], dtype=DTYPE)
_VARIANT_Q = tuple(xfn_nb.axis_angle_to_quaternion(ax, ang) for ax, ang in zip(_axes, _angles))

def _project(rotation_quat, dir_cos, m_u, m_l, npx, npy, scale):
    """Project one simulated pattern for masked pixels only."""
    return _project_single_pattern_from_master_pattern(
        rotation=rotation_quat,
        direction_cosines=dir_cos,
        master_upper=m_u,
        master_lower=m_l,
        npx=npx, npy=npy, scale=scale,
        rescale=False, out_min=0, out_max=1, dtype_out=DTYPE,
    )


def compute_wcc_single_pixel(B, rot0_quat, dir_cos, m_u, m_l, npx, npy, scale, mask_idx, debug=False):
    """Compute WCC components for a single pixel. Set debug=True to print shapes/dtypes.

    Notes
    -----
    This function expects `dir_cos` to be a 2D array of shape (N_pixels, 3) for the
    detector pixels included in `mask_idx`. If a 3D array (per-map-pixel) is passed,
    callers should slice it to the per-pixel 2D array before calling.
    """
    # Defensive check: ensure dir_cos passed in is 2D (Npix,3)
    if getattr(dir_cos, 'ndim', None) == 3:
        raise ValueError(
            "compute_wcc_single_pixel expects direction_cosines shaped (Npix,3). "
            "Pass dir_cos[map_idx] (per-map-pixel slice) or call compute_wcc_map which slices it for you."
        )

    # Extract masked pixels from experimental pattern
    B_full = np.asarray(B, dtype=DTYPE, order="C").ravel()
    B = B_full[mask_idx]

    if debug:
        print("[debug] B_full.shape:", B_full.shape, "mask_idx.shape:", mask_idx.shape, "B.shape:", B.shape)
        print("[debug] dir_cos.shape:", getattr(dir_cos, 'shape', None), "dir_cos.dtype:", getattr(dir_cos, 'dtype', None))
        print("[debug] rot0_quat.shape:", np.asarray(rot0_quat).shape, "dtype:", np.asarray(rot0_quat).dtype)

    # Check dir_cos and B shapes match
    if dir_cos.shape[0] != len(mask_idx):
        raise ValueError(
            f"Direction cosines length ({dir_cos.shape[0]}) does not match masked pixels ({len(mask_idx)}). "
            "This usually means the signal_mask was applied inconsistently between setup and compute."
        )

    # simulate base (k=0)
    sim0 = _project(rot0_quat, dir_cos, m_u, m_l, npx, npy, scale)

    if debug:
        print("[debug] sim0.shape:", sim0.shape, "sim0.dtype:", sim0.dtype)

    # center patterns
    sim0_mean = sim0.mean(dtype=SUM_DTYPE)
    Ai = (sim0 - sim0_mean).astype(DTYPE, copy=False)
    B_mean = B.mean(dtype=SUM_DTYPE)
    Bc = (B - B_mean).astype(DTYPE, copy=False)

    # Xi_e (unweighted)
    num_e = np.add.reduce(Ai.astype(SUM_DTYPE) * Bc.astype(SUM_DTYPE), dtype=SUM_DTYPE)
    den_e = np.sqrt(
        np.add.reduce(Ai.astype(SUM_DTYPE) * Ai.astype(SUM_DTYPE), dtype=SUM_DTYPE) *
        np.add.reduce(Bc.astype(SUM_DTYPE) * Bc.astype(SUM_DTYPE), dtype=SUM_DTYPE)
    )
    Xi_e = np.float64(0.0 if den_e <= 0 else num_e / den_e)

    Xi_t  = np.ones(6, dtype=DTYPE)
    Xi_eV = np.ones(6, dtype=DTYPE)
    Xi_e0 = np.ones(6, dtype=DTYPE)

    Ai64 = Ai.astype(SUM_DTYPE, copy=False)
    B64  = Bc.astype(SUM_DTYPE, copy=False)
    sim0_64 = sim0.astype(SUM_DTYPE, copy=False)

    for k, qv in enumerate(_VARIANT_Q, start=1):
        rotk = xfn_nb.quaternion_multiply(qv, rot0_quat)
        rotk = xfn_nb.q_normalize(rotk)
        simk = _project(rotk, dir_cos, m_u, m_l, npx, npy, scale)
        simk_mean = simk.mean(dtype=SUM_DTYPE)
        Aj = (simk.astype(SUM_DTYPE, copy=False) - simk_mean)

        w  = np.abs(sim0_64 - Aj - sim0_mean)
        w2 = w * w

        num_t    = np.add.reduce(w2 * Ai64 * Aj,               dtype=SUM_DTYPE)
        den_t_ai = np.add.reduce(w2 * Ai64 * Ai64,             dtype=SUM_DTYPE)
        den_t_aj = np.add.reduce(w2 * Aj    * Aj,              dtype=SUM_DTYPE)
        Xi_t[k]  = 0.0 if den_t_ai <= 0.0 or den_t_aj <= 0.0 else np.float64(num_t / np.sqrt(den_t_ai * den_t_aj))

        num_e0 = np.add.reduce(w2 * Ai64 * B64, dtype=SUM_DTYPE)
        den_e0 = np.sqrt(den_t_ai * np.add.reduce(w2 * B64 * B64, dtype=SUM_DTYPE))
        Xi_e0[k] = 0.0 if den_e0 <= 0.0 else np.float64(num_e0 / den_e0)

        num_eV = np.add.reduce(w2 * Aj * B64,  dtype=SUM_DTYPE)
        den_eV = np.sqrt(den_t_aj * np.add.reduce(w2 * B64 * B64, dtype=SUM_DTYPE))
        Xi_eV[k] = 0.0 if den_eV <= 0.0 else np.float64(num_eV / den_eV)

        del simk, Aj, w, w2

    Xi_eN = Xi_eV / Xi_e0
    eta   = np.float64(np.mean(np.abs(Xi_eN - Xi_t), dtype=SUM_DTYPE))
    CI    = np.float64(Xi_e - eta)
    return Xi_e, Xi_t, Xi_eN, eta, CI


def compute_wcc_map(
    xpat,                  # hyperspy signal for experimental patterns (Ny,Nx,H,W) or ndarray
    xmap,               # rotations matching detector shape (Ny,Nx) orix.quaternion.Rotation
    detector,               # detector matching rotations shape
    master_pattern,         # master pattern
    energy_kV,
    signal_mask=None,           # boolean mask for detector (False = include point)
    stripe_x=4, stripe_y=2  # compute in small tiles to save RAM
):
    import numpy as np
    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_for_varying_pc
    from kikuchipy.indexing._refinement._refinement import _get_master_pattern_data

    # Get master pattern data
    m_u, m_l, npx, npy, scale = _get_master_pattern_data(master_pattern, energy_kV)

    # Prepare signal mask - in our code False means include the point
    if signal_mask is None:
        # No mask = include all points
        mask_bool_flat = np.ones(detector.size, dtype=bool)  # True == include
        mask_idx = np.arange(detector.size)
    else:
        if signal_mask.shape != (detector.nrows, detector.ncols):
            raise ValueError(f"Signal mask shape {signal_mask.shape} does not match detector shape {(detector.nrows, detector.ncols)}")
        # Flatten mask; incoming convention: False == include; convert to True==include
        mask_2d = np.asarray(signal_mask, dtype=bool)
        mask_bool_flat = (~mask_2d).ravel()
        mask_idx = np.where(mask_bool_flat)[0]

    # Get pattern center values for the whole map
    pcx, pcy, pcz = detector.pc_flattened.T.astype(np.float64)

    # shapes
    Ny, Nx, py, px = xpat.data.shape
    N = Ny*Nx
    idx_1D = np.arange(N)
    idx_2D = idx_1D.reshape(Ny, Nx)   #1D indices arranged in 2D navigation shape and order

    # for progress counter
    total_col_stripes = (Ny + stripe_y - 1) // stripe_y
    total_row_stripes = (Nx + stripe_x - 1) // stripe_x
    total_stripes = total_col_stripes * total_row_stripes
    progress_every=10
    done = 0

    # outputs (all float64 now)
    Xi_e_map   = np.empty((Ny, Nx), dtype=DTYPE)
    Xi_t_map   = np.ones((Ny, Nx, 6), dtype=DTYPE)
    Xi_eN_map  = np.ones((Ny, Nx, 6), dtype=DTYPE)
    eta_map    = np.empty((Ny, Nx), dtype=DTYPE)
    CI_map     = np.empty((Ny, Nx), dtype=DTYPE)

    # iterate small tiles
    for x0 in range(0, Nx, stripe_x):
        x1 = min(Nx, x0 + stripe_x)
        for y0 in range(0, Ny, stripe_y):
            y1 = min(Ny, y0 + stripe_y)

            done += 1
            if done % progress_every == 0 or done == total_stripes:
                print(f"[WCC] Finished stripe {done}/{total_stripes} ({done/total_stripes:.1%})")

            # experimental tile
            if hasattr(xpat, "inav"):
                Bt = xpat.inav[x0:x1, y0:y1] #inav uses Hyperspy indexing [col(x), row(y)]
                Bt.compute(show_progressbar=False)
                B_arr = np.asarray(Bt.data, dtype=DTYPE)
            else:
                B_arr = np.asarray(xpat[y0:y1, x0:x1], dtype=DTYPE) #if xpat is a numpy array, then use numpy indexing [row(y), col(x)]
            Ny_t, Nx_t = B_arr.shape[:2] #height and width of tile
            #print("h, w:", Ny_t, Nx_t)
            Nt = Ny_t*Nx_t
            idx_1D_tile = np.arange(Nt)
            idx_2D_tile = idx_1D_tile.reshape(Ny_t, Nx_t)

            # Get PC values for this tile
            idx2D_tile = idx_2D[y0:y1, x0:x1]
            idx1D_tile = idx2D_tile.reshape(-1)
            pc_x_tile = pcx[idx1D_tile]
            pc_y_tile = pcy[idx1D_tile]
            pc_z_tile = pcz[idx1D_tile]

            # Compute direction cosines for this tile's PC values
            # _get_direction_cosines_for_varying_pc expects a 1D boolean signal mask where True==include
            dir_cos_tile = _get_direction_cosines_for_varying_pc(
                pcx=pc_x_tile,
                pcy=pc_y_tile,
                pcz=pc_z_tile,
                nrows=detector.nrows,
                ncols=detector.ncols,
                tilt=detector.tilt,
                azimuthal=detector.azimuthal,
                sample_tilt=detector.sample_tilt,
                signal_mask=mask_bool_flat
            )

            # rotations tile as quaternions
            rots_tile = xmap.rotations[idx1D_tile]
            if hasattr(rots_tile, 'data'):
                rot_tile_array = np.asarray(rots_tile.data, dtype=DTYPE)
            else:
                rot_tile_array = np.asarray(rots_tile, dtype=DTYPE)
            #print(rot_tile_array.shape)

            # pixel loop
            for iy in range(Ny_t):
                for ix in range(Nx_t):
                    Y = y0 + iy; X = x0 + ix
                    tile_idx = idx_2D_tile[iy,ix] # Index into the tile's direction cosines
                    #print(iy, ix, y0, x0, Y, X, tile_idx)

                    Xi_e, Xi_t, Xi_eN, eta, CI = compute_wcc_single_pixel(
                        B=B_arr[iy, ix],
                        rot0_quat=rot_tile_array[tile_idx,:],
                        dir_cos=dir_cos_tile[tile_idx, :],  # Use this tile's precomputed direction cosines
                        m_u=m_u, m_l=m_l,
                        npx=npx, npy=npy, scale=scale,
                        mask_idx=mask_idx
                    )

                    # Bounds check before assignment
                    if not (0 <= Y < Ny and 0 <= X < Nx):
                        raise IndexError(f"Output index out of bounds Y={Y}, X={X}, shape=({Ny},{Nx})")

                    Xi_e_map[Y, X]    = Xi_e
                    Xi_t_map[Y, X, :] = Xi_t
                    Xi_eN_map[Y, X, :] = Xi_eN
                    eta_map[Y, X]     = eta
                    CI_map[Y, X]      = CI

    return CI_map, eta_map, Xi_e_map, Xi_t_map, Xi_eN_map
