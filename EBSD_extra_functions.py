import math
import numpy as np
import imageio.v3 as iio
import kikuchipy as kp
from orix.crystal_map import CrystalMap
import h5py
import hyperspy.api as hs
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

def load_oxford_mp(mp_path, xmap=None):
    """
    Load an EBSD master pattern created using Oxford AZtecCrystal.

    Parameters
    ----------
    mp_path : str
        Path to the Oxford master pattern file.
    xmap : orix.crystal_map.CrystalMap, optional
        If provided, assign the master pattern phase from the indexed phase
        in the crystal map. Handles phase IDs 1 or 0, and ignores -1
        not-indexed pixels.

    Returns
    -------
    mp : kikuchipy.signals.EBSDMasterPattern
        Master pattern converted to Lambert projection, with phase assigned
        if xmap is provided.
    """
    with h5py.File(mp_path, "r") as f:
        lower_hemisphere = f["Data/Master/Dynamical/Lower"][()]
        upper_hemisphere = f["Data/Master/Dynamical/Upper"][()]

    south_signal = hs.signals.Signal2D(lower_hemisphere)
    north_signal = hs.signals.Signal2D(upper_hemisphere)

    mp = kp.signals.EBSDMasterPattern(
        [north_signal, south_signal],
        hemisphere="both",
    )

    mp.hemispheres = {"north", "south"}
    mp.projection = "stereographic"
    mp = mp.as_lambert()

    if xmap is not None:
        phase_ids = np.unique(xmap.phase_id)
        phase_ids = phase_ids[phase_ids != -1]

        if phase_ids.size == 0:
            raise ValueError("No indexed phases found in xmap.")

        # Prefer phase 1 if present, otherwise phase 0 if present,
        # otherwise use the first indexed phase ID.
        if 1 in xmap.phases.ids and 1 in phase_ids:
            phase_id = 1
        elif 0 in xmap.phases.ids and 0 in phase_ids:
            phase_id = 0
        else:
            phase_id = int(phase_ids[0])

        mp.phase = xmap.phases[phase_id]

        print(f"Assigned master pattern phase from xmap phase ID {phase_id}:")
        print(mp.phase)

    return mp

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

def crop_ebsd_to_square(ebsd, mask=None, p=1.0):
    """
    Crop EBSD patterns to a centered square.

    Parameters
    ----------
    ebsd : kikuchipy.signals.EBSD
        EBSD signal with data shape (Ny, Nx, py, px).
    mask : None or "circular"
        If None, crop to the largest centered square inside the rectangular pattern.
        If "circular", crop to the largest centered square inside the circular mask.
    p : float
        Fraction of the square side to keep. Default is 1.0.
        For example, p=0.9 crops an additional 10% of the square width.

    Returns
    -------
    ebsd : kikuchipy.signals.EBSD
        Cropped EBSD signal.
    crop_info : dict
        Dictionary with crop coordinates and final pattern shape.
    """
    if not (0 < p <= 1):
        raise ValueError(f"p must be in the range (0, 1], got p={p}")

    if mask not in [None, "circular"]:
        raise ValueError("mask must be None or 'circular'")

    Ny, Nx, py, px = ebsd.data.shape
    dtype = ebsd.data.dtype

    # Reset static background so crop_signal does not fail due to shape mismatch
    ebsd.static_background = np.zeros((py, px), dtype=dtype)

    cy = py // 2
    cx = px // 2

    if mask == "circular":
        # Assume circular mask is centered and has diameter equal to the smaller
        # pattern dimension. Largest inscribed square has side = diameter / sqrt(2).
        diameter = min(py, px)
        base_side = diameter / np.sqrt(2)

    else:
        # Largest square inside rectangular image
        base_side = min(py, px)

    # Apply additional fractional crop
    side = int(np.floor(base_side * p))

    # Make side even so the crop is symmetric around the center
    if side % 2 == 1:
        side -= 1

    half_side = side // 2

    top = cy - half_side
    bottom = cy + half_side
    left = cx - half_side
    right = cx + half_side

    if top < 0 or left < 0 or bottom > py or right > px:
        raise ValueError(
            "Computed crop extends outside the pattern. "
            f"Pattern shape is ({py}, {px}), crop is "
            f"top={top}, bottom={bottom}, left={left}, right={right}."
        )

    ebsd.crop_signal(
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )

    crop_info = {
        "original_pattern_shape": (py, px),
        "cropped_pattern_shape": ebsd.data.shape[-2:],
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "side": side,
        "mask": mask,
        "p": p,
    }

    return ebsd, crop_info

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

def crop_detector(det, corners):
    """
    Crop detector and update PC values.

    corners should match kikuchipy's EBSDDetector.crop() extent:
    (y0, y1, x0, x1)
    """
    det_cropped = det.crop(corners)

    # det.crop() should already preserve geometry, but ensure twist is retained
    if hasattr(det, "twist") and hasattr(det_cropped, "twist"):
        det_cropped.twist = det.twist

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

# ======================= Parallel pattern processing ========================

def cast_processed_patterns_to_dtype(block, out_dtype, cast_mode="rescale_per_pattern"):
    """
    Cast processed EBSD patterns to the target H5 dtype.

    Parameters
    ----------
    block : ndarray
        Pattern block with shape (..., py, px).
    out_dtype : dtype
        Target dtype, usually the H5 pattern dataset dtype.
    cast_mode : {"clip", "rescale_per_pattern", "rescale_block"}
        "clip":
            Clip directly to dtype range and cast.
        "rescale_per_pattern":
            Rescale each pattern independently to dtype range.
        "rescale_block":
            Rescale whole block using one min/max.
    """
    import numpy as np

    out_dtype = np.dtype(out_dtype)
    block = np.asarray(block)

    if np.issubdtype(out_dtype, np.floating):
        return block.astype(out_dtype, copy=False)

    if not np.issubdtype(out_dtype, np.integer):
        return block.astype(out_dtype, copy=False)

    info = np.iinfo(out_dtype)
    out_min = float(info.min)
    out_max = float(info.max)

    block = np.asarray(block, dtype=np.float32)

    if cast_mode == "clip":
        out = np.clip(block, out_min, out_max)

    elif cast_mode == "rescale_block":
        mn = np.nanmin(block)
        mx = np.nanmax(block)

        if mx > mn:
            out = (block - mn) / (mx - mn)
            out = out_min + out * (out_max - out_min)
        else:
            out = np.zeros_like(block, dtype=np.float32) + out_min

    elif cast_mode == "rescale_per_pattern":
        mn = np.nanmin(block, axis=(-2, -1), keepdims=True)
        mx = np.nanmax(block, axis=(-2, -1), keepdims=True)
        denom = mx - mn

        out = np.zeros_like(block, dtype=np.float32)
        np.divide(block - mn, denom, out=out, where=denom > 0)
        out = out_min + out * (out_max - out_min)

    else:
        raise ValueError(
            f"Unknown cast_mode={cast_mode!r}. "
            "Use 'clip', 'rescale_per_pattern', or 'rescale_block'."
        )

    return np.rint(out).astype(out_dtype)


def process_part1a_block(
    block,
    best_params,
    out_dtype=None,
    cast_mode="rescale_per_pattern",
):
    """
    Process one EBSD navigation block using the optimized Part1A parameters.

    Parameters
    ----------
    block : ndarray
        Pattern block with shape (by, bx, py, px).
    best_params : dict
        Optimized Part1A processing parameters.
    out_dtype : dtype or None
        Target output dtype. If None, keeps processed dtype.
    cast_mode : str
        Passed to cast_processed_patterns_to_dtype().

    Returns
    -------
    out : ndarray
        Processed block with shape (by, bx, py, px).
    """
    import numpy as np
    import kikuchipy as kp
    import gc

    block = np.asarray(block)
    by, bx, py, px = block.shape

    sig = kp.signals.EBSD(block)
    sig.static_background = np.zeros((py, px), dtype=block.dtype)

    # 1) Dynamic background subtraction
    sig = sig.remove_dynamic_background(
        operation="subtract",
        filter_domain="frequency",
        std=best_params["DBS_std"],
        truncate=best_params["DBS_trunc"],
        inplace=False,
        show_progressbar=False,
    )

    # 2) Adaptive histogram equalization
    if bool(best_params.get("AHE_on", True)):
        sig = sig.adaptive_histogram_equalization(
            kernel_size=(best_params["AHE_kernel"], best_params["AHE_kernel"]),
            clip_limit=best_params["AHE_clip"],
            nbins=best_params["AHE_nbins"],
            inplace=False,
            show_progressbar=False,
        )

    # 3) FFT filter
    pattern_shape = (py, px)

    w_low = kp.filters.Window(
        window="lowpass",
        cutoff=best_params["FFT_cutL"],
        cutoff_width=10,
        shape=pattern_shape,
    )
    w_high = kp.filters.Window(
        window="highpass",
        cutoff=best_params["FFT_cutH"],
        cutoff_width=2,
        shape=pattern_shape,
    )

    sig = sig.fft_filter(
        transfer_function=w_low * w_high,
        function_domain="frequency",
        shift=True,
        inplace=False,
        show_progressbar=False,
    )

    out = np.asarray(sig.data)

    if out_dtype is not None:
        out = cast_processed_patterns_to_dtype(
            out,
            out_dtype=out_dtype,
            cast_mode=cast_mode,
        )

    del sig
    gc.collect()

    return out


def write_dask_pattern_array_to_h5_from_driver(
    data,
    h5_template,
    h5_out,
    client,
    Ny,
    Nx,
    py,
    px,
    pattern_path="Scan 1/EBSD/Data/patterns",
    chunk_y=32,
    chunk_x=32,
    max_in_flight=4,
    overwrite=True,
    out_dtype=None,
    cast_mode=None,
):
    """
    Compute a Dask pattern array chunk-by-chunk and write it to H5 from the driver.

    This avoids collecting the full pattern stack into driver memory and avoids
    sending writable h5py objects to Dask workers.

    Parameters
    ----------
    data : dask.array.Array
        Pattern array with shape (Ny, Nx, py, px).
    h5_template : str
        Existing H5 file to copy as metadata/template.
    h5_out : str
        Output H5 file path. Must not equal h5_template unless h5_out is a temp file.
    client : dask.distributed.Client
        Existing Dask client.
    Ny, Nx, py, px : int
        EBSD dimensions.
    pattern_path : str
        H5 path to pattern dataset.
    chunk_y, chunk_x : int
        Navigation chunk sizes.
    max_in_flight : int
        Number of chunks submitted to workers at once.
    overwrite : bool
        Whether to overwrite h5_out if it exists.
    out_dtype : dtype or None
        If provided, cast data to this dtype before writing. If None, use H5 dtype.
    cast_mode : None or str
        If provided, use cast_processed_patterns_to_dtype() per chunk. This is useful
        for per-pattern normalization. If None, simple Dask astype/clip behavior is used.
    """
    import os
    import shutil
    import time
    import numpy as np
    import h5py
    import dask.array as da
    from dask.array.core import slices_from_chunks
    from dask.distributed import as_completed

    t0 = time.time()

    if os.path.abspath(h5_out) == os.path.abspath(h5_template):
        raise ValueError(
            "h5_out equals h5_template. Use a temporary output path, then os.replace()."
        )

    if os.path.exists(h5_out):
        if overwrite:
            print(f"Removing existing output file: {h5_out}")
            os.remove(h5_out)
        else:
            raise FileExistsError(f"Output already exists: {h5_out}")

    print(f"Copying H5 template:\n  from: {h5_template}\n  to:   {h5_out}")
    shutil.copyfile(h5_template, h5_out)

    with h5py.File(h5_out, "r") as f:
        if pattern_path not in f:
            raise ValueError(f"{pattern_path} not found in {h5_out}")

        dset_shape = f[pattern_path].shape
        dset_dtype = np.dtype(f[pattern_path].dtype)

    if dset_shape == (Ny, Nx, py, px):
        h5_layout = "nav"
    elif dset_shape == (Ny * Nx, py, px):
        h5_layout = "flat"
    else:
        raise ValueError(
            f"Unexpected H5 pattern shape {dset_shape}. "
            f"Expected {(Ny, Nx, py, px)} or {(Ny * Nx, py, px)}."
        )

    if out_dtype is None:
        out_dtype = dset_dtype
    out_dtype = np.dtype(out_dtype)

    print("Output H5 pattern shape:", dset_shape)
    print("Output H5 pattern dtype:", dset_dtype)
    print("Write dtype:", out_dtype)
    print("H5 layout:", h5_layout)

    arr = data.rechunk((min(chunk_y, Ny), min(chunk_x, Nx), -1, -1))

    if cast_mode is not None:
        # Use map_blocks so each block can be normalized/cast independently.
        arr = arr.map_blocks(
            cast_processed_patterns_to_dtype,
            out_dtype=out_dtype,
            cast_mode=cast_mode,
            dtype=out_dtype,
        )
    elif np.dtype(arr.dtype) != out_dtype:
        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            arr = da.clip(arr, info.min, info.max).astype(out_dtype)
        else:
            arr = arr.astype(out_dtype)

    print("Dask array to write:", arr)
    print("Write chunks:", arr.chunks)

    block_slices = list(slices_from_chunks(arr.chunks))
    block_delayed = arr.to_delayed().flatten().tolist()

    if len(block_slices) != len(block_delayed):
        raise RuntimeError(
            f"Mismatch: {len(block_slices)} slices but "
            f"{len(block_delayed)} delayed chunks."
        )

    total_chunks = len(block_delayed)
    max_in_flight = max(1, int(max_in_flight))

    print(f"Total chunks to write: {total_chunks}")
    print(f"Max in-flight chunks: {max_in_flight}")

    with h5py.File(h5_out, "r+") as f:
        dset = f[pattern_path]

        futures = {}
        submitted = 0
        completed = 0

        def submit_one(submitted_idx):
            if submitted_idx >= total_chunks:
                return submitted_idx, None

            fut = client.compute(block_delayed[submitted_idx])
            futures[fut] = block_slices[submitted_idx]
            submitted_idx += 1
            return submitted_idx, fut

        for _ in range(min(max_in_flight, total_chunks)):
            submitted, _ = submit_one(submitted)

        ac = as_completed(list(futures.keys()))

        for fut in ac:
            block_slice = futures.pop(fut)
            block = fut.result()

            sy, sx, spy, spx = block_slice
            y0, y1 = sy.start, sy.stop
            x0, x1 = sx.start, sx.stop

            if spy != slice(0, py, None) or spx != slice(0, px, None):
                raise RuntimeError(f"Unexpected signal slice: {spy}, {spx}")

            if h5_layout == "nav":
                dset[y0:y1, x0:x1, :, :] = block

            elif h5_layout == "flat":
                for local_y, global_y in enumerate(range(y0, y1)):
                    row0 = global_y * Nx + x0
                    row1 = global_y * Nx + x1
                    dset[row0:row1, :, :] = block[local_y, :, :, :]

            completed += 1

            if completed % max(1, total_chunks // 20) == 0 or completed == total_chunks:
                elapsed = time.time() - t0
                print(
                    f"Wrote {completed}/{total_chunks} chunks "
                    f"({100 * completed / total_chunks:.1f}%) "
                    f"after {elapsed:.2f} s"
                )
                f.flush()

            submitted, new_fut = submit_one(submitted)
            if new_fut is not None:
                ac.add(new_fut)

        f.flush()

    print(f"Finished writing H5: {h5_out}")
    print(f"Total write time: {time.time() - t0:.2f} s")
    return h5_out


def create_kp_h5_template(h5_path, ebsd, Ny, Nx, py_c, px_c, chunk_y=32, chunk_x=32):
    """
    Create a kikuchipy-format h5 template with correct metadata and a
    pre-allocated empty patterns dataset of shape (Ny*Nx, py_c, px_c).

    Writes the file directly with h5py using the same internal serialisation
    helpers that kikuchipy's own writer uses (_dict2hdf5group, crystalmap2dict),
    so the resulting structure is identical to a file produced by
    kp.signals.EBSD.save() and can be read back with kp.load().

    No pattern data is read from the full stack; the patterns dataset is
    pre-allocated with zeros and filled in later by
    write_dask_pattern_array_to_h5_from_driver().

    Parameters
    ----------
    h5_path : str
        Output path (will be overwritten if it exists).
    ebsd : kikuchipy.signals.EBSD
        Lazy EBSD signal with xmap, detector, and static_background already
        set to their final values (post-crop, post-convention-conversion).
    Ny, Nx : int
        Full map navigation dimensions (rows, columns).
    py_c, px_c : int
        Cropped pattern dimensions (signal shape).
    chunk_y, chunk_x : int
        Navigation chunk sizes used when pre-allocating the patterns dataset.
    """
    import numpy as np
    import h5py
    from kikuchipy import __version__ as kp_version
    from orix import __version__ as orix_version
    from orix.io.plugins.orix_hdf5 import crystalmap2dict
    from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group

    N = Ny * Nx
    det = ebsd.detector

    # --- PC: normalise to (Ny, Nx) regardless of original storage shape ---
    pc = np.asarray(det.pc).reshape(-1, 3)
    if pc.shape[0] == 1:
        pc = np.tile(pc, (N, 1))
    pc = pc.reshape(Ny, Nx, 3)
    pcx = pc[:, :, 0].astype(np.float64)
    pcy = pc[:, :, 1].astype(np.float64)
    pcz = pc[:, :, 2].astype(np.float64)

    # --- static background ---
    static_bg = ebsd.static_background
    if static_bg is None:
        static_bg = -1  # kikuchipy convention for missing background
    else:
        static_bg = np.asarray(static_bg)

    with h5py.File(h5_path, "w") as f:

        # Root datasets "manufacturer" and "version" are written by
        # kikuchipy.io.plugins._h5ebsd._dict2hdf5group, which is the same
        # helper KikuchipyH5EBSDWriter.write() uses.  kikuchipy checks
        # manufacturer == "kikuchipy" when opening the file.
        _dict2hdf5group(
            {"manufacturer": "kikuchipy", "version": kp_version},
            f["/"],
        )

        scan = f.create_group("Scan 1")

        # Crystal map group written via orix.io.plugins.orix_hdf5.crystalmap2dict,
        # which is the same serialiser orix uses internally.  On load, kikuchipy
        # passes this group to dict2crystalmap (orix_hdf5) to reconstruct the
        # CrystalMap, including all phase / lattice / space-group information.
        # Using crystalmap2dict here avoids any CrystalMap / signal navigation-
        # shape validation (the source of the previous ValueError).
        _dict2hdf5group(
            {
                "manufacturer": "orix",
                "version": orix_version,
                "crystal_map": crystalmap2dict(ebsd.xmap),
            },
            scan.create_group("EBSD/CrystalMap"),
        )

        # Patterns dataset: pre-allocated with zeros.  Shape (N, py_c, px_c)
        # matches the "flat" layout expected by write_dask_pattern_array_to_h5_from_driver
        # and by KikuchipyH5EBSDReader.scan2dict (which reshapes to (Ny, Nx, py, px)
        # using n_rows / n_columns from the EBSD header).
        data_grp = scan.create_group("EBSD/Data")
        data_grp.create_dataset(
            "patterns",
            shape=(N, py_c, px_c),
            dtype=ebsd.data.dtype,
            chunks=(min(chunk_y * chunk_x, N), py_c, px_c),
            fillvalue=0,
        )

        # EBSD header written via _dict2hdf5group.  Field names and value sources
        # match KikuchipyH5EBSDWriter.write() exactly.  On load, KikuchipyH5EBSDReader
        # .scan2dict reads these fields to reconstruct ny/nx/sy/sx and to build
        # the EBSDDetector (elevation_angle -> tilt, azimuth_angle -> azimuthal,
        # detector_pixel_size -> px_size, sample_tilt -> sample_tilt, pcx/pcy/pcz).
        _dict2hdf5group(
            {
                "azimuth_angle":        float(det.azimuthal),
                "binning":              float(det.binning),
                "elevation_angle":      float(det.tilt),
                "n_columns":            Nx,
                "n_rows":               Ny,
                "pattern_width":        px_c,
                "pattern_height":       py_c,
                "pcx":                  pcx,          # (Ny, Nx) float64
                "pcy":                  pcy,          # (Ny, Nx) float64
                "pcz":                  pcz,          # (Ny, Nx) float64
                "detector_pixel_size":  float(det.px_size),
                "sample_tilt":          float(det.sample_tilt),
                "static_background":    static_bg,    # array or -1 if not set
                "step_x":               float(ebsd.xmap.dx),
                "step_y":               float(ebsd.xmap.dy),
            },
            scan.create_group("EBSD/Header"),
        )

        # SEM header required by KikuchipyH5EBSDReader.scan2dict for the
        # "Acquisition_instrument.SEM" metadata block; not used by the pipeline.
        _dict2hdf5group(
            {
                "beam_energy":      0.0,
                "magnification":    0.0,
                "microscope":       "",
                "working_distance": 0.0,
            },
            scan.create_group("SEM/Header"),
        )

    print(
        f"Created h5 template: Ny={Ny}, Nx={Nx}, py={py_c}, px={px_c}, "
        f"patterns=({N}, {py_c}, {px_c}), chunk=({min(chunk_y*chunk_x, N)}, {py_c}, {px_c})"
    )


# ======================= CI_wcc caluclations ================================
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
            dir_cos_tile = xfn_nb.get_direction_cosines_for_varying_pc_local(
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
