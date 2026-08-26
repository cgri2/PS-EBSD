import numpy as np
import kikuchipy as kp
import os
import matplotlib.pyplot as plt
import hyperspy.api as hs
import h5py
from orix.io import plugins
from orix.crystal_map import Phase, CrystalMap, PhaseList
from orix.quaternion import Rotation
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import shutil
import gc
import dask
import dask.array as da
from pathlib import Path
from collections import Counter
from dask.distributed import Client, wait
from dask_mpi import initialize
import time
import EBSD_extra_functions as xfn
import pipeline_io as pio

start_time = time.time()

# ------ Read variables from config file -------------------------------------
config_path = os.environ["CONFIG_PATH"]
config = pio.load_config(config_path)
paths = pio.resolve_pipeline_paths(config, config_path)

pname    = paths["pname"]
mapname  = paths["mapname"]
mp_path  = paths["mp_path"]

energy_kV    = float(config["global"]["energy_kV"])
PS_rotations = pio.get_ps_rotations(config)

cfg1A      = config.get("part1A", {})
instrument = pio.get_instrument(config)
det_cfg    = pio.get_detector_config(config)

map_num     = 1
h5oina_path = os.path.join(pname, f"{mapname}.h5oina")

overwrite_h5 = pio.get_overwrite_h5(config)
h5_out = pio.h5_path_for_stage(config, config_path, "Part1A_output")

# ============================================================================
# Dask / MPI cluster setup — must happen before any expensive work so that
# non-driver MPI ranks fork off as workers here and never execute Part 0/1A.
# ============================================================================
mem = (
    1024 * 1024 * int(os.environ["SLURM_MEM_PER_CPU"])
    if os.environ.get("SLURM_MEM_PER_CPU")
    else "auto"
)
dask_tmp = Path(pname) / "dask-temp" / "save-workers"
dask_tmp.mkdir(parents=True, exist_ok=True)
initialize(
    nthreads=1,
    memory_limit=mem,
    local_directory=str(dask_tmp),
)
client = Client()

print(f"Instrument:       {instrument}")
print(f"Part1A h5 output: {h5_out}")
print(f"overwriteH5:      {overwrite_h5}")

# ============================================================================
# All driver-only serial work runs under the synchronous scheduler so that
# any .compute() calls (e.g. the single-pattern read in create_kp_h5_template,
# pattern reads during optimization/GIFs) execute directly on rank 0 rather
# than being dispatched over the network to workers.
# The parallel save section below exits this context to use the distributed
# scheduler and the full worker pool.
# ============================================================================
with dask.config.set(scheduler="synchronous"):

    # ========================================================================
    # Part 0 — load patterns, crop, build xmap and detector
    # Branches on instrument; both branches produce a common `ebsd` object
    # with ebsd.xmap, ebsd.detector, and ebsd.static_background set.
    # ========================================================================

    if instrument == "oxford":
        # --- Safety checks ---------------------------------------------------
        if os.path.abspath(h5oina_path) == os.path.abspath(h5_out):
            raise ValueError(
                f"h5 output path would overwrite the h5oina source: {h5oina_path}\n"
                "Check your config paths."
            )
        if not os.path.isfile(h5oina_path):
            raise FileNotFoundError(f"h5oina source not found: {h5oina_path}")

        print(f"Part0 [Oxford] source: {h5oina_path}")

        # --- Load patterns ---------------------------------------------------
        ebsd = kp.load(h5oina_path, lazy=True)
        Ny, Nx, py, px = ebsd.data.shape

        ebsd, crop_info = xfn.crop_ebsd_to_square(ebsd)
        py_c, px_c = ebsd.data.shape[-2:]
        x0, x1 = crop_info["left"], crop_info["right"]
        y0, y1 = crop_info["top"], crop_info["bottom"]

        # --- Detector geometry -----------------------------------------------
        auto_load_det = bool(det_cfg.get("auto_load", True))
        if auto_load_det:
            with h5py.File(h5oina_path, "r") as hf:
                det_ori = hf[f"/{map_num}/EBSD/Header/Detector Orientation Euler"][()]
            det_ori_rad = np.asarray(det_ori, dtype=float).squeeze().reshape(3,)
            det_ori_deg = np.rad2deg(det_ori_rad)
            ebsd.detector.px_size   = 13.2315 * 1024 / py
            ebsd.detector.tilt      = det_ori_deg[1] - 90
            ebsd.detector.azimuthal = det_ori_deg[0]
            ebsd.detector.twist     = det_ori_deg[2]

        # Apply any explicit overrides from [detector] config
        for key in ("sample_tilt", "tilt", "azimuthal", "twist", "px_size", "binning"):
            if key in det_cfg:
                setattr(ebsd.detector, key, float(det_cfg[key]))

        print(ebsd.detector)
        ebsd.detector.save(filename=os.path.join(pname, f"{mapname}_InitialDetector.txt"))

        # --- Crystal map -----------------------------------------------------
        with h5py.File(h5oina_path, "r") as hf:
            euler_raw     = hf[f"/{map_num}/EBSD/Data/Euler"][()]
            x_raw         = hf[f"/{map_num}/EBSD/Data/X"][()]
            y_raw         = hf[f"/{map_num}/EBSD/Data/Y"][()]
            phaseID_raw   = hf[f"/{map_num}/EBSD/Data/Phase"][()]
            phaseName_raw = hf[f"/{map_num}/EBSD/Header/Phases/1/Phase Name"][()]
            phaseSG_raw   = hf[f"/{map_num}/EBSD/Header/Phases/1/Space Group"][()]

        phaseName   = phaseName_raw.item().decode("utf-8")
        space_group = int(np.asarray(phaseSG_raw).squeeze())
        euler    = np.asarray(euler_raw, dtype=float)
        x        = np.asarray(x_raw, dtype=float).reshape(-1)
        y        = np.asarray(y_raw, dtype=float).reshape(-1)
        phase_id = np.asarray(phaseID_raw, dtype=int).reshape(-1)

        rot_ox = Rotation.from_euler(euler, degrees=False)
        R_oxford_to_kp = Rotation.from_axes_angles([0, 0, 1], -np.pi / 2)
        rot_kp = rot_ox * R_oxford_to_kp

        phase_id[phase_id == 0] = -1
        unique_phase_ids = np.unique(phase_id)
        indexed_ids = unique_phase_ids[unique_phase_ids >= 0]
        if indexed_ids.size == 0:
            raise ValueError(f"No indexed phase IDs found. Unique IDs: {unique_phase_ids}")
        if indexed_ids.size != 1:
            raise ValueError(
                "Expected one indexed phase plus optional -1 not-indexed pixels, "
                f"but found phase IDs: {unique_phase_ids}"
            )
        pid = int(indexed_ids[0])
        phase = Phase(name=phaseName, space_group=space_group)
        phase_list = PhaseList(phases=[phase], ids=[pid])

        ebsd.xmap = CrystalMap(
            rotations=rot_kp,
            phase_id=phase_id,
            x=x,
            y=y,
            phase_list=phase_list,
        )

    elif instrument == "edax":
        edax_h5_path = os.path.join(pname, f"{mapname}.h5")
        ang_path     = os.path.join(pname, f"{mapname}.ang")

        # EDAX source and pipeline output share the same .h5 name when overwriting.
        if overwrite_h5 and os.path.abspath(edax_h5_path) == os.path.abspath(h5_out):
            raise ValueError(
                "EDAX: source pattern file and pipeline output would be the same path\n"
                f"({edax_h5_path}).\n"
                "Set overwriteH5 = false so processed patterns are written to a "
                "separate _PP.h5 file."
            )
        if not os.path.isfile(edax_h5_path):
            raise FileNotFoundError(f"EDAX pattern file not found: {edax_h5_path}")
        if not os.path.isfile(ang_path):
            raise FileNotFoundError(f"EDAX .ang map file not found: {ang_path}")

        print(f"Part0 [EDAX] pattern source: {edax_h5_path}")
        print(f"Part0 [EDAX] ang source:     {ang_path}")

        # --- Load patterns and xmap ------------------------------------------
        xpat = kp.load(edax_h5_path, lazy=True)
        Ny, Nx, py, px = xpat.data.shape
        xmap = plugins.ang.file_reader(ang_path)

        xpat, crop_info = xfn.crop_ebsd_to_square(xpat)
        py_c, px_c = xpat.data.shape[-2:]
        x0, x1 = crop_info["left"], crop_info["right"]
        y0, y1 = crop_info["top"], crop_info["bottom"]

        # --- Detector --------------------------------------------------------
        if "pc" not in det_cfg:
            raise ValueError(
                "[detector].pc = [pcx, pcy, pcz] is required for instrument = 'edax'."
            )
        pc_init = list(det_cfg["pc"])
        convention = str(det_cfg.get("convention", "edax"))
        det = kp.detectors.EBSDDetector(
            shape=(py, px),
            pc=pc_init,
            convention=convention,
            sample_tilt=float(det_cfg.get("sample_tilt", 70.0)),
            tilt=float(det_cfg.get("tilt", 0.0)),
            azimuthal=float(det_cfg.get("azimuthal", 0.0)),
            twist=float(det_cfg.get("twist", 0.0)),
            px_size=float(det_cfg.get("px_size", 66.67)),
            binning=int(det_cfg.get("binning", 1)),
        )
        det = xfn.crop_detector(det, (y0, y1, x0, x1))
        iy0 = Ny // 2
        ix0 = Nx // 2
        det = det.extrapolate_pc(
            pc_indices=[iy0, ix0],
            navigation_shape=xmap.shape,
            step_sizes=(xmap.dy, xmap.dx),
        )
        print(det)
        det.save(filename=os.path.join(pname, f"{mapname}_InitialDetector.txt"))

        ebsd = xpat
        ebsd.xmap = xmap
        ebsd.detector = det
        ebsd.static_background = np.zeros((py_c, px_c), dtype=xpat.data.dtype)

    else:
        raise ValueError(f"Unknown instrument '{instrument}'. Must be 'oxford' or 'edax'.")

    ckpt0 = time.time() - start_time
    print(f"Time to finish loading and building EBSD signal: {ckpt0:.2f} s")

    # ========================================================================
    # Create h5 template — fast (1 pattern + h5py patches, no full pattern write)
    # ========================================================================
    cy = int(cfg1A.get("save_chunk_y", 32))
    cx = int(cfg1A.get("save_chunk_x", 32))
    cy = min(cy, Ny)
    cx = min(cx, Nx)

    xfn.create_kp_h5_template(h5_out, ebsd, Ny, Nx, py_c, px_c, cy, cx)

    ckpt_tmpl = time.time() - start_time
    print(f"Time to create h5 template: {ckpt_tmpl:.2f} s")

    # ========================================================================
    # Optimization, plots, example GIFs + processing figures
    # ========================================================================

    # --- Load master pattern -------------------------------------------------
    mp = xfn.load_oxford_mp(mp_path, xmap=ebsd.xmap)

    # --- Select optimization point -------------------------------------------
    opt_y = int(cfg1A.get("opt_y", 3))
    opt_x = int(cfg1A.get("opt_x", 3))

    if not (0 <= opt_y < Ny and 0 <= opt_x < Nx):
        raise ValueError(
            f"Requested Part1A optimization point (opt_y={opt_y}, opt_x={opt_x}) "
            f"is outside the map shape (Ny={Ny}, Nx={Nx})."
        )

    k_opt = np.ravel_multi_index((opt_y, opt_x), dims=(Ny, Nx), order="C")

    if ebsd.xmap.phase_id[k_opt] == -1:
        raise ValueError(
            f"Requested Part1A optimization point (opt_y={opt_y}, opt_x={opt_x}, "
            f"k={k_opt}) is not indexed (phase_id = -1). Choose an indexed point."
        )

    print(f"Part1A optimization point: opt_y={opt_y}, opt_x={opt_x}, flat k={k_opt}")
    print("x:", ebsd.xmap.x[k_opt], "y:", ebsd.xmap.y[k_opt])

    nav_mask = np.ones((Ny, Nx), dtype=bool)
    nav_mask[opt_y, opt_x] = False

    # --- Refine orientation and PC for the optimization point ----------------
    xmap_ref, pc_ref = ebsd.refine_orientation_projection_center(
        xmap=ebsd.xmap,
        detector=ebsd.detector,
        master_pattern=mp,
        energy=energy_kV,
        pseudo_symmetry_ops=PS_rotations,
        navigation_mask=nav_mask,
        method="LN_NELDERMEAD",
        trust_region=[2, 2, 2, 0.05, 0.05, 0.05],
        rtol=1e-3,
    )

    rotations = xmap_ref.rotations.reshape(*xmap_ref.shape)
    sim = mp.get_patterns(rotations=rotations, detector=pc_ref, energy=energy_kV, compute=True)

    # --- Build 1-pattern signal for optimization -----------------------------
    p0_arr = ebsd.inav[opt_x, opt_y].data.compute()
    p0_arr = np.squeeze(np.asarray(p0_arr))
    p0 = kp.signals.EBSD(p0_arr)
    p0.static_background = np.zeros(p0_arr.shape, dtype=p0_arr.dtype)

    s_opt = sim.inav[0, 0]

    def norm_cross_cor(exp, sim_pat):
        exp     = np.asarray(exp,     dtype=np.float32)
        sim_pat = np.asarray(sim_pat, dtype=np.float32)
        A = exp     - np.mean(exp)
        B = sim_pat - np.mean(sim_pat)
        denom = np.sqrt(np.sum(A * A) * np.sum(B * B))
        if denom == 0:
            return 0.0
        return float(np.sum(A * B) / denom)

    def process_pipeline(p0, DBS_std, DBS_trunc, FFT_cutH, FFT_cutL,
                         AHE_kernel, AHE_clip, AHE_nbins, AHE_on, sim_ref=None):
        """Run the full processing pipeline on a single-pattern EBSD signal.

        sim_ref : kp.signals.EBSD with navigation shape (), used for NCC.
                  Defaults to s_opt (the optimization-point simulation).
        """
        if sim_ref is None:
            sim_ref = s_opt
        q   = np.zeros(4)
        NCC = np.zeros(4)
        q[0]   = kp.pattern.get_image_quality(np.asarray(p0.data), normalize=True)
        NCC[0] = norm_cross_cor(p0.data, sim_ref.data)

        p1 = p0.remove_dynamic_background(
            operation="subtract", filter_domain="frequency",
            std=DBS_std, truncate=DBS_trunc,
            inplace=False, show_progressbar=False,
        )
        q[1]   = kp.pattern.get_image_quality(np.asarray(p1.data), normalize=True)
        NCC[1] = norm_cross_cor(p1.data, sim_ref.data)

        if not AHE_on:
            p2 = p1
        else:
            p2 = p1.adaptive_histogram_equalization(
                kernel_size=(AHE_kernel, AHE_kernel),
                clip_limit=AHE_clip, nbins=AHE_nbins,
                inplace=False, show_progressbar=False,
            )
        q[2]   = kp.pattern.get_image_quality(np.asarray(p2.data), normalize=True)
        NCC[2] = norm_cross_cor(p2.data, sim_ref.data)

        pattern_shape = p2.axes_manager.signal_shape[::-1]
        w_low  = kp.filters.Window(window="lowpass",  cutoff=FFT_cutL, cutoff_width=10, shape=pattern_shape)
        w_high = kp.filters.Window(window="highpass", cutoff=FFT_cutH, cutoff_width=2,  shape=pattern_shape)
        p3 = p2.fft_filter(
            transfer_function=w_low * w_high, function_domain="frequency",
            shift=True, inplace=False, show_progressbar=False,
        )
        q[3]   = kp.pattern.get_image_quality(np.asarray(p3.data), normalize=True)
        NCC[3] = norm_cross_cor(p3.data, sim_ref.data)

        return p1, p2, p3, q, NCC

    dimensions = [
        Integer(8, 40,   name="DBS_std"),
        Integer(2, 10,   name="DBS_trunc"),
        Integer(1,  7,   name="FFT_cutH"),
        Integer(50, 100, name="FFT_cutL"),
        Categorical([48, 64, 80, 96, 112, 128, 256], name="AHE_kernel"),
        Categorical([float(f"{v:.6f}") for v in np.logspace(np.log10(1e-4), np.log10(5e-3), 7)], name="AHE_clip"),
        Categorical([128, 256, 512], name="AHE_nbins"),
        Categorical([False, True],   name="AHE_on"),
    ]

    @use_named_args(dimensions)
    def objective(DBS_std, DBS_trunc, FFT_cutH, FFT_cutL,
                  AHE_kernel, AHE_clip, AHE_nbins, AHE_on):
        _, _, _, _, NCC = process_pipeline(
            p0, DBS_std, DBS_trunc, FFT_cutH, FFT_cutL,
            AHE_kernel, AHE_clip, AHE_nbins, AHE_on,
        )
        return -NCC[-1]

    res = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=int(cfg1A.get("n_calls", 150)),
        n_initial_points=int(cfg1A.get("n_initial_points", 12)),
        random_state=int(cfg1A.get("random_state", 0)),
    )

    best_params  = dict(zip([d.name for d in dimensions], res.x))
    best_quality = -res.fun
    print("Best Q =", best_quality)
    print("at params:", best_params)

    def plot_pattern_processing(patterns, titles, save_path):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                                 gridspec_kw={"height_ratios": [3, 1.5]})
        for ax, pat, title in zip(axes[0], patterns, titles):
            ax.imshow(pat, cmap="gray", vmin=pat.min(), vmax=pat.max())
            ax.set_title(title)
            ax.axis("off")
        for ax, pat in zip(axes[1], patterns):
            ax.hist(pat.ravel(), bins=100)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    p1, p2, p3, q, NCC = process_pipeline(p0, **best_params)
    plot_pattern_processing(
        [np.squeeze(s_opt.data), np.squeeze(p0.data),
         np.squeeze(p1.data), np.squeeze(p2.data), np.squeeze(p3.data)],
        ["Simulated", "No processing", "DBS", "DBS + AHE", "DBS + AHE + FFT"],
        os.path.join(pname, f"{mapname}_PatProc_y{opt_y}_x{opt_x}.png"),
    )
    print("Image Quality:", q)
    print("Normalized Cross Correlation:", NCC)

    out_txt = os.path.join(pname, f"{mapname}_ProcessingParameters.txt")
    with open(out_txt, "w") as f:
        f.write(
            f"Optimization point: opt_y={opt_y}, opt_x={opt_x}, k={k_opt}\n"
            f"x = {ebsd.xmap.x[k_opt]}, y = {ebsd.xmap.y[k_opt]}\n"
            f"Best Q = {best_quality}\n"
            f"at params: {best_params}\n"
            f"Image Quality: {q}\n"
            f"Normalized Cross Correlation: {NCC}\n"
        )

    ckpt1 = time.time() - start_time
    print(f"Time to finish optimization: {ckpt1:.2f} s")

    # --- Example GIFs and pattern processing figures -------------------------
    example_dir = os.path.join(pname, "example_patterns")
    os.makedirs(example_dir, exist_ok=True)
    rng = np.random.default_rng()
    valid_k = np.where(ebsd.xmap.phase_id != -1)[0]
    k_examples = rng.choice(valid_k, size=3, replace=False)

    for n, k in enumerate(k_examples, start=1):
        i, j = np.unravel_index(k, shape=(Ny, Nx), order="C")
        print(f"\nExample {n}  i={i}, j={j}, k={k}")
        print("x:", ebsd.xmap.x[k], "y:", ebsd.xmap.y[k])

        det1 = ebsd.detector.deepcopy()
        if np.ndim(det1.pc) == 3:
            det1.pc = ebsd.detector.pc[i, j]

        rot1 = ebsd.xmap.rotations[k]
        sim1 = mp.get_patterns(rotations=rot1, detector=det1, energy=energy_kV, compute=True)

        exp_pat = ebsd.inav[j, i].data
        sim_pat = sim1.data[0] if sim1.data.ndim == 3 else sim1.data
        if hasattr(exp_pat, "compute"):
            exp_pat = exp_pat.compute()
        if hasattr(sim_pat, "compute"):
            sim_pat = sim_pat.compute()

        # Flash GIF
        gif_path = os.path.join(
            example_dir,
            f"{mapname}_ExpSimFlash_random{n}_Id{k}_i{i}_j{j}.gif",
        )
        xfn.make_flash_gif(exp_pat, sim_pat, gif_path, duration=0.5, n_flashes=2)
        print("Saved GIF:", gif_path)

        # Pattern processing figure — uses this point's simulation as NCC reference
        p_ex = kp.signals.EBSD(exp_pat)
        p_ex.static_background = np.zeros(exp_pat.shape, dtype=exp_pat.dtype)
        s_ex = kp.signals.EBSD(sim_pat)

        p1_ex, p2_ex, p3_ex, q_ex, NCC_ex = process_pipeline(
            p_ex, **best_params, sim_ref=s_ex,
        )
        fig_path = os.path.join(
            example_dir,
            f"{mapname}_PatProc_random{n}_Id{k}_i{i}_j{j}.png",
        )
        plot_pattern_processing(
            [sim_pat, exp_pat,
             np.squeeze(p1_ex.data), np.squeeze(p2_ex.data), np.squeeze(p3_ex.data)],
            ["Simulated", "No processing", "DBS", "DBS + AHE", "DBS + AHE + FFT"],
            fig_path,
        )
        print("Saved processing figure:", fig_path)
        print(f"  IQ={q_ex}  NCC={NCC_ex}")

# ============================================================================
# Parallel processing and save — uses the distributed scheduler / worker pool
# ============================================================================
pattern_path = "Scan 1/EBSD/Data/patterns"

num_workers = len(client.scheduler_info()["workers"])
num_workers = max(1, num_workers)
print(f"Dask workers available: {num_workers}")

max_in_flight = int(cfg1A.get("save_max_in_flight", min(num_workers, 4)))
cast_mode     = cfg1A.get("save_cast_mode", "clip")
out_dtype     = ebsd.data.dtype

print(f"Processing chunks: {cy}×{cx}")
print(f"Max in-flight: {max_in_flight}")
print(f"Cast mode: {cast_mode}")
print(f"Output dtype: {out_dtype}")

# write_dask_pattern_array_to_h5_from_driver requires h5_template != h5_out.
# When overwriting, write to a tmp path then atomically replace h5_out.
if overwrite_h5:
    h5_out_path  = Path(h5_out)
    write_target = str(h5_out_path.with_name(h5_out_path.stem + "_tmp" + h5_out_path.suffix))
else:
    write_target = h5_out

# Rechunk and persist raw patterns to workers
print("Rechunking raw lazy patterns...")
raw = ebsd.data.rechunk((cy, cx, -1, -1))
print("Persisting raw chunks to Dask workers...")
raw = raw.persist()
wait(raw)

client.rebalance(raw)
owners = Counter(w for ws in client.who_has(raw).values() for w in ws)
print("Raw chunks per worker:", dict(owners))

# Build the processed dask array
print("Building processed Dask array...")
processed = raw.map_blocks(
    xfn.process_part1a_block,
    best_params=best_params,
    out_dtype=out_dtype,
    cast_mode=cast_mode,
    dtype=out_dtype,
)

# Write to h5 (h5_out is the template; write_target is the output)
xfn.write_dask_pattern_array_to_h5_from_driver(
    data=processed,
    h5_template=h5_out,
    h5_out=write_target,
    client=client,
    Ny=Ny,
    Nx=Nx,
    py=py_c,
    px=px_c,
    pattern_path=pattern_path,
    chunk_y=cy,
    chunk_x=cx,
    max_in_flight=max_in_flight,
    overwrite=True,
    out_dtype=out_dtype,
    cast_mode=None,  # already cast inside process_part1a_block
)

if overwrite_h5:
    print(f"Replacing template:\n  tmp: {write_target}\n  dst: {h5_out}")
    os.replace(write_target, h5_out)

ckpt_total = time.time() - start_time
print(f"EBSD dataset saved as {h5_out}")
print(f"Total Part1A time: {ckpt_total:.2f} s")
print(f"Time for parallel processing+save: {ckpt_total - ckpt1:.2f} s")
