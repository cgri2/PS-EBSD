import numpy as np
import kikuchipy as kp
import os
import matplotlib.pyplot as plt
import hyperspy.api as hs
import h5py
from orix.io import plugins
from orix.quaternion import Rotation
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import shutil
import gc
import dask.array as da
from pathlib import Path
from dask.distributed import Client, wait
from dask_mpi import initialize
import time
import EBSD_extra_functions as xfn
import pipeline_io as pio
start_time = time.time()

# ------ Read variables from config file -----------------------------
config_path = os.environ["CONFIG_PATH"]
config = pio.load_config(config_path)
paths = pio.resolve_pipeline_paths(config, config_path)

pname = paths["pname"]
mapname = paths["mapname"]
mp_path = paths["mp_path"]
h5_path = paths["h5_path"]

energy_kV = float(config["global"]["energy_kV"])
PS_rotations = pio.get_ps_rotations(config)

cfg1A = config.get("part1A", {})

overwrite_h5 = pio.get_overwrite_h5(config)
h5_in = pio.h5_path_for_stage(config, config_path, "Part1A_input")
h5_out = pio.h5_path_for_stage(config, config_path, "Part1A_output")
print(f"Part1A input H5:  {h5_in}")
print(f"Part1A output H5: {h5_out}")
print(f"overwriteH5:      {overwrite_h5}")

# ------ Load data  -----------------------------------------
ebsd = kp.load(h5_in,lazy=True)
xmap = ebsd.xmap
Ny, Nx, py, px = ebsd.data.shape
det = ebsd.detector
det.save(filename=os.path.join(pname,f"{mapname}_InitialDetector.txt"))
mp = xfn.load_oxford_mp(mp_path, xmap=xmap)

# Select the single pattern used to optimize the pattern-processing recipe.
# opt_y/opt_x are row/column indices in the EBSD map.
opt_y = int(cfg1A.get("opt_y", 3))
opt_x = int(cfg1A.get("opt_x", 3))

if not (0 <= opt_y < Ny and 0 <= opt_x < Nx):
    raise ValueError(
        f"Requested Part1A optimization point (opt_y={opt_y}, opt_x={opt_x}) "
        f"is outside the map shape (Ny={Ny}, Nx={Nx})."
    )

k_opt = np.ravel_multi_index((opt_y, opt_x), dims=(Ny, Nx), order="C")

if xmap.phase_id[k_opt] == -1:
    raise ValueError(
        f"Requested Part1A optimization point (opt_y={opt_y}, opt_x={opt_x}, k={k_opt}) "
        "is not indexed, phase_id = -1. Choose an indexed point."
    )

print(f"Part1A optimization point: opt_y={opt_y}, opt_x={opt_x}, flat index k={k_opt}")
print("x:", xmap.x[k_opt], "y:", xmap.y[k_opt])

#Create nav mask to only reindex one point
nav_mask = np.ones((Ny, Nx), dtype=bool)
nav_mask[opt_y, opt_x] = False

# ------ Refine a subset of orientations to use for matching --------
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
print(sim.data.shape)
# ------ Define pattern processing workflow- ------------------------
# build a 1-pattern signal for processing optimization/plotting
p0_arr = ebsd.inav[opt_x, opt_y].data
p0_arr = p0_arr.compute()
p0_arr = np.squeeze(np.asarray(p0_arr))
p0 = kp.signals.EBSD(p0_arr)
p0.static_background = np.zeros(p0_arr.shape, dtype=p0_arr.dtype)

s = sim.inav[0,0]

# Define NCC function
def norm_cross_cor(exp, sim):
    exp = np.asarray(exp, dtype=np.float32)
    sim = np.asarray(sim, dtype=np.float32)

    A = exp - np.mean(exp)
    B = sim - np.mean(sim)

    denom = np.sqrt(np.sum(A * A) * np.sum(B * B))
    if denom == 0:
        return 0.0

    return np.sum(A * B) / denom

# Define pattern processing functions
#    order of functions was determined to be the best by a quick manual optimization approach

def process_pipeline(p0, DBS_std, DBS_trunc, FFT_cutH, FFT_cutL, AHE_kernel, AHE_clip, AHE_nbins, AHE_on):
    q=np.zeros(4)
    NCC=np.zeros(4)
    q[0] = kp.pattern.get_image_quality(np.asarray(p0.data), normalize=True)
    NCC[0] = norm_cross_cor(p0.data, s.data)
    #1) dynamic background subtraction
    p1 = p0.remove_dynamic_background(
         operation='subtract',
            filter_domain='frequency',
            std=DBS_std,
            truncate=DBS_trunc,
            inplace=False,
            show_progressbar=False,
            )
    q[1] = kp.pattern.get_image_quality(np.asarray(p1.data), normalize=True)
    NCC[1] = norm_cross_cor(p1.data, s.data)

    #2) adaptive histogram equalization
    if not AHE_on:
        p2 = p1
    else:
        p2 = p1.adaptive_histogram_equalization(
                kernel_size=(AHE_kernel, AHE_kernel),
                clip_limit=AHE_clip,
                nbins=AHE_nbins,
                inplace=False,
                show_progressbar=False,
                )
    q[2] = kp.pattern.get_image_quality(np.asarray(p2.data), normalize=True)
    NCC[2] = norm_cross_cor(p2.data, s.data)

    #3) fft filter (low pass to filter noise, high pass to filterlarge variations across detector)
    pattern_shape = p2.axes_manager.signal_shape[::-1]
    w_low = kp.filters.Window(
            window="lowpass", cutoff=FFT_cutL, cutoff_width=10, shape=pattern_shape
            )
    w_high = kp.filters.Window(
            window="highpass", cutoff=FFT_cutH, cutoff_width=2, shape=pattern_shape)
    p3 = p2.fft_filter(
            transfer_function=w_low * w_high,
            function_domain="frequency",
            shift=True,
            inplace=False,
            show_progressbar=False,
            )
    q[3] = kp.pattern.get_image_quality(np.asarray(p3.data), normalize=True)
    NCC[3] = norm_cross_cor(p3.data, s.data)
    
    return p1, p2, p3, q, NCC

# ------  Define objective for optimization ----------------------
dimensions = [
    Integer(8, 40, name='DBS_std'),
    Integer(2, 10, name='DBS_trunc'),
    Integer(1, 7, name='FFT_cutH'),
    Integer(50, 100, name='FFT_cutL'),
    Categorical([48, 64, 80, 96, 112, 128, 256],  name='AHE_kernel'),
    #Real(0.003, 0.03, prior='log-uniform', name='AHE_clip'),
    Categorical([float(f"{v:.6f}") for v in np.logspace(np.log10(1e-4), np.log10(5e-3), 7)], name='AHE_clip'),
    #Categorical([0.003, 0.005, 0.007, 0.009, 0.01, 0.02, 0.03], name='AHE_clip'),
    Categorical([128, 256, 512], name='AHE_nbins'),
    Categorical([False, True], name='AHE_on'),             # simple on/off
]

@use_named_args(dimensions)

def objective(DBS_std, DBS_trunc, FFT_cutH, FFT_cutL, AHE_kernel, AHE_clip, AHE_nbins, AHE_on):
    p1, p2, p3, q, NCC = process_pipeline(
            p0, DBS_std, DBS_trunc, FFT_cutH, FFT_cutL, AHE_kernel, AHE_clip, AHE_nbins, AHE_on)
    # maximize Q → minimize -Q
    return -NCC[-1]

# ------ Run the Bayesian optimizer ----------------------------------------
res = gp_minimize(
    func = objective,
    dimensions = dimensions,
    n_calls=int(cfg1A.get("n_calls", 150)),
    n_initial_points=int(cfg1A.get("n_initial_points", 12)),
    random_state=int(cfg1A.get("random_state", 0)),
)

# ------ Inspect best result -----------------------------------------------
best_params  = dict(zip([d.name for d in dimensions], res.x))
best_quality = -res.fun
print("Best Q =", best_quality)
print("at params:", best_params)

#out_path = os.path.join(pname, f"{mapname}_ProcessingParameters.npz")
#np.savez(out_path, **{k: np.asarray(v) for k, v in best_params.items()})

#Define plot function
def plot_pattern_processing(patterns, titles):
    fig, axes = plt.subplots(2, 5, figsize=(15,6),
                             gridspec_kw={'height_ratios':[3,1.5]})
    for ax, pat, title in zip(axes[0], patterns, titles):
        ax.imshow(pat, cmap='gray', vmin=pat.min(), vmax=pat.max())
        ax.set_title(title)
        ax.axis('off')
    for ax, pat in zip(axes[1], patterns):
        ax.hist(pat.ravel(), bins=100)
    fig.tight_layout()
    plt.savefig(os.path.join(pname, f"{mapname}_PatProc_y{opt_y}_x{opt_x}.png"),dpi=300)

p1, p2, p3, q, NCC = process_pipeline(p0, **best_params)
patterns = [
    np.squeeze(s.data),
    np.squeeze(p0.data),
    np.squeeze(p1.data),
    np.squeeze(p2.data),
    np.squeeze(p3.data),
]
plot_pattern_processing(
    patterns, ["Simulated","No processing", "DBS", "DBS + AHE", "DBS + AHE + FFT"]
)
print('Image Quality:', q)
print('Normalized Cross Correlation:', NCC)
ckpt1 = time.time() - start_time
print(f"Time to finish optimization: {ckpt1:.2f} seconds")

#write to a txt file
out_path = os.path.join(pname, f"{mapname}_ProcessingParameters.txt")
with open(out_path, "w") as f:
    f.write(
        f"Optimization point: opt_y={opt_y}, opt_x={opt_x}, k={k_opt}\n"
        f"x = {xmap.x[k_opt]}, y = {xmap.y[k_opt]}\n"
        f"Best Q = {best_quality}\n"
        f"at params: {best_params}\n"
        f"Image Quality: {q}\n"
        f"Normalized Cross Correlation: {NCC}\n"
    )

# ------ Process all patterns (builds lazy processing graph)  -------------------
#dynamic background subtraction
ebsd = ebsd.remove_dynamic_background(
            operation='subtract',
            filter_domain='frequency',
            std=best_params["DBS_std"],
            truncate=best_params["DBS_trunc"],
            show_progressbar=False,
            inplace=False,
            )
#adaptive histogram equalization
ebsd = ebsd.adaptive_histogram_equalization(
            kernel_size=(best_params["AHE_kernel"], best_params["AHE_kernel"]),
            clip_limit=best_params["AHE_clip"],
            nbins=best_params["AHE_nbins"],
            inplace=False,
            show_progressbar=False,
            )
#fft (low pass to filter noise, high pass to filter variations across detector)
pattern_shape = (py, px)
w_low = kp.filters.Window(window="lowpass", cutoff=best_params["FFT_cutL"], cutoff_width=10, shape=pattern_shape)
w_high = kp.filters.Window(window="highpass", cutoff=best_params["FFT_cutH"], cutoff_width=2, shape=pattern_shape)
ebsd = ebsd.fft_filter(
            transfer_function=w_low * w_high,
            function_domain="frequency",
            shift=True,
            inplace=False,
            show_progressbar=False,
            )
# Save (this may take a long time since processing steps need to be computed)
save_chunk_y = int(cfg1A.get("save_chunk_y", 8))
save_chunk_x = int(cfg1A.get("save_chunk_x", 8))
ebsd.data = ebsd.data.rechunk((save_chunk_y, save_chunk_x, -1, -1))
print("Chunks before save:", ebsd.data.chunks)
# save patterns to h5 file
if os.path.abspath(h5_out) == os.path.abspath(h5_in):
    h5_out_path = Path(h5_out)
    tmp_h5 = str(h5_out_path.with_name(h5_out_path.stem + "_tmp" + h5_out_path.suffix))

    if os.path.exists(tmp_h5):
        os.remove(tmp_h5)

    print(f"h5_out equals h5_in, saving to temporary file first:\n  tmp: {tmp_h5}")
    ebsd.save(tmp_h5, overwrite=True)

    print(f"Replacing original H5:\n  tmp: {tmp_h5}\n  dst: {h5_out}")
    os.replace(tmp_h5, h5_out)

else:
    ebsd.save(h5_out, overwrite=True)

ckpt3 = time.time() - start_time
print(f"EBSD dataset saved as {h5_out}")
print(f"Total Part1A time: {ckpt3:.2f} seconds")
