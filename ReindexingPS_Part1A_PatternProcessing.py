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
import time
import sys
sys.path.append("/cluster/work/mandm/cgriesbach/EBSDindexing")
import EBSD_extra_functions as xfn
start_time = time.time()

# ------ File paths -------------------------------------------------
#read common environment variables
pname = os.environ["PNAME"]
mapname = os.environ["MAPNAME"]
mp_path = os.environ.get("MP_PATH","")
energy_kV = float(os.environ.get("ENERGY_KV", "25"))
pcx = float(os.environ["PCX"])
pcy = float(os.environ["PCY"])
pcz = float(os.environ["PCZ"])
sample_tilt_deg = float(os.environ["SAMPLE_TILT_DEG"])


# ------ Load data and crop -----------------------------------------
xpat = kp.load(os.path.join(pname,f"{mapname}.h5"),lazy=True)
xmap = plugins.ang.file_reader(os.path.join(pname,f'{mapname}.ang')) #map

#crop patterns to square mask
Ny, Nx, py, px = xpat.data.shape
print(xpat.data.shape)
sig_shape = (py,px)
dtype = xpat.data.dtype
xpat.static_background = np.zeros(sig_shape, dtype=dtype)
side   = int(np.floor(py/(2*np.sqrt(2))))
cx = px//2
x0, x1 = cx-side, cx+side
xpat.crop_signal(top=x0, bottom=x1, left=x0, right=x1)
_, _, py_c, px_c = xpat.data.shape

#Define detector
"""
#load from Si cal data
det_cal=kp.detectors.EBSDDetector.load(os.path.join(pname,"20250821_BTO101PFM_SiCal1_DetCal_crop.txt"))
SiCal_map = plugins.ang.file_reader(os.path.join(pname,"20250821_BTO101PFM_SiCal1_refined_crop.ang"))
#extrapolate calibrated detector pattern centers to new dataset
det_xmap = det_cal.extrapolate_pc(
    pc_indices=[SiCal_map.x,SiCal_map.y],
    navigation_shape=xmap.shape,
    step_sizes=(xmap.dx, xmap.dy),
)
"""
#pc from vendor
det = kp.detectors.EBSDDetector(
        shape=(py,px),
        pc=(pcx, pcy, pcz),
        convention='edax',
        sample_tilt=sample_tilt_deg,
        tilt=10,
        azimuthal=-2,
        px_size=66.67,
        binning=1,
        )
det = xfn.crop_detector(det, (py_c, px_c), (x0,x1,x0,x1))
iy0 = Ny // 2
ix0 = Nx // 2
det_xmap = det.extrapolate_pc(
        pc_indices=[iy0, ix0],   # (row, col)#[Ny/2, Nx/2],
        navigation_shape=xmap.shape,
        step_sizes=(xmap.dy, xmap.dx),
        )
det_xmap.save(filename=os.path.join(pname,f"{mapname}_Detector.txt"))

#Package into one ebsd signal
Ny, Nx, py, px = xpat.data.shape
sig_shape = (py,px)
dtype = xpat.data.dtype
xpat.static_background = np.zeros(sig_shape, dtype=dtype)

EBSDdat = kp.signals.EBSD(
    xpat,
    xmap=xmap,
    detector=det_xmap,
    static_background=xpat.static_background,
)
EBSDdat.set_scan_calibration(step_x=xmap.dx, step_y=xmap.dy)

#crop map to only use a subset of patterns
nav_mask = np.ones((Ny, Nx), dtype=bool)
nav_mask[0:4, 0:4] = False

#Load master pattern
f = h5py.File(mp_path,'r')
lower_hemisphere = f["Data/Master/Dynamical/Lower"]
upper_hemisphere = f["Data/Master/Dynamical/Upper"]
south_signal = hs.signals.Signal2D(lower_hemisphere)
north_signal = hs.signals.Signal2D(upper_hemisphere)
mp = kp.signals.EBSDMasterPattern([north_signal, south_signal], hemisphere='both',) # Create the EBSDMasterPattern signal, explicitly setting hemispheres
mp.hemispheres = {"north", "south"} # Assign hemispheres
mp.phase=xmap.phases[0]
mp.projection = 'stereographic' #Assign projection
mp = mp.as_lambert() # Convert the master pattern to the square Lambert projection

# ------ Refine a subset of orientations to use for matching --------
#Define variants
PS_rotations=Rotation.from_axes_angles(((1, 0, 0),(1, 0, 0),(1, 0, 0),(0, 1, 0),(0, 1, 0)), (180, 90, -90, 90, -90) ,degrees=True)

xmap_ref, pc_ref = EBSDdat.refine_orientation_projection_center(
    xmap = EBSDdat.xmap,
    detector = EBSDdat.detector,
    master_pattern = mp,
    energy = energy_kV,
    pseudo_symmetry_ops = PS_rotations,
    navigation_mask = nav_mask,
    #signal_mask = sig_mask,
    method = "LN_NELDERMEAD",
    trust_region = [2, 2, 2, 0.05, 0.05, 0.05],
    rtol = 1e-3,
    )

# Prepare simulated patterns
rotations = xmap_ref.rotations.reshape(*xmap_ref.shape)
sim = mp.get_patterns(
    rotations=rotations,
    detector=pc_ref,
    energy=energy_kV,  # Energy in keV
    compute=True,
    )


# ------ Define pattern processing workflow- ------------------------
# Select pattern to optimize
p0 = EBSDdat.inav[3,3]
#p0.crop_signal(top=y0, bottom=y1, left=x0, right=x1)
s = sim.inav[3,3]
#s.crop_signal(top=y0, bottom=y1, left=x0, right=x1)

# Define NCC function
def norm_cross_cor(exp, sim):
    A = exp - np.mean(exp)
    B = sim - np.mean(sim)
    return np.sum(A*B)/np.sqrt(np.sum(np.square(A))*np.sum(np.square(B)))

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
    n_calls = 150,
    n_initial_points=12,
    random_state = 0,
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
    plt.savefig(os.path.join(pname,f"{mapname}_PatProc.png"),dpi=300)

p1, p2, p3, q, NCC = process_pipeline(p0, **best_params)
patterns = [s.data, p0.data, p1.data, p2.data, p3.data]
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
    f.write(f"Best Q = {best_quality}\n"
            f"at params: {best_params}\n"
            f"Image Quality: {q}\n"
            f"Normalized Cross Correlation: {NCC}\n")

# ------ Process all patterns and save to a new h5 file -------------------
#dynamic background subtraction
xpat = xpat.remove_dynamic_background(
            operation='subtract',
            filter_domain='frequency',
            std=best_params["DBS_std"],
            truncate=best_params["DBS_trunc"],
            show_progressbar=False,
            inplace=False,
            )
#adaptive histogram equalization
xpat = xpat.adaptive_histogram_equalization(
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
xpat = xpat.fft_filter(
            transfer_function=w_low * w_high,
            function_domain="frequency",
            shift=True,
            inplace=False,
            show_progressbar=False,
            )
ckpt2 = time.time() - start_time
print(f"Time to finish processing of entire dataset: {ckpt2:.2f} seconds")

#save patterns to h5 file
xpat.compute(show_progressbar=True)
xpat.save(os.path.join(pname,f"{mapname}_PP.h5"), overwrite=True)
