# EBSD Pseudosymmetry Reindexing Pipeline

A Python-based EBSD reindexing pipeline for distinguishing pseudosymmetry-related ferroelectric domain variants using simulated pattern matching, optimized pattern preprocessing, neighbor pattern averaging, global sample-detector geometry refinement, and CI<sub>wcc</sub>-based final variant selection.

<p align="center">
  <img src="workflow.svg" alt="Workflow schematic for the EBSD pseudosymmetry reindexing pipeline" width="950">
</p>

## Overview

This repository implements a multi-step EBSD reindexing workflow designed for challenging pseudosymmetry materials. The code is organized as a sequential pipeline:

1. **Pattern processing and initial detector setup**  
   Optimize preprocessing parameters on a representative pattern and apply the selected workflow to the full dataset.
2. **Pseudosymmetry-sensitive neighbor pattern averaging (PSS-NPA)**  
   Improve pattern quality before downstream reindexing by averaging locally similar neighbors.
3. **Global geometry refinement**  
   Refine detector/sample geometry using map-wide displacement signatures between experimental and simulated patterns.
4. **Final pseudosymmetry-aware refinement and CI<sub>wcc</sub> selection**  
   Refine each pseudosymmetry seed separately, compute confidence metrics, and select the best variant per pixel.

The scripts are intended for **high-performance execution on ETH Euler / Slurm-based systems**, with MPI/Dask used in the more computationally intensive steps.

---

## Workflow

### Part 1A — Pattern processing (`ReindexingPS_Part1A_PatternProcessing.py`)

This step:

- loads the raw EBSD patterns from `MAPNAME.h5`
- loads the initial Hough / orientation map from `MAPNAME.ang`
- constructs an initial detector from vendor pattern center values passed through environment variables
- crops the detector and patterns to a square signal region
- extrapolates the detector PC field over the full map
- performs a small orientation + projection-center refinement on a subset of the map with pseudosymmetry operators enabled
- optimizes a preprocessing pipeline using Bayesian optimization (`gp_minimize`)
- applies the selected preprocessing to the entire dataset

### Pattern-processing stages

The optimized preprocessing pipeline includes:

- dynamic background subtraction
- optional adaptive histogram equalization
- FFT-based bandpass filtering

### Part 1A outputs

- `MAPNAME_Detector.txt` — initial extrapolated detector field  
- `MAPNAME_ProcessingParameters.txt` — selected preprocessing parameters and summary metrics  
- `MAPNAME_PatProc.png` — diagnostic figure showing processing stages  
- `MAPNAME_PP.h5` — processed EBSD patterns for downstream analysis

---

### Part 1B — Pseudosymmetry-sensitive neighbor pattern averaging (`ReindexingPS_Part1B_NPA_MPI.py`)

This stage performs non-local neighbor pattern averaging using MPI/Dask. It pads the pattern stack in navigation space, distributes the work across workers, computes normalized cross-correlation between each pattern and its local neighborhood, and uses a **first-jump cutoff** strategy to identify a suitable neighbor set for averaging.

The goal is to improve pattern quality while avoiding averaging across dissimilar neighborhoods.

### Part 1B outputs

This stage writes the averaged patterns to a new HDF5 dataset used by later stages:

- `MAPNAME_PP_NPA<RADIUS>.h5`

It can also generate diagnostic plots for selected target pixels, depending on the runtime settings.

---

### Part 1C — Global geometry refinement (`ReindexingPS_Part1C_GlobalGeomRefine.py`)

This step loads the NPA-processed patterns, the original map, the initial detector, and the master pattern, then performs iterative **global sample-detector geometry refinement** on a subset of points.

The geometry refinement is handled by `optimize_geometry_and_orientations()` in `EBSD_refine_geometry.py`. The implementation couples:

- simulated pattern generation from the current geometry
- displacement-field estimation between experimental and simulated patterns
- Jacobian-based geometry updates for parameters such as `pcx`, `pcy`, `pcz`, `sample_tilt`, `azimuthal`, and `tilt`
- orientation refinement under the updated geometry
- acceptance/rejection of each update based on post-refinement NCC improvement

### Part 1C outputs

- `MAPNAME_CalibratedDetector.txt` — refined detector field extrapolated back to the full map  
- `MAPNAME_GeomRefine/` — geometry-refinement diagnostics and iteration outputs  
- `MAPNAME_GeomRefine/job.out` — moved log file from the Slurm run

---

### Part 2 — Final pseudosymmetry-aware refinement and confidence scoring (`ReindexingPS_Part2_NCCrefPScheck.py`)

This is the final reindexing stage.

The script:

- loads the NPA-processed patterns and calibrated detector
- constructs six pseudosymmetry-related seeds from the original Hough map
- refines each seed **without allowing switching during the refinement pass**
- computes CI<sub>wcc</sub>-related metrics for each variant using `compute_wcc_map()`
- stores per-variant metrics in HDF5
- selects the best variant per pixel based on the scalar confidence metric
- reconstructs the final selected crystal map and writes it to `.ang`

### Part 2 outputs

Inside:

- `MAPNAME_NPA<RADIUS>_Refine-1step/`

The main outputs are:

- `MAPNAME_Refine1step_V0.ang` … `MAPNAME_Refine1step_V5.ang` — refined maps for each pseudosymmetry seed  
- `MAPNAME_CIwcc_data.h5` — per-variant and final confidence datasets  
- `MAPNAME_FinalSelected.ang` — final selected map after variant competition

The final HDF5 file contains datasets such as:

- `CI_stack`
- `V0/` … `V5/` groups with `CI_map`, `eta_map`, `Xi_e_map`, `Xi_t_map`, and `Xi_eN_map`
- `Final/best_idx`
- `Final/cross_variant_margin`
- `Final/Xi_t`
- `Final/Xi_eN`
- `Final/Xi_e`
- `Final/CI_sel`

---

## Repository structure

```text
.
├── EBSD_extra_functions.py
├── EBSD_refine_geometry.py
├── ReindexingPS_Part1A_PatternProcessing.py
├── ReindexingPS_Part1B_NPA_MPI.py
├── ReindexingPS_Part1C_GlobalGeomRefine.py
├── ReindexingPS_Part2_NCCrefPScheck.py
├── run_Part1A.sh
├── run_Part1B_MPI.sh
├── run_Part1C.sh
├── run_Part2_MPI.sh
└── submit_pipeline.sh
```

### Helper modules

#### `EBSD_extra_functions.py`
Contains utility functions used across the pipeline, including for example:

- detector cropping
- chunk-size selection for Dask
- EBSD subset extraction (`EBSD_subset`)
- binned detector construction
- CI<sub>wcc</sub> calculations (`compute_wcc_map`)
- diagnostic GIF generation (`make_flash_gif`)

#### `EBSD_refine_geometry.py`
Contains the geometry-refinement implementation, including:

- pattern simulation helpers
- ROI subdivision and visualization tools
- PCC- and sparse-feature-based displacement estimation
- iterative geometry refinement with optional PC updates
- orientation-refinement / NCC-based acceptance logic

---

## Input data assumptions

The pipeline assumes the following input files are already available:

- `MAPNAME.h5` — raw EBSD patterns
- `MAPNAME.ang` — initial orientation map
- `MP_PATH` — path to an EBSD master pattern HDF5 file with:
  - `Data/Master/Dynamical/Lower`
  - `Data/Master/Dynamical/Upper`

It also expects initial detector parameters to be passed through environment variables or command-line submission arguments:

- `PCX`
- `PCY`
- `PCZ`
- `SAMPLE_TILT_DEG`
- `ENERGY_KV`
- `RADIUS`

---

## Running the pipeline

### Submit the full pipeline

```bash
./submit_pipeline.sh \
  --pname /cluster/work/.../your_dataset/ \
  --mapname YOUR_MAP_NAME \
  --mp_path /cluster/work/.../MasterPatterns/BTOsc_25kV.sdf5 \
  --energy 25 \
  --pcx 0.50 \
  --pcy 0.75 \
  --pcz 0.60 \
  --sample_tilt 70 \
  --radius 7
```

### Restart from an intermediate stage

```bash
./submit_pipeline.sh \
  --start-from 1C \
  --pname /cluster/work/.../your_dataset/ \
  --mapname YOUR_MAP_NAME \
  --mp_path /cluster/work/.../MasterPatterns/BTOsc_25kV.sdf5 \
  --energy 25 \
  --pcx 0.50 \
  --pcy 0.75 \
  --pcz 0.60 \
  --sample_tilt 70 \
  --radius 7
```

Allowed values for `--start-from` are:

- `1A`
- `1B`
- `1C`
- `2`

The submission script chains the Slurm jobs using `afterok` dependencies.

---

## Slurm wrappers

The repository includes ready-to-use Slurm scripts for each stage:

- `run_Part1A.sh`
- `run_Part1B_MPI.sh`
- `run_Part1C.sh`
- `run_Part2_MPI.sh`

Current resource requests in the uploaded versions are:

- **Part 1A:** 1 task, 96 GB / CPU, 5 h  
- **Part 1B:** 15 tasks, 16 GB / CPU, 5 h  
- **Part 1C:** 1 task, 16 GB / CPU, 5 h  
- **Part 2:** 50 tasks, 15 GB / CPU, 48 h

These settings are useful starting points, but they will likely need adjustment for different map sizes and cluster environments.

---

## Dependencies

The codebase appears to rely on the following Python packages:

- `numpy`
- `matplotlib`
- `h5py`
- `kikuchipy`
- `orix`
- `hyperspy`
- `scikit-optimize`
- `scikit-image`
- `opencv-python`
- `scipy`
- `dask`
- `distributed`
- `dask-mpi`
- `imageio`
- `pandas`
- `pyvista`
- `tqdm`

Additional notes:

- the code imports a local compiled helper module named `EBSD_extra_functions_numba`
- the Slurm scripts target the ETH Euler software stack `stack/2024-06` with Python `3.12.8`
- MPI-enabled stages load `openmpi/4.1.6`

Because no environment file was included in the uploaded material, this dependency list should be treated as a **best-effort reconstruction** from the scripts rather than a locked environment specification.

---

## Notes for a public repository

Before making the repository public, you may want to:

- replace hard-coded cluster paths such as `/cluster/work/...` and `/cluster/project/...`
- include an `environment.yml` or `requirements.txt`
- add a small example dataset or a synthetic demo
- document the meaning of CI<sub>wcc</sub>, `Xi_e`, `Xi_t`, and `Xi_eN` in more detail
- add a license
- add a citation section if this accompanies a manuscript or preprint

---

## Suggested citation text

If you plan to attach this repository to a manuscript, a short citation block could be added here later, for example with the paper title, DOI, and versioned GitHub release / Zenodo archive.

---

## Contact

For questions about the method or implementation, please open an issue or contact the repository maintainer.
