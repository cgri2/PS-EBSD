import numpy as np
import os
from orix.crystal_map import CrystalMap
from orix import io
import h5py
import time
import pipeline_io as pio

start = time.time()

# ---- Read environment variables ----
config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    raise RuntimeError("CONFIG_PATH is not set.")

config = pio.load_config(config_path)
paths = pio.resolve_pipeline_paths(config, config_path)

pname   = paths["pname"]
mapname = paths["mapname"]
cfg1B   = config.get("part1B", {})
r       = int(cfg1B.get("radius", 7))
cfg2    = config.get("part2", {})

outdir_suffix = cfg2.get("outdir_suffix", f"NPA{r}_Refine-1step")
outdir = os.path.join(pname, f"{mapname}_{outdir_suffix}")

PS_rotations = pio.get_ps_rotations(config)
n_variants = len(PS_rotations) + 1

print(f"Part2B: assembling {n_variants} variants from {outdir}")

# ---- Verify all Part2A outputs exist ----
for vi in range(n_variants):
    ang_path = os.path.join(outdir, f"{mapname}_Refine1step_V{vi}.ang")
    h5_path  = os.path.join(outdir, f"{mapname}_CIwcc_V{vi}.h5")
    if not os.path.exists(ang_path):
        raise FileNotFoundError(f"Missing Part2A .ang output for V{vi}: {ang_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing Part2A .h5 output for V{vi}: {h5_path}")

# ---- Read Ny, Nx and tensor dimension from V0 HDF5 ----
h5_v0 = os.path.join(outdir, f"{mapname}_CIwcc_V0.h5")
with h5py.File(h5_v0, "r") as f:
    Ny, Nx   = f["V0"]["CI_map"].shape
    n_tensor = f["V0"]["Xi_t_map"].shape[2]

print(f"Map shape: {Ny}×{Nx}, tensor components: {n_tensor}")

# ---- Load per-variant .ang files to get CrystalMap objects ----
refined_maps = []
for vi in range(n_variants):
    ang_path = os.path.join(outdir, f"{mapname}_Refine1step_V{vi}.ang")
    xmap_v = io.load(ang_path)
    refined_maps.append(xmap_v)
    print(f"Loaded V{vi} from {ang_path}")

# ---- Assemble CI stack from per-variant HDF5 files ----
CI_stack = np.empty((Ny, Nx, n_variants), dtype="f4")
for vi in range(n_variants):
    h5_v = os.path.join(outdir, f"{mapname}_CIwcc_V{vi}.h5")
    with h5py.File(h5_v, "r") as f:
        CI_stack[:, :, vi] = f[f"V{vi}"]["CI_map"][:]
    print(f"Loaded CI map for V{vi}")

# ---- Select best variant per pixel ----
best_idx = np.argmax(CI_stack, axis=-1).astype(np.uint8)
ci_sorted = np.sort(CI_stack, axis=-1)
cross_variant_margin = (ci_sorted[..., -1] - ci_sorted[..., -2]).astype("f4")

# ---- Allocate final selection arrays ----
Xi_t_sel  = np.empty((Ny, Nx, n_tensor), dtype="f4")
Xi_eN_sel = np.empty((Ny, Nx, n_tensor), dtype="f4")
Xi_e_sel  = np.empty((Ny, Nx), dtype="f4")
CI_sel    = np.empty((Ny, Nx), dtype="f4")

# ---- Write merged HDF5, filling per-variant groups and Final group ----
final_h5 = os.path.join(outdir, f"{mapname}_CIwcc_data.h5")
with h5py.File(final_h5, "w") as f:
    f.create_dataset(
        "CI_stack", data=CI_stack,
        chunks=(min(Ny, 128), min(Nx, 128), 1), compression="gzip",
    )

    for vi in range(n_variants):
        h5_v = os.path.join(outdir, f"{mapname}_CIwcc_V{vi}.h5")
        with h5py.File(h5_v, "r") as fv:
            g_src = fv[f"V{vi}"]
            g_dst = f.create_group(f"V{vi}")
            for key in g_src.keys():
                g_dst.create_dataset(key, data=g_src[key][:], compression="gzip")

        # Fill per-pixel best-variant selections
        mask = (best_idx == vi)
        g = f[f"V{vi}"]
        Xi_t_v  = g["Xi_t_map"][...]
        Xi_eN_v = g["Xi_eN_map"][...]
        Xi_e_v  = g["Xi_e_map"][...]
        CI_v    = g["CI_map"][...]

        Xi_e_sel[mask] = Xi_e_v[mask]
        CI_sel[mask]   = CI_v[mask]
        m3 = np.repeat(mask[:, :, None], n_tensor, axis=2)
        Xi_t_sel[m3]  = Xi_t_v[m3]
        Xi_eN_sel[m3] = Xi_eN_v[m3]

    gfin = f.create_group("Final")
    gfin.create_dataset("best_idx",             data=best_idx,             compression="gzip")
    gfin.create_dataset("cross_variant_margin", data=cross_variant_margin, compression="gzip")
    gfin.create_dataset("Xi_t",                 data=Xi_t_sel,             compression="gzip")
    gfin.create_dataset("Xi_eN",                data=Xi_eN_sel,            compression="gzip")
    gfin.create_dataset("Xi_e",                 data=Xi_e_sel,             compression="gzip")
    gfin.create_dataset("CI_sel",               data=CI_sel,               compression="gzip")

print(f"Saved merged HDF5: {final_h5}")

# ---- Build final crystal map from best rotations per pixel ----
print("Selecting best variant per pixel...")
N = Ny * Nx
variant = best_idx.reshape(-1)
xmap_ref = refined_maps[0]

rots = refined_maps[0].rotations
for v in range(n_variants):
    m = (variant == v)
    if np.any(m):
        rots[m] = refined_maps[v].rotations[m]

assert rots.size == N

xmap_final = CrystalMap(
    rotations=rots,
    phase_id=xmap_ref.phase_id,
    x=xmap_ref.x,
    y=xmap_ref.y,
    phase_list=xmap_ref.phases,
    prop=xmap_ref.prop,
    scan_unit=xmap_ref.scan_unit,
    is_in_data=np.ones(xmap_ref.x.size, dtype=bool),
)

final_ang = os.path.join(outdir, f"{mapname}_FinalSelected.ang")
io.save(final_ang, xmap_final, overwrite=True)

tf = time.time()
print(f"All done. Total time: {tf - start:.2f} s. Final outputs saved in: {outdir}")
