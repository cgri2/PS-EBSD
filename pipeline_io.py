from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

def load_config(config_path: str | os.PathLike) -> dict[str, Any]:
    """Load a TOML pipeline config file."""
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        return tomllib.load(f)


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return one config section, or an empty dict if missing."""
    return dict(config.get(section, {}))


def get_global(config: dict[str, Any]) -> dict[str, Any]:
    """Return the global config section."""
    if "global" not in config:
        raise KeyError("Config file must contain a [global] section.")
    return dict(config["global"])


def resolve_pipeline_paths(config: dict[str, Any], config_path: str | os.PathLike | None = None) -> dict[str, str]:
    """
    Resolve pname, mapname, mp_path, and canonical h5_path.

    Relative paths are interpreted relative to the config file location.
    """
    g = get_global(config)

    base = Path(config_path).resolve().parent if config_path else Path.cwd()

    pname = Path(g["pname"])
    if not pname.is_absolute():
        pname = (base / pname).resolve()

    mp_path = Path(g["mp_path"])
    if not mp_path.is_absolute():
        mp_path = (base / mp_path).resolve()

    mapname = str(g["mapname"])
    h5_path = pname / f"{mapname}.h5"

    return {
        "pname": str(pname),
        "mapname": mapname,
        "mp_path": str(mp_path),
        "h5_path": str(h5_path),
    }


def get_ps_rotations(config: dict[str, Any]):
    """
    Convert config [global].PS_rotations into an orix Rotation object.

    Expected config format:
        PS_rotations = [
          [1, 0, 0, 180],
          [1, 0, 0,  90],
          ...
        ]
    """
    import numpy as np
    from orix.quaternion import Rotation

    rows = np.asarray(config["global"]["PS_rotations"], dtype=float)

    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError("global.PS_rotations must have shape (N, 4): [ax, ay, az, angle_deg]")

    axes = rows[:, :3]
    angles = rows[:, 3]

    return Rotation.from_axes_angles(axes, angles, degrees=True)

def get_overwrite_h5(config: dict) -> bool:
    """Return global overwriteH5 setting."""
    return bool(config.get("global", {}).get("overwriteH5", True))


def get_part1b_radius(config: dict) -> int:
    """Return Part1B NPA radius."""
    return int(config.get("part1B", {}).get("radius", 7))


def h5_path_for_stage(config: dict, config_path, stage: str) -> str:
    """
    Return the H5 path to read/write for a pipeline stage.

    Stages:
        Part0_output
        Part1A_input
        Part1A_output
        Part1B_input
        Part1B_output
        Part1C_input
        Part1C_output
        Part2_input
    """
    paths = resolve_pipeline_paths(config, config_path)
    pname = Path(paths["pname"])
    mapname = paths["mapname"]

    overwrite = get_overwrite_h5(config)
    radius = get_part1b_radius(config)

    h5_original = pname / f"{mapname}.h5"
    h5_pp = pname / f"{mapname}_PP.h5"
    h5_npa = pname / f"{mapname}_PP_NPA{radius}.h5"

    if overwrite:
        return str(h5_original)

    if stage in ("Part0_output", "Part1A_input"):
        return str(h5_original)

    if stage == "Part1A_output":
        return str(h5_pp)

    if stage == "Part1B_input":
        return str(h5_pp)

    if stage in ("Part1B_output", "Part1C_input", "Part1C_output", "Part2_input"):
        return str(h5_npa)

    raise ValueError(f"Unknown stage: {stage}")
