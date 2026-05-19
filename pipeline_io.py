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

