"""WEC-Sim data adapter (P2-T2).

Loads hydrodynamic coefficients from WEC-Sim output files into the xarray
schema used by the NOSSO-MAR training pipeline.

Supported formats:
  - JSON  : synthetic / hand-crafted benchmarks (used in CI, no extra deps)
  - HDF5  : real WEC-Sim .h5 output (requires h5py; install with pip install h5py)

Output schema (matches wec_dataset.py from Phase 1):
  Coordinates : omega  (N_omega,)  rad/s  -- strictly ascending
  Data vars   : added_mass   (case, omega)  kg
                damping      (case, omega)  kg/s
                radius       (case,)        m
                draft        (case,)        m
                depth        (case,)        m
                bpto         (case,)        N.s/m
  Attributes  : source, body_name
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

_REQUIRED_JSON_KEYS = (
    "omega",
    "added_mass_heave",
    "radiation_damping_heave",
    "draft",
    "depth",
    "bpto",
    "radius",
)


def _as_1d(x: object, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be a 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' contains non-finite values")
    return arr


def _to_dataset(
    omega: np.ndarray,
    added_mass: np.ndarray,
    damping: np.ndarray,
    radius: float,
    draft: float,
    depth: float,
    bpto: float,
    source: str,
    body_name: str,
) -> xr.Dataset:
    """Pack arrays into the standard NOSSO-MAR xarray schema."""
    # Sort by omega to guarantee ascending coordinate
    idx = np.argsort(omega)
    omega = omega[idx]
    added_mass = added_mass[idx]
    damping = damping[idx]

    return xr.Dataset(
        {
            "added_mass": (("case", "omega"), added_mass[np.newaxis, :]),
            "damping":    (("case", "omega"), damping[np.newaxis, :]),
            "radius":     (("case",), np.array([radius])),
            "draft":      (("case",), np.array([draft])),
            "depth":      (("case",), np.array([depth])),
            "bpto":       (("case",), np.array([bpto])),
        },
        coords={"omega": omega},
        attrs={"source": source, "body_name": body_name},
    )


def _load_json(path: Path) -> xr.Dataset:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    missing = [k for k in _REQUIRED_JSON_KEYS if k not in raw]
    if missing:
        raise KeyError(f"Benchmark JSON missing required keys: {missing}")

    omega = _as_1d(raw["omega"], "omega")
    added_mass = _as_1d(raw["added_mass_heave"], "added_mass_heave")
    damping = _as_1d(raw["radiation_damping_heave"], "radiation_damping_heave")

    if omega.shape != added_mass.shape or omega.shape != damping.shape:
        raise ValueError(
            "omega, added_mass_heave, and radiation_damping_heave must have the same length"
        )

    if not (added_mass > 0).all():
        raise ValueError("added_mass_heave must be strictly positive")
    if not (damping >= 0).all():
        raise ValueError("radiation_damping_heave must be non-negative")

    return _to_dataset(
        omega=omega,
        added_mass=added_mass,
        damping=damping,
        radius=float(raw["radius"]),
        draft=float(raw["draft"]),
        depth=float(raw["depth"]),
        bpto=float(raw["bpto"]),
        source=str(raw.get("source", "json")),
        body_name=str(raw.get("body_name", "unknown")),
    )


def _load_h5(path: Path) -> xr.Dataset:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load WEC-Sim HDF5 files.\n"
            "Install it with: pip install h5py"
        ) from exc

    with h5py.File(path, "r") as f:
        # WEC-Sim stores hydro data under 'body1' by convention
        # Adapt key paths as needed for your WEC-Sim version
        body_key = list(f.keys())[0]  # take first body
        body = f[body_key]

        omega = np.asarray(body["omega"], dtype=float).ravel()
        # Added mass: shape (6,6,N_omega) — take heave-heave (index 2,2)
        added_mass = np.asarray(body["added_mass"], dtype=float)
        if added_mass.ndim == 3:
            added_mass = added_mass[2, 2, :]
        else:
            added_mass = added_mass.ravel()

        damping = np.asarray(body["radiation_damping"], dtype=float)
        if damping.ndim == 3:
            damping = damping[2, 2, :]
        else:
            damping = damping.ravel()

        attrs = dict(body.attrs)
        radius = float(attrs.get("radius", 1.0))
        draft = float(attrs.get("draft", 1.0))
        depth = float(attrs.get("depth", 50.0))
        bpto = float(attrs.get("bpto", 1e4))
        body_name = str(attrs.get("body_name", body_key))

    return _to_dataset(
        omega=omega,
        added_mass=added_mass,
        damping=damping,
        radius=radius,
        draft=draft,
        depth=depth,
        bpto=bpto,
        source="wecsim",
        body_name=body_name,
    )


def load_wecsim_dataset(path: Union[str, Path]) -> xr.Dataset:
    """Load a WEC hydrodynamic dataset from a JSON or HDF5 file.

    Args:
        path: Path to a ``.json`` (synthetic/benchmark) or ``.h5`` (WEC-Sim) file.

    Returns:
        xr.Dataset with schema matching NOSSO-MAR Phase 1 wec_dataset.py.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is not supported.
        KeyError: If required keys are missing from a JSON file.
        ImportError: If h5py is not installed when loading an HDF5 file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    elif suffix in (".h5", ".hdf5", ".mat"):
        return _load_h5(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Expected .json, .h5, .hdf5, or .mat"
        )
