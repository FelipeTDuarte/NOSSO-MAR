"""Core contracts for the local NOSSO-MAR Phase 1 baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array


def _as_3d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (time, y, x).")
    return array


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(slots=True)
class WECState:
    """Frequency-domain hydrodynamic state for a single WEC."""

    freq: np.ndarray
    added_mass: np.ndarray
    damping: np.ndarray
    excitation_real: np.ndarray
    excitation_imag: np.ndarray
    device_type: str
    radius: float
    draft: float
    mass: float
    bpto: float
    depth: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq = _as_1d_array(self.freq, "freq")
        self.added_mass = _as_1d_array(self.added_mass, "added_mass")
        self.damping = _as_1d_array(self.damping, "damping")
        self.excitation_real = _as_1d_array(self.excitation_real, "excitation_real")
        self.excitation_imag = _as_1d_array(self.excitation_imag, "excitation_imag")
        self.radius = float(self.radius)
        self.draft = float(self.draft)
        self.mass = float(self.mass)
        self.bpto = float(self.bpto)
        self.depth = float(self.depth)
        self.validate()

    def validate(self) -> None:
        expected_shape = self.freq.shape
        arrays = {
            "added_mass": self.added_mass,
            "damping": self.damping,
            "excitation_real": self.excitation_real,
            "excitation_imag": self.excitation_imag,
        }
        for name, array in arrays.items():
            if array.shape != expected_shape:
                raise ValueError(f"{name} must match freq shape {expected_shape}.")

        if len(self.freq) == 0:
            raise ValueError("freq must not be empty.")
        if np.any(self.freq <= 0.0):
            raise ValueError("freq values must be positive.")
        if np.any(np.diff(self.freq) <= 0.0):
            raise ValueError("freq must be strictly increasing.")
        if np.any(self.added_mass < 0.0):
            raise ValueError("added_mass must be non-negative.")
        if np.any(self.damping < 0.0):
            raise ValueError("damping must be non-negative.")
        if self.radius <= 0.0 or self.draft <= 0.0 or self.mass <= 0.0 or self.depth <= 0.0:
            raise ValueError("radius, draft, mass, and depth must be positive.")
        if self.bpto < 0.0:
            raise ValueError("bpto must be non-negative.")

    @property
    def excitation_complex(self) -> np.ndarray:
        return self.excitation_real + 1j * self.excitation_imag

    def to_dict(self) -> dict[str, Any]:
        return {
            "freq": self.freq.tolist(),
            "added_mass": self.added_mass.tolist(),
            "damping": self.damping.tolist(),
            "excitation_real": self.excitation_real.tolist(),
            "excitation_imag": self.excitation_imag.tolist(),
            "device_type": self.device_type,
            "radius": self.radius,
            "draft": self.draft,
            "mass": self.mass,
            "bpto": self.bpto,
            "depth": self.depth,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WECState":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        _json_write(Path(path), self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "WECState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(slots=True)
class WaveField:
    """Minimal phase-resolved wave field representation."""

    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    elevation: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = _as_1d_array(self.x, "x")
        self.y = _as_1d_array(self.y, "y")
        self.time = _as_1d_array(self.time, "time")
        self.elevation = _as_3d_array(self.elevation, "elevation")
        self.validate()

    def validate(self) -> None:
        expected_shape = (len(self.time), len(self.y), len(self.x))
        if self.elevation.shape != expected_shape:
            raise ValueError(
                f"elevation must have shape (time, y, x) = {expected_shape}, "
                f"got {self.elevation.shape}."
            )
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time must be strictly increasing.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "time": self.time.tolist(),
            "elevation": self.elevation.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WaveField":
        return cls(**payload)


@dataclass(slots=True)
class OceanState:
    """Portable container for one local baseline case."""

    wec_states: list[WECState]
    wave_field: WaveField | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.wec_states:
            raise ValueError("OceanState requires at least one WECState.")
        for state in self.wec_states:
            state.validate()
        if self.wave_field is not None:
            self.wave_field.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "wec_states": [state.to_dict() for state in self.wec_states],
            "wave_field": None if self.wave_field is None else self.wave_field.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OceanState":
        wec_states = [WECState.from_dict(item) for item in payload["wec_states"]]
        wave_field_payload = payload.get("wave_field")
        wave_field = None if wave_field_payload is None else WaveField.from_dict(wave_field_payload)
        return cls(wec_states=wec_states, wave_field=wave_field, metadata=payload.get("metadata", {}))

    def to_json(self, path: str | Path) -> None:
        _json_write(Path(path), self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "OceanState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
