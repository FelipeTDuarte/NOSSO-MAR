"""Field schemas that bridge gridded, graph and spectral fidelities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class GridDefinition:
    """Minimal grid descriptor used by regular-grid operator modules."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    t: np.ndarray | None = None
    crs: str = "EPSG:4326"
    metadata: dict[str, Any] = field(default_factory=dict)

    def shape(self) -> tuple[int, ...]:
        dims = [self.x.size, self.y.size]
        if self.z is not None:
            dims.append(self.z.size)
        if self.t is not None:
            dims.append(self.t.size)
        return tuple(dims)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["x"] = self.x.tolist()
        payload["y"] = self.y.tolist()
        payload["z"] = None if self.z is None else self.z.tolist()
        payload["t"] = None if self.t is None else self.t.tolist()
        return payload


@dataclass(slots=True)
class MultiFidelitySample:
    """Container describing aligned data products across fidelity levels."""

    sample_id: str
    grid: GridDefinition
    forcing: dict[str, Any]
    l0_observations: dict[str, Any] = field(default_factory=dict)
    l1_spectral: dict[str, Any] = field(default_factory=dict)
    l2_phase_resolved: dict[str, Any] = field(default_factory=dict)
    l3_wec_response: dict[str, Any] = field(default_factory=dict)
    l4_cfd: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def available_levels(self) -> list[str]:
        levels = {
            "l0_observations": self.l0_observations,
            "l1_spectral": self.l1_spectral,
            "l2_phase_resolved": self.l2_phase_resolved,
            "l3_wec_response": self.l3_wec_response,
            "l4_cfd": self.l4_cfd,
        }
        return [name for name, values in levels.items() if values]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grid"] = self.grid.to_dict()
        return payload
