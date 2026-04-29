"""Operator factory for NOSSO-MAR experiments."""

from __future__ import annotations

from typing import Any

from .deeponet.deeponet import DeepONet
from .deeponet.physics_deeponet import PhysicsInformedDeepONet
from .fno.ffno import FFNO2d
from .fno.fno2d import FNO2d
from .fno.fno3d import FNO3d
from .fno.geo_fno import GeoFNO2d
from .gno.graph_neural_operator import GraphNeuralOperator
from .rino import RINO2d
from .wno.wavelet_neural_operator import WaveletNeuralOperator


def build_operator(name: str, cfg: dict[str, Any], **kwargs: Any):
    """Instantiate one operator family by name."""

    normalized = name.strip().lower()
    registry = {
        "fno2d": FNO2d,
        "fno3d": FNO3d,
        "ffno2d": FFNO2d,
        "ffno": FFNO2d,
        "geofno2d": GeoFNO2d,
        "geo_fno2d": GeoFNO2d,
        "wno": WaveletNeuralOperator,
        "wavelet_neural_operator": WaveletNeuralOperator,
        "gno": GraphNeuralOperator,
        "graph_neural_operator": GraphNeuralOperator,
        "deeponet": DeepONet,
        "physics_deeponet": PhysicsInformedDeepONet,
        "pi_deeponet": PhysicsInformedDeepONet,
        "rino2d": RINO2d,
        "rino": RINO2d,
    }
    operator_cls = registry.get(normalized)
    if operator_cls is None:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown operator '{name}'. Available: {available}")
    if operator_cls is PhysicsInformedDeepONet:
        return operator_cls(cfg, pde_residual_fn=kwargs.get("pde_residual_fn"))
    return operator_cls(cfg)
