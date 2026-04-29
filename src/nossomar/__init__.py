"""Multi-fidelity neural-operator workspace for NOSSO-MAR."""

from .core.contracts import OceanState, WaveField, WECState
from .core.field_schema import GridDefinition, MultiFidelitySample

__all__ = [
    "GridDefinition",
    "MultiFidelitySample",
    "OceanState",
    "WaveField",
    "WECState",
]
__version__ = "0.1.0"
