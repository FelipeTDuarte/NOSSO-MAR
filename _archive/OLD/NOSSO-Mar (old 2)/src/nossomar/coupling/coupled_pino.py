"""Coupled PINO execution loop entrypoint.

This module wires together the wave-field operator and the multi-object FSI
adapter into a single coupled forward pass. The current implementation is a
contract stub that defines the interface without executing real physics.
"""

from ..core.contracts import WECState, WaveField
from ..modules.multi_object_fsi import MultiObjectFSI


class CoupledPINO:
    """Coupled physics-informed neural operator for wave + WSI.

    Args:
        fsi: MultiObjectFSI adapter instance.
    """

    def __init__(self, fsi: MultiObjectFSI | None = None):
        self.fsi = fsi or MultiObjectFSI()

    def step(
        self, wave_field: WaveField, state: WECState
    ) -> tuple[WaveField, WECState]:
        """Advance one coupled step.

        Args:
            wave_field: current wave field.
            state: current WEC state.

        Returns:
            next_wave_field: propagated wave field (identity in stub).
            next_state: updated WEC state after FSI.
        """
        next_state, _ = self.fsi(wave_field, state)
        return wave_field, next_state
