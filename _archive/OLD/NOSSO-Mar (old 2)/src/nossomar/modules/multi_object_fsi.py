"""Multi-object FSI contract adapter.

This module defines the contract-safe interface for wave-structure interaction
with multiple floating or fixed objects. The current implementation is a
contract-correct placeholder that returns zero force fields with the expected
shapes. Replace the internals in Phase 2 with a real learned or analytical FSI
operator.
"""

import torch
import torch.nn as nn
from ..core.contracts import WECState, WaveField


class MultiObjectFSI(nn.Module):
    """Contract-safe multi-object FSI adapter.

    Args:
        n_max_objects: maximum number of WEC objects supported in one forward pass.
        hidden: hidden dimension for internal projections (reserved for future use).
    """

    def __init__(self, n_max_objects: int = 8, hidden: int = 64):
        super().__init__()
        self.n_max_objects = n_max_objects
        self.hidden = hidden

    def forward(
        self, wave_field: WaveField, state: WECState
    ) -> tuple[WECState, torch.Tensor]:
        """Compute per-object forces and a spatial force field.

        Args:
            wave_field: WaveField with eta, u, v of shape (batch, H, W, T).
            state: WECState with pos (n_obj, 2), vel (n_obj, 6), force (n_obj, 6).

        Returns:
            out_state: updated WECState with forces applied (zero in placeholder).
            force_field: tensor of shape (batch, n_obj, H, W).
        """
        batch = wave_field.eta.shape[0]
        H = wave_field.eta.shape[1]
        W = wave_field.eta.shape[2]
        n_obj = state.pos.shape[0]

        if n_obj > self.n_max_objects:
            raise ValueError(
                f"n_objects={n_obj} exceeds n_max_objects={self.n_max_objects}"
            )

        force_out = torch.zeros_like(state.force)
        out_state = WECState(pos=state.pos, vel=state.vel, force=force_out)
        force_field = torch.zeros(batch, n_obj, H, W, device=wave_field.eta.device)
        return out_state, force_field
