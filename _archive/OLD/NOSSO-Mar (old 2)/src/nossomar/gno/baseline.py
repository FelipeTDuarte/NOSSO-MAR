"""LinearSuperpositionBaseline: position-independent WEC farm baseline.

Wraps SixDOFSurrogate and returns per-device isolated (A, B) matrices.
No pairwise coupling. The GNO learns the residual correction on top of this.

The wrapper is frozen: no own trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..models.sixdof_surrogate import SixDOFSurrogate


class LinearSuperpositionBaseline(nn.Module):
    """Position-independent per-device hydrodynamic baseline.

    Args:
        surrogate : Trained or untrained SixDOFSurrogate instance.
                    The baseline does NOT own or copy the surrogate;
                    it holds a reference and calls it in eval mode.
    """

    def __init__(self, surrogate: SixDOFSurrogate) -> None:
        super().__init__()
        # Register as a non-trainable sub-module (no own parameters)
        self.surrogate = surrogate

    def forward(self, props: Tensor, omega: Tensor) -> tuple[Tensor, Tensor]:
        """Compute per-device isolated A and B matrices.

        Args:
            props : (N, 4)  WEC property vectors [radius, draft, depth, Bpto]
            omega : (K,)    frequency query points (rad/s)

        Returns:
            A_base : (N, 6, 6)  added-mass matrices (no coupling)
            B_base : (N, 6, 6)  radiation-damping matrices (no coupling)
        """
        N = props.shape[0]
        # SixDOFSurrogate expects omega shape (N, K)
        omega_exp = omega.unsqueeze(0).expand(N, -1)   # (N, K)
        return self.surrogate(props, omega_exp)         # (N, 6, 6), (N, 6, 6)

    def parameters(self, recurse: bool = True):
        """Expose no own parameters; surrogate params are separate."""
        if recurse:
            return self.surrogate.parameters(recurse=True)
        return iter([])   # no own parameters
