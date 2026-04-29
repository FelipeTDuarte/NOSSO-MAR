"""
Physics-Informed DeepONet (PI-DeepONet).

Adds PDE residual terms to the training loss so that the operator
satisfies the governing equations in the interior, not just on data.

For Module 2 (WEC FSI), the physics residual enforces the equation of motion:
    R(ω) = [-ω²(M + A(ω)) + iω(B(ω) + B_pto) + C] X(ω) - F_ex(ω) = 0

For Module 1 (wave propagation), the residual can enforce the mild-slope equation.
"""
from __future__ import annotations
from typing import Dict, Callable, Optional
import torch

from .deeponet import DeepONet


class PhysicsInformedDeepONet(DeepONet):
    """
    PI-DeepONet with pluggable PDE residual function.

    cfg keys: all DeepONet keys + pde_weight (float, default 0.1)
    """

    def __init__(self, cfg: Dict, pde_residual_fn: Optional[Callable] = None):
        super().__init__(cfg)
        self.pde_weight       = cfg.get("pde_weight", 0.1)
        self.pde_residual_fn  = pde_residual_fn  # callable(outputs, inputs) -> residual

    def physics_loss(self, u: torch.Tensor,
                     query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss at collocation points.
        Requires query_points to have requires_grad=True for auto-diff.
        """
        if self.pde_residual_fn is None:
            return torch.tensor(0.0, device=u.device)

        query_points = query_points.requires_grad_(True)
        out = self.forward(u, query_points)
        residual = self.pde_residual_fn(out, query_points)
        return (residual ** 2).mean()
