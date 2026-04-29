"""
Refinement criteria for adaptive mesh refinement (AMR) in wave simulations.

Criteria decide which cells to refine or coarsen based on solution features.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class WaveGradientCriterion(nn.Module):
    """
    Refine where |∇η| exceeds a threshold (wave front / steep gradients).
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        """
        eta     : (B, 1, H, W) free-surface elevation
        Returns : (B, 1, H, W) bool tensor — True where refinement needed
        """
        # Gradient via finite differences
        dx = torch.roll(eta, -1, dims=-1) - torch.roll(eta, 1, dims=-1)
        dy = torch.roll(eta, -1, dims=-2) - torch.roll(eta, 1, dims=-2)
        grad_mag = (dx ** 2 + dy ** 2).sqrt()
        return grad_mag > self.threshold


class EnergyFluxCriterion(nn.Module):
    """
    Refine where wave energy flux P = E * c_g exceeds a threshold.
    Ensures adequate resolution in high-energy propagation corridors.
    """

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, eta: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
        """
        eta : (B, 1, H, W)  — surface elevation proxy for energy
        cg  : (B, 1, H, W)  — group velocity field
        """
        E = 0.5 * 9.81 * eta ** 2   # linear wave energy density
        flux = E * cg
        return flux.abs() > self.threshold
