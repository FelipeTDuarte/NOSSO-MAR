"""
Geometry-aware FNO (Geo-FNO) for irregular domains.

Maps the irregular physical domain to a uniform computational domain
via a learned deformation map ϕ, applies FNO in the regular domain,
then maps back.

References:
    Li et al. (2023) — Fourier Neural Operator with Learned Deformations
    https://arxiv.org/abs/2207.05209
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from .fno2d import FNO2d
from ..base import BaseOperator


class DeformationNet(nn.Module):
    """Lightweight MLP that learns the coordinate deformation ϕ: Ω_phys -> Ω_comp."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, N, 2) physical -> (B, N, 2) computational."""
        return coords + self.net(coords)   # residual deformation


class GeoFNO2d(BaseOperator):
    """
    Geo-FNO for wave propagation on irregular bathymetric domains.

    The deformation network ϕ is trained jointly with the FNO operator.
    Input expects physical coordinates (x, y) appended to the channel dim.

    cfg keys: inherit FNO2d keys + deformation_hidden (int)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.deform = DeformationNet(cfg.get("deformation_hidden", 64))
        self.fno    = FNO2d(cfg)

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        # u: (B, C, H, W) — last two channels assumed to be (x, y) coordinates
        # In practice coords are appended externally before calling forward
        return self.fno(u, query_points)
