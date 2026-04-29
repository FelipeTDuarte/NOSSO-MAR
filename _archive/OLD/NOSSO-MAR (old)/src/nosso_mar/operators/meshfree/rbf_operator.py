"""
Radial Basis Function Neural Operator (RBF-NO).

Mesh-free operator using RBF kernel interpolation combined with
a learned correction network. Suited for irregular WEC positions
and scattered measurement points.

Architecture:
    Input scattered points → RBF interpolation → uniform grid →
    convolutional correction → query points
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseOperator


class RBFLayer(nn.Module):
    """
    Learnable Gaussian RBF layer.
    centres: (M, d) — learnable centre locations
    log_scales: (M,) — learnable bandwidth (log for positivity)
    """

    def __init__(self, n_centres: int, in_dim: int, out_dim: int):
        super().__init__()
        self.centres    = nn.Parameter(torch.randn(n_centres, in_dim))
        self.log_scales = nn.Parameter(torch.zeros(n_centres))
        self.linear     = nn.Linear(n_centres, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) -> (..., out_dim)"""
        scales = self.log_scales.exp()                        # (M,)
        diff   = x.unsqueeze(-2) - self.centres              # (..., M, in_dim)
        r2     = (diff ** 2).sum(-1)                          # (..., M)
        phi    = torch.exp(-r2 * scales ** 2)                 # (..., M)  Gaussian RBF
        return self.linear(phi)


class RBFOperator(BaseOperator):
    """
    Mesh-free operator for scattered data (e.g. wave buoy observations,
    irregular computational grids around WEC structures).

    cfg keys:
        in_dim      : int  — spatial dim + field channels
        out_dim     : int
        hidden      : int
        n_centres   : int  (RBF centres, default 256)
        n_layers    : int  (MLP layers after RBF, default 3)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        in_dim    = cfg.get("in_dim",     5)
        out_dim   = cfg.get("out_dim",    3)
        hidden    = cfg.get("hidden",    64)
        n_centres = cfg.get("n_centres", 256)
        n_layers  = cfg.get("n_layers",    3)

        self.rbf = RBFLayer(n_centres, in_dim, hidden)
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u: (N, in_dim) scattered input (coordinates + field values)
        Returns: (N, out_dim)
        """
        return self.mlp(self.rbf(u))
