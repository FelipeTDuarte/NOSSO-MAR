"""
KAN-inspired operator (Kolmogorov-Arnold Network operator).

Uses learnable univariate spline functions instead of fixed activations,
potentially more parameter-efficient for smooth physical operators.

Reference: Liu et al. (2024) — KAN: Kolmogorov-Arnold Networks
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..base import BaseOperator


class BSplineActivation(nn.Module):
    """Learnable B-spline activation function."""

    def __init__(self, n_basis: int = 8, degree: int = 3):
        super().__init__()
        self.n_basis = n_basis
        self.degree  = degree
        self.coefs   = nn.Parameter(torch.randn(n_basis))
        # Fixed uniform knots
        self.register_buffer(
            "knots",
            torch.linspace(-1, 1, n_basis + degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple polynomial approximation (full B-spline basis in development)
        t  = x.clamp(-1, 1)
        bs = torch.stack([t ** k for k in range(self.n_basis)], dim=-1)
        return (bs * self.coefs).sum(-1)


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 8):
        super().__init__()
        self.activations = nn.ModuleList([
            BSplineActivation(n_basis) for _ in range(in_dim * out_dim)])
        self.in_dim  = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.zeros(B, self.out_dim, device=x.device)
        for j in range(self.out_dim):
            for i in range(self.in_dim):
                out[:, j] += self.activations[j * self.in_dim + i](x[:, i])
        return out


class KANOperator(BaseOperator):
    """Small KAN-based operator for smooth 1-D/low-dim mappings (e.g. RAO curves)."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        dims    = cfg.get("dims", [4, 32, 32, 4])
        n_basis = cfg.get("n_basis", 8)
        self.layers = nn.ModuleList([
            KANLayer(dims[i], dims[i + 1], n_basis)
            for i in range(len(dims) - 1)
        ])

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = u
        for layer in self.layers:
            x = layer(x)
        return x
