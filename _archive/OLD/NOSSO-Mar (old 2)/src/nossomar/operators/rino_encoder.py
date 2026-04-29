"""RINO — Resolution-Independent Neural Operator encoder.

Maps an arbitrary point cloud of geometry samples to a fixed-size latent
vector using a PointNet-style shared MLP + global max-pool. The design is
resolution-invariant by construction: the aggregation (max-pool) does not
depend on N_pts, so the same geometry sampled at different densities produces
a consistent latent representation.

Reference: Qi et al. (2017) PointNet, CVPR. https://arxiv.org/abs/1612.00593
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _SharedMLP(nn.Module):
    """Per-point MLP applied identically across all N_pts (shared weights)."""

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C_in) → (B, N, C_out)
        return self.net(x)


class RINOEncoder(nn.Module):
    """Resolution-Independent Neural Operator encoder.

    Args:
        in_features: Number of per-point feature channels F (e.g. 4 for
                     normal_x, normal_y, normal_z, panel_area).
        d_latent:    Output latent dimension. Default 64.
        hidden:      Hidden channel width for the shared MLPs. Default 128.
    """

    def __init__(
        self,
        in_features: int,
        d_latent: int = 64,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        in_channels = 3 + in_features  # xyz coords + per-point features

        # Local feature extractor (per-point)
        self.local_mlp = _SharedMLP([in_channels, hidden, hidden])

        # Global context after max-pool
        self.global_mlp = _SharedMLP([hidden, hidden, d_latent])

        self.d_latent = d_latent

    def forward(
        self,
        points: torch.Tensor,   # (B, N_pts, 3)
        features: torch.Tensor, # (B, N_pts, F)
    ) -> torch.Tensor:          # (B, d_latent)
        # Concatenate spatial coords and per-point features
        x = torch.cat([points, features], dim=-1)  # (B, N, 3+F)

        # Per-point projection
        x = self.local_mlp(x)                      # (B, N, hidden)

        # Resolution-invariant global aggregation
        x, _ = x.max(dim=1)                        # (B, hidden)  — global max-pool

        # Project to latent space
        x = x.unsqueeze(1)                         # (B, 1, hidden)
        x = self.global_mlp(x)                     # (B, 1, d_latent)
        return x.squeeze(1)                        # (B, d_latent)
