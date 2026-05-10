"""
2-D Fourier Neural Operator.

Architecture:
    Lifting layer  → P FNO layers → Projection layer
    Each FNO layer: SpectralConv2d (global) + Conv1x1 (local) + activation

References:
    Li et al. (2021) — https://arxiv.org/abs/2010.08895
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv import SpectralConv2d
from ..base import BaseOperator


class FNOBlock2d(nn.Module):
    """Single FNO layer: spectral path + residual path + activation."""

    def __init__(self, width: int, modes_x: int, modes_y: int,
                 activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.residual = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(width)
        self.act      = {"gelu": F.gelu, "relu": F.relu,
                         "silu": F.silu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.residual(x)))


class FNO2d(BaseOperator):
    """
    Full 2-D FNO for phase-resolving wave propagation and bathymetry fields.

    Input:  (B, C_in,  H, W) — e.g., [eta, u, v, h, mask]
    Output: (B, C_out, H, W) — e.g., [eta_next, u_next, v_next]

    Args:
        cfg (dict):
            in_channels  : int
            out_channels : int
            width        : int    (lifted channel width, default 64)
            modes_x      : int    (Fourier modes in x, default 16)
            modes_y      : int    (Fourier modes in y, default 16)
            n_layers     : int    (FNO blocks, default 4)
            activation   : str    (gelu | relu | silu)
            padding      : int    (zero-pad before FFT if domain non-periodic)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        C_in  = cfg["in_channels"]
        C_out = cfg["out_channels"]
        W     = cfg.get("width",      64)
        mx    = cfg.get("modes_x",    16)
        my    = cfg.get("modes_y",    16)
        n_l   = cfg.get("n_layers",    4)
        act   = cfg.get("activation", "gelu")
        self.padding = cfg.get("padding", 0)

        # Lifting: R^{C_in} -> R^W
        self.lift = nn.Conv2d(C_in, W, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock2d(W, mx, my, act) for _ in range(n_l)])

        # Projection: R^W -> R^{C_out}
        self.proj = nn.Sequential(
            nn.Conv2d(W, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, C_out, kernel_size=1),
        )

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """u: (B, C_in, H, W) -> (B, C_out, H, W)."""
        x = self.lift(u)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            x = block(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        return self.proj(x)
