"""
3-D Fourier Neural Operator (x, y, t) for space-time wave fields.

Extends FNO2d to handle temporal evolution directly, useful for
time-series prediction of wave surface elevation η(x,y,t).

References:
    Li et al. (2021) — https://arxiv.org/abs/2010.08895
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv import SpectralConv3d
from ..base import BaseOperator


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, mx: int, my: int, mz: int, act: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, mx, my, mz)
        self.residual = nn.Conv3d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm3d(width)
        self.act_fn   = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[act]

    def forward(self, x):
        return self.act_fn(self.norm(self.spectral(x) + self.residual(x)))


class FNO3d(BaseOperator):
    """
    FNO operating on (x, y, t) volumes.

    Input:  (B, C_in,  H, W, T)
    Output: (B, C_out, H, W, T)

    cfg keys: in_channels, out_channels, width, modes_x, modes_y, modes_t,
              n_layers, activation, padding
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        C_in  = cfg["in_channels"]
        C_out = cfg["out_channels"]
        W     = cfg.get("width",    64)
        mx    = cfg.get("modes_x",  12)
        my    = cfg.get("modes_y",  12)
        mt    = cfg.get("modes_t",  12)
        n_l   = cfg.get("n_layers",  4)
        act   = cfg.get("activation", "gelu")
        self.padding = cfg.get("padding", 0)

        self.lift   = nn.Conv3d(C_in, W, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock3d(W, mx, my, mt, act) for _ in range(n_l)])
        self.proj   = nn.Sequential(
            nn.Conv3d(W, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, C_out, kernel_size=1),
        )

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.lift(u)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding] * 3)
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]
        return self.proj(x)
