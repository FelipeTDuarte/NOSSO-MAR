"""
Factorised FNO (F-FNO) — reduces memory by factorising 2D/3D spectral
convolutions into 1D passes along each spatial dimension.

References:
    Tran et al. (2023) — Factorized Fourier Neural Operators
    https://arxiv.org/abs/2111.13802
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseOperator


class FactorisedSpectralConv2d(nn.Module):
    """
    F(x) ≈ F_y^{-1}[ W_y · F_x^{-1}[ W_x · F_x[u] ] ]
    Two sequential 1-D spectral convolutions instead of one 2-D.
    """

    def __init__(self, channels: int, modes_x: int, modes_y: int):
        super().__init__()
        scale = 1.0 / channels
        self.wx = nn.Parameter(scale * torch.randn(channels, channels, modes_x, 2))
        self.wy = nn.Parameter(scale * torch.randn(channels, channels, modes_y, 2))

    def _mul1d(self, a, w):
        return torch.view_as_real(
            torch.einsum("bcx,ocx->box",
                         torch.view_as_complex(a),
                         torch.view_as_complex(w)))

    def forward(self, x):
        B, C, H, W = x.shape
        # --- x-direction ---
        xf = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="ortho"))
        out = torch.zeros_like(xf)
        out[:, :, :, :self.wx.shape[2]] = self._mul1d(
            xf[:, :, :, :self.wx.shape[2]], self.wx)
        x2 = torch.fft.irfft(torch.view_as_complex(out), n=W, dim=-1, norm="ortho")
        # --- y-direction ---
        yf = torch.view_as_real(torch.fft.rfft(x2, dim=-2, norm="ortho"))
        out2 = torch.zeros_like(yf)
        x_ft_c = torch.view_as_complex(yf[:, :, :self.wy.shape[2], :])
        out_c = torch.einsum("bcx,ocx->box",
                              x_ft_c,
                              torch.view_as_complex(self.wy))
        out2[:, :, :self.wy.shape[2], :] = torch.view_as_real(out_c)
        return torch.fft.irfft(torch.view_as_complex(out2), n=H, dim=-2, norm="ortho")


class FFNO2d(BaseOperator):
    """Memory-efficient factorised FNO for large spatial domains."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        C_in  = cfg["in_channels"]
        C_out = cfg["out_channels"]
        W     = cfg.get("width",   64)
        mx    = cfg.get("modes_x", 16)
        my    = cfg.get("modes_y", 16)
        n_l   = cfg.get("n_layers", 4)

        self.lift   = nn.Conv2d(C_in, W, 1)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "spectral": FactorisedSpectralConv2d(W, mx, my),
                "residual": nn.Conv2d(W, W, 1),
                "norm":     nn.InstanceNorm2d(W),
            }) for _ in range(n_l)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(W, 128, 1), nn.GELU(), nn.Conv2d(128, C_out, 1))

    def forward(self, u, query_points=None):
        x = self.lift(u)
        for layer in self.layers:
            x = F.gelu(layer["norm"](layer["spectral"](x) + layer["residual"](x)))
        return self.proj(x)
