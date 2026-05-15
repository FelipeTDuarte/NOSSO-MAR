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

    Weights are stored as complex parameters to avoid view_as_complex
    on the weight tensors at every forward pass.

    Shape conventions:
        rfft(x, dim=-1) → (B, C, H, W//2+1) — H is a free spatial dim for x-conv
        rfft(x, dim=-2) → (B, C, H//2+1, W) — W is a free spatial dim for y-conv
        Both einsums therefore carry the free spatial dimension explicitly.
    """

    def __init__(self, channels: int, modes_x: int, modes_y: int):
        super().__init__()
        scale = 1.0 / channels
        # Complex parameters: (C_out, C_in, modes)
        self.wx = nn.Parameter(
            scale * torch.randn(channels, channels, modes_x, dtype=torch.cfloat))
        self.wy = nn.Parameter(
            scale * torch.randn(channels, channels, modes_y, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # --- x-direction: rfft along last dim, H is the free spatial dim ---
        xf = torch.fft.rfft(x, dim=-1, norm="ortho")  # (B, C, H, W//2+1)
        Mx = self.wx.shape[2]
        out = torch.zeros_like(xf)
        out[:, :, :, :Mx] = torch.einsum(
            "bchx,ocx->bohx", xf[:, :, :, :Mx], self.wx)  # h is free
        x2 = torch.fft.irfft(out, n=W, dim=-1, norm="ortho")  # (B, C, H, W)

        # --- y-direction: rfft along second-to-last dim, W is the free spatial dim ---
        yf = torch.fft.rfft(x2, dim=-2, norm="ortho")  # (B, C, H//2+1, W)
        My = self.wy.shape[2]
        out2 = torch.zeros_like(yf)
        out2[:, :, :My, :] = torch.einsum(
            "bcyw,ocy->boyw", yf[:, :, :My, :], self.wy)  # w is free
        return torch.fft.irfft(out2, n=H, dim=-2, norm="ortho")  # (B, C, H, W)


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
