"""
Wavelet convolution layer for the Wavelet Neural Operator (WNO).

Uses a multi-level 2-D Haar/Daubechies DWT to decompose the input,
applies learnable filters at each scale, and reconstructs via iDWT.

References:
    Navaneeth & Chakraborty (2023) — Wavelet Neural Operator for solving PDEs
    https://arxiv.org/abs/2205.02191
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Haar DWT / iDWT (pure PyTorch, no pywavelets dependency) ──────────────

def haar_dwt2d(x: torch.Tensor) -> tuple:
    """
    One-level 2-D Haar DWT.
    x     : (B, C, H, W)
    Returns (LL, LH, HL, HH) each (B, C, H/2, W/2)
    """
    x_e  = x[:, :, 0::2, :]
    x_o  = x[:, :, 1::2, :]
    x_lo = (x_e + x_o) / 2.0
    x_hi = (x_e - x_o) / 2.0
    LL = (x_lo[:, :, :, 0::2] + x_lo[:, :, :, 1::2]) / 2.0
    LH = (x_lo[:, :, :, 0::2] - x_lo[:, :, :, 1::2]) / 2.0
    HL = (x_hi[:, :, :, 0::2] + x_hi[:, :, :, 1::2]) / 2.0
    HH = (x_hi[:, :, :, 0::2] - x_hi[:, :, :, 1::2]) / 2.0
    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH) -> torch.Tensor:
    """Inverse 2-D Haar DWT."""
    x_lo = torch.stack([LL + LH, LL - LH], dim=-1).flatten(-2)
    x_hi = torch.stack([HL + HH, HL - HH], dim=-1).flatten(-2)
    out  = torch.stack([x_lo + x_hi, x_lo - x_hi], dim=-2).flatten(-3, -2)
    return out


class WaveletConv2d(nn.Module):
    """
    Multi-level wavelet convolution operator.

    For each DWT level, learns separate 1x1 conv weights for each
    sub-band (LL, LH, HL, HH), enabling multi-scale filtering.
    """

    def __init__(self, channels: int, levels: int = 3):
        super().__init__()
        self.levels = levels
        # One conv per sub-band per level (LL is shared across levels as coarse)
        self.w_ll = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])
        self.w_lh = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])
        self.w_hl = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])
        self.w_hh = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Multi-level forward DWT
        coeff_list = []
        cur = x
        for lv in range(self.levels):
            LL, LH, HL, HH = haar_dwt2d(cur)
            coeff_list.append((LH, HL, HH, lv))
            cur = LL

        # Apply learnable weights at each level
        for LH, HL, HH, lv in reversed(coeff_list):
            LL_w  = self.w_ll[lv](cur)
            LH_w  = self.w_lh[lv](LH)
            HL_w  = self.w_hl[lv](HL)
            HH_w  = self.w_hh[lv](HH)
            cur   = haar_idwt2d(LL_w, LH_w, HL_w, HH_w)

        return cur
