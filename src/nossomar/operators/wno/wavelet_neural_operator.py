"""
Wavelet Neural Operator (WNO) for multi-resolution wave field prediction.

Particularly suited for shallow-to-deep water wave propagation where
phenomena span multiple spatial scales (swell vs. wind-sea, shoaling).

Architecture:
    Lift → [WaveletConv + residual conv + norm + act] x L → Project
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_conv import WaveletConv2d
from ..base import BaseOperator


class WNOBlock(nn.Module):
    def __init__(self, width: int, levels: int, act: str = "gelu"):
        super().__init__()
        self.wav  = WaveletConv2d(width, levels)
        self.res  = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.act  = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[act]

    def forward(self, x):
        return self.act(self.norm(self.wav(x) + self.res(x)))


class WaveletNeuralOperator(BaseOperator):
    """
    WNO for phase-resolving wave propagation (Module 1 primary candidate).

    cfg keys:
        in_channels  : int
        out_channels : int
        width        : int  (default 64)
        levels       : int  (DWT levels, default 3)
        n_layers     : int  (WNO blocks, default 4)
        activation   : str  (default gelu)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        C_in  = cfg["in_channels"]
        C_out = cfg["out_channels"]
        W     = cfg.get("width",      64)
        lv    = cfg.get("levels",      3)
        n_l   = cfg.get("n_layers",    4)
        act   = cfg.get("activation", "gelu")

        self.lift   = nn.Conv2d(C_in, W, 1)
        self.blocks = nn.ModuleList([WNOBlock(W, lv, act) for _ in range(n_l)])
        self.proj   = nn.Sequential(
            nn.Conv2d(W, 128, 1), nn.GELU(), nn.Conv2d(128, C_out, 1))

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.lift(u)
        for b in self.blocks:
            x = b(x)
        return self.proj(x)
