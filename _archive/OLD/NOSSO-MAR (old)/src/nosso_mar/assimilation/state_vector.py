"""
State vector definition and manipulation for NOSSO-MAR data assimilation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np


@dataclass
class WaveStateVector:
    """
    State vector x = [η, u, v, h_bathy, Hs, Tp]
    flattened for EnKF / 4D-Var.

    Attributes:
        eta      : (H, W)   free-surface elevation [m]
        u_vel    : (H, W)   x-velocity [m/s]
        v_vel    : (H, W)   y-velocity [m/s]
        bathy    : (H, W)   bathymetry [m]
        Hs       : float    significant wave height [m]
        Tp       : float    peak period [s]
    """
    eta:   torch.Tensor
    u_vel: torch.Tensor
    v_vel: torch.Tensor
    bathy: torch.Tensor
    Hs:    float = 2.0
    Tp:    float = 8.0

    def to_vector(self) -> torch.Tensor:
        return torch.cat([
            self.eta.flatten(),
            self.u_vel.flatten(),
            self.v_vel.flatten(),
            self.bathy.flatten(),
            torch.tensor([self.Hs, self.Tp]),
        ])

    @classmethod
    def from_vector(cls, v: torch.Tensor, H: int, W: int) -> "WaveStateVector":
        n = H * W
        return cls(
            eta   = v[:n].reshape(H, W),
            u_vel = v[n:2*n].reshape(H, W),
            v_vel = v[2*n:3*n].reshape(H, W),
            bathy = v[3*n:4*n].reshape(H, W),
            Hs    = float(v[4*n]),
            Tp    = float(v[4*n + 1]),
        )

    def state_dim(self, H: int, W: int) -> int:
        return 4 * H * W + 2
