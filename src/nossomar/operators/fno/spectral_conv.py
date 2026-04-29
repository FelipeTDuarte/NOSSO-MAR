"""
Spectral convolution layers for Fourier Neural Operators.

References:
    Li et al. (2021) — Fourier Neural Operator for Parametric PDEs
    https://arxiv.org/abs/2010.08895
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2-D Fourier integral operator.

    Computes: (K u)(x) = F^{-1}[ R(k) · F[u](k) ]
    where R(k) are learnable complex weights truncated to modes_x x modes_y.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x      = modes_x
        self.modes_y      = modes_y

        scale = 1.0 / (in_channels * out_channels)
        # Real and imaginary parts stored separately for compatibility
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))

    @staticmethod
    def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (B, Ci, X, Y) x (Ci, Co, X, Y) -> (B, Co, X, Y)."""
        a_c = torch.view_as_complex(a)
        b_c = torch.view_as_complex(b)
        return torch.view_as_real(torch.einsum("bixy,ioxy->boxy", a_c, b_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H, W)."""
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1)
        x_ft = torch.view_as_real(x_ft)           # (B, C, H, W//2+1, 2)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, 2,
                             device=x.device, dtype=x.dtype)

        # Low-frequency modes (quadrant 1 & 2)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self._compl_mul2d(
            x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self._compl_mul2d(
            x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2)

        out_ft = torch.view_as_complex(out_ft)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class SpectralConv3d(nn.Module):
    """
    3-D Fourier integral operator (x, y, t or x, y, z).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z, 2)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def _compl_mul3d(a, b):
        ac = torch.view_as_complex(a)
        bc = torch.view_as_complex(b)
        return torch.view_as_real(torch.einsum("bixyz,ioxyz->boxyz", ac, bc))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W, D = x.shape
        x_ft = torch.view_as_real(torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho"))
        Dh = D // 2 + 1

        out_ft = torch.zeros(B, self.out_channels, H, W, Dh, 2,
                             device=x.device, dtype=x.dtype)
        mx, my, mz = self.modes_x, self.modes_y, self.modes_z
        out_ft[:, :, :mx, :my, :mz]   = self._compl_mul3d(x_ft[:, :, :mx, :my, :mz],   self.weights1)
        out_ft[:, :, -mx:, :my, :mz]  = self._compl_mul3d(x_ft[:, :, -mx:, :my, :mz],  self.weights2)
        out_ft[:, :, :mx, -my:, :mz]  = self._compl_mul3d(x_ft[:, :, :mx, -my:, :mz],  self.weights3)
        out_ft[:, :, -mx:, -my:, :mz] = self._compl_mul3d(x_ft[:, :, -mx:, -my:, :mz], self.weights4)

        return torch.fft.irfftn(torch.view_as_complex(out_ft),
                                s=(H, W, D), dim=[-3, -2, -1], norm="ortho")
