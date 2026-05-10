"""
Physics-informed loss terms for NOSSO-MAR neural operators.

Enforces PDE residuals at collocation points during training.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LinearWaveLoss(nn.Module):
    """
    Residual of the 2-D linear wave equation:
        ∂²η/∂t² - c²(∂²η/∂x² + ∂²η/∂y²) = 0

    Applied to predicted η(x,y,t) — requires time-dependent operator output.
    """

    def forward(self, eta: torch.Tensor, c: float, dt: float, dx: float,
                dy: float) -> torch.Tensor:
        d2t = (eta[:, :, :, 2:] - 2*eta[:, :, :, 1:-1] + eta[:, :, :, :-2]) / dt**2
        d2x = (eta[:, :, 2:, :] - 2*eta[:, :, 1:-1, :] + eta[:, :, :-2, :]) / dx**2
        d2y = (eta[:, 2:, :, :] - 2*eta[:, 1:-1, :, :] + eta[:, :-2, :, :]) / dy**2
        # Trim to common shape
        min_H = min(d2t.shape[2], d2x.shape[2], d2y.shape[2])
        min_W = min(d2t.shape[3], d2x.shape[3], d2y.shape[3])
        res = (d2t[:, :, :min_H, :min_W] -
               c**2 * (d2x[:, :, :min_H, :min_W] + d2y[:, :, :min_H, :min_W]))
        return (res ** 2).mean()


class MildSlopeLoss(nn.Module):
    """
    Residual of the mild-slope equation for wave propagation over bathymetry:
        ∇·(c c_g ∇η) + ω² / c_p * c_g * η = 0
    """

    def forward(self, eta: torch.Tensor, cc_g: torch.Tensor,
                omega: float) -> torch.Tensor:
        # Finite difference divergence of (cc_g * ∇η)
        deta_x = torch.gradient(eta,   dim=-1)[0]
        deta_y = torch.gradient(eta,   dim=-2)[0]
        flux_x = cc_g * deta_x
        flux_y = cc_g * deta_y
        div    = torch.gradient(flux_x, dim=-1)[0] + torch.gradient(flux_y, dim=-2)[0]
        k2_eta = (omega ** 2 / cc_g.clamp(min=1e-6)) * eta
        res    = div + k2_eta
        return (res ** 2).mean()


class EOMResidualLoss(nn.Module):
    """
    Residual of the WEC equation of motion in frequency domain:
        R = |[-ω²(M+A) + iω(B+Bpto) + C] X - F_ex|²

    Penalises DeepONet predictions that violate the physical EOM.
    """

    def forward(self, omega: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                X: torch.Tensor, F_ex: torch.Tensor,
                M: float, Bpto: float, C: float) -> torch.Tensor:
        impedance = (-omega**2 * (M + A) +
                     1j * omega * (B + Bpto) + C)
        residual  = impedance * X - F_ex
        return (residual.abs() ** 2).mean()
