"""Physics-informed loss terms for WEC frequency-domain training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from nossomar.physics.residuals_torch import residual_mse, wec_frequency_domain_residual


def damping_nonneg_loss(B_pred: torch.Tensor) -> torch.Tensor:
    """Quadratic soft penalty for negative radiation damping."""

    negative_part = F.relu(-B_pred)
    return torch.mean(negative_part.square())


def wec_eom_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    Fex_real: torch.Tensor,
    Fex_imag: torch.Tensor,
    freq: torch.Tensor,
    mass: torch.Tensor | float,
    bpto: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    displacement: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared residual of the linear frequency-domain WEC equation."""

    omega = 2.0 * torch.pi * freq
    excitation = torch.complex(Fex_real, Fex_imag)
    if displacement is None:
        total_mass = mass + A
        total_damping = B + bpto
        dynamic_stiffness = -omega.to(A.dtype).square() * total_mass + stiffness
        damping_term = omega.to(A.dtype) * total_damping
        denominator = torch.complex(dynamic_stiffness, damping_term)
        displacement = excitation / torch.where(
            torch.abs(denominator) > 1.0e-12,
            denominator,
            torch.full_like(denominator, 1.0e-12 + 0.0j),
        )
    residual = wec_frequency_domain_residual(
        omega,
        mass=mass,
        added_mass=A,
        damping=B,
        stiffness=stiffness,
        displacement=displacement,
        excitation=excitation,
        pto_damping=bpto,
    )
    return residual_mse(residual)


def total_loss(
    supervised: torch.Tensor,
    physics: torch.Tensor | None = None,
    cross_fidelity: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Combine supervised, physics, and cross-fidelity objectives."""

    active_weights = weights or {}
    loss = supervised * float(active_weights.get("supervised", 1.0))
    if physics is not None:
        loss = loss + physics * float(active_weights.get("physics", 1.0))
    if cross_fidelity is not None:
        loss = loss + cross_fidelity * float(active_weights.get("cross_fidelity", 1.0))
    return loss


@dataclass(frozen=True, slots=True)
class CurriculumWeight:
    """Linear scalar schedule for gradually activating a loss term."""

    start_epoch: int
    end_epoch: int
    start_val: float
    end_val: float

    def __call__(self, epoch: int) -> float:
        if self.end_epoch <= self.start_epoch:
            return float(self.end_val if epoch >= self.end_epoch else self.start_val)
        if epoch <= self.start_epoch:
            return float(self.start_val)
        if epoch >= self.end_epoch:
            return float(self.end_val)
        fraction = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return float(self.start_val + fraction * (self.end_val - self.start_val))
