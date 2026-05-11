"""Physics-informed loss terms for WEC frequency-domain training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol

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



class LossFn(Protocol):
    def __call__(self, predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
        ...

@_dataclass(frozen=True)
class LossSpec:
    loss_id: str
    name: str
    model_classes: tuple[str, ...]
    required_observables: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    supports_complex: bool = False
    default_weight: float = 1.0
    notes: str = ""

@_dataclass
class LossTerm:
    spec: LossSpec
    fn: LossFn

    def validate_inputs(self, predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        available = set(predictions.keys()) | set(targets.keys()) | set(context.keys())
        missing_obs = [key for key in self.spec.required_observables if key not in available]
        missing_par = [key for key in self.spec.required_parameters if key not in available]
        if missing_obs or missing_par:
            raise KeyError(
                f"Loss {self.spec.loss_id} missing inputs: observables={missing_obs}, parameters={missing_par}"
            )

    def __call__(self, predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
        self.validate_inputs(predictions, targets, context)
        return self.fn(predictions, targets, context)


class PhysicsLossRegistry:
    """Lightweight registry for declarative loss activation from YAML configs."""

    def __init__(self) -> None:
        self._terms: _Dict[str, LossTerm] = {}

    def register(self, term: LossTerm) -> None:
        if term.spec.loss_id in self._terms:
            raise ValueError(f"Duplicate loss_id: {term.spec.loss_id}")
        self._terms[term.spec.loss_id] = term

    def get(self, loss_id: str) -> LossTerm:
        if loss_id not in self._terms:
            raise KeyError(f"Unknown loss_id: {loss_id}")
        return self._terms[loss_id]

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._terms.keys()))

    def build_from_config(self, loss_cfgs: Iterable[Mapping[str, Any]]) -> list[tuple[LossTerm, float]]:
        built = []
        for cfg in loss_cfgs:
            if not cfg.get("enabled", False):
                continue
            lid = cfg["loss_id"]
            weight = float(cfg.get("weight", self.get(lid).spec.default_weight))
            built.append((self.get(lid), weight))
        return built

def _loss_l00_data_fidelity(predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    pred = predictions['prediction']
    target = targets['target']
    return ((pred - target) ** 2).mean()

def _loss_l30_wsi_eom(predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    xi = predictions['xi']
    omega = context['omega']
    M = context['M']
    A = predictions['added_mass']
    B = predictions['radiation_damping']
    F_exc = predictions['excitation_force']
    C_pto = context['C_pto']
    K_h = context['K_h']
    K_pto = context['K_pto']
    residual = (-omega**2 * (M + A) + 1j * omega * (B + C_pto) + (K_h + K_pto)) * xi - F_exc
    return (residual.real ** 2 + residual.imag ** 2).mean()


def _loss_l31_passivity(predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    damping = predictions['radiation_damping']
    penalty = (0.0 - damping).clip(min=0.0)
    return (penalty ** 2).mean()


def _loss_l32_smooth_frequency_response(predictions: Mapping[str, Any], targets: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    curve = predictions['rao_amplitude']
    second_diff = curve[..., 2:] - 2 * curve[..., 1:-1] + curve[..., :-2]
    return (second_diff ** 2).mean()

def build_default_registry() -> PhysicsLossRegistry:
    registry = PhysicsLossRegistry()
    registry.register(
        LossTerm(
            spec=LossSpec(
                loss_id='L-00',
                name='data_fidelity',
                model_classes=('M1', 'M2', 'M3', 'M4', 'M5'),
                required_observables=('prediction', 'target'),
                default_weight=1.0,
                notes='Reference supervised loss. Safe default for current Phase 1.',
            ),
            fn=_loss_l00_data_fidelity,
        )
    )
    registry.register(
        LossTerm(
            spec=LossSpec(
                loss_id='L-30',
                name='wsi_equation_of_motion',
                model_classes=('M1',),
                required_observables=('xi', 'added_mass', 'radiation_damping', 'excitation_force'),
                required_parameters=('omega', 'M', 'C_pto', 'K_h', 'K_pto'),
                supports_complex=True,
                default_weight=0.1,
                notes='Primary physics loss candidate for Capytaine-anchored F1A.',
            ),
            fn=_loss_l30_wsi_eom,
        )
    )
    registry.register(
        LossTerm(
            spec=LossSpec(
                loss_id='L-31',
                name='passivity_positivity',
                model_classes=('M1',),
                required_observables=('radiation_damping',),
                default_weight=0.01,
                notes='Regularization for non-physical negative damping outputs.',
            ),
            fn=_loss_l31_passivity,
        )
    )
    registry.register(
        LossTerm(
            spec=LossSpec(
                loss_id='L-32',
                name='smooth_frequency_response',
                model_classes=('M1',),
                required_observables=('rao_amplitude',),
                default_weight=0.001,
                notes='Regularization that should be used carefully near resonance peaks.',
            ),
            fn=_loss_l32_smooth_frequency_response,
        )
    )
    return registry
