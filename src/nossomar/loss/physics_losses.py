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


# ---------------------------------------------------------------------------
# Physics loss registry — additive extension (specs 11 and 12)
# All functions and classes above are unchanged. This section adds a registry
# for declarative activation of loss terms via YAML config.
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Any as _Any, Dict as _Dict, Iterable as _Iterable, Protocol as _Protocol


class _LossFnProtocol(_Protocol):
    def __call__(self, preds: dict, targets: dict, ctx: dict) -> torch.Tensor: ...


@_dataclass(frozen=True)
class LossSpec:
    loss_id: str
    name: str
    model_classes: tuple
    required_observables: tuple = ()
    required_parameters: tuple = ()
    supports_complex: bool = False
    default_weight: float = 1.0
    notes: str = ""


@_dataclass
class LossTerm:
    spec: LossSpec
    fn: _LossFnProtocol

    def validate_inputs(self, preds: dict, targets: dict, ctx: dict) -> None:
        available = set(preds.keys()) | set(targets.keys()) | set(ctx.keys())
        missing_obs = [k for k in self.spec.required_observables if k not in available]
        missing_par = [k for k in self.spec.required_parameters if k not in available]
        if missing_obs or missing_par:
            raise KeyError(
                f"Loss {self.spec.loss_id} missing: observables={missing_obs} parameters={missing_par}"
            )

    def __call__(self, preds: dict, targets: dict, ctx: dict) -> torch.Tensor:
        self.validate_inputs(preds, targets, ctx)
        return self.fn(preds, targets, ctx)


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

    def available(self) -> tuple:
        return tuple(sorted(self._terms.keys()))

    def build_from_config(self, loss_cfgs: _Iterable[dict]) -> list:
        built = []
        for cfg in loss_cfgs:
            if not cfg.get("enabled", False):
                continue
            lid = cfg["loss_id"]
            weight = float(cfg.get("weight", self.get(lid).spec.default_weight))
            built.append((self.get(lid), weight))
        return built


def _registry_l00_fn(preds: dict, targets: dict, ctx: dict) -> torch.Tensor:
    pred = preds["prediction"] if isinstance(preds["prediction"], torch.Tensor) else torch.as_tensor(preds["prediction"])
    tgt = targets["target"] if isinstance(targets["target"], torch.Tensor) else torch.as_tensor(targets["target"])
    return ((pred - tgt) ** 2).mean()


def _registry_l30_fn(preds: dict, targets: dict, ctx: dict) -> torch.Tensor:
    return wec_eom_loss(
        A=preds["added_mass"],
        B=preds["radiation_damping"],
        Fex_real=preds["excitation_force"].real,
        Fex_imag=preds["excitation_force"].imag,
        freq=ctx["freq"],
        mass=ctx["mass"],
        bpto=ctx.get("bpto", 0.0),
        stiffness=ctx.get("stiffness", 0.0),
        displacement=preds.get("displacement"),
    )


def _registry_l31_fn(preds: dict, targets: dict, ctx: dict) -> torch.Tensor:
    return damping_nonneg_loss(preds["radiation_damping"])


def build_default_registry() -> PhysicsLossRegistry:
    """Build the default registry wiring existing loss functions to registry IDs."""
    reg = PhysicsLossRegistry()
    reg.register(LossTerm(
        spec=LossSpec(
            loss_id="L-00", name="data_fidelity",
            model_classes=("M1", "M2", "M3", "M4", "M5"),
            required_observables=("prediction", "target"),
            default_weight=1.0,
            notes="Reference supervised loss. Safe default for Phase 1.",
        ),
        fn=_registry_l00_fn,
    ))
    reg.register(LossTerm(
        spec=LossSpec(
            loss_id="L-30", name="wsi_equation_of_motion",
            model_classes=("M1",),
            required_observables=("added_mass", "radiation_damping", "excitation_force"),
            required_parameters=("freq", "mass"),
            supports_complex=True,
            default_weight=0.1,
            notes="Wraps existing wec_eom_loss. Capytaine-anchored F1A.",
        ),
        fn=_registry_l30_fn,
    ))
    reg.register(LossTerm(
        spec=LossSpec(
            loss_id="L-31", name="passivity_positivity",
            model_classes=("M1",),
            required_observables=("radiation_damping",),
            default_weight=0.01,
            notes="Wraps existing damping_nonneg_loss.",
        ),
        fn=_registry_l31_fn,
    ))
    return reg
