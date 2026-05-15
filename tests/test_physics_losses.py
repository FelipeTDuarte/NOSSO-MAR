from __future__ import annotations

import torch
import pytest
from nossomar.loss.physics_losses import (
    CurriculumWeight,
    damping_nonneg_loss,
    total_loss,
    wec_eom_loss,
    build_default_registry,
)

def test_damping_nonneg_loss_penalizes_only_negative_values() -> None:
    positive = damping_nonneg_loss(torch.tensor([0.0, 1.0, 5.0]))
    mixed = damping_nonneg_loss(torch.tensor([-1.0, 2.0, -3.0]))
    assert positive.item() == pytest.approx(0.0)
    assert mixed.item() > 0.0


def test_curriculum_weight_ramps_up() -> None:
    cw = CurriculumWeight(warmup_steps=100, final_weight=1.0)
    w0 = cw(step=0)
    w50 = cw(step=50)
    w100 = cw(step=100)
    assert w0 == pytest.approx(0.0)
    assert 0.0 < w50 < 1.0
    assert w100 == pytest.approx(1.0)


def test_total_loss_combines_data_and_physics() -> None:
    pred = torch.randn(4, 3, 16, 16)
    target = torch.randn(4, 3, 16, 16)
    dummy_physics = torch.tensor(0.5)
    loss = total_loss(
        pred=pred,
        target=target,
        physics_residuals={"swe": dummy_physics},
        physics_weight=0.1,
    )
    assert loss.item() > 0.0


def test_build_default_registry_contains_swe() -> None:
    registry = build_default_registry()
    assert "swe" in registry


def test_wec_eom_loss_zero_for_exact_solution() -> None:
    """If the predicted response satisfies EOM exactly, loss should be ~0."""
    t = torch.linspace(0, 1, 50)
    # trivial case: zero motion, zero forcing
    x = torch.zeros(1, 50)
    x_dot = torch.zeros(1, 50)
    x_ddot = torch.zeros(1, 50)
    F_ext = torch.zeros(1, 50)
    m, c, k = 1.0, 0.0, 0.0
    loss = wec_eom_loss(x_ddot, x_dot, x, F_ext, m=m, c=c, k=k)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
