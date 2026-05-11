from __future__ import annotations

import torch

from nossomar.loss.physics_losses import CurriculumWeight, damping_nonneg_loss, total_loss, build_default_registry


def test_damping_nonneg_loss_penalizes_only_negative_values() -> None:
    positive = damping_nonneg_loss(torch.tensor([0.0, 1.0, 5.0]))
    mixed = damping_nonneg_loss(torch.tensor([-2.0, 0.0, 3.0]))

    assert float(positive) == 0.0
    assert float(mixed) > 0.0


def test_wec_eom_loss_is_finite_on_valid_inputs() -> None:
    freq = torch.linspace(0.1, 2.0, 8)
    A = torch.full_like(freq, 2.0e5)
    B = torch.full_like(freq, 8.0e4)
    Fex_real = torch.full_like(freq, 1.0e6)
    Fex_imag = torch.full_like(freq, 2.0e5)

    loss = wec_eom_loss(
        A=A,
        B=B,
        Fex_real=Fex_real,
        Fex_imag=Fex_imag,
        freq=freq,
        mass=6.0e5,
        bpto=5.0e4,
        stiffness=1.0e6,
    )

    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_total_loss_and_curriculum_weight() -> None:
    combined = total_loss(
        supervised=torch.tensor(2.0),
        physics=torch.tensor(3.0),
        cross_fidelity=torch.tensor(5.0),
        weights={"supervised": 1.0, "physics": 0.5, "cross_fidelity": 0.1},
    )
    ramp = CurriculumWeight(start_epoch=2, end_epoch=6, start_val=0.0, end_val=1.0)

    assert torch.isclose(combined, torch.tensor(4.0))
    assert ramp(1) == 0.0
    assert ramp(4) == 0.5
    assert ramp(8) == 1.0

def test_default_registry_contains_expected_losses():
    registry = build_default_registry()
    assert {'L-00', 'L-30', 'L-31', 'L-32'}.issubset(set(registry.available()))


def test_registry_rejects_missing_inputs():
    registry = build_default_registry()
    term = registry.get('L-30')
    with pytest.raises(KeyError):
        term(predictions={'xi': 1}, targets={}, context={})