"""RED tests for multi-object hydrodynamic interaction module (P2-T3)."""
from __future__ import annotations

import pytest
import torch
from nossomar.core.contracts import WECState
from nossomar.modules.multi_object_interaction import MultiObjectInteraction


@pytest.fixture
def mod():
    return MultiObjectInteraction(d_latent=64, hidden=32, n_heads=2)


def _state(n: int, spread: float = 100.0) -> WECState:
    return WECState(
        pos=torch.rand(n, 2) * spread,
        vel=torch.zeros(n, 6),
        force=torch.zeros(n, 6),
    )


def test_output_shape(mod):
    states = _state(4)
    latents = torch.randn(4, 64)
    delta = mod(states, latents)
    assert delta.shape == (4, 6), f"Expected (4,6), got {delta.shape}"


def test_no_nan_two_wecs(mod):
    states = _state(2)
    latents = torch.randn(2, 64)
    delta = mod(states, latents)
    assert not torch.isnan(delta).any()


def test_no_nan_many_wecs(mod):
    states = _state(10)
    latents = torch.randn(10, 64)
    delta = mod(states, latents)
    assert not torch.isnan(delta).any()


def test_single_wec_returns_zero(mod):
    """One WEC has no neighbour — interaction correction must be zero."""
    states = _state(1)
    latents = torch.randn(1, 64)
    delta = mod(states, latents)
    assert delta.abs().max().item() < 1e-6, \
        f"Single WEC should give zero delta, got max={delta.abs().max().item()}"


def test_permutation_equivariance(mod):
    """Shuffling WEC order must shuffle output in same order."""
    torch.manual_seed(7)
    mod.eval()
    pos = torch.rand(3, 2) * 100
    lat = torch.randn(3, 64)
    s = WECState(pos=pos, vel=torch.zeros(3, 6), force=torch.zeros(3, 6))
    with torch.no_grad():
        delta_orig = mod(s, lat)
    perm = [2, 0, 1]
    s_p = WECState(pos=pos[perm], vel=torch.zeros(3, 6), force=torch.zeros(3, 6))
    with torch.no_grad():
        delta_perm = mod(s_p, lat[perm])
    assert torch.allclose(delta_orig[perm], delta_perm, atol=1e-4), \
        "Output not equivariant under WEC permutation"


def test_output_finite_at_close_range(mod):
    """Two WECs very close together must not produce inf/nan."""
    pos = torch.tensor([[0.0, 0.0], [0.1, 0.0]])  # 10 cm apart
    states = WECState(pos=pos, vel=torch.zeros(2, 6), force=torch.zeros(2, 6))
    latents = torch.randn(2, 64)
    delta = mod(states, latents)
    assert torch.isfinite(delta).all(), "Output not finite at close WEC range"


def test_distance_scaling(mod):
    """WECs far apart should produce smaller interaction than WECs close together."""
    mod.eval()
    torch.manual_seed(0)
    lat = torch.randn(2, 64)

    pos_close = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    pos_far = torch.tensor([[0.0, 0.0], [500.0, 0.0]])

    s_close = WECState(pos=pos_close, vel=torch.zeros(2, 6), force=torch.zeros(2, 6))
    s_far = WECState(pos=pos_far, vel=torch.zeros(2, 6), force=torch.zeros(2, 6))

    with torch.no_grad():
        d_close = mod(s_close, lat).abs().mean().item()
        d_far = mod(s_far, lat).abs().mean().item()

    assert d_close >= d_far, \
        f"Close WECs ({d_close:.4f}) should interact more than far WECs ({d_far:.4f})"
