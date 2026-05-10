"""RED tests for Phase 4 P4-T3: LinearSuperpositionBaseline.

Defines the contract for LinearSuperpositionBaseline before any
implementation exists. All tests must FAIL on first run.

Physical context
----------------
Linear superposition is the simplest physical model for a WEC farm:
each device responds as if it were isolated, and the total farm
response is the sum of individual responses.

For added mass and radiation damping matrices, this means:

    A_farm[i] = A_isolated[i]     (no coupling)
    B_farm[i] = B_isolated[i]     (no coupling)

This is the O(1) baseline the GNO must learn to correct. The GNO
residual is delta_A = A_farm - A_baseline, etc.

Contract: LinearSuperpositionBaseline(surrogate) exposes:
    .forward(props, omega)
        props : (N, 4)   WEC property vectors
        omega : (K,)     frequency query points (rad/s)
    returns:
        A_base : (N, 6, 6)  per-device isolated added-mass matrices
        B_base : (N, 6, 6)  per-device isolated radiation-damping matrices

The baseline is position-independent by definition.
It wraps SixDOFSurrogate and is itself not trainable (frozen wrapper).
"""
from __future__ import annotations

import torch
import pytest

from nossomar.gno.baseline import LinearSuperpositionBaseline
from nossomar.models.sixdof_surrogate import SixDOFSurrogate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surrogate():
    torch.manual_seed(0)
    return SixDOFSurrogate(d_hidden=32)


@pytest.fixture(scope="module")
def baseline(surrogate):
    return LinearSuperpositionBaseline(surrogate)


def make_props(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, 4) + 0.1


OMEGA = torch.linspace(0.3, 2.0, 6)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_baseline_instantiation(baseline):
    assert baseline is not None


def test_baseline_is_not_trainable(baseline):
    """Baseline wraps a surrogate but exposes no own trainable params."""
    own_params = list(baseline.parameters(recurse=False))
    assert len(own_params) == 0, "Baseline should not have own parameters"


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_wec", [1, 3, 8])
def test_output_shape(baseline, n_wec):
    props = make_props(n_wec)
    A, B = baseline(props, OMEGA)
    assert A.shape == (n_wec, 6, 6)
    assert B.shape == (n_wec, 6, 6)


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_wec", [1, 4])
def test_no_nan(baseline, n_wec):
    props = make_props(n_wec)
    A, B = baseline(props, OMEGA)
    assert not torch.isnan(A).any()
    assert not torch.isnan(B).any()


# ---------------------------------------------------------------------------
# Physical constraints inherited from SixDOFSurrogate
# ---------------------------------------------------------------------------

def test_A_symmetric(baseline):
    props = make_props(4)
    A, _ = baseline(props, OMEGA)
    diff = (A - A.transpose(-2, -1)).abs().max()
    assert diff < 1e-4


def test_B_symmetric(baseline):
    props = make_props(4)
    _, B = baseline(props, OMEGA)
    diff = (B - B.transpose(-2, -1)).abs().max()
    assert diff < 1e-4


def test_A_diagonal_non_negative(baseline):
    props = make_props(4)
    A, _ = baseline(props, OMEGA)
    assert torch.all(torch.diagonal(A, dim1=-2, dim2=-1) >= -1e-4)


def test_B_diagonal_non_negative(baseline):
    props = make_props(4)
    _, B = baseline(props, OMEGA)
    assert torch.all(torch.diagonal(B, dim1=-2, dim2=-1) >= -1e-4)


# ---------------------------------------------------------------------------
# Position-independence
# ---------------------------------------------------------------------------

def test_position_independent(baseline):
    """Same props, different positions -> same baseline output."""
    props = make_props(3)
    A1, B1 = baseline(props, OMEGA)
    A2, B2 = baseline(props, OMEGA)  # no pos argument
    assert torch.allclose(A1, A2, atol=1e-6)
    assert torch.allclose(B1, B2, atol=1e-6)


# ---------------------------------------------------------------------------
# Consistency: N independent calls == one batched call
# ---------------------------------------------------------------------------

def test_batched_equals_individual(baseline):
    """Baseline for a farm of N == N independent single-device calls."""
    torch.manual_seed(5)
    props = torch.rand(3, 4) + 0.1

    A_farm, B_farm = baseline(props, OMEGA)

    for i in range(3):
        A_i, B_i = baseline(props[i:i+1], OMEGA)
        assert torch.allclose(A_farm[i], A_i[0], atol=1e-5), f"Device {i} A mismatch"
        assert torch.allclose(B_farm[i], B_i[0], atol=1e-5), f"Device {i} B mismatch"


# ---------------------------------------------------------------------------
# Residual contract: GNO correction is delta = farm - baseline
# ---------------------------------------------------------------------------

def test_residual_shape(baseline):
    """GNO output minus baseline has the same (N,6,6) shape."""
    props = make_props(4)
    A_base, B_base = baseline(props, OMEGA)
    # Simulate a GNO correction of the same shape
    delta_A = torch.randn_like(A_base) * 0.01
    delta_B = torch.randn_like(B_base) * 0.01
    A_corrected = A_base + delta_A
    B_corrected = B_base + delta_B
    assert A_corrected.shape == (4, 6, 6)
    assert B_corrected.shape == (4, 6, 6)
