"""RED tests for Phase 3 farm layout optimiser (P3-T1).

These tests define the public contract of FarmOptimiser before any
implementation exists. All tests must FAIL on first run.

Contract:
  FarmOptimiser(surrogate, wave_resource) exposes:
    .optimise(n_wec, bounds, n_iter) -> OptResult
    .aep(layout)                     -> float  [kWh/year]
    .power_matrix(layout, Hs, Tp)    -> array  (n_wec,)

  OptResult:
    .layout   (n_wec, 2) float array  [x, y] in metres
    .aep      float  [kWh/year]
    .history  list[float]  AEP per iteration
"""
from __future__ import annotations

import numpy as np
import pytest

from nossomar.optimisation.farm_optimiser import FarmOptimiser
from nossomar.data.wec_dataset import generate_analytic_dataset
from nossomar.training.train_phase2 import _Phase2Model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surrogate():
    """Untrained Phase 2 model — good enough for interface tests."""
    return _Phase2Model(d_latent=64)


@pytest.fixture(scope="module")
def wave_resource():
    """Minimal synthetic wave resource: uniform sea state."""
    return {
        "Hs": np.array([1.0, 2.0, 3.0]),        # significant wave height [m]
        "Tp": np.array([6.0, 8.0, 10.0]),        # peak period [s]
        "prob": np.array([0.5, 0.3, 0.2]),        # occurrence probability (sums to 1)
    }


@pytest.fixture(scope="module")
def optimiser(surrogate, wave_resource):
    return FarmOptimiser(surrogate=surrogate, wave_resource=wave_resource)


@pytest.fixture(scope="module")
def simple_layout():
    """2-WEC layout on a 200 m spacing."""
    return np.array([[0.0, 0.0], [200.0, 0.0]])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_instantiation(surrogate, wave_resource):
    opt = FarmOptimiser(surrogate=surrogate, wave_resource=wave_resource)
    assert opt is not None


def test_wave_resource_probs_must_sum_to_one(surrogate):
    bad_resource = {
        "Hs": np.array([1.0, 2.0]),
        "Tp": np.array([6.0, 8.0]),
        "prob": np.array([0.3, 0.3]),   # sums to 0.6, not 1
    }
    with pytest.raises(ValueError, match="prob"):
        FarmOptimiser(surrogate=surrogate, wave_resource=bad_resource)


# ---------------------------------------------------------------------------
# AEP evaluation
# ---------------------------------------------------------------------------

def test_aep_returns_positive_float(optimiser, simple_layout):
    aep = optimiser.aep(simple_layout)
    assert isinstance(aep, float)
    assert aep > 0.0


def test_aep_single_wec_less_than_two(optimiser):
    layout_1 = np.array([[0.0, 0.0]])
    layout_2 = np.array([[0.0, 0.0], [200.0, 0.0]])
    aep_1 = optimiser.aep(layout_1)
    aep_2 = optimiser.aep(layout_2)
    assert aep_2 > aep_1


def test_aep_shape_agnostic(optimiser):
    """AEP must work for any N >= 1."""
    for n in [1, 2, 3, 5]:
        layout = np.random.default_rng(n).uniform(0, 500, size=(n, 2))
        aep = optimiser.aep(layout)
        assert aep > 0.0, f"AEP non-positive for n={n}"


# ---------------------------------------------------------------------------
# Power matrix
# ---------------------------------------------------------------------------

def test_power_matrix_shape(optimiser, simple_layout):
    pm = optimiser.power_matrix(simple_layout, Hs=1.5, Tp=7.0)
    assert pm.shape == (len(simple_layout),)


def test_power_matrix_non_negative(optimiser, simple_layout):
    pm = optimiser.power_matrix(simple_layout, Hs=1.5, Tp=7.0)
    assert np.all(pm >= 0.0)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def test_optimise_returns_opt_result(optimiser):
    bounds = np.array([[0.0, 1000.0], [0.0, 1000.0]])  # 1 km x 1 km site
    result = optimiser.optimise(n_wec=2, bounds=bounds, n_iter=3)
    assert hasattr(result, "layout")
    assert hasattr(result, "aep")
    assert hasattr(result, "history")


def test_optimise_layout_shape(optimiser):
    bounds = np.array([[0.0, 1000.0], [0.0, 1000.0]])
    result = optimiser.optimise(n_wec=3, bounds=bounds, n_iter=3)
    assert result.layout.shape == (3, 2)


def test_optimise_layout_within_bounds(optimiser):
    bounds = np.array([[0.0, 500.0], [0.0, 500.0]])
    result = optimiser.optimise(n_wec=2, bounds=bounds, n_iter=5)
    assert np.all(result.layout[:, 0] >= 0.0)
    assert np.all(result.layout[:, 0] <= 500.0)
    assert np.all(result.layout[:, 1] >= 0.0)
    assert np.all(result.layout[:, 1] <= 500.0)


def test_optimise_history_length(optimiser):
    bounds = np.array([[0.0, 1000.0], [0.0, 1000.0]])
    n_iter = 4
    result = optimiser.optimise(n_wec=2, bounds=bounds, n_iter=n_iter)
    assert len(result.history) == n_iter


def test_optimise_aep_improves(optimiser):
    """AEP at end of optimisation should be >= AEP at start."""
    bounds = np.array([[0.0, 1000.0], [0.0, 1000.0]])
    result = optimiser.optimise(n_wec=2, bounds=bounds, n_iter=10)
    assert result.history[-1] >= result.history[0]


def test_optimise_result_aep_matches_layout(optimiser):
    """result.aep must equal optimiser.aep(result.layout)."""
    bounds = np.array([[0.0, 1000.0], [0.0, 1000.0]])
    result = optimiser.optimise(n_wec=2, bounds=bounds, n_iter=3)
    assert abs(result.aep - optimiser.aep(result.layout)) < 1.0  # within 1 kWh
