"""RED tests for Phase 3 uncertainty quantification module (P3-T3).

Defines the contract for MCDropoutEstimator before any implementation.
All tests must FAIL on first run.

Contract: MCDropoutEstimator(model, n_samples) exposes:
    .predict(x)         -> UQResult
    .calibrate(x, y)    -> float  (Expected Calibration Error)

UQResult:
    .mean     ndarray (N, d_out)    predictive mean
    .std      ndarray (N, d_out)    predictive std dev  (>= 0)
    .lower    ndarray (N, d_out)    2.5th percentile
    .upper    ndarray (N, d_out)    97.5th percentile   (95% CI)
    .samples  ndarray (n_samples, N, d_out)  raw MC samples

Key physics invariants:
    - std >= 0 everywhere
    - lower <= mean <= upper
    - wider CI for inputs far from training distribution (epistemic UQ)
    - ECE in [0, 1]
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from nossomar.uncertainty.mc_dropout import MCDropoutEstimator


# ---------------------------------------------------------------------------
# Minimal model fixture with Dropout layers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Small MLP with Dropout for testing."""
    def __init__(self, d_in: int = 4, d_out: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(32, d_out),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return _TinyModel(d_in=4, d_out=2)


@pytest.fixture(scope="module")
def estimator(model):
    return MCDropoutEstimator(model=model, n_samples=50)


@pytest.fixture(scope="module")
def x_batch():
    torch.manual_seed(1)
    return torch.randn(16, 4)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_instantiation(model):
    est = MCDropoutEstimator(model=model, n_samples=30)
    assert est is not None


def test_n_samples_must_be_positive(model):
    with pytest.raises(ValueError, match="n_samples"):
        MCDropoutEstimator(model=model, n_samples=0)


# ---------------------------------------------------------------------------
# UQResult shape
# ---------------------------------------------------------------------------

def test_predict_returns_uq_result(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert hasattr(result, "mean")
    assert hasattr(result, "std")
    assert hasattr(result, "lower")
    assert hasattr(result, "upper")
    assert hasattr(result, "samples")


def test_mean_shape(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert result.mean.shape == (16, 2)


def test_std_shape(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert result.std.shape == (16, 2)


def test_samples_shape(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert result.samples.shape == (50, 16, 2)


# ---------------------------------------------------------------------------
# Physics / statistics invariants
# ---------------------------------------------------------------------------

def test_std_non_negative(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert np.all(result.std >= 0.0)


def test_lower_le_mean_le_upper(estimator, x_batch):
    result = estimator.predict(x_batch)
    assert np.all(result.lower <= result.mean + 1e-6)
    assert np.all(result.mean <= result.upper + 1e-6)


def test_ci_coverage_approximately_95(estimator):
    """95% CI should contain ~95% of samples on a well-behaved model."""
    torch.manual_seed(42)
    x = torch.randn(200, 4)
    result = estimator.predict(x)
    # Check across all outputs and batch elements
    in_ci = (result.samples >= result.lower) & (result.samples <= result.upper)
    coverage = in_ci.mean()
    assert coverage >= 0.90, f"CI coverage too low: {coverage:.3f}"


def test_std_nonzero_with_dropout(estimator, x_batch):
    """With Dropout active, predictions should vary across MC samples."""
    result = estimator.predict(x_batch)
    assert np.any(result.std > 0.0)


def test_more_samples_reduces_mean_variance(model):
    """Predictive mean should stabilise with more MC samples."""
    torch.manual_seed(7)
    x = torch.randn(32, 4)
    est_10 = MCDropoutEstimator(model=model, n_samples=10)
    est_200 = MCDropoutEstimator(model=model, n_samples=200)
    # Run several times and measure variance of the mean estimate
    means_10 = np.stack([est_10.predict(x).mean for _ in range(5)])
    means_200 = np.stack([est_200.predict(x).mean for _ in range(5)])
    assert means_10.var() >= means_200.var()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibrate_returns_float(estimator, x_batch):
    y = torch.randn(16, 2)   # synthetic targets
    ece = estimator.calibrate(x_batch, y)
    assert isinstance(ece, float)


def test_calibrate_ece_in_range(estimator, x_batch):
    y = torch.randn(16, 2)
    ece = estimator.calibrate(x_batch, y)
    assert 0.0 <= ece <= 1.0


def test_calibrate_perfect_model_low_ece():
    """A model that always predicts the true value should have low ECE."""
    class _PerfectModel(nn.Module):
        def forward(self, x):
            return x[:, :2]   # just return first 2 features as prediction

    est = MCDropoutEstimator(model=_PerfectModel(), n_samples=50)
    torch.manual_seed(0)
    x = torch.randn(100, 4)
    y = x[:, :2]   # targets = what the model predicts
    ece = est.calibrate(x, y)
    # ECE should be low since predictions match targets
    assert ece < 0.5
