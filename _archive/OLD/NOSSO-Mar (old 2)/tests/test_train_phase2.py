"""RED tests for Phase 2 end-to-end training loop (P2-T4)."""
from __future__ import annotations

import numpy as np
import pytest
from nossomar.data.wec_dataset import generate_analytic_dataset
from nossomar.training.train_phase2 import train_phase2


def test_loss_decreases():
    ds = generate_analytic_dataset(n_cases=200, seed=0)
    history = train_phase2(ds, epochs=10, lr=1e-3, batch_size=32)
    assert min(history) < history[0], (
        f"Loss never improved: min={min(history):.4f}, start={history[0]:.4f}"
    )


def test_no_nan_in_history():
    ds = generate_analytic_dataset(n_cases=50, seed=1)
    history = train_phase2(ds, epochs=3, lr=1e-3, batch_size=16)
    assert all(np.isfinite(h) for h in history), "NaN or Inf detected in loss history"


def test_returns_list_of_floats():
    ds = generate_analytic_dataset(n_cases=50, seed=2)
    history = train_phase2(ds, epochs=3, lr=1e-3, batch_size=16)
    assert isinstance(history, list)
    assert all(isinstance(h, float) for h in history)


def test_history_length_matches_epochs():
    ds = generate_analytic_dataset(n_cases=50, seed=3)
    history = train_phase2(ds, epochs=5, lr=1e-3, batch_size=16)
    assert len(history) == 5, f"Expected 5 epochs, got {len(history)}"


def test_loss_positive():
    ds = generate_analytic_dataset(n_cases=50, seed=4)
    history = train_phase2(ds, epochs=3, lr=1e-3, batch_size=16)
    assert all(h > 0 for h in history), "Loss must be positive (MSE)"
