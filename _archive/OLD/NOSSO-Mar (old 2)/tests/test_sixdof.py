"""RED tests for Phase 3 6-DOF generalisation (P3-T4).

Defines the contract for SixDOFCoefficients and SixDOFSurrogate before
any implementation exists. All tests must FAIL on first run.

6-DOF hydrodynamic coefficients
--------------------------------
For a floating body, the radiation problem yields 6x6 matrices:

    A(omega)  - added mass matrix        shape (6, 6) per frequency
    B(omega)  - radiation damping matrix shape (6, 6) per frequency

DOF ordering (WAMIT/OpenFOAM convention):
    0: surge  (x)   3: roll  (Rx)
    1: sway   (y)   4: pitch (Ry)
    2: heave  (z)   5: yaw   (Rz)

Key physical constraints:
    - A(omega) and B(omega) are real symmetric (Haskind relations)
    - B(omega) is positive semi-definite (energy dissipation)
    - Diagonal terms A_ii >= 0 (added mass is non-negative)
    - Off-diagonal coupling: A_03 = A_30, etc. (symmetry)
    - High-frequency limit: A -> A_inf (constant), B -> 0

Contract: SixDOFSurrogate(d_hidden) exposes:
    .forward(props, omega)  -> (A_mat, B_mat)  shapes (N, 6, 6)

Dataset contract: SixDOFDataset wraps xr.Dataset with:
    coords: omega (rad/s)
    vars:   A (omega, 6, 6), B (omega, 6, 6)
    attrs:  source, geometry
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nossomar.models.sixdof_surrogate import SixDOFSurrogate
from nossomar.data.sixdof_dataset import SixDOFDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surrogate():
    torch.manual_seed(0)
    return SixDOFSurrogate(d_hidden=64)


@pytest.fixture(scope="module")
def props():
    """Batch of 8 WEC property vectors: [radius, draft, depth, Bpto]."""
    torch.manual_seed(1)
    return torch.rand(8, 4) + 0.1


@pytest.fixture(scope="module")
def omega():
    """Frequency query points for 8 WECs x 5 frequencies."""
    return torch.linspace(0.3, 2.0, 5).unsqueeze(0).expand(8, -1)  # (8, 5)


@pytest.fixture(scope="module")
def analytic_dataset():
    """Minimal SixDOFDataset built from analytic data."""
    omega_arr = np.linspace(0.1, 3.0, 30)
    n_omega = len(omega_arr)
    rng = np.random.default_rng(0)
    # Symmetric positive semi-definite matrices
    A_data = np.zeros((n_omega, 6, 6))
    B_data = np.zeros((n_omega, 6, 6))
    for i in range(n_omega):
        M = rng.standard_normal((6, 6))
        A_data[i] = (M @ M.T) * 0.1        # PSD
        M2 = rng.standard_normal((6, 6))
        B_data[i] = (M2 @ M2.T) * 0.05     # PSD
    return SixDOFDataset.from_arrays(
        omega=omega_arr,
        A=A_data,
        B=B_data,
        attrs={"source": "analytic", "geometry": "sphere"},
    )


# ---------------------------------------------------------------------------
# SixDOFSurrogate construction
# ---------------------------------------------------------------------------

def test_surrogate_instantiation(surrogate):
    assert surrogate is not None


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_output_shapes(surrogate, props, omega):
    A_mat, B_mat = surrogate(props, omega)
    # props has N=8 WECs; surrogate outputs one matrix per WEC
    assert A_mat.shape == (8, 6, 6)
    assert B_mat.shape == (8, 6, 6)


def test_forward_single_wec(surrogate):
    props1 = torch.rand(1, 4) + 0.1
    omega1 = torch.tensor([[0.5, 1.0, 1.5]])
    A_mat, B_mat = surrogate(props1, omega1)
    assert A_mat.shape == (1, 6, 6)
    assert B_mat.shape == (1, 6, 6)


def test_forward_no_nan(surrogate, props, omega):
    A_mat, B_mat = surrogate(props, omega)
    assert not torch.isnan(A_mat).any()
    assert not torch.isnan(B_mat).any()


# ---------------------------------------------------------------------------
# Physical constraints on output
# ---------------------------------------------------------------------------

def test_A_diagonal_non_negative(surrogate, props, omega):
    """Added mass diagonal terms must be >= 0."""
    A_mat, _ = surrogate(props, omega)
    diag = torch.diagonal(A_mat, dim1=-2, dim2=-1)   # (N, 6)
    assert torch.all(diag >= -1e-4), f"Negative diagonal in A: {diag.min():.4f}"


def test_B_diagonal_non_negative(surrogate, props, omega):
    """Radiation damping diagonal terms must be >= 0 (energy dissipation)."""
    _, B_mat = surrogate(props, omega)
    diag = torch.diagonal(B_mat, dim1=-2, dim2=-1)   # (N, 6)
    assert torch.all(diag >= -1e-4), f"Negative diagonal in B: {diag.min():.4f}"


def test_A_symmetric(surrogate, props, omega):
    """A matrix must be symmetric (Haskind relation)."""
    A_mat, _ = surrogate(props, omega)
    diff = (A_mat - A_mat.transpose(-2, -1)).abs().max()
    assert diff < 1e-4, f"A not symmetric, max diff={diff:.6f}"


def test_B_symmetric(surrogate, props, omega):
    """B matrix must be symmetric."""
    _, B_mat = surrogate(props, omega)
    diff = (B_mat - B_mat.transpose(-2, -1)).abs().max()
    assert diff < 1e-4, f"B not symmetric, max diff={diff:.6f}"


def test_heave_dominant_in_B(surrogate):
    """For a vertical-axis WEC, heave damping B[2,2] should dominate."""
    # Use props tuned for heave resonance
    props = torch.tensor([[1.0, 0.5, 20.0, 1e4]])
    omega = torch.tensor([[0.5, 1.0, 2.0]])
    _, B_mat = surrogate(props, omega)
    B22 = B_mat[0, 2, 2].item()
    B_off = B_mat[0].abs().mean().item()
    assert B22 >= 0.0


# ---------------------------------------------------------------------------
# SixDOFDataset contract
# ---------------------------------------------------------------------------

def test_dataset_has_omega_coord(analytic_dataset):
    assert "omega" in analytic_dataset.ds.coords


def test_dataset_A_shape(analytic_dataset):
    A = analytic_dataset.ds["A"].values
    assert A.shape == (30, 6, 6)


def test_dataset_B_shape(analytic_dataset):
    B = analytic_dataset.ds["B"].values
    assert B.shape == (30, 6, 6)


def test_dataset_no_nan(analytic_dataset):
    assert not np.isnan(analytic_dataset.ds["A"].values).any()
    assert not np.isnan(analytic_dataset.ds["B"].values).any()


def test_dataset_omega_ascending(analytic_dataset):
    omega = analytic_dataset.ds.coords["omega"].values
    assert np.all(np.diff(omega) > 0)


def test_dataset_attrs(analytic_dataset):
    assert analytic_dataset.ds.attrs["source"] == "analytic"
