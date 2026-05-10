"""RED tests for Phase 4 P4-T2: WECFarmGNO variable-N forward pass.

Defines the contract for WECFarmGNO before any implementation exists.
All tests must FAIL on first run (ModuleNotFoundError expected).

Physical context
----------------
A WEC farm has N devices at 2D positions. The GNO:
  1. Lifts each device's property vector to a node embedding via
     SixDOFSurrogate (already implemented in Phase 3).
  2. Builds a fully-connected graph: E = N*(N-1) directed edges.
  3. Computes edge features from pairwise positions + query frequency.
  4. Runs K rounds of GNOEdge message passing.
  5. Reads out a per-device response: (A_mat, B_mat) each (N, 6, 6).

Contract: WECFarmGNO(d_model, d_edge, n_layers) exposes:
    .forward(props, pos, omega)
        props : (N, 4)    WEC property vectors
        pos   : (N, 2)    WEC positions in metres [x, y]
        omega : (K,)      frequency query points (rad/s)
    returns:
        A_farm : (N, 6, 6)  added-mass matrices (farm-corrected)
        B_farm : (N, 6, 6)  radiation-damping matrices (farm-corrected)

Key invariants
--------------
- Works for any N in [1, N_max] without recompiling
- N=1 reduces to isolated device (no interaction edges contribute)
- Output matrices remain symmetric and diagonal-non-negative
- Gradients flow back to props and pos
"""
from __future__ import annotations

import math
import torch
import pytest

from nossomar.gno.farm import WECFarmGNO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
OMEGA = torch.linspace(0.3, 2.0, 8)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return WECFarmGNO(d_model=D_MODEL, d_edge=6, n_layers=2)


def make_farm(n: int, seed: int = 42):
    torch.manual_seed(seed)
    props = torch.rand(n, 4) + 0.1           # [radius, draft, depth, Bpto]
    pos   = torch.rand(n, 2) * 500.0         # positions in [0, 500] m
    return props, pos


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_farm_gno_instantiation(model):
    assert model is not None


def test_farm_gno_has_parameters(model):
    n = sum(p.numel() for p in model.parameters())
    assert n > 0


# ---------------------------------------------------------------------------
# Output shape for various N
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_wec", [1, 2, 4, 8])
def test_output_shape(model, n_wec):
    props, pos = make_farm(n_wec)
    A, B = model(props, pos, OMEGA)
    assert A.shape == (n_wec, 6, 6), f"A shape wrong for N={n_wec}"
    assert B.shape == (n_wec, 6, 6), f"B shape wrong for N={n_wec}"


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_wec", [1, 3, 6])
def test_no_nan(model, n_wec):
    props, pos = make_farm(n_wec)
    A, B = model(props, pos, OMEGA)
    assert not torch.isnan(A).any(), f"NaN in A for N={n_wec}"
    assert not torch.isnan(B).any(), f"NaN in B for N={n_wec}"


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_wec", [2, 5])
def test_A_symmetric(model, n_wec):
    props, pos = make_farm(n_wec)
    A, _ = model(props, pos, OMEGA)
    diff = (A - A.transpose(-2, -1)).abs().max()
    assert diff < 1e-4, f"A not symmetric for N={n_wec}, max diff={diff:.6f}"


@pytest.mark.parametrize("n_wec", [2, 5])
def test_B_symmetric(model, n_wec):
    props, pos = make_farm(n_wec)
    _, B = model(props, pos, OMEGA)
    diff = (B - B.transpose(-2, -1)).abs().max()
    assert diff < 1e-4, f"B not symmetric for N={n_wec}, max diff={diff:.6f}"


@pytest.mark.parametrize("n_wec", [1, 3])
def test_A_diagonal_non_negative(model, n_wec):
    props, pos = make_farm(n_wec)
    A, _ = model(props, pos, OMEGA)
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    assert torch.all(diag >= -1e-4), f"Negative A diagonal for N={n_wec}"


@pytest.mark.parametrize("n_wec", [1, 3])
def test_B_diagonal_non_negative(model, n_wec):
    props, pos = make_farm(n_wec)
    _, B = model(props, pos, OMEGA)
    diag = torch.diagonal(B, dim1=-2, dim2=-1)
    assert torch.all(diag >= -1e-4), f"Negative B diagonal for N={n_wec}"


# ---------------------------------------------------------------------------
# N=1 isolation: farm output close to isolated surrogate
# ---------------------------------------------------------------------------

def test_n1_no_interaction_edges(model):
    """With a single WEC there are zero interaction edges.
    The model must not crash and output must be finite."""
    props, pos = make_farm(1)
    A, B = model(props, pos, OMEGA)
    assert A.shape == (1, 6, 6)
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows_to_props(model):
    props, pos = make_farm(3)
    props.requires_grad_(True)
    A, B = model(props, pos, OMEGA)
    (A.sum() + B.sum()).backward()
    assert props.grad is not None
    assert not torch.isnan(props.grad).any()


def test_gradient_flows_to_pos(model):
    """Gradients must reach WEC positions so layout can be optimised."""
    props, pos = make_farm(3)
    pos.requires_grad_(True)
    A, B = model(props, pos, OMEGA)
    (A.sum() + B.sum()).backward()
    assert pos.grad is not None
    assert not torch.isnan(pos.grad).any()


# ---------------------------------------------------------------------------
# Farm interaction: different layouts -> different outputs
# ---------------------------------------------------------------------------

def test_layout_changes_output(model):
    """Two farms with different positions must produce different outputs."""
    torch.manual_seed(0)
    props = torch.rand(4, 4) + 0.1
    pos1 = torch.rand(4, 2) * 200.0
    pos2 = torch.rand(4, 2) * 200.0 + 300.0   # shifted far away
    A1, B1 = model(props, pos1, OMEGA)
    A2, B2 = model(props, pos2, OMEGA)
    # Same devices, different layout -> different farm corrections
    assert not torch.allclose(A1, A2, atol=1e-4)
