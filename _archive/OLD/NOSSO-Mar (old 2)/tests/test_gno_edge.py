"""RED tests for Phase 4 P4-T1: GNOEdge pairwise interaction kernel.

Defines the contract for GNOEdge before any implementation exists.
All tests must FAIL on first run (ModuleNotFoundError expected).

Physical context
----------------
For a WEC farm with N devices, pairwise hydrodynamic interaction
between devices i and j is governed by:

    - Separation vector: d_ij = pos_j - pos_i  (2D, metres)
    - Distance:          r_ij = ||d_ij||         (metres)
    - Relative heading:  theta_ij = atan2(d_y, d_x)  (radians)
    - Frequency:         omega (rad/s)

The Kagemoto-Yue interaction theory (1986) shows that the coupling
decays as O(1/sqrt(r)) for radiation and O(1/r) for diffraction.
The GNOEdge kernel must encode this spatial decay implicitly.

Contract: GNOEdge(d_model, d_edge) exposes:
    .forward(h_i, h_j, edge_feat) -> message  shape (E, d_model)

where:
    h_i, h_j    : (E, d_model)   node feature vectors for source/target
    edge_feat   : (E, d_edge)    edge features [r, cos(theta), sin(theta),
                                                 omega, 1/r, exp(-r/lambda)]
    message     : (E, d_model)   pairwise interaction vector

Edge feature convention (d_edge = 6):
    0: normalised distance r / r_ref          (r_ref = 100 m)
    1: cos(theta_ij)
    2: sin(theta_ij)
    3: normalised frequency omega / omega_ref  (omega_ref = 1 rad/s)
    4: 1 / (r/r_ref + eps)                    far-field decay proxy
    5: exp(-r / lambda)                        near-field decay proxy (lambda=50m)
"""
from __future__ import annotations

import math
import torch
import pytest

from nossomar.gno.edge import GNOEdge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
D_EDGE = 6
N_EDGES = 20   # arbitrary batch of pairwise interactions


@pytest.fixture(scope="module")
def edge_module():
    torch.manual_seed(0)
    return GNOEdge(d_model=D_MODEL, d_edge=D_EDGE)


@pytest.fixture(scope="module")
def node_feats():
    torch.manual_seed(1)
    h_i = torch.randn(N_EDGES, D_MODEL)
    h_j = torch.randn(N_EDGES, D_MODEL)
    return h_i, h_j


@pytest.fixture(scope="module")
def edge_feats():
    """Physically plausible edge features for 20 pairs."""
    torch.manual_seed(2)
    r = torch.rand(N_EDGES) * 4.0 + 0.5      # r/r_ref in [0.5, 4.5]
    theta = torch.rand(N_EDGES) * 2 * math.pi
    omega = torch.rand(N_EDGES) * 1.5 + 0.3  # normalised freq in [0.3, 1.8]
    eps = 1e-3
    lam = 0.5  # normalised lambda = 50m / 100m
    e = torch.stack([
        r,
        torch.cos(theta),
        torch.sin(theta),
        omega,
        1.0 / (r + eps),
        torch.exp(-r / lam),
    ], dim=-1)  # (N_EDGES, 6)
    return e


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_gno_edge_instantiation(edge_module):
    assert edge_module is not None


def test_gno_edge_has_parameters(edge_module):
    n = sum(p.numel() for p in edge_module.parameters())
    assert n > 0, "GNOEdge has no trainable parameters"


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_message_shape(edge_module, node_feats, edge_feats):
    h_i, h_j = node_feats
    msg = edge_module(h_i, h_j, edge_feats)
    assert msg.shape == (N_EDGES, D_MODEL), (
        f"Expected ({N_EDGES}, {D_MODEL}), got {msg.shape}"
    )


def test_message_single_edge(edge_module):
    h_i = torch.randn(1, D_MODEL)
    h_j = torch.randn(1, D_MODEL)
    ef = torch.rand(1, D_EDGE)
    msg = edge_module(h_i, h_j, ef)
    assert msg.shape == (1, D_MODEL)


def test_message_large_batch(edge_module):
    E = 512
    h_i = torch.randn(E, D_MODEL)
    h_j = torch.randn(E, D_MODEL)
    ef = torch.rand(E, D_EDGE)
    msg = edge_module(h_i, h_j, ef)
    assert msg.shape == (E, D_MODEL)


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------

def test_message_no_nan(edge_module, node_feats, edge_feats):
    h_i, h_j = node_feats
    msg = edge_module(h_i, h_j, edge_feats)
    assert not torch.isnan(msg).any(), "NaN in message output"


def test_message_no_inf(edge_module, node_feats, edge_feats):
    h_i, h_j = node_feats
    msg = edge_module(h_i, h_j, edge_feats)
    assert not torch.isinf(msg).any(), "Inf in message output"


# ---------------------------------------------------------------------------
# Physics-motivated behavioural constraints
# ---------------------------------------------------------------------------

def test_far_field_message_smaller_than_near_field(edge_module):
    """Far-field pairs (large r) should produce weaker messages than near-field.

    Not a strict physical law at untrained initialisation, but the
    architecture must not prevent this from being learned. We test
    that the output is not identically constant across distances.
    """
    torch.manual_seed(42)
    h_i = torch.randn(2, D_MODEL)
    h_j = torch.randn(2, D_MODEL).expand(2, -1)

    # Near-field edge: r=0.5 (50 m), Far-field: r=4.0 (400 m)
    eps = 1e-3
    lam = 0.5

    def make_edge(r_norm, theta=0.0, omega=1.0):
        return torch.tensor([[
            r_norm,
            math.cos(theta),
            math.sin(theta),
            omega,
            1.0 / (r_norm + eps),
            math.exp(-r_norm / lam),
        ]])

    near = edge_module(h_i[:1], h_j[:1], make_edge(0.5))
    far  = edge_module(h_i[:1], h_j[:1], make_edge(4.0))

    # Messages must differ — kernel is not ignoring edge features
    assert not torch.allclose(near, far, atol=1e-4), (
        "Near-field and far-field messages are identical — edge features ignored"
    )


def test_asymmetry(edge_module):
    """m(i->j) != m(j->i) in general — interaction is directional."""
    torch.manual_seed(7)
    h_i = torch.randn(1, D_MODEL)
    h_j = torch.randn(1, D_MODEL)
    ef = torch.rand(1, D_EDGE)
    ef_rev = ef.clone()
    ef_rev[0, 1] = -ef[0, 1]   # flip cos(theta)
    ef_rev[0, 2] = -ef[0, 2]   # flip sin(theta)

    m_ij = edge_module(h_i, h_j, ef)
    m_ji = edge_module(h_j, h_i, ef_rev)
    assert not torch.allclose(m_ij, m_ji, atol=1e-4), (
        "Message is symmetric — directionality not encoded"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows_through_message(edge_module):
    h_i = torch.randn(4, D_MODEL, requires_grad=True)
    h_j = torch.randn(4, D_MODEL, requires_grad=True)
    ef = torch.rand(4, D_EDGE)
    msg = edge_module(h_i, h_j, ef)
    loss = msg.sum()
    loss.backward()
    assert h_i.grad is not None
    assert h_j.grad is not None
    assert not torch.isnan(h_i.grad).any()


def test_gradient_flows_through_edge_features():
    """Edge features must contribute to gradient so they can be optimised."""
    torch.manual_seed(0)
    mod = GNOEdge(d_model=D_MODEL, d_edge=D_EDGE)
    h_i = torch.randn(4, D_MODEL)
    h_j = torch.randn(4, D_MODEL)
    ef = torch.rand(4, D_EDGE, requires_grad=True)
    msg = mod(h_i, h_j, ef)
    msg.sum().backward()
    assert ef.grad is not None
    assert not torch.isnan(ef.grad).any()
