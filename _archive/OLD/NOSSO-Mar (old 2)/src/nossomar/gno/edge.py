"""GNOEdge: pairwise hydrodynamic interaction kernel for WEC farms.

Maps a pair of node feature vectors (h_i, h_j) and a 6-dimensional
edge feature vector to a message vector of size d_model.

Edge feature convention (d_edge = 6)
-------------------------------------
    0: r / r_ref              normalised separation (r_ref = 100 m)
    1: cos(theta_ij)          heading
    2: sin(theta_ij)          heading
    3: omega / omega_ref      normalised frequency (omega_ref = 1 rad/s)
    4: 1 / (r/r_ref + eps)    far-field radiation decay proxy
    5: exp(-r / lambda)       near-field diffraction decay proxy

Architecture
------------
The message is computed as:

    gate   = sigmoid( W_g [h_i || h_j || e_ij] )
    content = tanh(   W_c [h_i || h_j || e_ij] )
    msg    = gate * content * scale(e_ij)

The multiplicative gate ensures that edge features are not ignored:
even at random initialisation, different edge vectors produce
different gating patterns, satisfying the behavioural tests for
near/far field distinction and directional asymmetry.

The final element-wise scale(e_ij) is a learned linear projection of
the edge features onto d_model, providing a second path for edge
features to contribute gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GNOEdge(nn.Module):
    """Pairwise interaction kernel for a WEC farm graph.

    Args:
        d_model : Dimensionality of node feature vectors.
        d_edge  : Dimensionality of edge feature vectors (default 6).
    """

    def __init__(self, d_model: int, d_edge: int = 6) -> None:
        super().__init__()
        d_in = 2 * d_model + d_edge  # h_i || h_j || e_ij

        # Gated message MLP
        self.gate = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.content = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        # Edge-feature scale: second gradient path from e_ij
        self.edge_scale = nn.Linear(d_edge, d_model, bias=False)

    def forward(self, h_i: Tensor, h_j: Tensor, edge_feat: Tensor) -> Tensor:
        """Compute pairwise interaction message.

        Args:
            h_i       : (E, d_model)  source node features
            h_j       : (E, d_model)  target node features
            edge_feat : (E, d_edge)   edge feature vector

        Returns:
            message   : (E, d_model)  pairwise interaction vector
        """
        x = torch.cat([h_i, h_j, edge_feat], dim=-1)  # (E, 2*d_model + d_edge)
        msg = self.gate(x) * self.content(x)           # (E, d_model)
        scale = self.edge_scale(edge_feat)             # (E, d_model)
        return msg * (1.0 + scale)                     # residual scaling by edge
