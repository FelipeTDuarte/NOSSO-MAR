"""Multi-object hydrodynamic interaction module (P2-T3).

Replaces the Phase 1 MultiObjectFSI stub with a real pairwise interaction
model. Each WEC is a graph node; edges connect every pair of WECs and carry
relative-position and distance features. A two-layer Graph Attention Network
(GAT) aggregates neighbour messages into a per-WEC interaction correction
vector in 6-DOF force space.

Key design decisions:
  - Fully-connected graph: every WEC influences every other. For farm sizes
    up to ~30 WECs this is tractable; sparse graphs can be introduced later.
  - Distance modulation: edge features include 1/(1+d) so far-away WECs
    contribute less attention weight even before the GAT learns anything.
  - Single-WEC short-circuit: returns zeros without running the graph to
    avoid NaN from empty edge sets.
  - Permutation equivariance: the architecture is equivariant by construction
    (symmetric message passing, no positional indexing).

Reference: Velickovic et al. (2018) Graph Attention Networks, ICLR.
           https://arxiv.org/abs/1710.10903
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from nossomar.core.contracts import WECState


class _EdgeMLP(nn.Module):
    """MLP that produces a message from concatenated node features + edge features."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GATLayer(nn.Module):
    """Single GAT layer with multi-head attention over a fully-connected graph.

    Args:
        d_node:   Node feature dimension.
        d_edge:   Edge feature dimension.
        hidden:   Hidden width for attention scoring MLP.
        n_heads:  Number of attention heads.
        d_out:    Output node dimension (= d_node by default for residual add).
    """

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        hidden: int,
        n_heads: int,
        d_out: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = max(hidden // n_heads, 1)

        # Attention scoring: concat(h_i, h_j, e_ij) → scalar per head
        self.attn_mlp = _EdgeMLP(
            in_dim=d_node * 2 + d_edge,
            hidden=hidden,
            out_dim=n_heads,
        )
        # Message: h_j + edge → value per head
        self.msg_mlp = _EdgeMLP(
            in_dim=d_node + d_edge,
            hidden=hidden,
            out_dim=n_heads * self.d_head,
        )
        # Output projection
        self.out_proj = nn.Linear(n_heads * self.d_head, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(
        self,
        h: torch.Tensor,        # (N, d_node)
        edge_feats: torch.Tensor,  # (N, N, d_edge)
    ) -> torch.Tensor:             # (N, d_out)
        N = h.shape[0]

        # Expand node features for all pairs
        hi = h.unsqueeze(1).expand(N, N, -1)   # (N, N, d_node)  sender
        hj = h.unsqueeze(0).expand(N, N, -1)   # (N, N, d_node)  receiver

        # Attention scores
        attn_in = torch.cat([hi, hj, edge_feats], dim=-1)  # (N, N, 2*d_node+d_edge)
        scores = self.attn_mlp(attn_in)                     # (N, N, n_heads)

        # Mask self-loops (diagonal)
        mask = torch.eye(N, device=h.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        alpha = F.softmax(scores, dim=1)   # (N, N, n_heads) -- normalise over senders

        # Messages from each sender
        msg_in = torch.cat([hj, edge_feats], dim=-1)          # (N, N, d_node+d_edge)
        msgs = self.msg_mlp(msg_in)                            # (N, N, H*d_head)
        msgs = msgs.view(N, N, self.n_heads, self.d_head)     # (N, N, H, d_head)

        # Weighted aggregation
        alpha = alpha.unsqueeze(-1)        # (N, N, H, 1)
        agg = (alpha * msgs).sum(dim=1)    # (N, H, d_head)
        agg = agg.view(N, -1)              # (N, H*d_head)

        out = self.out_proj(agg)           # (N, d_out)
        return self.norm(out)


class MultiObjectInteraction(nn.Module):
    """GAT-based pairwise hydrodynamic interaction correction.

    For a farm of N WECs, computes a 6-DOF force correction delta_force(N,6)
    that accounts for wave-radiation coupling between devices.

    Args:
        d_latent: RINO latent dimension (node feature size).
        hidden:   Hidden width for attention and message MLPs.
        n_heads:  Number of GAT attention heads.
    """

    # Reference separation for distance normalisation (metres)
    _D_REF: float = 100.0

    def __init__(
        self,
        d_latent: int = 64,
        hidden: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        # Edge feature dim: [dx, dy, dist, dx/D_REF, dy/D_REF, 1/(1+dist)]
        d_edge = 6

        self.gat1 = _GATLayer(
            d_node=d_latent, d_edge=d_edge,
            hidden=hidden, n_heads=n_heads, d_out=hidden,
        )
        self.gat2 = _GATLayer(
            d_node=hidden, d_edge=d_edge,
            hidden=hidden, n_heads=n_heads, d_out=hidden,
        )
        # Final projection to 6-DOF force correction
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )

    def _edge_features(self, pos: torch.Tensor) -> torch.Tensor:
        """Build (N, N, 6) edge feature tensor from WEC positions (N, 2)."""
        N = pos.shape[0]
        pi = pos.unsqueeze(1).expand(N, N, 2)   # (N, N, 2)
        pj = pos.unsqueeze(0).expand(N, N, 2)   # (N, N, 2)
        diff = pj - pi                           # (N, N, 2)  [dx, dy]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-3)  # (N, N, 1)
        inv_dist = 1.0 / (1.0 + dist)            # decays with separation
        norm_diff = diff / self._D_REF
        return torch.cat([diff, dist, norm_diff, inv_dist], dim=-1)  # (N, N, 6)

    def forward(
        self,
        states: WECState,
        latents: torch.Tensor,   # (N, d_latent)
    ) -> torch.Tensor:           # (N, 6)
        N = latents.shape[0]

        # Single-WEC short-circuit: no neighbours, no interaction
        if N == 1:
            return torch.zeros(1, 6, device=latents.device, dtype=latents.dtype)

        edge_feats = self._edge_features(states.pos)  # (N, N, 6)

        h = self.gat1(latents, edge_feats)    # (N, hidden)
        h = F.gelu(h)
        h = self.gat2(h, edge_feats)          # (N, hidden)

        return self.out(h)                    # (N, 6)
