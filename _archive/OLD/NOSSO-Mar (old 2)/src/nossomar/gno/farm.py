"""WECFarmGNO: variable-N WEC farm Graph Neural Operator.

Forward pass
------------
1. Lift each device's property vector to a node embedding h_i (d_model)
   via a shared MLP (farm_encoder).
2. Build fully-connected directed graph: E = N*(N-1) edges.
3. Compute 6-dim edge features from pairwise positions + omega.
4. Run n_layers rounds of GNOEdge message passing with sum aggregation.
5. Read out per-device (A_mat, B_mat) via two _SymmetricHead modules
   (same architecture as SixDOFSurrogate, guaranteeing symmetry and
   diagonal non-negativity).

Edge feature construction (d_edge = 6)
---------------------------------------
    r_ref   = 100.0 m
    lambda  = 50.0 m  (normalised: 0.5)
    eps     = 1e-3
    omega_ref = 1.0 rad/s
    mean_omega = mean(omega)  (scalar frequency summary)

    e[0] = r_ij / r_ref
    e[1] = cos(theta_ij)
    e[2] = sin(theta_ij)
    e[3] = mean_omega / omega_ref
    e[4] = 1 / (r_ij / r_ref + eps)
    e[5] = exp(-r_ij / (lambda / r_ref))  = exp(-r_norm / 0.5)

N=1 edge case
-------------
With a single WEC there are no edges. Message passing is skipped and
the node embedding is passed directly to the read-out heads.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from .edge import GNOEdge
from ..models.sixdof_surrogate import _SymmetricHead


R_REF = 100.0      # metres
LAM_NORM = 0.5     # lambda / r_ref  = 50 m / 100 m
EPS = 1e-3


def _build_edge_features(pos: Tensor, omega: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Build directed edge index and 6-dim edge features.

    Args:
        pos   : (N, 2)  WEC positions in metres
        omega : (K,)    frequency query points (rad/s)

    Returns:
        src   : (E,)       source node indices
        dst   : (E,)       destination node indices
        ef    : (E, 6)     edge feature matrix
    """
    N = pos.shape[0]
    if N == 1:
        # No edges
        device = pos.device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, 6, device=device)

    # Fully-connected directed graph
    idx = torch.arange(N, device=pos.device)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")  # (N, N)
    mask = src != dst
    src = src[mask]   # (E,)
    dst = dst[mask]   # (E,)

    d = pos[dst] - pos[src]             # (E, 2)  separation vectors
    r = d.norm(dim=-1).clamp(min=1.0)  # (E,)    distance in metres
    r_norm = r / R_REF
    theta = torch.atan2(d[:, 1], d[:, 0])  # (E,)

    omega_mean = omega.mean().expand(src.shape[0])  # (E,)

    ef = torch.stack([
        r_norm,
        torch.cos(theta),
        torch.sin(theta),
        omega_mean,
        1.0 / (r_norm + EPS),
        torch.exp(-r_norm / LAM_NORM),
    ], dim=-1)  # (E, 6)

    return src, dst, ef


class WECFarmGNO(nn.Module):
    """Variable-N WEC farm Graph Neural Operator.

    Args:
        d_model  : Node embedding dimension.
        d_edge   : Edge feature dimension (default 6).
        n_layers : Number of GNOEdge message-passing rounds.
    """

    D_PROPS = 4

    def __init__(self, d_model: int = 64, d_edge: int = 6, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Node encoder: props (N, 4) -> (N, d_model)
        self.farm_encoder = nn.Sequential(
            nn.Linear(self.D_PROPS, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        # Stack of edge kernels
        self.edge_layers = nn.ModuleList(
            [GNOEdge(d_model=d_model, d_edge=d_edge) for _ in range(n_layers)]
        )

        # Node update MLP (applied after each aggregation)
        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
            )
            for _ in range(n_layers)
        ])

        # Symmetric read-out heads (same as SixDOFSurrogate)
        self.head_A = _SymmetricHead(d_model)
        self.head_B = _SymmetricHead(d_model)

    def forward(
        self,
        props: Tensor,   # (N, 4)
        pos: Tensor,     # (N, 2)
        omega: Tensor,   # (K,)
    ) -> tuple[Tensor, Tensor]:
        """Compute farm-corrected A and B matrices.

        Returns:
            A_farm : (N, 6, 6)
            B_farm : (N, 6, 6)
        """
        N = props.shape[0]

        # 1. Node embeddings
        h = self.farm_encoder(props)       # (N, d_model)

        # 2. Edge construction
        src, dst, ef = _build_edge_features(pos, omega)

        # 3. Message passing
        for edge_fn, update_fn in zip(self.edge_layers, self.node_update):
            if src.shape[0] > 0:           # skip if N=1
                msg = edge_fn(h[src], h[dst], ef)   # (E, d_model)
                # Sum-aggregate messages at each destination node
                agg = torch.zeros_like(h)           # (N, d_model)
                agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
                h = update_fn(h + agg)
            else:
                h = update_fn(h)

        # 4. Read-out: symmetric heads guarantee symmetry + diag >= 0
        A_farm = self.head_A(h)   # (N, 6, 6)
        B_farm = self.head_B(h)   # (N, 6, 6)

        return A_farm, B_farm
