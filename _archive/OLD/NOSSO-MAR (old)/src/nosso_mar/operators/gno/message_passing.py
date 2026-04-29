"""
Message-passing layers for Graph Neural Operators.

WaveInteractionLayer models the hydrodynamic interaction between
WEC nodes in the wave farm graph.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class WaveInteractionLayer(nn.Module):
    """
    Edge-conditioned message passing for WEC-WEC wave interaction.

    Node features:  device state + local wave field
    Edge features:  relative position, distance, wave direction cosines

    Update:
        m_{ij} = MLP_e([h_i, h_j, e_{ij}])
        h_i'   = MLP_n([h_i, Σ_j m_{ij}])
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, node_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden), nn.GELU(),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        h          : (N, node_dim)
        edge_index : (2, E)  — [source, target]
        edge_attr  : (E, edge_dim)
        Returns    : (N, node_dim)
        """
        src, tgt = edge_index
        msg = self.edge_mlp(
            torch.cat([h[src], h[tgt], edge_attr], dim=-1))  # (E, node_dim)

        agg = torch.zeros_like(h)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)

        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)   # residual
