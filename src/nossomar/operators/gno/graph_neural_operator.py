"""
Graph Neural Operator (GNO) for irregular mesh wave simulation
and WEC-farm interaction modelling.

Architecture per Li et al. (2020) GNO + message-passing extension:
    Encoder → L message-passing layers → Decoder

References:
    Li et al. (2020) — Neural Operator: Graph Kernel Network for PDEs
    https://arxiv.org/abs/2003.03485
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from .message_passing import WaveInteractionLayer
from .mesh_utils import compute_edge_features
from ..base import BaseOperator


class GNOEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


class GNODecoder(nn.Module):
    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphNeuralOperator(BaseOperator):
    """
    GNO operating on unstructured node sets (irregular meshes or WEC positions).

    cfg keys:
        node_in_dim   : int  — input node feature dimension
        node_out_dim  : int  — output node feature dimension
        edge_dim      : int  — edge feature dimension (default 7 from mesh_utils)
        hidden        : int  — hidden width (default 128)
        n_layers      : int  — message-passing layers (default 4)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        node_in  = cfg["node_in_dim"]
        node_out = cfg["node_out_dim"]
        edge_dim = cfg.get("edge_dim",  7)
        hidden   = cfg.get("hidden",  128)
        n_layers = cfg.get("n_layers",  4)

        self.encoder = GNOEncoder(node_in, hidden)
        self.layers  = nn.ModuleList([
            WaveInteractionLayer(hidden, edge_dim, hidden)
            for _ in range(n_layers)
        ])
        self.decoder = GNODecoder(hidden, node_out)

    def forward(self, u: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u          : (N, node_in_dim)  node features
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)
        Returns    : (N, node_out_dim)
        """
        assert edge_index is not None, "GNO requires edge_index"
        h = self.encoder(u)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.decoder(h)
