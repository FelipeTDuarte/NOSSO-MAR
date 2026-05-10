"""
DeepONet — Deep Operator Network.

Learns the operator G: u → G(u)(y) via
    G(u)(y) ≈ Σ_k branch_k(u) · trunk_k(y)

Primary candidate for Module 2: WEC FSI surrogate (BEM response as
a function of frequency ω for any device geometry).

References:
    Lu et al. (2021) — Learning Nonlinear Operators via DeepONet
    https://arxiv.org/abs/1910.03193
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from ..base import BaseOperator


class MLP(nn.Module):
    """Fully-connected network with optional layer norm."""

    def __init__(self, dims: List[int], activation: str = "tanh",
                 layer_norm: bool = False):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "gelu": nn.GELU,
                   "relu": nn.ReLU, "silu": nn.SiLU}
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_map[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(BaseOperator):
    """
    DeepONet with separate branch and trunk networks.

    Branch net: encodes input function u (e.g., WEC properties, boundary
                wave conditions) evaluated at fixed sensor locations.
    Trunk  net: encodes query coordinates y (e.g., frequency ω, spatial
                point (x,y)).

    G(u)(y) = Σ_{k=1}^{p} b_k(u) · t_k(y) + bias

    cfg keys:
        branch_input_dim  : int  — branch input size
        trunk_input_dim   : int  — trunk input size (e.g. 1 for omega)
        hidden_dim        : int  — hidden layer width
        n_hidden          : int  — number of hidden layers per net
        output_dim        : int  — output dimension per query point
        p                 : int  — trunk/branch output dim (basis size)
        activation        : str
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        b_in  = cfg["branch_input_dim"]
        t_in  = cfg["trunk_input_dim"]
        H     = cfg.get("hidden_dim",   128)
        n_h   = cfg.get("n_hidden",       4)
        C_out = cfg.get("output_dim",     1)
        p     = cfg.get("p",            128)  # basis functions
        act   = cfg.get("activation", "tanh")

        b_dims = [b_in] + [H] * n_h + [p * C_out]
        t_dims = [t_in] + [H] * n_h + [p]

        self.branch   = MLP(b_dims, activation=act, layer_norm=True)
        self.trunk    = MLP(t_dims, activation=act, layer_norm=True)
        self.bias     = nn.Parameter(torch.zeros(C_out))
        self.p        = p
        self.C_out    = C_out

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u            : (B, branch_input_dim)
        query_points : (Q, trunk_input_dim) or (B, Q, trunk_input_dim)
        Returns      : (B, Q, output_dim)
        """
        assert query_points is not None, "DeepONet requires query_points (trunk input)"

        b_out = self.branch(u)                      # (B, p * C_out)
        b_out = b_out.view(-1, self.C_out, self.p)  # (B, C_out, p)

        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0).expand(u.shape[0], -1, -1)

        # Trunk: (B, Q, trunk_in) -> (B, Q, p)
        t_out = self.trunk(query_points)            # (B, Q, p)

        # Dot product over basis: (B, C_out, p) x (B, Q, p) -> (B, Q, C_out)
        out = torch.einsum("bcp,bqp->bqc", b_out, t_out)
        return out + self.bias
