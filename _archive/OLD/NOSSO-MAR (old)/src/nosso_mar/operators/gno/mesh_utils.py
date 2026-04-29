"""
Utility functions for building graphs from unstructured meshes.
"""
from __future__ import annotations
import torch
import numpy as np


def build_radius_graph(pos: torch.Tensor, radius: float,
                       max_neighbors: int = 32) -> torch.Tensor:
    """
    Build edge_index for all pairs within a given radius.

    pos    : (N, 2) node positions
    radius : connectivity radius [m]
    Returns: (2, E) edge_index
    """
    N = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N, 2)
    dist = diff.norm(dim=-1)                      # (N, N)
    mask = (dist < radius) & (dist > 0)
    idx  = mask.nonzero(as_tuple=False).t()       # (2, E)
    return idx


def build_knn_graph(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
    """K-nearest-neighbours graph."""
    N = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = diff.norm(dim=-1)
    dist.fill_diagonal_(float("inf"))
    _, idx_k = dist.topk(k, dim=-1, largest=False)
    src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k).reshape(-1)
    tgt = idx_k.reshape(-1)
    return torch.stack([src, tgt], dim=0)


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor,
                           wave_dir: float = 0.0) -> torch.Tensor:
    """
    Edge features: [dx, dy, dist, cos_theta_wave, sin_theta_wave]
    wave_dir: principal wave direction [rad]
    """
    src, tgt = edge_index
    diff  = pos[tgt] - pos[src]               # (E, 2)
    dist  = diff.norm(dim=-1, keepdim=True)   # (E, 1)
    d_hat = diff / (dist + 1e-8)              # unit vector
    cos_w = torch.full((diff.shape[0], 1), float(np.cos(wave_dir)), device=pos.device)
    sin_w = torch.full((diff.shape[0], 1), float(np.sin(wave_dir)), device=pos.device)
    return torch.cat([diff, dist, d_hat, cos_w, sin_w], dim=-1)   # (E, 7)
