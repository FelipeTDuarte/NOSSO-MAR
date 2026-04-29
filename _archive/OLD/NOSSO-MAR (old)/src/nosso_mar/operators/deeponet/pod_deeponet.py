"""
POD-DeepONet — Proper Orthogonal Decomposition enhanced DeepONet.

Uses POD basis from training data to initialise the trunk network,
accelerating convergence and improving accuracy for wave problems
where the solution manifold has low effective dimension.

References:
    Lu et al. (2022) — A comprehensive study of non-adaptive and residual-based
    adaptive sampling for physics-informed neural networks
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np

from .deeponet import DeepONet, MLP


class PODDeepONet(DeepONet):
    """
    DeepONet whose trunk basis is initialised from POD modes.

    Call `set_pod_basis(modes)` after computing modes from BEM/CFD data
    before training to warm-start the trunk network.
    """

    def set_pod_basis(self, modes: np.ndarray):
        """
        modes: (n_points, p) POD spatial modes from SVD of snapshot matrix.
        Replaces the last trunk linear layer weights.
        """
        with torch.no_grad():
            t = torch.from_numpy(modes.T.astype(np.float32))  # (p, n_points)
            last_layer = list(self.trunk.net.children())[-1]
            if isinstance(last_layer, nn.Linear):
                last_layer.weight.data = t
