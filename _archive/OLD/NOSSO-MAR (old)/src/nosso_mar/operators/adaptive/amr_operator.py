"""
Adaptive Multi-Resolution Operator (AMR-NO).

Combines an adaptive mesh with a mesh-free neural operator to allocate
compute where needed — fine resolution near WECs and wave fronts,
coarse resolution in calm deep-water regions.

Strategy:
    1. Run coarse FNO pass on uniform grid
    2. Apply refinement criterion -> identify high-gradient regions
    3. Interpolate coarse solution to AMR leaf points
    4. Run local mesh-free operator (RBF-NO) at refined points
    5. Merge coarse + fine solutions
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..base import BaseOperator
from .adaptive_mesh import AdaptiveMesh2D
from .refinement_criterion import WaveGradientCriterion


class AMROperator(BaseOperator):
    """
    cfg keys:
        coarse_operator_cfg : dict  — FNO or WNO config for coarse pass
        fine_operator_cfg   : dict  — mesh-free operator config for fine pass
        refinement_threshold: float
        max_levels          : int
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        from ..fno.fno2d import FNO2d
        from ..meshfree.rbf_operator import RBFOperator

        self.coarse_op   = FNO2d(cfg.get("coarse_operator_cfg", {
            "in_channels": 5, "out_channels": 3, "width": 32, "n_layers": 3}))
        self.fine_op     = RBFOperator(cfg.get("fine_operator_cfg", {
            "in_dim": 5, "out_dim": 3, "hidden": 64}))
        self.criterion   = WaveGradientCriterion(
            cfg.get("refinement_threshold", 0.05))

    def forward(self, u: torch.Tensor,
                query_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Coarse pass
        coarse_out = self.coarse_op(u)

        # Identify regions needing refinement
        eta_coarse = coarse_out[:, :1]
        refine_mask = self.criterion(eta_coarse)

        # TODO: interpolate to AMR points, run fine_op, merge
        # For now return coarse output (full pipeline in development)
        return coarse_out
