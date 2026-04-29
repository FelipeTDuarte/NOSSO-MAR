"""
Adaptive 2-D Cartesian mesh with quad-tree refinement.
Provides the data structure and interpolation for AMR in NOSSO-MAR.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch


@dataclass
class Cell:
    x0: float; y0: float
    dx: float; dy: float
    level: int = 0
    children: List["Cell"] = field(default_factory=list)
    value: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        return self.x0 + self.dx / 2, self.y0 + self.dy / 2

    def refine(self) -> List["Cell"]:
        """Split into 4 children."""
        hdx, hdy = self.dx / 2, self.dy / 2
        self.children = [
            Cell(self.x0,        self.y0,        hdx, hdy, self.level + 1),
            Cell(self.x0 + hdx,  self.y0,        hdx, hdy, self.level + 1),
            Cell(self.x0,        self.y0 + hdy,  hdx, hdy, self.level + 1),
            Cell(self.x0 + hdx,  self.y0 + hdy,  hdx, hdy, self.level + 1),
        ]
        return self.children


class AdaptiveMesh2D:
    """
    Quad-tree adaptive mesh for wave propagation domains.

    Usage:
        mesh = AdaptiveMesh2D(Lx=10000, Ly=10000, base_nx=32, base_ny=32)
        mesh.refine_where(criterion_mask)   # (H, W) bool
        pts  = mesh.leaf_centers()           # (N, 2) query points for mesh-free ops
    """

    def __init__(self, Lx: float, Ly: float, base_nx: int, base_ny: int):
        self.Lx, self.Ly = Lx, Ly
        dx = Lx / base_nx
        dy = Ly / base_ny
        self.roots = [
            Cell(i * dx, j * dy, dx, dy)
            for j in range(base_ny)
            for i in range(base_nx)
        ]

    def leaf_cells(self) -> List[Cell]:
        def _leaves(cell):
            if not cell.children:
                return [cell]
            return [c for child in cell.children for c in _leaves(child)]
        return [c for root in self.roots for c in _leaves(root)]

    def leaf_centers(self) -> np.ndarray:
        cells = self.leaf_cells()
        return np.array([c.center for c in cells])

    def n_leaves(self) -> int:
        return len(self.leaf_cells())
