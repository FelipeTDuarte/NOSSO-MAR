"""FarmDataset: synthetic WEC farm configurations via Latin Hypercube Sampling.

Each sample contains:
    props  : (N, 4)     WEC property vectors [radius, draft, depth, Bpto]
    pos    : (N, 2)     WEC positions in metres
    omega  : (K,)       frequency query points (rad/s)
    A_base : (N, 6, 6)  per-device baseline added-mass (LinearSuperpositionBaseline)
    B_base : (N, 6, 6)  per-device baseline radiation-damping

Layout generation
-----------------
Devices are placed on a regular grid with the sampled spacing,
then rotated by the sampled heading angle. This guarantees:
    - Minimum inter-device distance >= spacing_min
    - Physically plausible rectangular array layouts
    - Heading diversity via rotation

Parameter ranges (LHS)
----------------------
    radius  : [0.5, 5.0]   m
    draft   : [0.5, 8.0]   m
    depth   : [10., 80.]   m
    Bpto    : [1e3, 1e5]   N.s/m
    spacing : [20., 150.]  m  (minimum inter-device distance)
    heading : [0, 2*pi]    rad
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..models.sixdof_surrogate import SixDOFSurrogate
from ..gno.baseline import LinearSuperpositionBaseline


# Default frequency grid shared across all samples
_OMEGA_DEFAULT = torch.linspace(0.2, 2.5, 12)


@dataclass
class FarmSample:
    """One synthetic WEC farm configuration."""
    props:  Tensor   # (N, 4)
    pos:    Tensor   # (N, 2)
    omega:  Tensor   # (K,)
    A_base: Tensor   # (N, 6, 6)
    B_base: Tensor   # (N, 6, 6)


def _grid_positions(n: int, spacing: float, heading: float) -> np.ndarray:
    """Place N devices on a rectangular grid, then rotate by heading.

    Args:
        n       : number of devices
        spacing : minimum inter-device distance (m)
        heading : rotation angle (rad)

    Returns:
        pos : (N, 2)  device positions in metres
    """
    # Determine grid dimensions (roughly square)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    xs = np.arange(n_cols) * spacing
    ys = np.arange(n_rows) * spacing
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (n_rows*n_cols, 2)
    pos = grid[:n]  # take first N points

    # Rotate by heading
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]])
    pos = pos @ R.T

    return pos.astype(np.float32)


class FarmDataset(Dataset):
    """Synthetic WEC farm dataset generated via LHS.

    Args:
        n_cases : number of farm configurations to generate
        n_min   : minimum number of WECs per farm
        n_max   : maximum number of WECs per farm
        seed    : random seed for reproducibility
        omega   : frequency query points (rad/s); defaults to 12-point grid
        d_hidden: hidden dim for the internal SixDOFSurrogate
    """

    # LHS parameter bounds: [radius, draft, depth, Bpto, spacing, heading]
    _LB = np.array([0.5,  0.5,  10.0, 1e3,  20.0, 0.0])
    _UB = np.array([5.0,  8.0,  80.0, 1e5, 150.0, 2 * math.pi])

    def __init__(
        self,
        n_cases: int = 1000,
        n_min: int = 2,
        n_max: int = 8,
        seed: int = 42,
        omega: Tensor | None = None,
        d_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.omega = omega if omega is not None else _OMEGA_DEFAULT

        # Build frozen baseline
        torch.manual_seed(seed)
        surrogate = SixDOFSurrogate(d_hidden=d_hidden)
        self.baseline = LinearSuperpositionBaseline(surrogate)

        # LHS sampling
        rng = np.random.default_rng(seed)
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        raw = sampler.random(n=n_cases)                      # (n_cases, 6) in [0,1]
        params = qmc.scale(raw, self._LB, self._UB)          # (n_cases, 6)

        # N per sample: uniform integer in [n_min, n_max]
        ns = rng.integers(n_min, n_max + 1, size=n_cases)   # (n_cases,)

        self._samples: list[FarmSample] = []
        with torch.no_grad():
            for i in range(n_cases):
                radius, draft, depth, bpto, spacing, heading = params[i]
                n = int(ns[i])

                # WEC props: all devices share the same type in this farm
                props_np = np.tile(
                    [radius, draft, depth, bpto], (n, 1)
                ).astype(np.float32)                         # (N, 4)
                props = torch.from_numpy(props_np)

                # Positions
                pos_np = _grid_positions(n, float(spacing), float(heading))
                pos = torch.from_numpy(pos_np)               # (N, 2)

                # Baseline A, B
                A_base, B_base = self.baseline(props, self.omega)

                self._samples.append(FarmSample(
                    props=props,
                    pos=pos,
                    omega=self.omega.clone(),
                    A_base=A_base,
                    B_base=B_base,
                ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> FarmSample:
        return self._samples[idx]
