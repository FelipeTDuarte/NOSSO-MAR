"""
Observation operators H: state space -> observation space.
Maps the model state to what sensors actually measure.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import numpy as np


class WaveBuoyObsOperator:
    """
    H: state -> wave buoy measurements (η at fixed locations).

    buoy_positions: List[(i, j)] grid indices of buoy locations
    """

    def __init__(self, buoy_positions: List[Tuple[int, int]]):
        self.positions = buoy_positions

    def observe(self, state_eta: torch.Tensor) -> torch.Tensor:
        """state_eta: (H, W) -> obs: (n_buoys,)"""
        return torch.stack([state_eta[i, j] for i, j in self.positions])

    def jacobian(self, H: int, W: int) -> torch.Tensor:
        """Sparse observation matrix (n_buoys, H*W) for Kalman update."""
        n_obs = len(self.positions)
        J = torch.zeros(n_obs, H * W)
        for k, (i, j) in enumerate(self.positions):
            J[k, i * W + j] = 1.0
        return J


class SatelliteSwathOperator:
    """
    H: state -> satellite altimeter swath (η along 1-D track).
    Simulates SWOT or conventional nadir altimetry.
    """

    def __init__(self, track_col: int):
        self.track_col = track_col   # column index of satellite track

    def observe(self, state_eta: torch.Tensor) -> torch.Tensor:
        """state_eta: (H, W) -> obs: (H,) along track"""
        return state_eta[:, self.track_col]
