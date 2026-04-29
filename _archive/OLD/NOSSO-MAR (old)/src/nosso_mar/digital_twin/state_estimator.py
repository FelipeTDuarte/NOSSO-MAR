"""
State estimator for the digital twin — combines EnKF with NO forecast model.
Continuously ingests sensor observations and updates the twin state.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime

from ..assimilation.enkf import EnsembleKalmanFilter
from ..assimilation.state_vector import WaveStateVector


class StateEstimator:
    """
    Real-time state estimator for NOSSO-MAR digital twin.

    Maintains a probabilistic estimate of the ocean state by running
    an EnKF assimilation cycle at every observation arrival.
    """

    def __init__(self, enkf: EnsembleKalmanFilter,
                 grid_shape: tuple, update_interval_s: float = 300.0):
        self.enkf           = enkf
        self.H, self.W      = grid_shape
        self.update_interval = update_interval_s
        self.last_update    = None
        self.current_state  = None

    def initialise(self, background: WaveStateVector):
        x0 = background.to_vector()
        self.enkf.initialise(x0)
        self.current_state = background

    def update(self, observations: Dict, R_diag: torch.Tensor):
        """
        Assimilate a batch of observations.
        observations: dict mapping sensor_id -> observed value tensor
        """
        if not observations:
            return

        y_obs  = torch.cat(list(observations.values()))
        x_mean = self.enkf.analysis(y_obs, R_diag)

        self.current_state = WaveStateVector.from_vector(x_mean, self.H, self.W)
        self.last_update   = datetime.utcnow()

    def get_state(self) -> WaveStateVector:
        return self.current_state

    def get_uncertainty(self) -> torch.Tensor:
        return self.enkf.get_spread()
