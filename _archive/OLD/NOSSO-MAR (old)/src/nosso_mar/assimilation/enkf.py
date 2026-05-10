"""
Ensemble Kalman Filter (EnKF) for NOSSO-MAR.

Assimilates wave buoy and satellite observations into the NO state.
Compatible with any NOSSO-MAR module that exposes a run() method as
the forecast model.

References:
    Evensen (1994, 2003) — The Ensemble Kalman Filter
    Sakov & Oke (2008)   — A Deterministic Formulation of the Ensemble Kalman Filter
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import torch
import numpy as np


class EnsembleKalmanFilter:
    """
    Stochastic EnKF with perturbed observations.

    Forecast model:  f(x) — wraps any NOSSO-MAR module
    Obs operator:    H(x) — maps state to observation space

    Parameters:
        n_ensemble     : ensemble size (typically 50–200)
        state_dim      : full state vector dimension
        obs_dim        : observation vector dimension
        inflation      : multiplicative covariance inflation (> 1.0)
        localisation_r : localisation radius (None = no localisation)
    """

    def __init__(self,
                 forecast_fn: Callable,
                 obs_operator: Callable,
                 n_ensemble: int,
                 state_dim: int,
                 obs_dim: int,
                 inflation: float = 1.05,
                 localisation_r: Optional[float] = None):
        self.forecast_fn    = forecast_fn
        self.H              = obs_operator
        self.n_ens          = n_ensemble
        self.state_dim      = state_dim
        self.obs_dim        = obs_dim
        self.inflation      = inflation
        self.localisation_r = localisation_r

        # Ensemble matrix X: (state_dim, n_ensemble)
        self.X = torch.zeros(state_dim, n_ensemble)

    def initialise(self, x_mean: torch.Tensor,
                   P0_diag: Optional[torch.Tensor] = None):
        """Initialise ensemble from background state + perturbations."""
        std = P0_diag.sqrt() if P0_diag is not None else torch.ones(self.state_dim)
        pert = torch.randn(self.state_dim, self.n_ens) * std.unsqueeze(1)
        self.X = x_mean.unsqueeze(1) + pert

    def forecast(self, inputs: Dict) -> None:
        """Propagate each ensemble member through the forecast model."""
        for j in range(self.n_ens):
            inputs["state"] = self.X[:, j]
            result = self.forecast_fn(inputs)
            if "state" in result:
                self.X[:, j] = result["state"]

    def analysis(self, y_obs: torch.Tensor,
                 R_diag: torch.Tensor) -> torch.Tensor:
        """
        EnKF analysis step (stochastic, perturbed observations).

        y_obs  : (obs_dim,)  observation vector
        R_diag : (obs_dim,)  observation error variances
        Returns: x_a — analysis ensemble mean (state_dim,)
        """
        # Apply covariance inflation
        x_mean = self.X.mean(dim=1, keepdim=True)
        A = self.X - x_mean
        A *= self.inflation

        # Project ensemble to observation space
        HX = torch.stack([self.H(self.X[:, j]) for j in range(self.n_ens)], dim=1)
        HA = HX - HX.mean(dim=1, keepdim=True)

        # Innovation covariance S = HA HA^T / (N-1) + R
        N  = self.n_ens
        S  = (HA @ HA.T) / (N - 1) + torch.diag(R_diag)

        # Kalman gain K = A HA^T (N-1)^{-1} S^{-1}
        K = (A @ HA.T) / (N - 1) @ torch.linalg.inv(S)   # (state_dim, obs_dim)

        # Perturbed observations
        y_pert = y_obs.unsqueeze(1) +                  torch.randn(self.obs_dim, N) * R_diag.sqrt().unsqueeze(1)

        # Analysis
        self.X = self.X + K @ (y_pert - HX)
        return self.X.mean(dim=1)

    def get_mean(self) -> torch.Tensor:
        return self.X.mean(dim=1)

    def get_spread(self) -> torch.Tensor:
        return self.X.std(dim=1)
