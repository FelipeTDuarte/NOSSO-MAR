"""
4D-Var data assimilation for NOSSO-MAR.

Minimises the cost function:
    J(x0) = ½(x0 - xb)^T B^{-1} (x0 - xb)
           + ½ Σ_t [H(M_t(x0)) - y_t]^T R^{-1} [H(M_t(x0)) - y_t]

Uses automatic differentiation through the neural operator M_t for
computing the gradient ∇J(x0), replacing the classical adjoint model.

References:
    Talagrand & Courtier (1987) — 4D-Var original
    Chen et al. (2021)          — Hybrid DA + ML methods
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import torch
import torch.optim as optim


class FourDVar:
    """
    Gradient-based 4D-Var using AD through neural operators.

    forecast_model: callable x_t -> x_{t+1} (must be differentiable)
    obs_operator:   callable x_t -> y_t
    """

    def __init__(self,
                 forecast_model: Callable,
                 obs_operator: Callable,
                 state_dim: int,
                 n_iter: int = 100,
                 lr: float = 1e-3):
        self.model    = forecast_model
        self.H        = obs_operator
        self.n_iter   = n_iter
        self.lr       = lr
        self.state_dim = state_dim

    def assimilate(
        self,
        x_background: torch.Tensor,
        observations:  List[torch.Tensor],
        obs_times:     List[int],
        B_inv_diag:    torch.Tensor,
        R_inv_diag:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns optimised initial state x0.

        x_background : (state_dim,)  background state
        observations : list of (obs_dim,) tensors at each assimilation time
        obs_times    : list of integer time steps when obs are available
        B_inv_diag   : (state_dim,) background error precision diagonal
        R_inv_diag   : (obs_dim,)   observation error precision diagonal
        """
        x0 = x_background.clone().requires_grad_(True)
        optimiser = optim.LBFGS([x0], lr=self.lr, max_iter=20)

        def closure():
            optimiser.zero_grad()
            # Background term
            diff_b = x0 - x_background
            J_b    = 0.5 * (diff_b * B_inv_diag * diff_b).sum()

            # Observation terms — propagate state forward
            x = x0
            J_o = torch.tensor(0.0)
            t   = 0
            obs_idx = 0
            for obs_idx, (y_t, t_obs) in enumerate(zip(observations, obs_times)):
                while t < t_obs:
                    x = self.model(x)
                    t += 1
                diff_o = self.H(x) - y_t
                J_o    = J_o + 0.5 * (diff_o * R_inv_diag * diff_o).sum()

            J = J_b + J_o
            J.backward()
            return J

        for _ in range(self.n_iter):
            optimiser.step(closure)

        return x0.detach()
