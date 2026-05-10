"""
Wave Farm MARL Environment.

Wraps the NOSSO-MAR coupled NO (Module 1 + Module 2) as a Gym-compatible
multi-agent environment for WEC layout and PTO optimisation.

Agents: one per WEC — controls (x,y) position AND/OR B_pto.
State:  wave field + all WEC positions + absorbed power history.
Reward: cooperative — total farm power (potentially multi-objective).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class WaveFarmEnv:
    """
    Multi-agent wave energy farm environment.

    cfg keys:
        n_wec            : int
        farm_bounds      : [[x_min,x_max],[y_min,y_max]] [m]
        min_wec_spacing  : float [m]  — minimum allowed inter-WEC distance
        episode_length   : int   — steps per episode
        dt               : float — time step [s]
        wave_scenarios   : list of dicts — sampled each episode reset
    """

    def __init__(self, cfg: Dict, wave_no=None, fsi_module=None,
                 coupling_manager=None):
        self.cfg             = cfg
        self.n_wec           = cfg["n_wec"]
        self.bounds          = cfg.get("farm_bounds", [[0,2000],[0,2000]])
        self.min_spacing     = cfg.get("min_wec_spacing", 50.0)
        self.episode_length  = cfg.get("episode_length", 200)
        self.wave_no         = wave_no
        self.fsi_module      = fsi_module
        self.coupling        = coupling_manager
        self._step           = 0
        self._positions      = None
        self._device_props   = cfg.get("device_properties", [{}] * self.n_wec)

    @property
    def n_agents(self): return self.n_wec

    def observation_space_dim(self) -> int:
        # Local eta spectrum (64) + position (2) + neighbour positions (2*(n-1))
        return 64 + 2 + 2 * (self.n_wec - 1)

    def action_space_dim(self) -> int:
        return 2  # Δx, Δy

    def reset(self) -> List[np.ndarray]:
        self._step = 0
        xb, yb = self.bounds
        self._positions = [
            [np.random.uniform(*xb), np.random.uniform(*yb)]
            for _ in range(self.n_wec)
        ]
        return self._get_observations()

    def step(self, actions: List[torch.Tensor])             -> Tuple[List, List[float], bool, Dict]:
        # Apply actions (position updates)
        for i, (pos, act) in enumerate(zip(self._positions, actions)):
            delta = act.detach().numpy() if hasattr(act, "numpy") else act
            new_x = np.clip(pos[0] + delta[0], *self.bounds[0])
            new_y = np.clip(pos[1] + delta[1], *self.bounds[1])
            self._positions[i] = [new_x, new_y]

        # Evaluate farm with NO (or fallback dummy)
        power = self._evaluate_farm()
        reward = [power / self.n_wec] * self.n_wec  # shared cooperative reward

        self._step += 1
        done = self._step >= self.episode_length
        return self._get_observations(), reward, done, {"total_power": power}

    def _evaluate_farm(self) -> float:
        if self.coupling is None:
            # Dummy evaluation during unit tests
            return float(np.random.uniform(1e5, 1e6))
        inputs = {
            "spectrum":          {"Hs": 2.0, "Tp": 8.0, "direction": 0.0},
            "bathymetry":        torch.ones(1, 64, 64) * 30.0,
            "wec_positions":     self._positions,
            "device_properties": self._device_props,
            "omega":             np.linspace(0.2, 3.0, 64),
        }
        out = self.coupling.run_coupled(inputs)
        return float(out.get("total_power", 0.0))

    def _get_observations(self) -> List[np.ndarray]:
        obs = []
        for i, pos in enumerate(self._positions):
            local_eta = np.zeros(64)   # placeholder — filled by Module 1
            neigh_pos = np.concatenate([
                p for j, p in enumerate(self._positions) if j != i])
            ob = np.concatenate([local_eta, pos, neigh_pos])
            obs.append(ob.astype(np.float32))
        return obs
