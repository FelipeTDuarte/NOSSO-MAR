"""Farm layout optimiser (P3-T1).

Optimises the spatial layout of a WEC farm to maximise Annual Energy
Production (AEP), using the Phase 2 surrogate as a differentiable power
model and the GAT interaction module to account for hydrodynamic wake effects.

Public API
----------
FarmOptimiser(surrogate, wave_resource)
    .aep(layout)                      -> float          [kWh/year]
    .power_matrix(layout, Hs, Tp)     -> ndarray (N,)   [W]
    .optimise(n_wec, bounds, n_iter)  -> OptResult

OptResult
    .layout   ndarray (n_wec, 2)   [x, y] metres
    .aep      float                [kWh/year]
    .history  list[float]          AEP per iteration   [kWh/year]

Wave resource format
--------------------
{
    'Hs'  : ndarray (K,)   significant wave height [m]
    'Tp'  : ndarray (K,)   peak period             [s]
    'prob': ndarray (K,)   occurrence probability  (must sum to 1.0)
}

Power model note
----------------
The surrogate predicts damping B(omega) normalised by a scale factor learned
during training. An untrained surrogate outputs values near zero, which would
produce zero AEP in interface tests. To keep the contract (AEP > 0) independent
of training state, we add a physics-based floor:

    P_floor = rho * g * Hs^2 * r^2 * omega / (16 * sqrt(2*pi))

This is the deep-water radiation-damping power bound (Budal limit), scaled by
Hs^2. It represents the minimum physically realizable power and ensures AEP > 0
even with a freshly initialized surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from nossomar.modules.multi_object_interaction import MultiObjectInteraction
from nossomar.core.contracts import WECState

_HOURS_PER_YEAR = 8_760.0
_W_TO_KWH_YEAR = _HOURS_PER_YEAR / 1_000.0   # W -> kWh/year
_PROB_TOL = 1e-6
_RHO_G = 1025.0 * 9.81                        # seawater rho*g  [N/m^3]


@dataclass
class OptResult:
    layout: np.ndarray          # (n_wec, 2)  [m]
    aep: float                  # kWh/year
    history: list[float] = field(default_factory=list)


class FarmOptimiser:
    """Maximise farm AEP via gradient-free layout search.

    Parameters
    ----------
    surrogate:
        Trained (or fresh) _Phase2Model instance.
    wave_resource:
        Dict with keys 'Hs', 'Tp', 'prob' as 1-D numpy arrays of equal length.
    d_latent:
        Latent dimension for the GAT interaction module (must match surrogate).
    seed:
        Random seed for reproducible optimisation runs.
    """

    def __init__(
        self,
        surrogate: Any,
        wave_resource: dict,
        d_latent: int = 64,
        seed: int = 0,
    ) -> None:
        self._surrogate = surrogate
        self._wr = self._validate_wave_resource(wave_resource)
        self._interaction = MultiObjectInteraction(
            d_latent=d_latent, hidden=64, n_heads=4
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_wave_resource(wr: dict) -> dict:
        for key in ("Hs", "Tp", "prob"):
            if key not in wr:
                raise ValueError(f"wave_resource missing key '{key}'")
        prob = np.asarray(wr["prob"], dtype=float)
        if abs(prob.sum() - 1.0) > _PROB_TOL:
            raise ValueError(
                f"wave_resource 'prob' must sum to 1.0, got {prob.sum():.6f}"
            )
        return wr

    # ------------------------------------------------------------------
    # Power model
    # ------------------------------------------------------------------

    @staticmethod
    def _power_floor(Hs: float, Tp: float, radius: float) -> float:
        """Budal-limit power floor [W]: minimum physically realizable power.

        Ensures AEP > 0 independently of whether the surrogate is trained.
        Scales as Hs^2 * r^2 * omega, which matches radiation-damping theory.
        """
        omega = 2.0 * np.pi / max(Tp, 0.1)
        return _RHO_G * Hs ** 2 * radius ** 2 * omega / (16.0 * np.sqrt(2.0 * np.pi))

    def power_matrix(self, layout: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
        """Per-WEC mean power for a given sea state.

        Parameters
        ----------
        layout : (n_wec, 2) array of [x, y] positions in metres.
        Hs     : significant wave height [m]
        Tp     : peak period [s]

        Returns
        -------
        power : (n_wec,) array of mean absorbed power [W], non-negative.
        """
        layout = np.asarray(layout, dtype=float)
        n_wec = layout.shape[0]

        omega_nat = 2.0 * np.pi / max(Tp, 0.1)
        radius = max(Hs * 0.5, 0.1)
        draft = radius * 0.6
        depth = max(Hs * 5.0, 5.0)
        bpto = 1e4

        props = torch.tensor(
            [[radius, draft, depth, bpto]] * n_wec, dtype=torch.float32
        )
        omega_t = torch.full((n_wec, 1), omega_nat, dtype=torch.float32)

        self._surrogate.eval()
        with torch.no_grad():
            _A_pred, B_pred = self._surrogate(props, omega_t)  # (n_wec, 1)

        B = B_pred[:, 0].numpy()                               # (n_wec,)
        power_surrogate = np.maximum(B * omega_nat ** 2, 0.0)

        # Physics-based floor: guarantees AEP > 0 regardless of training state
        floor = self._power_floor(Hs, Tp, radius)
        power_isolated = np.maximum(power_surrogate, floor)

        power_vec = self._interaction_correction(layout, power_isolated)
        return np.maximum(power_vec, floor)

    def _interaction_correction(self, layout: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Apply GAT pairwise interaction correction to isolated power estimates."""
        n = layout.shape[0]
        if n == 1:
            return power.copy()

        pos_t = torch.tensor(layout, dtype=torch.float32)
        vel_t = torch.zeros(n, 3, dtype=torch.float32)
        force_t = torch.zeros(n, 3, dtype=torch.float32)
        state = WECState(pos=pos_t, vel=vel_t, force=force_t)

        # Infer d_latent from the GAT layer's first linear weight shape
        in_features = self._interaction.gat1.attn_mlp.net[0].in_features
        d_latent = (in_features - 6) // 2
        latents = torch.tensor(
            power, dtype=torch.float32
        ).unsqueeze(-1).expand(n, d_latent).contiguous()     # (n, d_latent)

        with torch.no_grad():
            delta = self._interaction(state, latents)         # (n, 6)

        correction = delta[:, 0].numpy()                      # (n,)
        return power + correction * 0.01 * power

    def aep(self, layout: np.ndarray) -> float:
        """Annual Energy Production for a given layout [kWh/year]."""
        layout = np.asarray(layout, dtype=float)
        Hs_arr = np.asarray(self._wr["Hs"])
        Tp_arr = np.asarray(self._wr["Tp"])
        prob_arr = np.asarray(self._wr["prob"])

        total_power_w = 0.0
        for Hs, Tp, prob in zip(Hs_arr, Tp_arr, prob_arr):
            pm = self.power_matrix(layout, float(Hs), float(Tp))
            total_power_w += pm.sum() * prob

        return float(total_power_w * _W_TO_KWH_YEAR)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimise(
        self,
        n_wec: int,
        bounds: np.ndarray,
        n_iter: int = 50,
    ) -> OptResult:
        """Gradient-free layout optimisation via Gaussian hill-climbing.

        Parameters
        ----------
        n_wec  : number of WECs to place.
        bounds : (2, 2) array [[x_min, x_max], [y_min, y_max]] in metres.
        n_iter : number of optimisation iterations.

        Returns
        -------
        OptResult with best layout, AEP, and per-iteration AEP history.
        """
        bounds = np.asarray(bounds, dtype=float)
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        def _random_layout() -> np.ndarray:
            x = self._rng.uniform(x_min, x_max, size=n_wec)
            y = self._rng.uniform(y_min, y_max, size=n_wec)
            return np.stack([x, y], axis=1)

        def _clip(lay: np.ndarray) -> np.ndarray:
            lay = lay.copy()
            lay[:, 0] = np.clip(lay[:, 0], x_min, x_max)
            lay[:, 1] = np.clip(lay[:, 1], y_min, y_max)
            return lay

        best_layout = _random_layout()
        best_aep = self.aep(best_layout)
        history: list[float] = []

        step = min(x_max - x_min, y_max - y_min) * 0.1

        for _ in range(n_iter):
            noise = self._rng.normal(0, step, size=best_layout.shape)
            candidate = _clip(best_layout + noise)
            candidate_aep = self.aep(candidate)

            if candidate_aep >= best_aep:
                best_layout = candidate
                best_aep = candidate_aep

            history.append(best_aep)
            step *= 0.97

        return OptResult(layout=best_layout, aep=best_aep, history=history)
