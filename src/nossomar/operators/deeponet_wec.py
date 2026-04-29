"""Factorized DeepONet-style regressor for the local Phase 1 baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nossomar.core.contracts import WECState
from nossomar.data.analytic_wec import (
    BRANCH_DIM,
    TRUNK_DIM,
    branch_feature_vector_from_params,
    trunk_feature_matrix,
)


@dataclass(slots=True)
class DeepONetWECRegressor:
    """A lightweight CPU-friendly operator surrogate.

    The model is not a full PyTorch DeepONet yet. It keeps the branch/trunk
    factorization and learns a linear map on their outer product, which makes
    the local workspace runnable without a heavy ML stack.
    """

    ridge: float = 1.0e-8
    weights: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return BRANCH_DIM * TRUNK_DIM

    def _branch_matrix(self, device_matrix: np.ndarray) -> np.ndarray:
        params = np.asarray(device_matrix, dtype=float)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.shape[1] != 5:
            raise ValueError("device_matrix must have shape (n, 5).")
        return np.vstack([branch_feature_vector_from_params(row) for row in params])

    def design_matrix(self, device_matrix: np.ndarray, freq: np.ndarray | list[float]) -> np.ndarray:
        """Build the outer-product design matrix for all rows."""

        params = np.asarray(device_matrix, dtype=float)
        freq_array = np.asarray(freq, dtype=float).reshape(-1)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.shape[0] not in (1, len(freq_array)):
            raise ValueError("device_matrix rows must match freq length, or contain a single device row.")
        if params.shape[0] == 1 and len(freq_array) > 1:
            params = np.repeat(params, len(freq_array), axis=0)
        branch = self._branch_matrix(params)
        trunk = trunk_feature_matrix(freq_array)
        return (branch[:, :, None] * trunk[:, None, :]).reshape(len(freq_array), self.feature_dim)

    def fit(
        self,
        device_matrix: np.ndarray,
        freq: np.ndarray | list[float],
        targets: np.ndarray,
    ) -> "DeepONetWECRegressor":
        """Fit the factorized regressor using ridge regression."""

        x = self.design_matrix(device_matrix, freq)
        y = np.asarray(targets, dtype=float)
        if y.ndim != 2 or y.shape[1] != 4:
            raise ValueError("targets must have shape (n, 4).")
        if y.shape[0] != x.shape[0]:
            raise ValueError("targets must have the same number of rows as the design matrix.")
        if self.ridge <= 1.0e-12:
            self.weights = np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            identity = np.eye(x.shape[1], dtype=float)
            gram = x.T @ x + self.ridge * identity
            self.weights = np.linalg.solve(gram, x.T @ y)
        return self

    def predict(self, device_matrix: np.ndarray, freq: np.ndarray | list[float]) -> np.ndarray:
        """Predict [A, B, Fex_real, Fex_imag] rows."""

        if self.weights is None:
            raise RuntimeError("Model has not been fit yet.")
        x = self.design_matrix(device_matrix, freq)
        return x @ self.weights

    def predict_state(
        self,
        device_params: np.ndarray | list[float],
        freq: np.ndarray | list[float],
        device_type: str = "cylinder",
        metadata: dict[str, Any] | None = None,
    ) -> WECState:
        """Predict a full WECState for one device over a frequency grid."""

        params = np.asarray(device_params, dtype=float).reshape(5)
        freq_array = np.asarray(freq, dtype=float).reshape(-1)
        predictions = self.predict(params.reshape(1, 5), freq_array)
        return WECState(
            freq=freq_array,
            added_mass=predictions[:, 0],
            damping=np.maximum(predictions[:, 1], 0.0),
            excitation_real=predictions[:, 2],
            excitation_imag=predictions[:, 3],
            device_type=device_type,
            radius=params[0],
            draft=params[1],
            mass=params[2],
            bpto=params[3],
            depth=params[4],
            metadata=metadata or {},
        )

    def rmse_by_channel(self, targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        """Compute per-channel RMSE metrics."""

        errors = np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2, axis=0))
        return {
            "A": float(errors[0]),
            "B": float(errors[1]),
            "Fex_real": float(errors[2]),
            "Fex_imag": float(errors[3]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ridge": self.ridge,
            "weights": None if self.weights is None else self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeepONetWECRegressor":
        model = cls(ridge=float(payload.get("ridge", 1.0e-8)))
        if payload.get("weights") is not None:
            model.weights = np.asarray(payload["weights"], dtype=float)
        return model

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DeepONetWECRegressor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
