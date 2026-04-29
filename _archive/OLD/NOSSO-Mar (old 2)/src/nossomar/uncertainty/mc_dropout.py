"""MC-Dropout uncertainty quantification (P3-T3).

Wraps any nn.Module with Dropout layers and runs Monte Carlo forward passes
to produce predictive mean, std, and confidence intervals.

Method
------
Gal & Ghahramani (2016) showed that a neural network with Dropout applied
at inference time is equivalent to approximate Bayesian inference. Running
N stochastic forward passes (each with a different Dropout mask) gives a
Monte Carlo estimate of the predictive distribution:

    p(y* | x*, X, Y) ~= (1/T) sum_t p(y* | x*, w_t)

where w_t are the weights sampled by Dropout at pass t.

Use:
    est = MCDropoutEstimator(model, n_samples=100)
    result = est.predict(x)          # UQResult
    ece = est.calibrate(x, y_true)   # Expected Calibration Error

Reference: Gal & Ghahramani (2016) Dropout as a Bayesian Approximation,
           ICML. https://arxiv.org/abs/1506.02142
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class UQResult:
    """Predictive distribution from MC-Dropout.

    Attributes
    ----------
    mean    : (N, d_out)            predictive mean
    std     : (N, d_out)            predictive std dev  (>= 0)
    lower   : (N, d_out)            2.5th percentile  (95% CI lower bound)
    upper   : (N, d_out)            97.5th percentile (95% CI upper bound)
    samples : (n_samples, N, d_out) raw MC samples
    """
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: np.ndarray


class MCDropoutEstimator:
    """MC-Dropout wrapper for any nn.Module with Dropout layers.

    Parameters
    ----------
    model     : nn.Module with at least one nn.Dropout layer.
    n_samples : number of Monte Carlo forward passes (> 0).
    """

    def __init__(self, model: nn.Module, n_samples: int = 100) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        self._model = model
        self._n_samples = n_samples

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> UQResult:
        """Run MC-Dropout inference and return the predictive distribution.

        Keeps the model in train() mode so Dropout stays active, but wraps
        each forward pass in torch.no_grad() to save memory.

        Parameters
        ----------
        x : (N, d_in) input tensor.

        Returns
        -------
        UQResult with mean, std, lower, upper, samples.
        """
        self._model.train()   # activates Dropout
        samples = []
        with torch.no_grad():
            for _ in range(self._n_samples):
                out = self._model(x)   # (N, d_out)
                samples.append(out.cpu().numpy())

        samples_arr = np.stack(samples, axis=0)   # (T, N, d_out)

        mean = samples_arr.mean(axis=0)            # (N, d_out)
        std = samples_arr.std(axis=0)              # (N, d_out)
        lower = np.percentile(samples_arr, 2.5, axis=0)
        upper = np.percentile(samples_arr, 97.5, axis=0)

        return UQResult(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=samples_arr,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Expected Calibration Error (ECE) for regression.

        ECE is estimated by checking coverage of confidence intervals at
        10 quantile levels (10%, 20%, ..., 100%) and measuring the mean
        absolute deviation between expected and observed coverage.

        Parameters
        ----------
        x : (N, d_in)   input tensor.
        y : (N, d_out)  true target tensor.

        Returns
        -------
        ece : float in [0, 1].  0 = perfectly calibrated, 1 = worst case.
        """
        result = self.predict(x)
        y_np = y.cpu().numpy()              # (N, d_out)
        samples = result.samples            # (T, N, d_out)

        confidence_levels = np.linspace(0.1, 1.0, 10)
        errors = []
        for conf in confidence_levels:
            alpha = 1.0 - conf
            lo = np.percentile(samples, 100 * alpha / 2, axis=0)      # (N, d_out)
            hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
            in_interval = (y_np >= lo) & (y_np <= hi)                  # (N, d_out)
            observed_coverage = in_interval.mean()
            errors.append(abs(observed_coverage - conf))

        return float(np.mean(errors))
