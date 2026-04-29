"""
Wave boundary condition generators for Module 1 training data and inference.
"""
from __future__ import annotations
import numpy as np
import torch


class JONSWAPSpectrum:
    """
    JONSWAP directional wave spectrum E(ω, θ).

    E(ω) = α g² ω⁻⁵ exp[-5/4 (ωp/ω)⁴] γ^r
    where r = exp[-(ω-ωp)² / (2σ²ωp²)]
    """

    def __init__(self, Hs: float, Tp: float, gamma: float = 3.3,
                 direction: float = 0.0, spread: float = 30.0):
        self.Hs        = Hs
        self.Tp        = Tp
        self.gamma     = gamma
        self.direction = direction        # principal direction [deg]
        self.spread    = spread           # directional spread [deg]

    def spectrum_1d(self, omega: np.ndarray) -> np.ndarray:
        g   = 9.81
        wp  = 2 * np.pi / self.Tp
        sigma = np.where(omega <= wp, 0.07, 0.09)
        r   = np.exp(-(omega - wp) ** 2 / (2 * sigma ** 2 * wp ** 2))
        alpha = 0.0624 / (0.185 + 0.11 / (0.0185 * self.Hs ** 2))  # approx
        S = (alpha * g ** 2 * omega ** (-5) *
             np.exp(-5 / 4 * (wp / omega) ** 4) *
             self.gamma ** r)
        return S

    def to_tensor(self, omega: np.ndarray) -> torch.Tensor:
        S = self.spectrum_1d(omega)
        return torch.from_numpy(S.astype(np.float32))


class WaveBoundaryCondition:
    """Container for wave boundary conditions."""

    def __init__(self, spectrum: JONSWAPSpectrum, omega: np.ndarray):
        self.spectrum = spectrum
        self.omega    = omega
        self.S        = spectrum.spectrum_1d(omega)

    def as_dict(self) -> dict:
        return {
            "Hs": self.spectrum.Hs,
            "Tp": self.spectrum.Tp,
            "direction": self.spectrum.direction,
            "omega": self.omega,
            "S": self.S,
        }
