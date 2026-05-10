"""
Anomaly detection for WEC structural health and extreme wave events.

Methods:
    - Statistical control charts (CUSUM, EWMA) for sensor drift
    - Autoencoder reconstruction error for structural anomalies
    - Threshold-based extreme event detection
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np


class WECAnomalyAutoencoder(nn.Module):
    """
    Variational Autoencoder trained on nominal WEC operational data.
    High reconstruction error -> anomaly (structural damage, mooring failure).
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),    nn.GELU(),
            nn.Linear(hidden, latent_dim * 2),  # mean + log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),     nn.GELU(),
            nn.Linear(hidden, input_dim),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return ((x - recon) ** 2).mean(dim=-1)


class ExtremeWaveDetector:
    """Detect rogue/freak waves (η > threshold × Hs)."""

    def __init__(self, threshold_factor: float = 2.0):
        self.threshold_factor = threshold_factor

    def detect(self, eta: torch.Tensor, Hs: float) -> torch.Tensor:
        """Returns mask of extreme wave locations."""
        return eta.abs() > self.threshold_factor * Hs
