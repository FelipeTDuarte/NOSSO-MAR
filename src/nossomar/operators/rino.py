"""Resolution-invariant style coordinate-conditioned operator."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOperator


class RINO2d(BaseOperator):
    """
    Lightweight coordinate-conditioned operator.

    This local implementation is intentionally compact: it encodes a field on a
    regular grid and decodes either the full grid or arbitrary query points.
    That makes it useful for smoke-testing resolution-transfer workflows.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]
        latent_channels = cfg.get("latent_channels", 64)
        n_layers = cfg.get("n_layers", 4)
        decoder_hidden = cfg.get("decoder_hidden", 128)

        blocks: list[nn.Module] = [
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(0, n_layers - 1)):
            blocks.extend(
                [
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.grid_head = nn.Conv2d(latent_channels, out_channels, kernel_size=1)
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_channels + 2, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, out_channels),
        )

    def forward(
        self,
        u: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.encoder(u)
        if query_points is None:
            return self.grid_head(latent)

        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0).expand(u.shape[0], -1, -1)
        if query_points.shape[-1] != 2:
            raise ValueError("RINO2d query_points must have shape (..., 2).")

        batch_size, n_points, _ = query_points.shape
        sample_grid = query_points.view(batch_size, n_points, 1, 2)
        sampled = F.grid_sample(
            latent,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        decoder_input = torch.cat([sampled, query_points], dim=-1)
        return self.point_decoder(decoder_input)
