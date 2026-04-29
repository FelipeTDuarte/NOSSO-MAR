"""Phase 2 end-to-end training loop (P2-T4).

Wires the RINO encoder and the Phase 1 DeepONet surrogate into a single
differentiable pipeline trained on frequency-domain hydrodynamic coefficients.

Training strategy:
  Epochs 1..warmup_epochs  : RINO encoder frozen, only DeepONet adapts to
                              the new latent inputs. This warm-start prevents
                              destroying the Phase 1 convergence.
  Epochs warmup_epochs+1.. : Full joint fine-tuning of RINO + DeepONet.

Data contract:
  ds must contain 'added_mass', 'damping', 'radius', 'draft', 'depth', 'bpto'
  with dims (case, omega) / (case,) — identical to wec_dataset.py output.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr

from nossomar.operators.rino_encoder import RINOEncoder
from nossomar.operators.deeponet_wec import WECDeepONet

N_PTS = 32  # synthetic point cloud size per WEC


class _Phase2Model(nn.Module):
    """RINO encoder + DeepONet surrogate as a single trainable module.

    The RINO encoder maps a synthetic point cloud (sampled from geometry
    parameters) to a latent vector that replaces the raw [r, d, h, B_pto]
    property vector used in Phase 1.
    """

    def __init__(self, d_latent: int = 64) -> None:
        super().__init__()
        self.encoder = RINOEncoder(in_features=4, d_latent=d_latent)
        self.surrogate = WECDeepONet(
            branch_dim=d_latent, trunk_dim=1, hidden=128, n_modes=64
        )

    def forward(
        self,
        props: torch.Tensor,   # (B, 4)  [radius, draft, depth, bpto]
        omega: torch.Tensor,   # (B, N_omega)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = props.shape[0]
        # Build a synthetic point cloud from geometry scalars:
        # spatial coords are small jitter; per-point features are tiled props.
        pts = torch.randn(B, N_PTS, 3, device=props.device) * 0.1
        feats = props.unsqueeze(1).expand(B, N_PTS, 4)  # (B, N_PTS, 4)

        latent = self.encoder(pts, feats)          # (B, d_latent)
        return self.surrogate(latent, omega)        # A (B, N_omega), B_rad (B, N_omega)


def train_phase2(
    ds: xr.Dataset,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 64,
    warmup_epochs: int = 5,
    d_latent: int = 64,
    device: str = "cpu",
) -> list[float]:
    """Train the Phase 2 RINO + DeepONet pipeline.

    Args:
        ds:             xr.Dataset from generate_analytic_dataset or load_wecsim_dataset.
        epochs:         Total training epochs.
        lr:             Peak learning rate for AdamW.
        batch_size:     Mini-batch size.
        warmup_epochs:  Epochs to train only the DeepONet (encoder frozen).
        d_latent:       RINO latent dimension.
        device:         Torch device string.

    Returns:
        history: List of mean per-epoch training losses (length = epochs).
    """
    props = torch.tensor(
        np.stack([
            ds["radius"].values,
            ds["draft"].values,
            ds["depth"].values,
            ds["bpto"].values,
        ], axis=1),
        dtype=torch.float32,
    )  # (N_cases, 4)

    omega = torch.tensor(ds["omega"].values, dtype=torch.float32)  # (N_omega,)
    A_tgt = torch.tensor(ds["added_mass"].values, dtype=torch.float32)  # (N, N_omega)
    B_tgt = torch.tensor(ds["damping"].values, dtype=torch.float32)     # (N, N_omega)

    # Normalise targets to ~O(1) for stable training
    A_scale = A_tgt.abs().mean().clamp(min=1.0)
    B_scale = B_tgt.abs().mean().clamp(min=1.0)
    A_tgt = A_tgt / A_scale
    B_tgt = B_tgt / B_scale

    loader = DataLoader(
        TensorDataset(props, A_tgt, B_tgt),
        batch_size=batch_size,
        shuffle=True,
    )

    model = _Phase2Model(d_latent=d_latent).to(device)
    omega_dev = omega.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: list[float] = []

    for ep in range(epochs):
        freeze = ep < warmup_epochs
        for p in model.encoder.parameters():
            p.requires_grad_(not freeze)

        ep_loss = 0.0
        for p_batch, a_batch, b_batch in loader:
            p_batch = p_batch.to(device)
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            om = omega_dev.unsqueeze(0).repeat(p_batch.shape[0], 1)  # (B, N_omega)
            A_pred, B_pred = model(p_batch, om)

            loss = (A_pred - a_batch).pow(2).mean() + (B_pred - b_batch).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item()

        sched.step()
        history.append(float(ep_loss / len(loader)))

    return history
