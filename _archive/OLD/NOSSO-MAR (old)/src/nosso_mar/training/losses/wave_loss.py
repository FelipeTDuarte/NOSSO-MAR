"""
Loss functions for wave propagation Neural Operator training.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """Relative L2 loss — scale-invariant, standard for operator learning."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return (diff.norm(dim=(-2,-1)) / target.norm(dim=(-2,-1)).clamp(min=1e-8)).mean()


class WaveLoss(nn.Module):
    """
    Combined loss for wave propagation operator:
        L = w_data * L_data + w_grad * L_grad + w_energy * L_energy

    L_data   : relative L2 on η
    L_grad   : gradient matching (penalises phase errors)
    L_energy : energy conservation (E_pred ≈ E_target at boundaries)
    """

    def __init__(self, w_data: float = 1.0, w_grad: float = 0.1,
                 w_energy: float = 0.05):
        super().__init__()
        self.w_data   = w_data
        self.w_grad   = w_grad
        self.w_energy = w_energy
        self.l2 = RelativeL2Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L_data   = self.l2(pred, target)

        # Gradient loss
        gx_p = pred[..., 1:, :] - pred[..., :-1, :]
        gx_t = target[..., 1:, :] - target[..., :-1, :]
        gy_p = pred[..., :, 1:] - pred[..., :, :-1]
        gy_t = target[..., :, 1:] - target[..., :, :-1]
        L_grad = self.l2(gx_p, gx_t) + self.l2(gy_p, gy_t)

        # Energy conservation: (η² dx dy) should be conserved
        E_pred   = (pred   ** 2).mean(dim=(-2, -1))
        E_target = (target ** 2).mean(dim=(-2, -1))
        L_energy = ((E_pred - E_target) ** 2).mean()

        return self.w_data * L_data + self.w_grad * L_grad + self.w_energy * L_energy
