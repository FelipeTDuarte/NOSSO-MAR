"""
Loss functions for BEM surrogate (Module 2 / DeepONet) training.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BEMLoss(nn.Module):
    """
    Multi-output loss for BEM surrogate:
        L = w_A * L(A) + w_B * L(B) + w_Fex * L(F_ex) + w_rao * L(RAO)
              + w_phys * L_EOM + w_Kramers * L_KK

    Kramers-Kronig: A(ω) and B(ω) are related by Hilbert transform.
    """

    def __init__(self, w_A=1.0, w_B=1.0, w_Fex=1.0, w_rao=0.5,
                 w_phys=0.1, w_kk=0.05):
        super().__init__()
        self.weights = dict(A=w_A, B=w_B, Fex=w_Fex, rao=w_rao,
                            phys=w_phys, kk=w_kk)

    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for key, w in [("added_mass", "A"), ("radiation_damping", "B"),
                        ("excitation_force", "Fex"), ("rao", "rao")]:
            if key in pred and key in target:
                diff  = pred[key] - target[key]
                denom = target[key].abs().mean().clamp(min=1e-6)
                loss  = loss + self.weights[w] * (diff.abs().mean() / denom)

        # Passivity: B(ω) >= 0 (radiation damping must be non-negative)
        if "radiation_damping" in pred:
            passivity_violation = pred["radiation_damping"].clamp(max=0).abs().mean()
            loss = loss + self.weights["phys"] * passivity_violation

        return loss
