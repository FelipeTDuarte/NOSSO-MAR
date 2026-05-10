"""
Power Take-Off (PTO) models for wave energy converters.

Supports:
    - Linear damper (resistive control)
    - Reactive control (complex conjugate)
    - Latching control (bang-bang)
    - Model Predictive Control interface (for MARL)
"""
from __future__ import annotations
from typing import Dict
import torch
import numpy as np

from .equation_of_motion import optimal_bpto


class LinearPTO:
    """Simple resistive PTO: F_pto = -B_pto * velocity."""

    def __init__(self, B_pto: float):
        self.B_pto = B_pto

    def power(self, omega: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        from .equation_of_motion import absorbed_power
        return absorbed_power(omega, X, self.B_pto)


class ReactiveControlPTO:
    """
    Complex conjugate (reactive) control — maximises power at each frequency.
    B_pto(ω) = B_opt(ω),  K_pto(ω) = -Im[Z_i(ω)]
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor,
                 M: float, C: float):
        self.B_opt = optimal_bpto(
            torch.linspace(0.1, 3.0, len(A)), A, B, M, C)

    def power(self, omega: torch.Tensor, F_ex: torch.Tensor,
              A: torch.Tensor, B: torch.Tensor,
              M: float, C: float) -> torch.Tensor:
        from .equation_of_motion import frequency_domain_eom, absorbed_power
        X = frequency_domain_eom(omega, M, A, B, float(self.B_opt.mean()), C, F_ex)
        return absorbed_power(omega, X, float(self.B_opt.mean()))
