"""
Frequency-domain equation of motion for heaving point absorbers.

[-ω²(M + A(ω)) + iω(B(ω) + B_pto) + C] X(ω) = F_ex(ω)
P(ω) = ½ B_pto ω² |X(ω)|²
"""
from __future__ import annotations
import torch
import numpy as np


def frequency_domain_eom(
    omega: torch.Tensor,
    M: float,
    A: torch.Tensor,
    B: torch.Tensor,
    B_pto: float,
    C: float,
    F_ex: torch.Tensor,
) -> torch.Tensor:
    """
    Solve EOM in frequency domain.

    omega  : (N_omega,)  angular frequencies [rad/s]
    A      : (N_omega,)  added mass [kg]
    B      : (N_omega,)  radiation damping [N.s/m]
    B_pto  : float       PTO damping [N.s/m]
    C      : float       hydrostatic restoring coefficient [N/m]
    F_ex   : (N_omega,)  complex excitation force [N]
    Returns: X(omega) (N_omega,) complex heave displacement [m]
    """
    w2  = omega ** 2
    dyn = -w2 * (M + A) + 1j * omega * (B + B_pto) + C
    return F_ex / dyn


def absorbed_power(omega: torch.Tensor, X: torch.Tensor,
                   B_pto: float) -> torch.Tensor:
    """
    P(ω) = ½ B_pto ω² |X(ω)|²  [W / (rad/s)]  (per unit bandwidth)
    """
    return 0.5 * B_pto * omega ** 2 * X.abs() ** 2


def total_power(omega: torch.Tensor, P_omega: torch.Tensor) -> torch.Tensor:
    """Integrate P(ω) over frequency using trapezoidal rule."""
    return torch.trapz(P_omega, omega)


def optimal_bpto(omega: torch.Tensor, A: torch.Tensor,
                 B: torch.Tensor, M: float, C: float) -> torch.Tensor:
    """
    Optimal PTO damping at each frequency (maximum power extraction):
        B_pto_opt(ω) = sqrt( B² + [ω(M+A) - C/ω]² )
    """
    reactance = omega * (M + A) - C / omega.clamp(min=1e-6)
    return (B ** 2 + reactance ** 2).sqrt()
