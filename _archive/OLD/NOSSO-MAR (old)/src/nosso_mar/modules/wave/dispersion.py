"""
Linear wave dispersion relations and derived quantities.
All functions are torch-compatible for use inside neural operator training loops.
"""
import torch
import numpy as np


def linear_dispersion(omega: torch.Tensor, h: torch.Tensor,
                       n_iter: int = 30) -> torch.Tensor:
    """
    Solve the linear dispersion relation iteratively:
        ω² = g k tanh(kh)

    Newton-Raphson iteration.

    omega : (...,)  angular frequency [rad/s]
    h     : (...,)  water depth [m]
    Returns: k (...,) wave number [rad/m]
    """
    g   = 9.81
    # Deep-water initial guess
    k   = omega ** 2 / g
    for _ in range(n_iter):
        tanh_kh = torch.tanh(k * h)
        f       = omega ** 2 - g * k * tanh_kh
        df      = -(g * tanh_kh + g * k * h * (1.0 - tanh_kh ** 2))
        k       = k - f / df
    return k.clamp(min=1e-6)


def phase_velocity(omega: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """c = omega / k"""
    return omega / k.clamp(min=1e-9)


def group_velocity(omega: torch.Tensor, k: torch.Tensor,
                   h: torch.Tensor) -> torch.Tensor:
    """
    c_g = c/2 * [1 + 2kh / sinh(2kh)]
    """
    c      = phase_velocity(omega, k)
    kh     = k * h
    sinh2  = torch.sinh(2 * kh).clamp(min=1e-9)
    n      = 0.5 * (1.0 + 2 * kh / sinh2)
    return n * c


def wavelength(k: torch.Tensor) -> torch.Tensor:
    return 2 * np.pi / k.clamp(min=1e-9)
