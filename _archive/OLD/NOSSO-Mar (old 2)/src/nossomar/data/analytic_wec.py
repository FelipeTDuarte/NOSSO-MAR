import numpy as np


def _wavenumber(omega, depth, g=9.81, tol=1e-8, maxiter=50):
    omega = np.atleast_1d(omega)
    k = omega**2 / g
    for _ in range(maxiter):
        k_new = omega**2 / (g * np.tanh(k * depth))
        if np.max(np.abs(k_new - k)) < tol:
            return k_new
        k = k_new
    return k


def analytic_added_mass_cylinder(omega, radius, draft, depth, rho=1025.0):
    k = _wavenumber(omega, depth)
    kr = k * radius
    A_inf = rho * np.pi * radius**2 * draft
    correction = 1.0 - 0.5 * np.exp(-kr * draft)
    return A_inf * correction


def analytic_damping_cylinder(omega, radius, draft, depth, rho=1025.0, g=9.81):
    omega = np.atleast_1d(omega)
    k = _wavenumber(omega, depth)
    # Clamp the full sinh argument to avoid overflow at high frequency / deep water
    sinh_arg = np.clip(2 * k * depth, 0.0, 700.0)
    cg = g / (2 * omega) * (1 + 2 * k * depth / np.sinh(sinh_arg))
    a_rad = (np.pi * radius**2 * rho * omega) ** 2 / (rho * cg * k)
    return a_rad * np.exp(-2 * k * draft)
