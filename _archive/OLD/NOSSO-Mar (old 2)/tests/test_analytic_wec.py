import numpy as np
from nossomar.data.analytic_wec import analytic_added_mass_cylinder, analytic_damping_cylinder


def test_added_mass_deep_water_limit():
    rho, r, draft = 1025.0, 1.0, 2.0
    omega = np.array([5.0, 10.0, 20.0])
    A = analytic_added_mass_cylinder(omega, r, draft, depth=50.0, rho=rho)
    A_inf = rho * np.pi * r**2 * draft
    assert abs(A[-1] - A_inf) / A_inf < 0.20


def test_damping_decays_at_high_frequency():
    omega = np.linspace(0.1, 15.0, 50)
    B = analytic_damping_cylinder(omega, 1.0, 2.0, depth=50.0)
    assert B[-1] < 0.05 * B.max()
