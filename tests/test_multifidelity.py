from __future__ import annotations

import numpy as np
import torch

from nossomar.physics.multifidelity import (
    bulk_wave_statistics,
    cfd_snapshot_to_phase_fields,
    phase_series_to_spectrum,
    reconstruct_irregular_wave,
)
from nossomar.physics.residuals_torch import residual_mse, wec_frequency_domain_residual, compute_m1_eom_residual


def test_phase_series_and_reconstruction_produce_positive_bulk_statistics() -> None:
    freq = np.linspace(0.05, 0.5, 32)
    spectrum = np.exp(-((freq - 0.18) ** 2) / 0.002) * 0.8
    reconstruction = reconstruct_irregular_wave(freq, spectrum, dt=0.5, duration=512.0, seed=7)
    freq_hat, spectrum_hat = phase_series_to_spectrum(reconstruction["eta"], dt=0.5)
    stats = bulk_wave_statistics(freq_hat[1:], spectrum_hat[1:])
    assert stats["Hs"] > 0.0
    assert stats["Tm01"] > 0.0


def test_cfd_snapshot_bridge_depth_averages_velocity() -> None:
    u = np.ones((3, 5, 4))
    v = 2.0 * np.ones((3, 5, 4))
    pressure = 1025.0 * 9.81 * np.ones((3, 5, 4))
    bridged = cfd_snapshot_to_phase_fields(u, v, pressure=pressure)
    assert bridged["u"].shape == (5, 4)
    assert np.allclose(bridged["u"], 1.0)
    assert np.allclose(bridged["v"], 2.0)
    assert np.allclose(bridged["eta"], 1.0)


def test_wec_frequency_domain_residual_mse_is_zero_for_consistent_balance() -> None:
    omega = torch.tensor([1.0, 2.0], dtype=torch.float64)
    displacement = torch.tensor([1.0 + 0.5j, 0.25 - 0.2j], dtype=torch.complex128)
    mass = torch.tensor(5.0, dtype=torch.float64)
    added_mass = torch.tensor([1.0, 1.5], dtype=torch.float64)
    damping = torch.tensor([0.2, 0.3], dtype=torch.float64)
    stiffness = torch.tensor(4.0, dtype=torch.float64)
    pto = torch.tensor(0.1, dtype=torch.float64)
    excitation = (
        (-omega**2 * (mass + added_mass) + 1j * omega * (damping + pto) + stiffness)
        * displacement
    )
    residual = wec_frequency_domain_residual(
        omega,
        mass=mass,
        added_mass=added_mass,
        damping=damping,
        stiffness=stiffness,
        displacement=displacement,
        excitation=excitation,
        pto_damping=pto,
    )
    assert torch.isclose(residual_mse(residual), torch.tensor(0.0, dtype=torch.float64))

def test_compute_m1_eom_residual_returns_complex_array():
    obs = {
        'xi': np.array([1 + 0j]),
        'added_mass': np.array([0.1 + 0j]),
        'radiation_damping': np.array([0.05 + 0j]),
        'excitation_force': np.array([0.2 + 0j]),
    }
    params = {
        'omega': np.array([1.0]),
        'M': np.array([1.0]),
        'C_pto': np.array([0.0]),
        'K_h': np.array([0.0]),
        'K_pto': np.array([0.0]),
    }
    residual = compute_m1_eom_residual(obs, params)
    assert residual.shape == (1,)