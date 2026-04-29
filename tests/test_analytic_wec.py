from __future__ import annotations

import numpy as np

from nossomar.data.analytic_wec import (
    airy_wave_diagnostics,
    build_analytic_wec_state,
    dispersion_wavenumber,
    make_frequency_grid,
)


def test_dispersion_relation_residual_is_small() -> None:
    freq = np.array([0.2, 0.6, 1.2])
    depth = 50.0
    k = dispersion_wavenumber(freq, depth)
    omega = 2.0 * np.pi * freq
    residual = 9.81 * k * np.tanh(k * depth) - omega**2
    assert float(np.max(np.abs(residual))) < 1.0e-8


def test_analytic_wec_state_respects_basic_physics() -> None:
    freq = make_frequency_grid(count=12)
    state = build_analytic_wec_state(radius=5.5, draft=4.0, depth=70.0, freq=freq, bpto=10_000.0)
    diag = airy_wave_diagnostics(freq, depth=70.0)

    assert np.all(state.added_mass > 0.0)
    assert np.all(state.damping >= 0.0)
    assert state.excitation_real.shape == freq.shape
    assert state.excitation_imag.shape == freq.shape
    assert np.all(diag["phase_velocity"] > 0.0)
