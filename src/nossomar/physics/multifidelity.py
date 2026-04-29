"""Helpers for moving between CFD, phase-resolved and spectral descriptions."""

from __future__ import annotations

from typing import Any

import numpy as np


def spectral_moments(
    frequency_hz: np.ndarray,
    spectral_density: np.ndarray,
    orders: tuple[int, ...] = (-1, 0, 1, 2),
) -> dict[int, float]:
    """Compute spectral moments m_n = integral f^n S(f) df."""

    freq = np.asarray(frequency_hz, dtype=float)
    spec = np.asarray(spectral_density, dtype=float)
    if freq.ndim != 1 or spec.ndim != 1 or freq.shape != spec.shape:
        raise ValueError("frequency_hz and spectral_density must be 1D arrays of equal length.")
    positive_freq = np.maximum(freq, 1e-12)
    return {
        order: float(np.trapezoid((positive_freq**order) * spec, positive_freq))
        for order in orders
    }


def bulk_wave_statistics(frequency_hz: np.ndarray, spectral_density: np.ndarray) -> dict[str, float]:
    """Return common bulk statistics derived from a 1D spectrum."""

    moments = spectral_moments(frequency_hz, spectral_density)
    m0 = max(moments[0], 1e-12)
    m1 = max(moments[1], 1e-12)
    m2 = max(moments[2], 1e-12)
    m_1 = max(moments[-1], 1e-12)
    return {
        "m0": moments[0],
        "m1": moments[1],
        "m2": moments[2],
        "m_1": moments[-1],
        "Hs": 4.0 * np.sqrt(m0),
        "Tm01": m0 / m1,
        "Tm02": np.sqrt(m0 / m2),
        "Te": m_1 / m0,
    }


def phase_series_to_spectrum(
    surface_elevation: np.ndarray,
    dt: float,
    *,
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a phase-resolved eta(t) series into a one-sided variance spectrum."""

    eta = np.asarray(surface_elevation, dtype=float)
    if eta.ndim != 1:
        raise ValueError("surface_elevation must be a 1D array.")
    series = eta - np.mean(eta) if detrend else eta.copy()
    freq = np.fft.rfftfreq(series.size, d=dt)
    fft = np.fft.rfft(series)
    spectral_density = (dt / series.size) * np.abs(fft) ** 2
    if spectral_density.size > 2:
        spectral_density[1:-1] *= 2.0
    return freq, spectral_density


def reconstruct_irregular_wave(
    frequency_hz: np.ndarray,
    spectral_density: np.ndarray,
    *,
    dt: float,
    duration: float | None = None,
    phases: np.ndarray | None = None,
    seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Synthesize eta(t) from a target spectrum using random phases."""

    freq = np.asarray(frequency_hz, dtype=float)
    spec = np.asarray(spectral_density, dtype=float)
    if freq.ndim != 1 or spec.ndim != 1 or freq.shape != spec.shape:
        raise ValueError("frequency_hz and spectral_density must be 1D arrays of equal length.")

    if duration is None:
        positive = freq[freq > 0]
        df = float(np.min(np.diff(positive))) if positive.size > 1 else 0.05
        duration = max(1.0 / df, 128.0 * dt)

    n_steps = max(8, int(round(duration / dt)))
    time = np.arange(n_steps, dtype=float) * dt
    df = float(np.mean(np.diff(freq))) if freq.size > 1 else 0.0
    rng = np.random.default_rng(seed)
    phase_array = phases if phases is not None else rng.uniform(0.0, 2.0 * np.pi, size=freq.size)
    amplitudes = np.sqrt(np.maximum(2.0 * spec * max(df, 1e-12), 0.0))

    eta = np.zeros_like(time)
    for amplitude, f_i, phi in zip(amplitudes, freq, phase_array, strict=True):
        eta += amplitude * np.cos(2.0 * np.pi * f_i * time + phi)

    return {"time": time, "eta": eta, "phases": np.asarray(phase_array, dtype=float)}


def cfd_snapshot_to_phase_fields(
    velocity_u: np.ndarray,
    velocity_v: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    free_surface: np.ndarray | None = None,
    rho: float = 1025.0,
    gravity: float = 9.81,
) -> dict[str, np.ndarray]:
    """
    Collapse a high-fidelity CFD snapshot into a phase-resolved representation.

    The default assumes the first axis is the vertical dimension and produces
    depth-averaged velocities. If free-surface elevation is unavailable, a
    hydrostatic proxy from surface pressure is used.
    """

    u = np.asarray(velocity_u, dtype=float)
    v = np.asarray(velocity_v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("velocity_u and velocity_v must share the same shape.")

    phase_u = u.mean(axis=0) if u.ndim >= 3 else u
    phase_v = v.mean(axis=0) if v.ndim >= 3 else v

    if free_surface is not None:
        eta = np.asarray(free_surface, dtype=float)
    elif pressure is not None:
        pressure_array = np.asarray(pressure, dtype=float)
        eta = pressure_array[0] / (rho * gravity) if pressure_array.ndim >= 3 else pressure_array / (rho * gravity)
    else:
        eta = np.zeros_like(phase_u)

    return {
        "eta": eta,
        "u": phase_u,
        "v": phase_v,
        "speed": np.sqrt(phase_u**2 + phase_v**2),
    }


def summarize_frequency_response(
    omega_rad_s: np.ndarray,
    added_mass: np.ndarray,
    damping: np.ndarray,
    excitation: np.ndarray,
) -> dict[str, Any]:
    """Summarize hydrodynamic response curves for operator training metadata."""

    omega = np.asarray(omega_rad_s, dtype=float)
    added = np.asarray(added_mass, dtype=float)
    damp = np.asarray(damping, dtype=float)
    force = np.asarray(excitation)
    force_mag = np.abs(force)
    peak_idx = int(np.argmax(force_mag))
    return {
        "omega_peak_excitation": float(omega[peak_idx]),
        "peak_excitation_magnitude": float(force_mag[peak_idx]),
        "mean_added_mass": float(np.mean(added)),
        "mean_damping": float(np.mean(damp)),
        "min_damping": float(np.min(damp)),
        "max_damping": float(np.max(damp)),
    }
