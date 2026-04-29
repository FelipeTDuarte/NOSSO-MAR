"""Analytical and synthetic hydrodynamic helpers for the local baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from nossomar.core.contracts import WECState

BRANCH_DIM = 10
TRUNK_DIM = 7


def make_frequency_grid(start: float = 0.1, stop: float = 2.0, count: int = 32) -> np.ndarray:
    """Create a monotonic frequency grid in hertz."""

    if count < 2:
        raise ValueError("count must be at least 2.")
    if start <= 0.0 or stop <= start:
        raise ValueError("Expected 0 < start < stop for the frequency grid.")
    return np.linspace(start, stop, count, dtype=float)


def normalize_device_params(
    radius: float,
    draft: float,
    mass: float,
    bpto: float,
    depth: float,
) -> np.ndarray:
    """Normalize device parameters to O(1) scales used by the local baseline."""

    return np.array(
        [
            float(radius) / 12.0,
            float(draft) / 10.0,
            float(mass) / 1.0e6,
            float(bpto) / 1.0e5,
            float(depth) / 100.0,
        ],
        dtype=float,
    )


def branch_feature_vector_from_params(device_params: np.ndarray | list[float]) -> np.ndarray:
    """Branch features for the factorized DeepONet-style baseline."""

    params = np.asarray(device_params, dtype=float)
    if params.shape != (5,):
        raise ValueError("device_params must have shape (5,) ordered as radius, draft, mass, bpto, depth.")
    r, d, m, b, h = normalize_device_params(*params)
    safe_h = max(h, 0.2)
    return np.array(
        [
            1.0,
            r,
            d,
            m,
            b,
            h,
            r * d,
            r / safe_h,
            d / safe_h,
            0.5 * (r + d + m),
        ],
        dtype=float,
    )


def trunk_feature_matrix(freq: np.ndarray | list[float]) -> np.ndarray:
    """Frequency features for the factorized DeepONet-style baseline."""

    freq_array = np.asarray(freq, dtype=float).reshape(-1)
    if np.any(freq_array <= 0.0):
        raise ValueError("freq must be strictly positive.")
    freq_norm = freq_array / 2.0
    return np.column_stack(
        [
            np.ones_like(freq_norm),
            freq_norm,
            1.0 / (1.0 + freq_norm**2),
            np.exp(-freq_norm),
            np.sin(np.pi * freq_norm),
            np.cos(np.pi * freq_norm),
            freq_norm * np.exp(-freq_norm),
        ]
    )


def dispersion_wavenumber(
    freq: np.ndarray | list[float],
    depth: float,
    gravity: float = 9.81,
    max_iter: int = 40,
    tol: float = 1.0e-12,
) -> np.ndarray:
    """Solve the linear dispersion relation for the wavenumber."""

    freq_array = np.asarray(freq, dtype=float).reshape(-1)
    if depth <= 0.0:
        raise ValueError("depth must be positive.")
    omega = 2.0 * np.pi * freq_array
    k = np.maximum(omega**2 / gravity, 1.0e-8)
    for _ in range(max_iter):
        tanh_kh = np.tanh(k * depth)
        residual = gravity * k * tanh_kh - omega**2
        derivative = gravity * tanh_kh + gravity * k * depth * (1.0 - tanh_kh**2)
        step = residual / np.maximum(derivative, 1.0e-12)
        k_next = np.maximum(k - step, 1.0e-10)
        if np.max(np.abs(k_next - k)) < tol:
            return k_next
        k = k_next
    return k


def airy_wave_diagnostics(
    freq: np.ndarray | list[float],
    depth: float,
    gravity: float = 9.81,
) -> dict[str, np.ndarray]:
    """Return basic Airy-wave diagnostics for the supplied frequencies."""

    freq_array = np.asarray(freq, dtype=float).reshape(-1)
    omega = 2.0 * np.pi * freq_array
    k = dispersion_wavenumber(freq_array, depth, gravity=gravity)
    phase_velocity = omega / np.maximum(k, 1.0e-12)
    x = 2.0 * k * depth
    ratio = np.zeros_like(x)
    safe_mask = x <= 40.0
    safe_x = np.maximum(x[safe_mask], 1.0e-12)
    ratio[safe_mask] = safe_x / np.sinh(safe_x)
    n_factor = 0.5 * (1.0 + ratio)
    group_velocity = phase_velocity * n_factor
    return {
        "freq": freq_array,
        "wavenumber": k,
        "phase_velocity": phase_velocity,
        "group_velocity": group_velocity,
        "wavelength": 2.0 * np.pi / np.maximum(k, 1.0e-12),
    }


def _bilinear_response(
    branch_features: np.ndarray,
    trunk_features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate branch^T W trunk for all frequency rows."""

    if branch_features.shape != (BRANCH_DIM,):
        raise ValueError(f"Expected branch feature shape ({BRANCH_DIM},).")
    if trunk_features.ndim != 2 or trunk_features.shape[1] != TRUNK_DIM:
        raise ValueError(f"Expected trunk feature matrix shape (n, {TRUNK_DIM}).")
    trunk_projection = branch_features @ weights
    return trunk_features @ trunk_projection


_A_WEIGHTS = np.array(
    [
        [0.040, 0.000, 0.100, 0.030, 0.000, 0.000, 0.000],
        [0.015, 0.000, 0.080, 0.020, 0.000, 0.000, 0.000],
        [0.010, 0.000, 0.070, 0.015, 0.000, 0.000, 0.000],
        [0.005, 0.000, 0.030, 0.005, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.010, 0.000, 0.000, 0.000, 0.000],
        [0.010, 0.000, 0.020, 0.015, 0.000, 0.000, 0.000],
        [0.010, 0.000, 0.050, 0.010, 0.000, 0.000, 0.000],
        [0.005, 0.000, 0.020, 0.000, 0.000, 0.000, 0.000],
        [0.005, 0.000, 0.020, 0.005, 0.000, 0.000, 0.000],
        [0.010, 0.000, 0.030, 0.010, 0.000, 0.000, 0.000],
    ],
    dtype=float,
)

_B_WEIGHTS = np.array(
    [
        [0.020, 0.020, 0.020, 0.015, 0.000, 0.000, 0.020],
        [0.010, 0.015, 0.015, 0.010, 0.000, 0.000, 0.015],
        [0.010, 0.010, 0.015, 0.010, 0.000, 0.000, 0.015],
        [0.020, 0.010, 0.010, 0.000, 0.000, 0.000, 0.010],
        [0.030, 0.010, 0.020, 0.000, 0.000, 0.000, 0.020],
        [0.010, 0.005, 0.010, 0.000, 0.000, 0.000, 0.010],
        [0.010, 0.010, 0.010, 0.000, 0.000, 0.000, 0.010],
        [0.005, 0.005, 0.010, 0.000, 0.000, 0.000, 0.005],
        [0.005, 0.005, 0.010, 0.000, 0.000, 0.000, 0.005],
        [0.010, 0.005, 0.010, 0.000, 0.000, 0.000, 0.010],
    ],
    dtype=float,
)

_FEX_REAL_WEIGHTS = np.array(
    [
        [0.060, 0.000, 0.030, 0.020, 0.030, 0.040, 0.020],
        [0.080, 0.000, 0.020, 0.030, 0.030, 0.050, 0.020],
        [0.040, 0.000, 0.020, 0.020, 0.020, 0.030, 0.010],
        [0.010, 0.000, 0.010, 0.010, 0.000, 0.010, 0.000],
        [0.010, 0.000, 0.005, 0.005, 0.000, 0.010, 0.000],
        [0.020, 0.000, 0.010, 0.010, 0.010, 0.020, 0.010],
        [0.050, 0.000, 0.020, 0.020, 0.030, 0.040, 0.010],
        [0.020, 0.000, 0.010, 0.010, 0.010, 0.020, 0.005],
        [0.020, 0.000, 0.010, 0.010, 0.010, 0.015, 0.005],
        [0.030, 0.000, 0.010, 0.015, 0.010, 0.020, 0.005],
    ],
    dtype=float,
)

_FEX_IMAG_WEIGHTS = np.array(
    [
        [0.020, 0.000, 0.010, 0.010, 0.050, 0.000, 0.020],
        [0.030, 0.000, 0.010, 0.010, 0.060, 0.000, 0.020],
        [0.020, 0.000, 0.010, 0.010, 0.040, 0.000, 0.015],
        [0.005, 0.000, 0.005, 0.005, 0.010, 0.000, 0.000],
        [0.005, 0.000, 0.005, 0.000, 0.010, 0.000, 0.000],
        [0.010, 0.000, 0.005, 0.005, 0.020, 0.000, 0.005],
        [0.020, 0.000, 0.010, 0.010, 0.030, 0.000, 0.010],
        [0.010, 0.000, 0.005, 0.005, 0.015, 0.000, 0.005],
        [0.010, 0.000, 0.005, 0.005, 0.010, 0.000, 0.005],
        [0.010, 0.000, 0.005, 0.005, 0.020, 0.000, 0.005],
    ],
    dtype=float,
)


def approximate_cylinder_hydrodynamics(
    radius: float,
    draft: float,
    freq: np.ndarray | list[float],
    depth: float,
    mass: float | None = None,
    bpto: float = 0.0,
    rho: float = 1025.0,
    gravity: float = 9.81,
) -> dict[str, np.ndarray]:
    """Generate smooth, physically bounded frequency-domain responses.

    This is a lightweight closed-form baseline for the local workspace.
    It is designed to stay consistent with the spec contracts and to be easy
    to fit with a factorized operator surrogate.
    """

    freq_array = np.asarray(freq, dtype=float).reshape(-1)
    if radius <= 0.0 or draft <= 0.0 or depth <= 0.0:
        raise ValueError("radius, draft, and depth must be positive.")
    displaced_mass = rho * np.pi * radius**2 * draft
    body_mass = float(mass if mass is not None else 0.9 * displaced_mass)
    params = np.array([radius, draft, body_mass, bpto, depth], dtype=float)
    branch = branch_feature_vector_from_params(params)
    trunk = trunk_feature_matrix(freq_array)

    # Keep the local analytic generator bilinear in the same branch/trunk
    # features used by the surrogate so the baseline is exactly learnable.
    added_mass = 8.0e5 * _bilinear_response(branch, trunk, _A_WEIGHTS)
    damping = 1.2e6 * _bilinear_response(branch, trunk, _B_WEIGHTS)
    excitation_real = 1.5e7 * _bilinear_response(branch, trunk, _FEX_REAL_WEIGHTS)
    excitation_imag = 1.5e7 * _bilinear_response(branch, trunk, _FEX_IMAG_WEIGHTS)

    return {
        "A": np.maximum(added_mass, 1.0e-6),
        "B": np.maximum(damping, 1.0e-6),
        "Fex_real": excitation_real,
        "Fex_imag": excitation_imag,
    }


def build_analytic_wec_state(
    radius: float,
    draft: float,
    freq: np.ndarray | list[float],
    depth: float,
    mass: float | None = None,
    bpto: float = 0.0,
    device_type: str = "cylinder",
    metadata: dict[str, Any] | None = None,
) -> WECState:
    """Construct a validated WECState using the local analytic baseline."""

    responses = approximate_cylinder_hydrodynamics(
        radius=radius,
        draft=draft,
        freq=freq,
        depth=depth,
        mass=mass,
        bpto=bpto,
    )
    freq_array = np.asarray(freq, dtype=float).reshape(-1)
    displaced_mass = 1025.0 * np.pi * radius**2 * draft
    body_mass = float(mass if mass is not None else 0.9 * displaced_mass)
    merged_metadata = {
        "source": "analytic_phase1_baseline",
        "notes": "Closed-form synthetic hydrodynamics aligned with local factorized operator features.",
    }
    if metadata:
        merged_metadata.update(metadata)
    return WECState(
        freq=freq_array,
        added_mass=responses["A"],
        damping=responses["B"],
        excitation_real=responses["Fex_real"],
        excitation_imag=responses["Fex_imag"],
        device_type=device_type,
        radius=radius,
        draft=draft,
        mass=body_mass,
        bpto=bpto,
        depth=depth,
        metadata=merged_metadata,
    )


def analytic_hydrodynamic_response(
    radius: float,
    draft: float,
    freq: np.ndarray | list[float],
    depth: float,
    mass: float | None = None,
    bpto: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compatibility wrapper for the local analytic hydrodynamic generator."""

    return approximate_cylinder_hydrodynamics(
        radius=radius,
        draft=draft,
        freq=freq,
        depth=depth,
        mass=mass,
        bpto=bpto,
    )


def generate_analytic_wec_case(**kwargs: Any) -> WECState:
    """Compatibility wrapper returning a validated analytic WEC case."""

    return build_analytic_wec_state(**kwargs)
