"""Physics helpers for multi-fidelity NOSSO-MAR models."""

from .multifidelity import (
    bulk_wave_statistics,
    cfd_snapshot_to_phase_fields,
    phase_series_to_spectrum,
    reconstruct_irregular_wave,
    spectral_moments,
    summarize_frequency_response,
)
from .residuals_torch import (
    exner_residual,
    navier_stokes_2d_residual,
    navier_stokes_3d_residual,
    residual_mse,
    shallow_water_residual,
    wave_action_balance_residual,
    wec_frequency_domain_residual,
)

__all__ = [
    "bulk_wave_statistics",
    "cfd_snapshot_to_phase_fields",
    "exner_residual",
    "navier_stokes_2d_residual",
    "navier_stokes_3d_residual",
    "phase_series_to_spectrum",
    "reconstruct_irregular_wave",
    "residual_mse",
    "shallow_water_residual",
    "spectral_moments",
    "summarize_frequency_response",
    "wave_action_balance_residual",
    "wec_frequency_domain_residual",
]
