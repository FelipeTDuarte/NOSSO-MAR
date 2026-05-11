"""Torch residuals for physics-informed training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from typing import Any

import torch


def _grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Auto-diff gradient with graph retention for higher-order derivatives."""

    grad_outputs = torch.ones_like(outputs)
    gradient = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(inputs)
    return gradient


def navier_stokes_2d_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    *,
    rho: float = 1025.0,
    nu: float = 1.0e-6,
    fx: float = 0.0,
    fy: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Residuals of the incompressible 2D Navier-Stokes equations."""

    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    v_t = _grad(v, t)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    p_x = _grad(p, x)
    p_y = _grad(p, y)

    lap_u = _grad(u_x, x) + _grad(u_y, y)
    lap_v = _grad(v_x, x) + _grad(v_y, y)
    continuity = u_x + v_y
    momentum_x = u_t + u * u_x + v * u_y + (1.0 / rho) * p_x - nu * lap_u - fx
    momentum_y = v_t + u * v_x + v * v_y + (1.0 / rho) * p_y - nu * lap_v - fy
    return {
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }


def navier_stokes_3d_residual(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
    p: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
    z: torch.Tensor, t: torch.Tensor,
    *, rho: float = 1025.0, nu: float = 1.0e-6,
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, torch.Tensor]:
    """Residuals of the incompressible 3D Navier-Stokes equations."""

    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y); u_z = _grad(u, z)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y); v_z = _grad(v, z)
    w_t = _grad(w, t); w_x = _grad(w, x); w_y = _grad(w, y); w_z = _grad(w, z)
    p_x = _grad(p, x); p_y = _grad(p, y); p_z = _grad(p, z)
    lap_u = _grad(u_x, x) + _grad(u_y, y) + _grad(u_z, z)
    lap_v = _grad(v_x, x) + _grad(v_y, y) + _grad(v_z, z)
    lap_w = _grad(w_x, x) + _grad(w_y, y) + _grad(w_z, z)
    continuity = u_x + v_y + w_z
    fx, fy, fz = body_force
    momentum_x = u_t + u*u_x + v*u_y + w*u_z + (1.0/rho)*p_x - nu*lap_u - fx
    momentum_y = v_t + u*v_x + v*v_y + w*v_z + (1.0/rho)*p_y - nu*lap_v - fy
    momentum_z = w_t + u*w_x + v*w_y + w*w_z + (1.0/rho)*p_z - nu*lap_w - fz
    return {"continuity": continuity, "momentum_x": momentum_x,
            "momentum_y": momentum_y, "momentum_z": momentum_z}


def shallow_water_residual(
    eta: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    depth: torch.Tensor, *, gravity: float = 9.81,
) -> dict[str, torch.Tensor]:
    """Residuals of the depth-averaged shallow-water equations."""

    eta_t = _grad(eta, t); eta_x = _grad(eta, x); eta_y = _grad(eta, y)
    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    hu_x = _grad((depth + eta) * u, x)
    hv_y = _grad((depth + eta) * v, y)
    continuity = eta_t + hu_x + hv_y
    momentum_x = u_t + u * u_x + v * u_y + gravity * eta_x
    momentum_y = v_t + u * v_x + v * v_y + gravity * eta_y
    return {"continuity": continuity, "momentum_x": momentum_x, "momentum_y": momentum_y}


def wave_action_balance_residual(
    action_density: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    *, c_gx: torch.Tensor, c_gy: torch.Tensor, source_term: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual of a reduced wave-action balance advection equation."""

    source = torch.zeros_like(action_density) if source_term is None else source_term
    action_t = _grad(action_density, t)
    flux_x = _grad(c_gx * action_density, x)
    flux_y = _grad(c_gy * action_density, y)
    return action_t + flux_x + flux_y - source


def exner_residual(
    bed_elevation: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    *, sediment_flux_x: torch.Tensor, sediment_flux_y: torch.Tensor, porosity: float = 0.4,
) -> torch.Tensor:
    """Residual of the Exner morphodynamic equation."""

    zb_t = _grad(bed_elevation, t)
    qs_x = _grad(sediment_flux_x, x)
    qs_y = _grad(sediment_flux_y, y)
    return (1.0 - porosity) * zb_t + qs_x + qs_y


def wec_frequency_domain_residual(
    omega_rad_s: torch.Tensor,
    *, mass: torch.Tensor | float, added_mass: torch.Tensor,
    damping: torch.Tensor, stiffness: torch.Tensor | float,
    displacement: torch.Tensor, excitation: torch.Tensor,
    pto_damping: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Frequency-domain WEC equation-of-motion residual."""

    omega = omega_rad_s.to(displacement.dtype)
    total_mass = mass + added_mass
    total_damping = damping + pto_damping
    dynamic_stiffness = -omega**2 * total_mass + 1j * omega * total_damping + stiffness
    return dynamic_stiffness * displacement - excitation


def residual_mse(residual: Any) -> torch.Tensor:
    """Mean squared magnitude of one residual or a dictionary of residuals."""

    if isinstance(residual, dict):
        pieces = [residual_mse(value) for value in residual.values()]
        return torch.stack(pieces).mean()
    tensor = residual if torch.is_complex(residual) else residual.to(torch.float64)
    squared = torch.abs(tensor) ** 2
    return squared.mean()


# ---------------------------------------------------------------------------
# M1 residual helpers exposing metadata contract for physics-informed training.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualMetadata:
    residual_id: str
    model_class: str
    required_observables: tuple[str, ...]
    required_parameters: tuple[str, ...]
    output_layout: str

class ResidualInputError(ValueError):
    pass

M1_EOM_METADATA = ResidualMetadata(
    residual_id='M1_EOM',
    model_class='M1',
    required_observables=('xi', 'added_mass', 'radiation_damping', 'excitation_force'),
    required_parameters=('omega', 'M', 'C_pto', 'K_h', 'K_pto'),
    output_layout='batch x frequency x dof',
)


def validate_required_keys(bundle: Mapping[str, Any], required: Sequence[str], label: str) -> None:
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ResidualInputError(f'{label} missing keys: {missing}')


def compute_m1_eom_residual(observables: Mapping[str, Any], parameters: Mapping[str, Any]) -> Any:
    validate_required_keys(observables, M1_EOM_METADATA.required_observables, 'observables')
    validate_required_keys(parameters, M1_EOM_METADATA.required_parameters, 'parameters')
    xi = observables['xi']
    A = observables['added_mass']
    B = observables['radiation_damping']
    F_exc = observables['excitation_force']
    omega = parameters['omega']
    M = parameters['M']
    C_pto = parameters['C_pto']
    K_h = parameters['K_h']
    K_pto = parameters['K_pto']
    return (-omega**2 * (M + A) + 1j * omega * (B + C_pto) + (K_h + K_pto)) * xi - F_exc


def residual_energy_norm(residual: Any) -> Any:
    return (residual.real ** 2 + residual.imag ** 2).mean()


def reduce_m1_residual(observables: Mapping[str, Any], parameters: Mapping[str, Any]) -> Any:
    residual = compute_m1_eom_residual(observables, parameters)
    return residual_energy_norm(residual)


# Reserved extension points for future phases.
def compute_m2_spectral_residual(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError('M2 spectral residual remains planned and should follow specs 11 and 12.')


def compute_m3_field_residual(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError('M3 phase-resolved residual remains planned and should follow specs 11 and 12.')

