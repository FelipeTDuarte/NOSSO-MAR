"""RED tests for Phase 3 OpenFOAM CFD adapter (P3-T2).

Defines the contract for OpenFoamAdapter before any implementation exists.
All tests must FAIL on first run.

OpenFOAM case directory layout expected:

    case_dir/
        constant/
            waveProperties          # wave Hs, Tp, direction
            triSurface/wec.stl      # WEC geometry (optional)
        postProcessing/
            forces/
                0/
                    force.dat       # time-series: t Fx Fy Fz Mx My Mz
            surfaceElevation/
                0/
                    eta.dat         # time-series: t eta
        system/
            controlDict             # deltaT, endTime

Contract: OpenFoamAdapter(case_dir) exposes:
    .load()         -> xr.Dataset   (same schema as WecSimAdapter)
    .wave_props()   -> dict          {Hs, Tp, direction_deg}
    .geometry()     -> dict          {radius, draft}  or None if no STL
    .is_valid()     -> bool          checks required files exist

Dataset schema (must match wec_sim_adapter output):
    coords: omega (rad/s), ascending
    vars:   A(omega), B(omega)   -- added mass and radiation damping
    attrs:  Hs, Tp, source='openfoam'
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nossomar.data.openfoam_adapter import OpenFoamAdapter


# ---------------------------------------------------------------------------
# Helpers: synthetic OpenFOAM case fixtures
# ---------------------------------------------------------------------------

def _write_control_dict(case: Path, delta_t: float = 0.05, end_time: float = 20.0) -> None:
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(
        textwrap.dedent(f"""\
        deltaT          {delta_t};
        endTime         {end_time};
        """)
    )


def _write_wave_properties(case: Path, Hs: float = 2.0, Tp: float = 8.0, direction: float = 0.0) -> None:
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "waveProperties").write_text(
        textwrap.dedent(f"""\
        Hs              {Hs};
        Tp              {Tp};
        waveDirection   ({np.cos(np.radians(direction))} {np.sin(np.radians(direction))} 0);
        """)
    )


def _write_force_dat(case: Path, n_steps: int = 400, delta_t: float = 0.05) -> None:
    """Synthetic sinusoidal force time series."""
    forces_dir = case / "postProcessing" / "forces" / "0"
    forces_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_steps) * delta_t
    Fz = 1e4 * np.sin(2 * np.pi * t / 8.0)   # heave force at Tp=8s
    lines = ["# Time\tFx\tFy\tFz\tMx\tMy\tMz"]
    for i, ti in enumerate(t):
        lines.append(f"{ti:.4f}\t0.0\t0.0\t{Fz[i]:.6f}\t0.0\t0.0\t0.0")
    (forces_dir / "force.dat").write_text("\n".join(lines))


def _write_eta_dat(case: Path, n_steps: int = 400, delta_t: float = 0.05) -> None:
    """Synthetic wave elevation time series."""
    eta_dir = case / "postProcessing" / "surfaceElevation" / "0"
    eta_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_steps) * delta_t
    eta = np.sin(2 * np.pi * t / 8.0)
    lines = ["# Time\teta"]
    for i, ti in enumerate(t):
        lines.append(f"{ti:.4f}\t{eta[i]:.6f}")
    (eta_dir / "eta.dat").write_text("\n".join(lines))


def _make_full_case(tmp_path: Path, Hs: float = 2.0, Tp: float = 8.0) -> Path:
    case = tmp_path / "of_case"
    _write_control_dict(case)
    _write_wave_properties(case, Hs=Hs, Tp=Tp)
    _write_force_dat(case)
    _write_eta_dat(case)
    return case


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_case(tmp_path):
    return _make_full_case(tmp_path)


@pytest.fixture()
def adapter(full_case):
    return OpenFoamAdapter(full_case)


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------

def test_is_valid_full_case(adapter):
    assert adapter.is_valid() is True


def test_is_valid_missing_force_dat(tmp_path):
    case = tmp_path / "bad_case"
    _write_control_dict(case)
    _write_wave_properties(case)
    _write_eta_dat(case)
    # deliberately skip force.dat
    assert OpenFoamAdapter(case).is_valid() is False


def test_is_valid_missing_eta_dat(tmp_path):
    case = tmp_path / "bad_case2"
    _write_control_dict(case)
    _write_wave_properties(case)
    _write_force_dat(case)
    # deliberately skip eta.dat
    assert OpenFoamAdapter(case).is_valid() is False


# ---------------------------------------------------------------------------
# wave_props
# ---------------------------------------------------------------------------

def test_wave_props_keys(adapter):
    wp = adapter.wave_props()
    assert "Hs" in wp
    assert "Tp" in wp
    assert "direction_deg" in wp


def test_wave_props_values(full_case):
    adapter = OpenFoamAdapter(full_case)
    wp = adapter.wave_props()
    assert abs(wp["Hs"] - 2.0) < 1e-6
    assert abs(wp["Tp"] - 8.0) < 1e-6
    assert abs(wp["direction_deg"] - 0.0) < 1e-3


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_geometry_returns_none_without_stl(adapter):
    """No STL in case -> geometry() returns None."""
    assert adapter.geometry() is None


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def test_load_returns_dataset(adapter):
    ds = adapter.load()
    assert isinstance(ds, xr.Dataset)


def test_load_has_omega_coord(adapter):
    ds = adapter.load()
    assert "omega" in ds.coords


def test_load_omega_ascending(adapter):
    ds = adapter.load()
    omega = ds.coords["omega"].values
    assert np.all(np.diff(omega) > 0)


def test_load_has_A_and_B(adapter):
    ds = adapter.load()
    assert "A" in ds
    assert "B" in ds


def test_load_no_nan(adapter):
    ds = adapter.load()
    assert not np.isnan(ds["A"].values).any()
    assert not np.isnan(ds["B"].values).any()


def test_load_B_non_negative(adapter):
    """Radiation damping must be >= 0 (energy dissipation)."""
    ds = adapter.load()
    assert np.all(ds["B"].values >= 0.0)


def test_load_source_attr(adapter):
    ds = adapter.load()
    assert ds.attrs.get("source") == "openfoam"


def test_load_wave_attrs(adapter):
    ds = adapter.load()
    assert "Hs" in ds.attrs
    assert "Tp" in ds.attrs


def test_load_raises_on_invalid_case(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        OpenFoamAdapter(empty).load()


def test_load_different_tp_shifts_peak(tmp_path):
    """Peak frequency of B(omega) should shift with Tp."""
    case_8 = _make_full_case(tmp_path / "tp8", Tp=8.0)
    case_12 = _make_full_case(tmp_path / "tp12", Tp=12.0)
    # Write different force frequencies
    n, dt = 400, 0.05
    t = np.arange(n) * dt
    for Tp, case in [(8.0, case_8), (12.0, case_12)]:
        Fz = 1e4 * np.sin(2 * np.pi * t / Tp)
        lines = ["# Time\tFx\tFy\tFz\tMx\tMy\tMz"]
        for i, ti in enumerate(t):
            lines.append(f"{ti:.4f}\t0.0\t0.0\t{Fz[i]:.6f}\t0.0\t0.0\t0.0")
        (case / "postProcessing" / "forces" / "0" / "force.dat").write_text("\n".join(lines))

    ds_8 = OpenFoamAdapter(case_8).load()
    ds_12 = OpenFoamAdapter(case_12).load()
    omega_peak_8 = ds_8.coords["omega"].values[np.argmax(ds_8["B"].values)]
    omega_peak_12 = ds_12.coords["omega"].values[np.argmax(ds_12["B"].values)]
    # Tp=8 -> omega=pi/4 ~ 0.785; Tp=12 -> omega=pi/6 ~ 0.524; peak_8 > peak_12
    assert omega_peak_8 > omega_peak_12
