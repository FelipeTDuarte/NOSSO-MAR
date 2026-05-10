"""OpenFOAM CFD adapter (P3-T2).

Loads OpenFOAM postProcessing output and extracts frequency-domain
hydrodynamic coefficients A(omega) and B(omega) via FFT-based system
identification, producing an xr.Dataset compatible with WecSimAdapter.

Expected case directory layout
------------------------------
    case_dir/
        constant/
            waveProperties          # Hs, Tp, waveDirection
        postProcessing/
            forces/0/force.dat      # time-series: t Fx Fy Fz Mx My Mz
            surfaceElevation/0/eta.dat  # time-series: t eta
        system/
            controlDict             # deltaT, endTime
        constant/triSurface/wec.stl # (optional) WEC geometry

System identification method
----------------------------
For a linear WEC in heave, the equation of motion is:

    (m + A(omega)) * z_ddot + B(omega) * z_dot + K * z = F_wave(t)

Given F_wave(t) and eta(t) from CFD, we identify A and B in the frequency
domain by:
    1. FFT both signals on the steady-state window (drop first 20%).
    2. Compute transfer function H(omega) = F(omega) / Eta(omega).
    3. A(omega) = -Re(H) / omega^2    (inertia term)
       B(omega) =  Im(H) / omega      (damping term, clipped >= 0)
    4. Interpolate onto a uniform omega grid, 0.1 to 3.0 rad/s.

Reference: Falnes (2002) Ocean Waves and Oscillating Systems, Ch. 6.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# Uniform output frequency grid [rad/s]
_OMEGA_MIN = 0.1
_OMEGA_MAX = 3.0
_OMEGA_N = 50
_OMEGA_GRID = np.linspace(_OMEGA_MIN, _OMEGA_MAX, _OMEGA_N)

# Fraction of time series to discard as transient
_TRANSIENT_FRAC = 0.20


class OpenFoamAdapter:
    """Read an OpenFOAM case directory and return hydro coefficients.

    Parameters
    ----------
    case_dir : path-like
        Root of the OpenFOAM case (contains constant/, postProcessing/, system/).
    """

    def __init__(self, case_dir: Path | str) -> None:
        self._root = Path(case_dir)

    # ------------------------------------------------------------------
    # Required file paths
    # ------------------------------------------------------------------

    @property
    def _force_dat(self) -> Path:
        return self._root / "postProcessing" / "forces" / "0" / "force.dat"

    @property
    def _eta_dat(self) -> Path:
        return self._root / "postProcessing" / "surfaceElevation" / "0" / "eta.dat"

    @property
    def _wave_props_file(self) -> Path:
        return self._root / "constant" / "waveProperties"

    @property
    def _stl_file(self) -> Path:
        return self._root / "constant" / "triSurface" / "wec.stl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if all required postProcessing files exist."""
        return self._force_dat.exists() and self._eta_dat.exists()

    def wave_props(self) -> dict:
        """Parse waveProperties and return {Hs, Tp, direction_deg}."""
        text = self._wave_props_file.read_text()
        Hs = float(re.search(r"Hs\s+([\d.eE+\-]+)", text).group(1))
        Tp = float(re.search(r"Tp\s+([\d.eE+\-]+)", text).group(1))
        dir_match = re.search(r"waveDirection\s+\(([^)]+)\)", text)
        dx, dy = (float(v) for v in dir_match.group(1).split()[:2])
        direction_deg = float(np.degrees(np.arctan2(dy, dx)))
        return {"Hs": Hs, "Tp": Tp, "direction_deg": direction_deg}

    def geometry(self) -> Optional[dict]:
        """Extract {radius, draft} from STL bounding box, or None if absent."""
        if not self._stl_file.exists():
            return None
        verts = _parse_stl_vertices(self._stl_file)
        if verts.size == 0:
            return None
        x_range = verts[:, 0].max() - verts[:, 0].min()
        z_range = verts[:, 2].max() - verts[:, 2].min()
        return {"radius": float(x_range / 2.0), "draft": float(z_range)}

    def load(self) -> xr.Dataset:
        """Load CFD output and return xr.Dataset with A(omega), B(omega).

        Raises
        ------
        FileNotFoundError  if force.dat or eta.dat is missing.
        ValueError         if time series are too short to identify coefficients.
        """
        if not self._force_dat.exists():
            raise FileNotFoundError(f"force.dat not found: {self._force_dat}")
        if not self._eta_dat.exists():
            raise FileNotFoundError(f"eta.dat not found: {self._eta_dat}")

        t_f, Fz = _read_force_dat(self._force_dat)
        t_e, eta = _read_eta_dat(self._eta_dat)

        # Interpolate eta onto force time grid if needed
        if not np.allclose(t_f, t_e, rtol=1e-4):
            eta = np.interp(t_f, t_e, eta)

        A_grid, B_grid = _identify_coefficients(t_f, Fz, eta, _OMEGA_GRID)

        wp = self.wave_props() if self._wave_props_file.exists() else {}

        ds = xr.Dataset(
            {
                "A": xr.DataArray(A_grid, coords={"omega": _OMEGA_GRID}, dims=["omega"]),
                "B": xr.DataArray(B_grid, coords={"omega": _OMEGA_GRID}, dims=["omega"]),
            },
            attrs={
                "source": "openfoam",
                "Hs": wp.get("Hs", float("nan")),
                "Tp": wp.get("Tp", float("nan")),
            },
        )
        return ds


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _read_force_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse OpenFOAM force.dat: return (t, Fz) arrays."""
    times, fz = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            times.append(float(parts[0]))
            fz.append(float(parts[3]))   # Fz column
        except ValueError:
            continue
    return np.array(times), np.array(fz)


def _read_eta_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse OpenFOAM surfaceElevation eta.dat: return (t, eta) arrays."""
    times, eta = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            times.append(float(parts[0]))
            eta.append(float(parts[1]))
        except ValueError:
            continue
    return np.array(times), np.array(eta)


def _identify_coefficients(
    t: np.ndarray,
    Fz: np.ndarray,
    eta: np.ndarray,
    omega_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT-based system identification of A(omega) and B(omega).

    Drops the first _TRANSIENT_FRAC of the time series as transient,
    then computes the transfer function H = F_fft / Eta_fft and extracts
    A = -Re(H)/omega^2 and B = Im(H)/omega (clipped to >= 0).
    """
    n = len(t)
    n_drop = max(1, int(n * _TRANSIENT_FRAC))
    t_ss = t[n_drop:]
    Fz_ss = Fz[n_drop:]
    eta_ss = eta[n_drop:]

    n_ss = len(t_ss)
    if n_ss < 8:
        raise ValueError(f"Time series too short for system identification (n={n_ss})")

    dt = float(np.mean(np.diff(t_ss)))
    freqs = np.fft.rfftfreq(n_ss, d=dt)          # [Hz]
    omega_fft = 2.0 * np.pi * freqs               # [rad/s]

    F_fft = np.fft.rfft(Fz_ss)
    Eta_fft = np.fft.rfft(eta_ss)

    # Avoid division by near-zero eta spectrum
    eps = np.max(np.abs(Eta_fft)) * 1e-6 + 1e-12
    H = F_fft / (Eta_fft + eps)

    # Exclude DC (omega=0) from interpolation
    mask = omega_fft > 0.0
    omega_fft = omega_fft[mask]
    H = H[mask]

    A_raw = -np.real(H) / np.maximum(omega_fft ** 2, 1e-6)
    B_raw = np.maximum(np.imag(H) / np.maximum(omega_fft, 1e-6), 0.0)

    # Interpolate onto uniform output grid
    A_grid = np.interp(omega_grid, omega_fft, A_raw)
    B_grid = np.maximum(
        np.interp(omega_grid, omega_fft, B_raw), 0.0
    )
    return A_grid, B_grid


def _parse_stl_vertices(path: Path) -> np.ndarray:
    """Extract all vertex coordinates from an ASCII STL file."""
    verts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (IndexError, ValueError):
                continue
    return np.array(verts) if verts else np.empty((0, 3))
