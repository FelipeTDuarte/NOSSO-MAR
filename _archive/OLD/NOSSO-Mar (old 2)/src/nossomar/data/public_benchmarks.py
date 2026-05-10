"""Loaders for public WEC benchmark hydrodynamic datasets.

Each loader returns a dict with keys:
    omega            : np.ndarray (n_omega,)  [rad/s]
    added_mass_heave : np.ndarray (n_omega,)  [kg]
    damping_heave    : np.ndarray (n_omega,)  [kg/s]
    radius           : float  [m]
    draft            : float  [m]
    depth            : float  [m]
    bpto             : float  [N·s/m]  (nominal, set to 0 if unknown)
"""

from pathlib import Path
import numpy as np


def load_csv_benchmark(path: str | Path) -> dict:
    """Load a benchmark CSV with columns: omega, added_mass, damping.

    The CSV must have a header row. Extra metadata columns are ignored.
    Caller is responsible for providing geometry parameters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "omega": data["omega"].astype(float),
        "added_mass_heave": data["added_mass"].astype(float),
        "damping_heave": data["damping"].astype(float),
    }


def synthetic_benchmark(radius=1.0, draft=2.0, depth=50.0, bpto=0.0, n_omega=40):
    """Generate a synthetic reference using the analytic cylinder model.

    Useful for integration tests when no external file is available.
    """
    from .analytic_wec import analytic_added_mass_cylinder, analytic_damping_cylinder

    omega = np.linspace(0.2, 4.0, n_omega)
    return {
        "omega": omega,
        "added_mass_heave": analytic_added_mass_cylinder(omega, radius, draft, depth),
        "damping_heave": analytic_damping_cylinder(omega, radius, draft, depth),
        "radius": radius,
        "draft": draft,
        "depth": depth,
        "bpto": bpto,
    }
