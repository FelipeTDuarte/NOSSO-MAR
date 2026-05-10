import numpy as np
import xarray as xr
from scipy.stats import qmc
from .analytic_wec import analytic_added_mass_cylinder, analytic_damping_cylinder


def generate_analytic_dataset(
    n_cases=1000,
    seed=42,
    omega=None,
):
    if omega is None:
        omega = np.linspace(0.2, 4.0, 40)
    rng = qmc.LatinHypercube(d=4, seed=seed)
    samples = qmc.scale(
        rng.random(n_cases),
        l_bounds=[0.5, 0.5, 10.0, 1e3],
        u_bounds=[5.0, 8.0, 80.0, 1e5],
    )
    radii, drafts, depths, bptos = samples.T
    A = np.stack(
        [analytic_added_mass_cylinder(omega, r, d, h) for r, d, h in zip(radii, drafts, depths)]
    )
    B = np.stack(
        [analytic_damping_cylinder(omega, r, d, h) for r, d, h in zip(radii, drafts, depths)]
    )
    return xr.Dataset(
        {
            "added_mass": (["case", "omega"], A),
            "damping": (["case", "omega"], B),
            "radius": (["case"], radii),
            "draft": (["case"], drafts),
            "depth": (["case"], depths),
            "bpto": (["case"], bptos),
        },
        coords={"omega": omega},
    )


def load_dataset(path):
    return xr.open_zarr(path)
