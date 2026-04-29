"""SixDOFDataset: thin xarray wrapper for 6-DOF hydrodynamic coefficient data.

Schema
------
coords:
    omega  (n_omega,)    rad/s, must be strictly ascending
variables:
    A      (n_omega, 6, 6)   added-mass matrix
    B      (n_omega, 6, 6)   radiation-damping matrix
attrs:
    source    str   e.g. 'WAMIT', 'OpenFOAM', 'analytic'
    geometry  str   e.g. 'sphere', 'cylinder', 'box'
"""
from __future__ import annotations

import numpy as np
import xarray as xr


class SixDOFDataset:
    """Wrapper around xr.Dataset for 6-DOF hydrodynamic coefficient data."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._validate(ds)
        self.ds = ds

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        omega: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        attrs: dict | None = None,
    ) -> "SixDOFDataset":
        """Build from numpy arrays.

        Args:
            omega : (n_omega,)       frequency array, rad/s
            A     : (n_omega, 6, 6)  added-mass matrices
            B     : (n_omega, 6, 6)  radiation-damping matrices
            attrs : optional metadata dict
        """
        omega = np.asarray(omega, dtype=float)
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        n_omega = len(omega)
        if A.shape != (n_omega, 6, 6):
            raise ValueError(f"A must have shape ({n_omega}, 6, 6), got {A.shape}")
        if B.shape != (n_omega, 6, 6):
            raise ValueError(f"B must have shape ({n_omega}, 6, 6), got {B.shape}")
        if not np.all(np.diff(omega) > 0):
            raise ValueError("omega must be strictly ascending")

        ds = xr.Dataset(
            {
                "A": (["omega", "dof_i", "dof_j"], A),
                "B": (["omega", "dof_i", "dof_j"], B),
            },
            coords={"omega": omega},
            attrs=attrs or {},
        )
        return cls(ds)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(ds: xr.Dataset) -> None:
        if "omega" not in ds.coords:
            raise ValueError("Dataset must have 'omega' coordinate")
        for var in ("A", "B"):
            if var not in ds:
                raise ValueError(f"Dataset must have variable '{var}'")
            shape = ds[var].shape
            if len(shape) != 3 or shape[1] != 6 or shape[2] != 6:
                raise ValueError(f"Variable '{var}' must have shape (n_omega, 6, 6)")
        omega = ds.coords["omega"].values
        if not np.all(np.diff(omega) > 0):
            raise ValueError("omega must be strictly ascending")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.ds.coords["omega"])
        src = self.ds.attrs.get("source", "unknown")
        geo = self.ds.attrs.get("geometry", "unknown")
        return f"SixDOFDataset(n_omega={n}, source='{src}', geometry='{geo}')"
