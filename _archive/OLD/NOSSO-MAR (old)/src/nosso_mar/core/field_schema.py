"""Standardised field containers (xarray-friendly)."""
import numpy as np
import torch

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False


def create_grid(nx=128, ny=128, dx=1.0, dy=1.0, origin=(0.0, 0.0)):
    x = np.arange(origin[0], origin[0] + nx * dx, dx)
    y = np.arange(origin[1], origin[1] + ny * dy, dy)
    return x, y


def bathymetry_to_xarray(bathy_array, x, y, name="bathymetry"):
    if not HAS_XR:
        return bathy_array
    return xr.DataArray(bathy_array, coords=[("y", y), ("x", x)], name=name)


def wave_field_to_xarray(eta_array, x, y, omega, name="eta"):
    if not HAS_XR:
        return eta_array
    return xr.DataArray(eta_array,
                        coords=[("y", y), ("x", x), ("omega", omega)],
                        name=name)
