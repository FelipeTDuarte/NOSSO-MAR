"""
Zarr-based storage backend for large-scale NOSSO-MAR simulation data.
Supports lazy loading, chunked access, and cloud storage (S3/Azure/GCS).
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np


class NossoMarZarrStore:
    """
    Zarr store for ocean state fields and simulation results.

    Schema:
        /eta     (T, H, W)   free-surface elevation
        /bathy   (H, W)      bathymetry
        /u, /v   (T, H, W)   velocities
        /spectra (T, N_f, N_dir)  wave spectra
        /wec_power (T, N_wec)     absorbed power per WEC

    cfg keys:
        path    : str   (local path or s3://bucket/path)
        mode    : str   (r | r+ | w | a)
        chunks  : dict  (e.g. {"T":100,"H":128,"W":128})
    """

    def __init__(self, cfg: Dict):
        try:
            import zarr
            self._zarr = zarr
        except ImportError:
            raise ImportError("pip install zarr to use NossoMarZarrStore")
        self.cfg  = cfg
        self.root = None

    def open(self):
        path = self.cfg["path"]
        mode = self.cfg.get("mode", "r")
        if path.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem()
            store = self._zarr.open(s3fs.S3Map(path, s3=fs), mode=mode)
        else:
            store = self._zarr.open(path, mode=mode)
        self.root = store
        return self

    def write_eta(self, eta: np.ndarray, t_idx: int):
        """Write a single time slice of η."""
        self.root["eta"][t_idx] = eta

    def read_eta(self, t_start: int, t_end: int) -> np.ndarray:
        return self.root["eta"][t_start:t_end]

    def close(self):
        if hasattr(self.root, "store") and hasattr(self.root.store, "close"):
            self.root.store.close()
