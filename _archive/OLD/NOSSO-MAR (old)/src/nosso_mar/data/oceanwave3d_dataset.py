"""
Dataset wrapping OceanWave3D simulation output for Module 1 (WNO/FNO) training.

OceanWave3D output format: NetCDF with variables eta(x,y,t), u, v, bathy.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class OceanWave3DDataset(Dataset):
    """
    Training dataset for Wave Propagation NO.

    Each sample:
        input  : (C_in, H, W)  [bathy, Hs, Tp, dir, rad_mag, diff_mag]
        target : (C_out, H, W) [eta, u_vel, v_vel]
        omega  : (N_omega,)
    """

    def __init__(self, data_dir: str, grid_size: Tuple[int, int] = (128, 128),
                 normalise: bool = True, augment: bool = True):
        self.data_dir   = Path(data_dir)
        self.H, self.W  = grid_size
        self.normalise  = normalise
        self.augment    = augment
        self.files      = sorted(self.data_dir.glob("*.nc"))
        if not self.files:
            raise FileNotFoundError(f"No .nc files in {data_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("pip install xarray to use OceanWave3DDataset")

        ds = xr.open_dataset(self.files[idx])

        def _field(name):
            arr = ds[name].values.astype(np.float32)
            # Interpolate to target grid
            if arr.shape != (self.H, self.W):
                from scipy.ndimage import zoom
                arr = zoom(arr, (self.H/arr.shape[0], self.W/arr.shape[1]))
            return torch.from_numpy(arr)

        bathy  = _field("bathymetry")
        eta    = _field("eta")
        u_vel  = _field("u_velocity") if "u_velocity" in ds else torch.zeros_like(eta)
        v_vel  = _field("v_velocity") if "v_velocity" in ds else torch.zeros_like(eta)

        Hs     = float(ds.attrs.get("Hs",  2.0))
        Tp     = float(ds.attrs.get("Tp",  8.0))
        dirn   = float(ds.attrs.get("direction", 0.0))

        input_fields = torch.stack([
            bathy,
            torch.full_like(bathy, Hs),
            torch.full_like(bathy, Tp),
            torch.full_like(bathy, dirn),
            torch.zeros_like(bathy),   # rad_force (0 for uncoupled training)
            torch.zeros_like(bathy),   # diff_force
        ])
        target = torch.stack([eta, u_vel, v_vel])

        ds.close()
        return {"input": input_fields, "target": target}
