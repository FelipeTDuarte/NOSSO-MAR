"""
BEM Surrogate — DeepONet-based surrogate for Capytaine/HAMS/WAMIT output.

Maps device geometry + mechanical properties to hydrodynamic coefficients
as continuous functions of frequency:
    A(ω), B(ω), F_ex(ω), RAO(ω)

Generalises to any point absorber within training distribution.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import numpy as np

from ...operators.deeponet.deeponet import DeepONet
from ...operators.deeponet.physics_deeponet import PhysicsInformedDeepONet


class BEMSurrogate:
    """
    Wraps a trained DeepONet to provide BEM-like interface.

    device_props keys (per WEC):
        radius, draft, mass, volume, Bpto, cog, depth

    Returns per WEC:
        added_mass, radiation_damping, excitation_force, rao — as (N_omega,) tensors
    """

    BRANCH_KEYS = ["radius", "draft", "mass", "volume", "Bpto", "cog", "depth"]
    BRANCH_DIM  = 7

    def __init__(self, config: Dict, weights_path: str = None):
        op_cfg = {
            "branch_input_dim": self.BRANCH_DIM,
            "trunk_input_dim":  1,            # omega
            "hidden_dim":       256,
            "n_hidden":         5,
            "output_dim":       4,            # A, B, |F_ex|, RAO
            "p":                256,
            "activation":       "tanh",
        }
        op_cfg.update(config.get("deeponet_cfg", {}))

        use_physics = config.get("physics_informed", False)
        if use_physics:
            self.net = PhysicsInformedDeepONet(op_cfg)
        else:
            self.net = DeepONet(op_cfg)

        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            self.net.load_state_dict(state)
        self.net.eval()

    def predict(self, device_props: Dict, omega: np.ndarray) -> Dict:
        """
        device_props: dict with BRANCH_KEYS
        omega:        (N_omega,) array [rad/s]
        Returns dict with added_mass, radiation_damping, excitation_force, rao
        """
        branch = torch.tensor(
            [float(device_props.get(k, 0.0)) for k in self.BRANCH_KEYS],
            dtype=torch.float32).unsqueeze(0)                # (1, 7)

        trunk = torch.from_numpy(omega.astype(np.float32)).unsqueeze(-1)  # (N_omega, 1)

        with torch.no_grad():
            out = self.net(branch, trunk.unsqueeze(0))       # (1, N_omega, 4)
        out = out.squeeze(0)                                  # (N_omega, 4)

        return {
            "added_mass":          out[:, 0],
            "radiation_damping":   out[:, 1].clamp(min=0),   # physical constraint
            "excitation_force":    out[:, 2],
            "rao":                 out[:, 3],
        }
