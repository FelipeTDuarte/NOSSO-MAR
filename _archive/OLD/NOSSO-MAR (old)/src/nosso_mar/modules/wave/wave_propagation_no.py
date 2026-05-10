"""
Module 1 — Wave Propagation Neural Operator.
Phase-resolving, valid for any water depth (shallow → deep).
Operator backend: WNO (primary) or FNO2d / GeoFNO (configurable).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from ..base_module import BaseModule
from ...operators import WaveletNeuralOperator, FNO2d


OPERATOR_REGISTRY = {
    "wno":    WaveletNeuralOperator,
    "fno2d":  FNO2d,
}


class WavePropagationNO(BaseModule):
    """
    Inputs dict:
        spectrum        : dict — Hs, Tp, direction, omega [rad/s]
        bathymetry      : (H, W) or (B, H, W) array/tensor
        wec_positions   : List[(x,y)]
        radiation_forces: List[Tensor(omega)]     from Module 2 (FSI)
        diffraction_forces: List[Tensor(omega)]   from Module 2

    Outputs dict:
        wave_field      : Tensor (B, n_omega, H, W)
        local_eta       : List[Tensor(omega)]     at each WEC
        local_pressure  : List[Tensor(omega)]
        local_velocity  : List[Tensor(omega)]
    """

    REQUIRED_INPUTS = ["spectrum", "bathymetry"]

    def __init__(self, config: Dict):
        super().__init__(config)
        op_type = config.get("operator_type", "wno")
        op_cfg  = config.get("operator_cfg", {
            "in_channels": 6, "out_channels": 3,
            "width": 64, "n_layers": 4})
        self.operator = OPERATOR_REGISTRY[op_type](op_cfg)
        self._wec_positions: Optional[List[Tuple[float, float]]] = None

    def run(self, inputs: Dict) -> Dict:
        self.validate_inputs(inputs, self.REQUIRED_INPUTS)

        bathy  = self._to_tensor(inputs["bathymetry"])     # (B, H, W)
        B, H, W = bathy.shape

        # Build input tensor: [bathy, Hs, Tp, dir, rad_force_mag, diff_force_mag]
        rad_mag  = self._force_magnitude(inputs.get("radiation_forces",   []), B, H, W)
        diff_mag = self._force_magnitude(inputs.get("diffraction_forces", []), B, H, W)
        sp       = inputs["spectrum"]
        Hs_t     = torch.full((B, 1, H, W), sp.get("Hs", 2.0))
        Tp_t     = torch.full((B, 1, H, W), sp.get("Tp", 8.0))
        dir_t    = torch.full((B, 1, H, W), sp.get("direction", 0.0))

        u = torch.cat([bathy.unsqueeze(1), Hs_t, Tp_t, dir_t,
                       rad_mag, diff_mag], dim=1)  # (B, 6, H, W)

        wave_field = self.operator(u)   # (B, 3, H, W)  [eta, u_vel, v_vel]

        wec_pos = inputs.get("wec_positions", self._wec_positions or [])
        local_eta, local_p, local_v = self._extract_local(wave_field, wec_pos, H, W)

        return {
            "wave_field":     wave_field,
            "local_eta":      local_eta,
            "local_pressure": local_p,
            "local_velocity": local_v,
        }

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float() if x.dim() == 3 else x.unsqueeze(0).float()
        import numpy as np
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return t if t.dim() == 3 else t.unsqueeze(0)

    def _force_magnitude(self, forces, B, H, W) -> torch.Tensor:
        if not forces:
            return torch.zeros(B, 1, H, W)
        mag = sum(f.abs().mean() for f in forces if isinstance(f, torch.Tensor))
        return torch.full((B, 1, H, W), float(mag))

    def _extract_local(self, field, positions, H, W):
        eta_list, p_list, v_list = [], [], []
        for (xi, yi) in positions:
            px = min(int(xi), H - 1)
            py = min(int(yi), W - 1)
            eta_list.append(field[:, 0, px, py])
            p_list.append(field[:, 0, px, py] * 1025 * 9.81)
            v_list.append(field[:, 1:3, px, py])
        return eta_list, p_list, v_list

    def get_observation(self, agent_id: str) -> Dict:
        return {}

    def apply_agent_action(self, agent_id: str, action) -> None:
        """Accept updated WEC position from layout-optimisation agent."""
        if "position" in action:
            self._wec_positions = action["position"]
