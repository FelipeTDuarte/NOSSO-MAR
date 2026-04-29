"""
Coupling manager — iterative M1↔M2 loop and sequential coupling.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch


class CouplingManager:
    def __init__(self, wave_no=None, fsi_no=None,
                 max_iter: int = 10, tol: float = 1e-4):
        self.wave_no  = wave_no
        self.fsi_no   = fsi_no
        self.max_iter = max_iter
        self.tol      = tol

    def run_coupled(self, base_inputs: Dict) -> Dict:
        """Iterative M1↔M2 coupling until radiation forces converge."""
        n_wec       = len(base_inputs.get("wec_positions", []))
        rad_forces  = [np.zeros(1)] * n_wec
        diff_forces = [np.zeros(1)] * n_wec

        for iteration in range(self.max_iter):
            m1_inputs = {**base_inputs,
                         "radiation_forces":   rad_forces,
                         "diffraction_forces": diff_forces}
            m1_out = self.wave_no.run(m1_inputs) if self.wave_no else {}

            m2_inputs = {
                "local_eta":          m1_out.get("local_eta", []),
                "local_pressure":     m1_out.get("local_pressure", []),
                "device_properties":  base_inputs.get("device_properties", []),
                "omega":              base_inputs.get("omega",
                                        np.linspace(0.2, 3.0, 64)),
            }
            m2_out = self.fsi_no.run(m2_inputs) if self.fsi_no else {}

            new_rad  = m2_out.get("radiation_forces",   rad_forces)
            new_diff = m2_out.get("diffraction_forces", diff_forces)

            if n_wec > 0:
                delta = max(
                    np.linalg.norm(
                        np.asarray(new_rad[i]) - np.asarray(rad_forces[i]))
                    for i in range(n_wec))
                if delta < self.tol:
                    break

            rad_forces  = new_rad
            diff_forces = new_diff

        return {**m1_out, **m2_out, "_coupling_iterations": iteration + 1}

    def execute_sequential(self, modules: List, inputs: Dict) -> Dict:
        data = dict(inputs)
        for module in modules:
            data.update(module.run(data))
        return data
