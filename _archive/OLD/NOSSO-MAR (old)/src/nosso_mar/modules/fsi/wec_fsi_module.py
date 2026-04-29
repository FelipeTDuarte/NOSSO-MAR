"""
Module 2 — WEC Wave-Structure Interaction Module.
BEM surrogate (DeepONet) + frequency-domain EOM + PTO model.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import numpy as np

from ..base_module import BaseModule
from .bem_surrogate import BEMSurrogate
from .equation_of_motion import frequency_domain_eom, absorbed_power, total_power
from .pto_model import LinearPTO


class WecFSIModule(BaseModule):
    """
    Inputs dict:
        local_eta         : List[Tensor(omega)]
        local_pressure    : List[Tensor(omega)]
        device_properties : List[Dict]  — per WEC geometry + mechanical props
        omega             : ndarray [rad/s]

    Outputs dict:
        added_mass         : List[Tensor(omega)]
        radiation_damping  : List[Tensor(omega)]
        excitation_force   : List[Tensor(omega)]
        rao                : List[Tensor(omega)]
        absorbed_power     : List[Tensor(omega)]
        total_power        : float   [W]  total farm power
        radiation_forces   : List[Tensor(omega)]  → to Module 1
        diffraction_forces : List[Tensor(omega)]  → to Module 1
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.surrogate = BEMSurrogate(config)

    def run(self, inputs: Dict) -> Dict:
        omega  = inputs.get("omega", np.linspace(0.2, 3.0, 64))
        props  = inputs.get("device_properties", [])
        etas   = inputs.get("local_eta", [None] * len(props))

        results = {k: [] for k in [
            "added_mass", "radiation_damping", "excitation_force",
            "rao", "absorbed_power", "radiation_forces", "diffraction_forces"]}

        powers = []
        for i, (prop, eta) in enumerate(zip(props, etas)):
            bem = self.surrogate.predict(prop, omega)
            omega_t = torch.from_numpy(omega.astype(np.float32))

            M    = float(prop.get("mass",   50000))
            Bpto = float(prop.get("Bpto",   30000))
            C    = 1025 * 9.81 * float(prop.get("waterplane_area",
                                                  np.pi * prop.get("radius", 3.0) ** 2))

            F_ex_mag = bem["excitation_force"]
            # Use local eta as excitation force proxy if available
            if eta is not None:
                if isinstance(eta, torch.Tensor):
                    F_ex_mag = F_ex_mag * eta.abs()

            X   = frequency_domain_eom(omega_t, M,
                                        bem["added_mass"],
                                        bem["radiation_damping"],
                                        Bpto, C, F_ex_mag)
            P_w = absorbed_power(omega_t, X, Bpto)
            P   = total_power(omega_t, P_w)

            results["added_mass"].append(bem["added_mass"])
            results["radiation_damping"].append(bem["radiation_damping"])
            results["excitation_force"].append(F_ex_mag)
            results["rao"].append(bem["rao"])
            results["absorbed_power"].append(P_w)
            results["radiation_forces"].append(bem["radiation_damping"] * X)
            results["diffraction_forces"].append(F_ex_mag * 0.5)
            powers.append(float(P))

        results["total_power"] = sum(powers)
        return results

    def get_observation(self, agent_id: str) -> Dict:
        """PTO state observation for MARL agent."""
        return {"Bpto": self.config.get("Bpto", 30000)}

    def apply_agent_action(self, agent_id: str, action) -> None:
        """Accept Bpto update from MARL PTO agent."""
        if "Bpto" in action:
            self.config["Bpto"] = float(action["Bpto"])
