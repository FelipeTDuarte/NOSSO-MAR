"""
Phase 5 — Tracer Transport Module.
Surrogate for passive scalar transport (temperature, salinity, pollutants).
"""
from ..base_module import BaseModule


class TracerModule(BaseModule):
    """
    Inputs:  velocity_field, tracer_initial, diffusion_coeff, source_terms
    Outputs: tracer_field(t)
    Operator: FNO3d (x, y, t)
    Training data: ROMS / SCHISM tracer simulations
    """
    def run(self, inputs):
        # TODO: implement tracer transport surrogate
        return {"tracer_field": None}
