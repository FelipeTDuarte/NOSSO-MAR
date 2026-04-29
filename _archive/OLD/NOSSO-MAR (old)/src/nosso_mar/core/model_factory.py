"""
Model factory — instantiates operators and modules from YAML configs.
"""
from __future__ import annotations
from typing import Dict
import torch


OPERATOR_REGISTRY = {
    "fno2d":   "src.nosso_mar.operators.fno.fno2d.FNO2d",
    "fno3d":   "src.nosso_mar.operators.fno.fno3d.FNO3d",
    "geo_fno": "src.nosso_mar.operators.fno.geo_fno.GeoFNO2d",
    "ffno":    "src.nosso_mar.operators.fno.ffno.FFNO2d",
    "wno":     "src.nosso_mar.operators.wno.wavelet_neural_operator.WaveletNeuralOperator",
    "gno":     "src.nosso_mar.operators.gno.graph_neural_operator.GraphNeuralOperator",
    "deeponet":"src.nosso_mar.operators.deeponet.deeponet.DeepONet",
    "rbf":     "src.nosso_mar.operators.meshfree.rbf_operator.RBFOperator",
}

MODULE_REGISTRY = {
    "wave_propagation_no": "src.nosso_mar.modules.wave.wave_propagation_no.WavePropagationNO",
    "wec_fsi":             "src.nosso_mar.modules.fsi.wec_fsi_module.WecFSIModule",
    "tidal":               "src.nosso_mar.modules.hydrodynamics.tidal_module.TidalModule",
    "morpho":              "src.nosso_mar.modules.morphodynamics.morpho_module.MorphodynamicsModule",
    "tracer":              "src.nosso_mar.modules.tracers.tracer_module.TracerModule",
}


def _import(dotted_path: str):
    parts  = dotted_path.rsplit(".", 1)
    module = __import__(parts[0], fromlist=[parts[1]])
    return getattr(module, parts[1])


class ModelFactory:
    @staticmethod
    def build_operator(operator_type: str, cfg: Dict):
        cls = _import(OPERATOR_REGISTRY[operator_type])
        return cls(cfg)

    @staticmethod
    def build_module(module_type: str, cfg: Dict):
        cls = _import(MODULE_REGISTRY[module_type])
        return cls(cfg)

    @staticmethod
    def load_operator(operator_type: str, cfg: Dict, weights_path: str):
        op = ModelFactory.build_operator(operator_type, cfg)
        state = torch.load(weights_path, map_location="cpu")
        op.load_state_dict(state)
        op.eval()
        return op
