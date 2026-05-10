import torch, numpy as np
from src.nosso_mar.modules.fsi.bem_surrogate import BEMSurrogate
from src.nosso_mar.modules.fsi.equation_of_motion import (
    frequency_domain_eom, absorbed_power, total_power)
from src.nosso_mar.modules.fsi.wec_fsi_module import WecFSIModule


def _cfg():
    return {"deeponet_cfg":{"branch_input_dim":7,"trunk_input_dim":1,
                            "hidden_dim":16,"n_hidden":2,"output_dim":4,"p":16}}

def test_bem_surrogate():
    s = BEMSurrogate(_cfg())
    out = s.predict({"radius":3.,"draft":3.,"mass":5e4,"volume":85.,
                     "Bpto":3e4,"cog":0.,"depth":50.}, np.linspace(0.2,3.0,32))
    assert out["radiation_damping"].shape == (32,)
    assert (out["radiation_damping"] >= 0).all()

def test_eom_power_positive():
    omega = torch.linspace(0.2, 3.0, 32)
    A = torch.ones(32)*1e4; B = torch.ones(32)*5e3; F = torch.ones(32)*1e4
    X = frequency_domain_eom(omega, 5e4, A, B, 3e4, 2e5, F)
    P = absorbed_power(omega, X, 3e4)
    assert float(total_power(omega, P)) > 0

def test_wec_fsi_module():
    mod = WecFSIModule(_cfg())
    out = mod.run({"omega": np.linspace(0.2,3.0,32),
                   "device_properties":[
                       {"radius":3.,"draft":3.,"mass":5e4,"volume":85.,"Bpto":3e4,"cog":0.,"depth":50.},
                       {"radius":3.,"draft":3.,"mass":5e4,"volume":85.,"Bpto":3e4,"cog":0.,"depth":50.}]})
    assert "total_power" in out
    assert len(out["absorbed_power"]) == 2
