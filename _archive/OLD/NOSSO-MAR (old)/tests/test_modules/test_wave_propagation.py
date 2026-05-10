import torch, numpy as np
from src.nosso_mar.modules.wave.wave_propagation_no import WavePropagationNO
from src.nosso_mar.modules.wave.dispersion import linear_dispersion, group_velocity


def _small_cfg():
    return {"operator_type":"wno",
            "operator_cfg":{"in_channels":6,"out_channels":3,"width":16,"n_layers":2,"levels":2}}

def test_wave_no_runs():
    mod = WavePropagationNO(_small_cfg())
    out = mod.run({"spectrum":{"Hs":2.0,"Tp":8.0,"direction":0.0},
                   "bathymetry": np.ones((64,64))*30.0,
                   "wec_positions":[(16,16),(32,32)]})
    assert "wave_field" in out
    assert len(out["local_eta"]) == 2

def test_dispersion():
    omega = torch.linspace(0.5, 2.5, 20)
    h     = torch.ones(20) * 50.0
    k  = linear_dispersion(omega, h)
    cg = group_velocity(omega, k, h)
    assert (k  > 0).all()
    assert (cg > 0).all()
