import torch
from src.nosso_mar.operators.deeponet.deeponet import DeepONet


def test_deeponet_shape(small_deeponet_cfg):
    m = DeepONet(small_deeponet_cfg)
    b = torch.randn(4, 7)
    t = torch.linspace(0.2,3.0,32).view(1,-1,1).expand(4,-1,-1)
    assert m(b, t).shape == (4, 32, 4)

def test_deeponet_no_nan(small_deeponet_cfg):
    m = DeepONet(small_deeponet_cfg)
    b = torch.randn(2, 7)
    t = torch.randn(2, 16, 1)
    assert not torch.isnan(m(b, t)).any()
