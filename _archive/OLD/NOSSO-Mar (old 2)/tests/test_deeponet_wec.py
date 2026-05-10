import torch
from nossomar.operators.deeponet_wec import WECDeepONet


def test_forward_shape():
    model = WECDeepONet(branch_dim=4, trunk_dim=1, hidden=64, n_modes=32)
    props = torch.randn(8, 4)
    omega = torch.linspace(0.2, 4.0, 40).unsqueeze(0).repeat(8, 1)
    A, B = model(props, omega)
    assert A.shape == (8, 40)
    assert B.shape == (8, 40)
    assert not torch.isnan(A).any()
