import torch


def test_fno2d_forward(small_fno_cfg, dummy_batch_2d):
    from src.nosso_mar.operators.fno.fno2d import FNO2d
    out = FNO2d(small_fno_cfg)(dummy_batch_2d)
    assert out.shape == (2, 2, 32, 32)

def test_fno3d_forward():
    from src.nosso_mar.operators.fno.fno3d import FNO3d
    cfg = {"in_channels":3,"out_channels":2,"width":8,"modes_x":4,"modes_y":4,"modes_t":4,"n_layers":2}
    out = FNO3d(cfg)(torch.randn(2, 3, 16, 16, 8))
    assert out.shape == (2, 2, 16, 16, 8)

def test_spectral_conv2d_preserves_shape():
    from src.nosso_mar.operators.fno.spectral_conv import SpectralConv2d
    x = torch.randn(2, 8, 32, 32)
    assert SpectralConv2d(8, 8, 4, 4)(x).shape == x.shape

def test_ffno2d_forward():
    from src.nosso_mar.operators.fno.ffno import FFNO2d
    cfg = {"in_channels":3,"out_channels":2,"width":16,"modes_x":4,"modes_y":4,"n_layers":2}
    out = FFNO2d(cfg)(torch.randn(2, 3, 32, 32))
    assert out.shape[1] == 2

def test_fno2d_no_nan(small_fno_cfg, dummy_batch_2d):
    from src.nosso_mar.operators.fno.fno2d import FNO2d
    assert not torch.isnan(FNO2d(small_fno_cfg)(dummy_batch_2d)).any()
