import torch
from src.nosso_mar.operators.wno.wavelet_conv import haar_dwt2d, haar_idwt2d, WaveletConv2d
from src.nosso_mar.operators.wno.wavelet_neural_operator import WaveletNeuralOperator


def test_haar_invertibility():
    x = torch.randn(1, 4, 32, 32)
    assert haar_idwt2d(*haar_dwt2d(x)).shape == x.shape

def test_wavelet_conv_shape():
    x = torch.randn(2, 8, 32, 32)
    assert WaveletConv2d(8, levels=2)(x).shape == x.shape

def test_wno_forward(small_wno_cfg, dummy_batch_2d):
    out = WaveletNeuralOperator(small_wno_cfg)(dummy_batch_2d)
    assert out.shape == (2, 2, 32, 32)
