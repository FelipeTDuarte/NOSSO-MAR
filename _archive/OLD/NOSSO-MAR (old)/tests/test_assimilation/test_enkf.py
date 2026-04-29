import torch
from src.nosso_mar.assimilation.enkf import EnsembleKalmanFilter


def test_enkf_analysis():
    sd, od, ne = 50, 5, 20
    enkf = EnsembleKalmanFilter(lambda x: x, lambda x: x[:od], ne, sd, od)
    enkf.initialise(torch.zeros(sd))
    xa = enkf.analysis(torch.randn(od), torch.ones(od)*0.01)
    assert xa.shape == (sd,)
    assert not torch.isnan(xa).any()
