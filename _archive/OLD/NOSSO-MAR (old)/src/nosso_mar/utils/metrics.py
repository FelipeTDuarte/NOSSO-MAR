import numpy as np
import torch


def rmse(pred, target):
    return float(((np.asarray(pred) - np.asarray(target)) ** 2).mean() ** 0.5)

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / target.norm().clamp(min=1e-8))

def nrmse(pred, target):
    r = rmse(pred, target)
    return r / (np.asarray(target).max() - np.asarray(target).min() + 1e-8)
