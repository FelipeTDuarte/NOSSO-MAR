"""Farm power reward functions."""
import torch


class FarmPowerReward:
    def __init__(self, normalise: bool = True, baseline_power: float = 1e5):
        self.normalise      = normalise
        self.baseline_power = baseline_power

    def __call__(self, total_power: float, n_wec: int) -> float:
        r = total_power / n_wec
        if self.normalise:
            r = r / self.baseline_power
        return r
