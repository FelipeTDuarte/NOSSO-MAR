"""
Multi-objective reward for WEC farm MARL.

Balances: power generation, structural fatigue, environmental impact,
and grid stability.
"""
from typing import Dict


class MultiObjectiveReward:
    DEFAULT_WEIGHTS = {
        "power":       1.0,
        "fatigue":    -0.2,
        "env_impact": -0.05,
        "grid_stability": 0.1,
    }

    def __init__(self, weights: Dict = None):
        self.w = weights or self.DEFAULT_WEIGHTS

    def __call__(self, metrics: Dict) -> float:
        return sum(self.w.get(k, 0.0) * float(v)
                   for k, v in metrics.items())
