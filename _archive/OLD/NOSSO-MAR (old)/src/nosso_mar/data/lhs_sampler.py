"""
Latin Hypercube Sampler for BEM and wave propagation training data.
Ensures good coverage of the parameter space for the training dataset.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
try:
    from scipy.stats.qmc import LatinHypercube, scale
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LatinHypercubeSampler:
    """
    Generates LHS samples for BEM training data generation.

    Parameter ranges for point absorbers:
        radius  : 1.0 – 10.0  [m]
        draft   : 0.5 – 20.0  [m]
        mass    : 1e3 – 1e6   [kg]
        Bpto    : 1e3 – 1e6   [N.s/m]
        depth   : 10  – 500   [m]
    """

    DEFAULT_BOUNDS = {
        "radius": (1.0,  10.0),
        "draft":  (0.5,  20.0),
        "mass":   (1e3,  1e6),
        "Bpto":   (1e3,  1e6),
        "depth":  (10.0, 500.0),
    }

    def __init__(self, bounds: Dict = None, n_samples: int = 10000,
                 seed: int = 42):
        self.bounds    = bounds or self.DEFAULT_BOUNDS
        self.n_samples = n_samples
        self.seed      = seed
        self.param_names = list(self.bounds.keys())

    def sample(self) -> List[Dict]:
        n_dim = len(self.param_names)
        if HAS_SCIPY:
            sampler  = LatinHypercube(d=n_dim, seed=self.seed)
            unit     = sampler.random(self.n_samples)
            lo = np.array([self.bounds[k][0] for k in self.param_names])
            hi = np.array([self.bounds[k][1] for k in self.param_names])
            scaled = lo + unit * (hi - lo)
        else:
            rng    = np.random.default_rng(self.seed)
            lo = np.array([self.bounds[k][0] for k in self.param_names])
            hi = np.array([self.bounds[k][1] for k in self.param_names])
            scaled = lo + rng.random((self.n_samples, n_dim)) * (hi - lo)

        return [dict(zip(self.param_names, row)) for row in scaled]
