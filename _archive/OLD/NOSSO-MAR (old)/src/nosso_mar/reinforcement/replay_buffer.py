"""Experience replay buffer for MADDPG."""
from __future__ import annotations
from typing import Dict
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0
        self._data: Dict = {}

    def push(self, obs, actions, rewards, next_obs, dones):
        """Store one transition (all agents)."""
        def _t(x): return torch.as_tensor(x, dtype=torch.float32)
        for key, val in [("obs", obs), ("actions", actions),
                          ("rewards", rewards), ("next_obs", next_obs),
                          ("dones", dones)]:
            t = _t(val)
            if key not in self._data:
                self._data[key] = torch.zeros(self.capacity, *t.shape)
            self._data[key][self.ptr] = t
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self): return self.size
