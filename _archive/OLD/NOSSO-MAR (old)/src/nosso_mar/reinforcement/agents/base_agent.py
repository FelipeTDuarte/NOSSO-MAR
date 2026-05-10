"""Abstract base class for all NOSSO-MAR MARL agents."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseAgent(ABC):
    def __init__(self, agent_id: str, obs_dim: int, action_dim: int,
                 config: Dict):
        self.agent_id   = agent_id
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.config     = config

    @abstractmethod
    def select_action(self, obs: torch.Tensor,
                       deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (action, log_prob)."""
        ...

    @abstractmethod
    def update(self, batch: Dict) -> Dict[str, float]:
        """Update policy from experience batch. Returns loss metrics."""
        ...

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    @abstractmethod
    def state_dict(self) -> Dict:
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict):
        ...
