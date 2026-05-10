"""
MADDPG coordinator — manages multi-agent training with centralised critics.

References:
    Lowe et al. (2017) — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
    https://arxiv.org/abs/1706.02275
"""
from __future__ import annotations
from typing import Dict, List
import torch
import numpy as np

from ..replay_buffer import ReplayBuffer


class MADDPG:
    def __init__(self, agents: List, buffer_size: int = 1_000_000,
                 batch_size: int = 256, config: Dict = None):
        self.agents     = agents
        self.buffer     = ReplayBuffer(buffer_size, len(agents))
        self.batch_size = batch_size
        self.config     = config or {}
        self.step       = 0

    def act(self, observations: List[torch.Tensor],
             deterministic: bool = False) -> List[torch.Tensor]:
        return [a.select_action(obs, deterministic)[0]
                for a, obs in zip(self.agents, observations)]

    def store(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push(obs, actions, rewards, next_obs, dones)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}
        batch  = self.buffer.sample(self.batch_size)
        losses = {}
        for i, agent in enumerate(self.agents):
            agent_batch = self._extract_agent_batch(batch, i)
            metrics = agent.update(agent_batch)
            for k, v in metrics.items():
                losses[f"agent{i}/{k}"] = v
        self.step += 1
        return losses

    def _extract_agent_batch(self, batch, i):
        return {
            "obs":        batch["obs"][:, i],
            "actions":    batch["actions"][:, i],
            "rewards":    batch["rewards"][:, i].unsqueeze(1),
            "next_obs":   batch["next_obs"][:, i],
            "dones":      batch["dones"][:, i].unsqueeze(1),
            "all_obs":    batch["obs"].flatten(1),
            "all_acts":   batch["actions"].flatten(1),
            "all_next_obs":  batch["next_obs"].flatten(1),
            "all_next_acts": batch["actions"].flatten(1),  # simplified
        }

    def save(self, path: str):
        states = [a.state_dict() for a in self.agents]
        torch.save({"agents": states, "step": self.step}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu")
        for a, s in zip(self.agents, ck["agents"]):
            a.load_state_dict(s)
        self.step = ck.get("step", 0)
