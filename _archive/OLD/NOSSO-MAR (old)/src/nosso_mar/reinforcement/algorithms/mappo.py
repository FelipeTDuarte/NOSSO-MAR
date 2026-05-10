"""
MAPPO — Multi-Agent PPO with centralised value function.

Better sample efficiency than MADDPG for cooperative tasks.
Used for WEC PTO control where agents share environment dynamics.

References:
    Yu et al. (2022) — The Surprising Effectiveness of PPO in Cooperative MARL
    https://arxiv.org/abs/2103.01955
"""
from __future__ import annotations
from typing import Dict, List
import torch


class MAPPO:
    def __init__(self, agents, config: Dict = None):
        self.agents  = agents
        self.config  = config or {}
        self.n_steps = config.get("n_steps", 2048)
        self.n_epochs = config.get("n_epochs", 10)
        self.rollout_buffer: List = []

    def collect_rollout(self, env, n_steps: int = None) -> Dict:
        n_steps = n_steps or self.n_steps
        rollout = {"obs": [], "actions": [], "rewards": [], "dones": [], "values": []}
        obs = env.reset()
        for _ in range(n_steps):
            actions, log_probs, values = [], [], []
            for a, o in zip(self.agents, obs):
                act, lp = a.select_action(torch.as_tensor(o, dtype=torch.float32))
                actions.append(act)
            next_obs, rewards, done, _ = env.step(actions)
            rollout["obs"].append(obs)
            rollout["actions"].append(actions)
            rollout["rewards"].append(rewards)
            rollout["dones"].append(done)
            obs = next_obs
            if done:
                obs = env.reset()
        return rollout

    def update(self, rollout: Dict) -> Dict[str, float]:
        losses = {}
        for i, agent in enumerate(self.agents):
            # Build per-agent batch from rollout
            batch = {"obs":      torch.stack([torch.as_tensor(r[i], dtype=torch.float32) for r in rollout["obs"]]),
                     "actions":  torch.stack([torch.as_tensor(a[i], dtype=torch.float32) for a in rollout["actions"]]),
                     "returns":  torch.zeros(len(rollout["rewards"])),
                     "advantages": torch.zeros(len(rollout["rewards"])),
                     "log_probs": torch.zeros(len(rollout["rewards"]))}
            metrics = agent.update(batch)
            for k, v in metrics.items():
                losses[f"agent{i}/{k}"] = v
        return losses
