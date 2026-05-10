"""
WEC Layout Optimisation Agent (MADDPG-based).

Each agent controls the (x, y) position of one WEC in the farm.
Observation: local wave field + neighbouring WEC positions + current power.
Action: Δ(x, y) — incremental position update.

Reward: total farm power (cooperative) + layout diversity bonus.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from .base_agent import BaseAgent


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),  nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mu  = self.net(obs)
        std = self.log_std.exp().expand_as(mu)
        return mu, std


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int,
                 hidden: int = 256):
        super().__init__()
        # Centralised critic: sees all agents' obs and actions
        in_dim = obs_dim * n_agents + action_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


class WECLayoutAgent(BaseAgent):
    """
    MADDPG layout agent with centralised training / decentralised execution.

    obs_dim    : local_wave_field (flattened patch) + position + neighbour_pos
    action_dim : 2  (Δx, Δy) normalised to [-1, 1]
    """

    def __init__(self, agent_id: str, obs_dim: int, action_dim: int,
                 n_agents: int, config: Dict):
        super().__init__(agent_id, obs_dim, action_dim, config)
        H = config.get("hidden", 256)
        lr_actor  = config.get("lr_actor",  1e-4)
        lr_critic = config.get("lr_critic", 3e-4)
        self.max_delta = config.get("max_position_delta", 50.0)  # [m]

        self.actor  = PolicyNet(obs_dim, action_dim, H)
        self.critic = QNet(obs_dim, action_dim, n_agents, H)
        self.actor_target  = PolicyNet(obs_dim, action_dim, H)
        self.critic_target = QNet(obs_dim, action_dim, n_agents, H)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau = config.get("tau", 0.01)   # soft target update rate
        self.gamma = config.get("gamma", 0.99)

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            mu, std = self.actor(obs)
        if deterministic:
            return mu.tanh() * self.max_delta, None
        dist   = torch.distributions.Normal(mu, std)
        raw    = dist.rsample()
        action = raw.tanh() * self.max_delta
        log_p  = dist.log_prob(raw).sum(-1) -                  torch.log(1 - action.tanh() ** 2 + 1e-7).sum(-1)
        return action, log_p

    def update(self, batch: Dict) -> Dict[str, float]:
        obs        = batch["obs"]
        actions    = batch["actions"]
        rewards    = batch["rewards"]
        next_obs   = batch["next_obs"]
        dones      = batch["dones"]
        all_obs    = batch["all_obs"]
        all_acts   = batch["all_acts"]
        all_nobs   = batch["all_next_obs"]
        all_nacts  = batch["all_next_acts"]  # from target policies

        # Critic update
        with torch.no_grad():
            q_next = self.critic_target(all_nobs, all_nacts)
            q_tgt  = rewards + self.gamma * (1 - dones) * q_next
        q_pred = self.critic(all_obs, all_acts)
        loss_c = nn.functional.mse_loss(q_pred, q_tgt)
        self.opt_critic.zero_grad(); loss_c.backward(); self.opt_critic.step()

        # Actor update
        mu, _ = self.actor(obs)
        all_acts_new = all_acts.clone()
        all_acts_new[:, :self.action_dim] = mu.tanh() * self.max_delta
        loss_a = -self.critic(all_obs, all_acts_new).mean()
        self.opt_actor.zero_grad(); loss_a.backward(); self.opt_actor.step()

        # Soft target updates
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"loss_actor": float(loss_a), "loss_critic": float(loss_c)}

    def state_dict(self):
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
