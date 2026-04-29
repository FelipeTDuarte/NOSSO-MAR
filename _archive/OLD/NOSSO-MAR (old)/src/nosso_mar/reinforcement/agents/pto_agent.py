"""
PTO Control Agent (PPO-based).

Controls the PTO damping coefficient B_pto for each WEC to maximise
absorbed power given the current wave state.

obs:    local_eta_spectrum (N_omega,) + current_Bpto + phase_info
action: ΔB_pto (continuous) or Bpto directly
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),  nn.GELU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h   = self.shared(obs)
        mu  = self.actor_mean(h)
        std = self.actor_logstd.exp().expand_as(mu)
        val = self.critic(h)
        return mu, std, val


class PTOControlAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent for real-time PTO control.
    Can run at kHz rates (inference only) using the trained policy.
    """

    def __init__(self, agent_id: str, obs_dim: int, action_dim: int,
                 config: Dict):
        super().__init__(agent_id, obs_dim, action_dim, config)
        H  = config.get("hidden", 128)
        lr = config.get("lr", 3e-4)
        self.ac = ActorCritic(obs_dim, action_dim, H)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.vf_coef  = config.get("vf_coef",  0.5)
        self.ent_coef = config.get("ent_coef",  0.01)
        self.Bpto_min = config.get("Bpto_min", 1e3)
        self.Bpto_max = config.get("Bpto_max", 1e6)

    def select_action(self, obs, deterministic=False):
        mu, std, _ = self.ac(obs)
        if deterministic:
            raw = mu
        else:
            raw = torch.distributions.Normal(mu, std).rsample()
        # Map to [Bpto_min, Bpto_max] via sigmoid
        action = self.Bpto_min + (self.Bpto_max - self.Bpto_min) * raw.sigmoid()
        return action, None

    def update(self, batch: Dict) -> Dict[str, float]:
        obs, actions, returns, advantages, old_log_probs = (
            batch["obs"], batch["actions"], batch["returns"],
            batch["advantages"], batch["log_probs"])

        mu, std, values = self.ac(obs)
        dist     = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy   = dist.entropy().sum(-1).mean()

        ratio     = (log_probs - old_log_probs).exp()
        clip_adv  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss_p    = -torch.min(ratio * advantages, clip_adv).mean()
        loss_v    = nn.functional.mse_loss(values.squeeze(), returns)
        loss      = loss_p + self.vf_coef * loss_v - self.ent_coef * entropy

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"loss": float(loss), "loss_policy": float(loss_p),
                "loss_value": float(loss_v), "entropy": float(entropy)}

    def state_dict(self): return {"ac": self.ac.state_dict()}
    def load_state_dict(self, s): self.ac.load_state_dict(s["ac"])
