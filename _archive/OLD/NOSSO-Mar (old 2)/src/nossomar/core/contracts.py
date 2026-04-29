from dataclasses import dataclass
import torch


@dataclass
class WECState:
    pos: torch.Tensor
    vel: torch.Tensor
    force: torch.Tensor


@dataclass
class WaveField:
    eta: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


def validate_wec_state(s: WECState):
    for name, t in (("pos", s.pos), ("vel", s.vel), ("force", s.force)):
        if torch.isnan(t).any():
            raise ValueError(f"NaN detected in WECState.{name}")
