"""Mixed precision training utilities for NOSSO-MAR."""
from __future__ import annotations
from typing import Callable, Dict
import torch
import torch.nn as nn


class MixedPrecisionTrainer:
    """
    Wraps a training step with torch.cuda.amp for automatic mixed precision.
    Compatible with FNO, WNO, DeepONet.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def step(self, model: nn.Module, loss_fn: Callable,
              optimiser: torch.optim.Optimizer, batch: Dict) -> float:
        optimiser.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.enabled):
            loss = loss_fn(model, batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.scaler.step(optimiser)
        self.scaler.update()
        return float(loss)
