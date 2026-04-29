"""
Single-node NO trainer for NOSSO-MAR.
Supports both FNO (wave) and DeepONet (BEM) training pipelines.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class NOTrainer:
    def __init__(self, model: nn.Module, loss_fn: Callable,
                 optimiser: torch.optim.Optimizer,
                 lr_scheduler=None, cfg: Dict = None):
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimiser    = optimiser
        self.scheduler    = lr_scheduler
        self.cfg          = cfg or {}
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history: Dict = {"train": [], "val": []}

    def train(self, train_loader, val_loader=None,
               n_epochs: int = 100, callbacks=None) -> Dict:
        callbacks = callbacks or []
        for epoch in range(n_epochs):
            train_loss = self._epoch(train_loader, training=True)
            self.history["train"].append(train_loss)

            val_loss = None
            if val_loader:
                val_loss = self._epoch(val_loader, training=False)
                self.history["val"].append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            logger.info(f"Epoch {epoch+1}/{n_epochs} | "
                        f"train={train_loss:.4e}"
                        + (f" | val={val_loss:.4e}" if val_loss else ""))

            for cb in callbacks:
                stop = cb(epoch, train_loss, val_loss, self)
                if stop:
                    logger.info("Early stopping triggered.")
                    return self.history

        return self.history

    def _epoch(self, loader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        ctx   = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = {k: v.to(self.device) if hasattr(v, "to") else v
                         for k, v in batch.items()}
                loss = self.loss_fn(self.model, batch)
                if training:
                    self.optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimiser.step()
                total += float(loss)
        return total / max(len(loader), 1)
