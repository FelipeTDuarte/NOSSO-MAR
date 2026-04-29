"""Training callbacks for NOSSO-MAR."""
from __future__ import annotations
from typing import Optional
import math
import logging
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = math.inf
        self.wait      = 0

    def __call__(self, epoch, train_loss, val_loss, trainer) -> bool:
        loss = val_loss if val_loss is not None else train_loss
        if loss < self.best - self.min_delta:
            self.best = loss; self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


class WandBLogger:
    def __init__(self, project: str = "nosso-mar", run_name: str = None):
        try:
            import wandb
            wandb.init(project=project, name=run_name)
            self._wandb = wandb
        except ImportError:
            self._wandb = None
            logger.warning("wandb not installed — logging disabled")

    def __call__(self, epoch, train_loss, val_loss, trainer) -> bool:
        if self._wandb:
            log = {"epoch": epoch, "train_loss": train_loss}
            if val_loss: log["val_loss"] = val_loss
            self._wandb.log(log)
        return False


class LearningRateFinder:
    """Runs LR range test (Smith 2017) before main training."""

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 1.0, n_steps: int = 100):
        self.start_lr = start_lr
        self.end_lr   = end_lr
        self.n_steps  = n_steps

    def find(self, trainer, loader) -> float:
        import numpy as np
        lrs, losses = [], []
        factor = (self.end_lr / self.start_lr) ** (1 / self.n_steps)
        lr = self.start_lr
        for g in trainer.optimiser.param_groups:
            g["lr"] = lr
        for i, batch in enumerate(loader):
            if i >= self.n_steps: break
            batch = {k: v.to(trainer.device) if hasattr(v, "to") else v for k, v in batch.items()}
            loss = trainer.loss_fn(trainer.model, batch)
            trainer.optimiser.zero_grad(); loss.backward(); trainer.optimiser.step()
            lrs.append(lr); losses.append(float(loss))
            lr *= factor
            for g in trainer.optimiser.param_groups: g["lr"] = lr
        best_lr = lrs[int(np.argmin(losses))]
        logger.info(f"LR finder: best_lr = {best_lr:.2e}")
        return best_lr
