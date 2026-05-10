"""
Distributed training for NOSSO-MAR Neural Operators using PyTorch DDP.

Supports multi-GPU (single node) and multi-node (SLURM cluster) training.
Designed for large-scale NO training on European HPC infrastructure
(LUMI, MareNostrum, Leonardo, ARCHER2).

Usage:
    torchrun --nproc_per_node=8 scripts/train_distributed.py --config configs/fno_phase1.yaml
"""
from __future__ import annotations
from typing import Dict, Optional, Callable
import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


class DistributedNOTrainer:
    """
    DDP trainer for any NOSSO-MAR BaseOperator.

    Handles:
        - Process group initialisation (NCCL backend for GPU, Gloo for CPU)
        - DistributedSampler for dataset sharding
        - Gradient synchronisation
        - Checkpoint saving on rank 0 only
        - Mixed precision (torch.amp)
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._setup_dist()
        self.rank       = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device     = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available()
                                        else "cpu")
        self.use_amp    = cfg.get("mixed_precision", True) and torch.cuda.is_available()
        self.scaler     = torch.cuda.amp.GradScaler() if self.use_amp else None

    def _setup_dist(self):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

    def wrap_model(self, model: nn.Module) -> DDP:
        model = model.to(self.device)
        return DDP(model, device_ids=[self.rank] if torch.cuda.is_available() else None)

    def get_dataloader(self, dataset, batch_size: int,
                        shuffle: bool = True) -> DataLoader:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          pin_memory=True, num_workers=4)

    def train_epoch(self, model: DDP, loader: DataLoader,
                     loss_fn: Callable, optimiser: torch.optim.Optimizer,
                     lr_scheduler=None) -> Dict[str, float]:
        model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_batches  = 0
        loader.sampler.set_epoch(getattr(self, "_epoch", 0))

        for batch in loader:
            batch = {k: v.to(self.device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            optimiser.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimiser)
                self.scaler.update()
            else:
                loss = loss_fn(model, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.detach()
            n_batches  += 1

        if lr_scheduler:
            lr_scheduler.step()

        # Reduce loss across all ranks
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        return {"train_loss": float(total_loss / (n_batches * self.world_size))}

    def save_checkpoint(self, model: DDP, optimiser, epoch: int, path: str,
                         extra: Dict = None):
        if self.rank == 0:
            state = {
                "epoch":     epoch,
                "model":     model.module.state_dict(),
                "optimiser": optimiser.state_dict(),
                **(extra or {}),
            }
            torch.save(state, path)
            logger.info(f"Checkpoint saved: {path}")

    def cleanup(self):
        dist.destroy_process_group()
