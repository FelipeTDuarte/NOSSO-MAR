"""
Fault-tolerant checkpointing for long HPC runs.
Supports local filesystem, shared NFS, and object storage (S3/LUMI Allas).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class HybridCheckpointer:
    """
    Saves checkpoints locally and optionally syncs to object storage.

    cfg keys:
        local_dir    : str
        remote_uri   : str  (optional, e.g. s3://bucket/nosso-mar/)
        save_every   : int  (epochs)
        keep_last_n  : int
    """

    def __init__(self, cfg: Dict):
        self.local_dir  = Path(cfg.get("local_dir", "checkpoints"))
        self.remote_uri = cfg.get("remote_uri", None)
        self.save_every = cfg.get("save_every", 10)
        self.keep_last_n = cfg.get("keep_last_n", 5)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict, epoch: int, tag: str = "model"):
        path = self.local_dir / f"{tag}_epoch{epoch:04d}.pt"
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")
        self._prune_old(tag)
        if self.remote_uri:
            self._sync_remote(path)

    def load_latest(self, tag: str = "model") -> Optional[Dict]:
        ckpts = sorted(self.local_dir.glob(f"{tag}_epoch*.pt"))
        if not ckpts:
            return None
        return torch.load(ckpts[-1], map_location="cpu")

    def _prune_old(self, tag: str):
        ckpts = sorted(self.local_dir.glob(f"{tag}_epoch*.pt"))
        for old in ckpts[:-self.keep_last_n]:
            old.unlink()

    def _sync_remote(self, path: Path):
        # Uses rclone or aws cli depending on cluster config
        import subprocess
        cmd = ["rclone", "copy", str(path), self.remote_uri]
        subprocess.Popen(cmd)   # non-blocking
