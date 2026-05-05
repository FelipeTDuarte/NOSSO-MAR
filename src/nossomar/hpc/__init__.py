"""
HPC module — PHASE 2 (planned).

Distributed training and job scheduling on European HPC clusters.

Planned modules:
    distributed_trainer.py — DDP with NCCL
    slurm_launcher.py      — SLURM launcher for LUMI/MN5/Leonardo
    mixed_precision.py     — BF16/FP16 with GradScaler
    checkpointing.py       — Automatic checkpoint and resume

Supported clusters: LUMI (CSC), MareNostrum 5 (BSC), Leonardo (CINECA)
Reference: _archive/OLD/NOSSO-MAR (old)/src/nosso_mar/hpc/
Configs:   configs/hpc/
"""
