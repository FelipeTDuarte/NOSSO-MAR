# HPC Training Guide — NOSSO-MAR

> **Status**: Phase 2 (planned). Reference configs in `configs/hpc/`.

## Supported clusters (planned)

| Cluster | Institution | GPUs/node | GPU |
|---------|-------------|-----------|-----|
| LUMI (CSC) | Finland | 8 | AMD MI250X |
| MareNostrum 5 (BSC) | Spain | 4 | H100 |
| Leonardo (CINECA) | Italy | 4 | A100 |

## Distributed training (DDP + SLURM)

**Reference file**: `_archive/OLD/NOSSO-MAR (old)/src/nosso_mar/hpc/slurm_launcher.py`

```bash
# Example: 16 nodes on LUMI
python -c "
from nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/lumi.yaml').submit(
    'scripts/train_phase1.py', 'configs/training.yaml')
"
```

## HPC configs

See `configs/hpc/` for ready-to-use configs:
- `lumi.yaml` — 16 nodes, 24h walltime
- `marenostrum5.yaml` — 8 nodes, 12h walltime
- `leonardo.yaml` — 8 nodes, 12h walltime

These configs will be activated in Phase 2.

## Mixed precision

Enabled by default in training configs:
```yaml
mixed_precision: true  # BF16 on LUMI (MI250X), FP16 on A100/H100
```

## Checkpointing

```python
# Current (Phase 1)
checkpoints/deeponet_wec_local.json     # model weights
checkpoints/deeponet_wec_metrics.json   # training history
```

In Phase 2, checkpointing will be integrated with the HPC module to support
automatic job resumption on clusters with limited walltime.
