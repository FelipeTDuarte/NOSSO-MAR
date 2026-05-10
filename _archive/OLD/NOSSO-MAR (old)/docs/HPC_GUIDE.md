# HPC Training Guide

## Supported clusters
| Cluster | GPUs/node | GPU |
|---|---|---|
| LUMI (CSC) | 8 | AMD MI250X |
| MareNostrum 5 (BSC) | 4 | H100 |
| Leonardo (CINECA) | 4 | A100 |

## Launch distributed training
```bash
python -c "
from src.nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/lumi.yaml').submit(
    'scripts/train_fno.py', 'configs/training/ffno_large.yaml')
"
```

## Mixed precision
Enabled by default. Disable: `mixed_precision: false` in training config.
