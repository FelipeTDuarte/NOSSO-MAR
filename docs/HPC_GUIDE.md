# HPC Training Guide — NOSSO-MAR

> **Status**: Phase 2 (planned). Reference configs in `configs/hpc/`.

---

## Supported clusters (planned)

| Cluster | Institution | Country | GPUs/node | GPU | Config |
|---------|-------------|---------|-----------|-----|--------|
| LUMI (CSC) | CSC | Finland | 8 | AMD MI250X | `configs/hpc/lumi.yaml` |
| MareNostrum 5 (BSC) | BSC | Spain | 4 | H100 | `configs/hpc/marenostrum5.yaml` |
| Leonardo (CINECA) | CINECA | Italy | 4 | A100 | `configs/hpc/leonardo.yaml` |
| Deucalion (MACC) | FCT / FCCN | Portugal | 4 | A100 (40/80 GB) | `configs/hpc/deucalion.yaml` |

All four are EuroHPC Joint Undertaking supercomputers. Access is available
both via national allocations and transnational EuroHPC calls.

---

## Adding a new cluster

If your cluster is not listed above, use one of the generic templates:

| Scheduler | Template | Covers |
|-----------|----------|--------|
| **SLURM** | `configs/hpc/generic_slurm.yaml` | ~50% of HPC sites worldwide — LUMI, MN5, Leonardo, Deucalion, and most academic clusters |
| **PBS / OpenPBS** | `configs/hpc/generic_pbs.yaml` | ~32% of sites — older university clusters, some national labs |

### Which scheduler does my cluster use?

```bash
which sbatch   # SLURM — if this returns a path
which qsub     # PBS/OpenPBS/Torque — if this returns a path
which bsub     # IBM LSF — enterprise/commercial clusters
```

### Steps to add a new cluster

```bash
# 1. Copy the right template
cp configs/hpc/generic_slurm.yaml configs/hpc/my_cluster.yaml

# 2. Find your partition name
sinfo -s                          # SLURM
qstat -Q                          # PBS

# 3. Find your account name
sacctmgr show user $USER          # SLURM
qstat -u $USER                    # PBS

# 4. Find your scratch path
echo $SCRATCH                     # most clusters
df -h ~                           # or check quota and filesystem

# 5. Fill in the placeholders and commit
```

---

## Deucalion — notes for NOSSO-MAR

Deucalion is the most relevant cluster for this project given its Portuguese
funding and host institution (University of Minho / FCT). It has two GPU node
types in the same partition:

- **17 nodes**: 4× Nvidia A100 40 GB
- **16 nodes**: 4× Nvidia A100 80 GB

For Phase 1 operators (DeepONet, FNO2d) the 40 GB nodes are sufficient.
For Phase 2 large operators (F-FNO, 3D grids, full EnKF ensembles) prefer
80 GB nodes:

```bash
#SBATCH --constraint=a100_80gb   # check availability with MACC support
```

Access: national quota via FCT, or transnational via EuroHPC JU calls.
Apply at: https://www.eurohpc-ju.europa.eu/access-our-supercomputers

---

## Distributed training (DDP + SLURM)

**Reference file**: `_archive/OLD/NOSSO-MAR (old)/src/nosso_mar/hpc/slurm_launcher.py`

```bash
# Example: 4 nodes on Deucalion (16 A100 GPUs total)
python -c "
from nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/deucalion.yaml').submit(
    'scripts/train_phase1.py', 'configs/training/ffno_large.yaml')
"

# Example: 16 nodes on LUMI (128 MI250X GPUs total)
python -c "
from nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/lumi.yaml').submit(
    'scripts/train_phase1.py', 'configs/training/ffno_large.yaml')
"

# Example: any new cluster using a generic config
python -c "
from nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/my_cluster.yaml').submit(
    'scripts/train_phase1.py', 'configs/training/deeponet_wec.yaml')
"
```

---

## Mixed precision by GPU

| GPU | Recommended precision | Notes |
|-----|-----------------------|-------|
| AMD MI250X | BF16 (native) | LUMI |
| H100 | BF16 (native) | MareNostrum 5 |
| A100 | FP16 (with GradScaler) | Leonardo, Deucalion |
| V100 | FP16 (with GradScaler) | older clusters |
| RTX 30/40 series | BF16 | consumer / university clusters |

All configs have `mixed_precision: true`. The training script selects
BF16 or FP16 automatically based on GPU capability at runtime.

---

## Checkpointing

```python
# Current (Phase 1 — local)
checkpoints/deeponet_wec_local.json     # model weights
checkpoints/deeponet_wec_metrics.json   # training history
```

In Phase 2, checkpointing will be integrated with the HPC module to support
automatic job resumption on clusters with limited walltime. The `checkpoint_dir`
in each cluster config points to the correct scratch filesystem.

---

## Storage paths by cluster

| Cluster | Scratch filesystem |
|---------|--------------------|
| LUMI | `/scratch/project_*/nosso_mar/` |
| MareNostrum 5 | `/gpfs/scratch/*/nosso_mar/` |
| Leonardo | `/leonardo_scratch/*/nosso_mar/` |
| Deucalion | `/scratch/*/nosso_mar/` (DDN EXAScaler, 10.6 PB) |
| Generic | Set `checkpoint_dir` / `data_dir` in your cluster config |

Always use the cluster's **scratch filesystem**, not home — quota on home
directories is typically too small for ML datasets and checkpoints.
Update the `your_*_account` placeholders in the config before submitting.
