# Generic SLURM config — NOSSO-MAR
# Use this as a starting point for any SLURM-based HPC cluster.
# Copy, rename to your cluster name, and fill in the placeholders.
#
# SLURM is used by: LUMI, MareNostrum 5, Leonardo, Deucalion, and most
# academic HPC clusters worldwide (~50% of HPC sites globally).
#
# How to use:
#   1. Copy this file: cp generic_slurm.yaml my_cluster.yaml
#   2. Fill in every field marked with YOUR_*
#   3. Verify partition name with: sinfo -s
#   4. Verify account name with:   sacctmgr show user $USER
#   5. Test with a short job before full training runs

cluster:         YOUR_CLUSTER_NAME       # e.g. "ada", "hawk", "archer2"
institution:     YOUR_INSTITUTION
scheduler:       slurm

# ── Partition / queue ────────────────────────────────────────────────────────
# Find available partitions with: sinfo -s
# GPU partitions are commonly named: gpu, gpu-a100, accelerated, boost_usr_prod
partition:       YOUR_GPU_PARTITION      # e.g. "gpu", "gpu-a100"
account:         YOUR_ACCOUNT           # from: sacctmgr show user $USER

# ── Resources ────────────────────────────────────────────────────────────────
n_nodes:         1                      # start with 1 node to test
gpus_per_node:   4                      # check with: sinfo -o "%n %G"
walltime:        "04:00:00"             # HH:MM:SS — check cluster limits

# ── GPU type ─────────────────────────────────────────────────────────────────
# Common values: A100, H100, V100, A40, RTX3090, MI250X
# Check available GPU types with: sinfo -o "%n %G %f"
gpu_type:        YOUR_GPU_TYPE

# ── Mixed precision ──────────────────────────────────────────────────────────
# A100 / V100 / H100  → use FP16 (with GradScaler)
# MI250X (AMD)        → use BF16
# If unsure, set true — the training script detects GPU capability at runtime
mixed_precision: true

# ── Environment modules ──────────────────────────────────────────────────────
# List modules to load before running. Check with: module avail cuda
# Common pattern: cuda + python + MPI
modules_to_load:
  - YOUR_CUDA_MODULE        # e.g. "cuda/12.1", "nvidia/cuda-12.1"
  - YOUR_PYTHON_MODULE      # e.g. "python/3.11", "anaconda/2023"
  - YOUR_MPI_MODULE         # e.g. "openmpi/4.1", "mpt/2.25"

# ── Storage paths ────────────────────────────────────────────────────────────
# Use the cluster's scratch filesystem — NOT home (quota too small for ML data)
# Find your scratch path with: echo $SCRATCH or check cluster documentation
checkpoint_dir:  /scratch/YOUR_ACCOUNT/nosso_mar/checkpoints
data_dir:        /scratch/YOUR_ACCOUNT/nosso_mar/data
output_dir:      /scratch/YOUR_ACCOUNT/nosso_mar/outputs

# ── Optional: SLURM constraints ──────────────────────────────────────────────
# Uncomment and set if your cluster requires specific node features
# constraints:   "a100_80gb"           # e.g. to request high-memory GPU nodes
# exclude_nodes: "node[01-05]"         # e.g. to avoid known bad nodes
# qos:           "gpu"                 # Quality of Service, if required

# ── SLURM quick reference ────────────────────────────────────────────────────
# Submit job:      sbatch my_job.sh
# Check queue:     squeue -u $USER
# Cancel job:      scancel JOB_ID
# Job details:     scontrol show job JOB_ID
# Accounting:      sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS
# Available nodes: sinfo -o "%n %C %G %f %m"
# GPU usage:       squeue -o "%i %j %R %b"      (shows GRES/GPU info)