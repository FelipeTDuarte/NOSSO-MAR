"""
SLURM job launcher for NOSSO-MAR HPC training.
Generates SLURM batch scripts for LUMI, MareNostrum 5, Leonardo, ARCHER2.
"""
from __future__ import annotations
from typing import Dict, Optional
import os, subprocess, textwrap
from pathlib import Path

CLUSTER_TEMPLATES = {
    "lumi": dict(partition="standard-g", account="project_462000000",
        modules=["LUMI/23.09","rocm/5.4.6","cray-python/3.10.10"], gpus_per_node=8),
    "marenostrum5": dict(partition="gpp", account="bsc_project",
        modules=["cuda/12.0","python/3.11"], gpus_per_node=4),
    "leonardo": dict(partition="boost_usr_prod", account="EUHPC_PROJECT",
        modules=["cuda/12.1","python/3.10.8"], gpus_per_node=4),
}

class SLURMLauncher:
    def __init__(self, cfg): self.cfg = cfg

    @classmethod
    def from_config(cls, config_path):
        import yaml
        with open(config_path) as fh: return cls(yaml.safe_load(fh))

    def generate_script(self, train_script, train_config, output_dir="outputs"):
        c = self.cfg; cluster = c.get("cluster","lumi")
        tmpl = CLUSTER_TEMPLATES.get(cluster, {})
        n_nodes = c.get("n_nodes",4); n_gpus = tmpl.get("gpus_per_node",8)
        walltime = c.get("walltime","24:00:00")
        ml = chr(10).join(f"module load {m}" for m in tmpl.get("modules",[]))
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=nosso_mar_train",
            f"#SBATCH --nodes={n_nodes}",
            f"#SBATCH --ntasks-per-node={n_gpus}",
            f"#SBATCH --gpus-per-node={n_gpus}",
            f"#SBATCH --partition={tmpl.get(chr(39)+'partition'+chr(39),'gpu')}",
            f"#SBATCH --account={tmpl.get(chr(39)+'account'+chr(39),chr(39)+chr(39))}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={output_dir}/slurm_%j.out",
            f"#SBATCH --error={output_dir}/slurm_%j.err",
            "", ml, "",
            "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
            "export MASTER_PORT=29500",
            f"export WORLD_SIZE=$((SLURM_NNODES * {n_gpus}))",
            f"srun torchrun \\",
            "    --nnodes=$SLURM_NNODES \\",
            f"    --nproc_per_node={n_gpus} \\",
            "    --rdzv_id=$SLURM_JOB_ID \\",
            "    --rdzv_backend=c10d \\",
            "    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\",
            f"    {train_script} --config {train_config}",
]
        return chr(10).join(lines)

    def submit(self, train_script, train_config, dry_run=False):
        script = self.generate_script(train_script, train_config)
        p = Path("slurm_job.sh"); p.write_text(script)
        if dry_run: print(script); return None
        r = subprocess.run(["sbatch", str(p)], capture_output=True, text=True)
        job_id = r.stdout.strip().split()[-1]
        print(f"Submitted job {job_id}"); return job_id
