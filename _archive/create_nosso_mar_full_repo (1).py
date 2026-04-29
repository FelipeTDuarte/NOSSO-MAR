#!/usr/bin/env python3
"""
create_nosso_mar_full_repo.py
─────────────────────────────
Master scaffolder for the NOSSO-MAR research-grade repository.

Generates ~400 files covering:
  • Full FNO implementation (2D, 3D, F-FNO, Geo-FNO) with SpectralConv
  • Graph Neural Operator (GNO) for irregular meshes / WEC arrays
  • Wavelet Neural Operator (WNO) with Haar DWT
  • DeepONet, POD-DeepONet, PI-DeepONet for BEM surrogate
  • Adaptive Multi-Resolution Operator (AMR-NO)
  • Mesh-free operators (RBF-NO, KAN-NO)
  • Physics modules: wave propagation, WEC FSI, tides, morpho, tracers
  • Data assimilation: EnKF + 4D-Var (AD through NOs)
  • Digital twin: sensor ingestion, EnKF state estimator, anomaly detector, REST API
  • MARL: MADDPG layout agent, MAPPO PTO agent, WaveFarmEnv
  • HPC: DDP trainer, SLURM launcher (LUMI/MN5/Leonardo), mixed precision
  • Data pipeline: Capytaine, OceanWave3D, LHS sampler, Zarr store
  • Full test suite, CI/CD, docs, configs, benchmarks

USAGE:
    python create_nosso_mar_full_repo.py [--dir /path/to/repo] [--push]

OPTIONS:
    --dir   : output directory (default: ./NOSSO-MAR)
    --push  : push to GitHub (requires GITHUB_USER and GITHUB_TOKEN env vars)
    --dry   : print file list without writing
"""

import os
import sys
import stat
import json
import argparse
import subprocess
from pathlib import Path

# ── allow running from any directory ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── import file maps from the four build modules ─────────────────────────────
print("Loading build modules …")
from build_part1 import FILES as F1
from build_part2 import FILES as F2
from build_part3 import FILES as F3
from build_part4 import FILES as F4

ALL_FILES = {**F1, **F2, **F3, **F4}
print(f"  Total files defined: {len(ALL_FILES)}")


def write_file(base: Path, rel_path: str, content: str,
               make_exec: bool = False):
    target = base / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    if make_exec:
        mode = os.stat(target).st_mode
        os.chmod(target, mode | stat.S_IEXEC)


def make_exec_patterns(rel_path: str) -> bool:
    """Mark shell scripts and entry scripts as executable."""
    return (rel_path.endswith(".sh") or
            rel_path.startswith("scripts/") or
            rel_path == "Dockerfile")


def run_cmd(cmd, cwd, check: bool = False):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check,
                          capture_output=False)


def push_to_github(base: Path, repo_name: str):
    user  = os.environ.get("GITHUB_USER", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not (user and token):
        print("ERROR: set GITHUB_USER and GITHUB_TOKEN env vars to push.")
        return

    import urllib.request, urllib.error
    payload = json.dumps({
        "name":        repo_name,
        "private":     False,
        "description": "NOSSO-MAR — Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables",
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=payload,
        headers={"Authorization": f"token {token}",
                 "Content-Type":  "application/json",
                 "User-Agent":    "nosso-mar-scaffolder"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            info = json.loads(r.read())
            print(f"  GitHub repo created: {info.get('html_url')}")
    except urllib.error.HTTPError as e:
        print(f"  GitHub API error: {e.read().decode()}")
        return

    remote = f"https://{user}:{token}@github.com/{user}/{repo_name}.git"
    run_cmd(["git", "remote", "add", "origin", remote], base)
    run_cmd(["git", "push", "-u", "origin", "main"], base)


def main():
    parser = argparse.ArgumentParser(description="NOSSO-MAR scaffolder")
    parser.add_argument("--dir",  default="NOSSO-MAR",
                        help="Output directory (default: ./NOSSO-MAR)")
    parser.add_argument("--push", action="store_true",
                        help="Push to GitHub after creation")
    parser.add_argument("--dry",  action="store_true",
                        help="Print file list without writing")
    args = parser.parse_args()

    base = Path(args.dir).expanduser().resolve()

    if args.dry:
        print(f"\nDry run — {len(ALL_FILES)} files would be written to {base}:\n")
        for p in sorted(ALL_FILES):
            n = len(ALL_FILES[p])
            print(f"  {p:<80}  ({n:,} chars)")
        print(f"\nTotal: {len(ALL_FILES)} files, "
              f"{sum(len(v) for v in ALL_FILES.values()):,} chars")
        return

    print(f"\nCreating NOSSO-MAR at {base} …")
    base.mkdir(parents=True, exist_ok=True)

    written = 0
    for rel_path, content in sorted(ALL_FILES.items()):
        write_file(base, rel_path, content,
                   make_exec=make_exec_patterns(rel_path))
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{len(ALL_FILES)} files written …")

    print(f"  {written} files written ✓")

    # ── git init ────────────────────────────────────────────────────────────
    print("\nInitialising git repository …")
    run_cmd(["git", "init"], base)
    run_cmd(["git", "checkout", "-b", "main"], base)
    run_cmd(["git", "add", "."], base)
    run_cmd(["git", "commit", "-m",
             "Initial scaffold: NOSSO-MAR full architecture (Phase 0-7)"], base)
    run_cmd(["git", "checkout", "-b", "develop"], base)
    run_cmd(["git", "checkout", "main"], base)

    if args.push:
        print("\nPushing to GitHub …")
        push_to_github(base, base.name)

    # ── summary ─────────────────────────────────────────────────────────────
    total_chars = sum(len(v) for v in ALL_FILES.values())
    total_lines = sum(v.count("\n") for v in ALL_FILES.values())

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          NOSSO-MAR scaffold complete                        ║
╠══════════════════════════════════════════════════════════════╣
║  Files written : {written:<6}                                   ║
║  Total lines   : {total_lines:<8}                                 ║
║  Total chars   : {total_chars:<10}                               ║
║  Location      : {str(base):<44}  ║
╠══════════════════════════════════════════════════════════════╣
║  Next steps:                                                ║
║  1. cd {base.name:<55}  ║
║  2. conda env create -f environment.yml                     ║
║     conda activate nosso-mar                                ║
║  3. pre-commit install                                      ║
║  4. pytest tests/ -x                                        ║
║  5. python scripts/generate_bem_data.py --n_samples 10000   ║
║  6. python scripts/train_deeponet_bem.py \\                  ║
║         --config configs/training/deeponet_bem.yaml         ║
║  7. python scripts/train_fno.py \\                           ║
║         --config configs/training/fno_wave.yaml             ║
║  8. python scripts/benchmarks/operator_benchmark.py         ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
