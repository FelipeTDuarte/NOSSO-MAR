"""
Generate BEM training data for Module 2 (WecFSIModule / BEMSurrogate).
Runs Capytaine in parallel for LHS-sampled device configurations.

Usage:
    python scripts/generate_bem_data.py --n_samples 10000 --n_workers 32 \
        --output_dir data/processed/capytaine --omega_min 0.2 --omega_max 3.0 --n_omega 64
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single(args):
    idx, props, omega, outdir = args
    from src.nosso_mar.data.capytaine_dataset import CapytaineSampler
    sampler = CapytaineSampler(omega)
    path    = str(Path(outdir) / f"bem_{idx:06d}.nc")
    try:
        sampler.run_single(props, path)
        return idx, True
    except Exception as e:
        logger.warning(f"Sample {idx} failed: {e}")
        return idx, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int,   default=10000)
    parser.add_argument("--n_workers",  type=int,   default=4)
    parser.add_argument("--output_dir", type=str,   default="data/processed/capytaine")
    parser.add_argument("--omega_min",  type=float, default=0.2)
    parser.add_argument("--omega_max",  type=float, default=3.0)
    parser.add_argument("--n_omega",    type=int,   default=64)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)

    from src.nosso_mar.data.lhs_sampler import LatinHypercubeSampler
    sampler = LatinHypercubeSampler(n_samples=args.n_samples)
    configs = sampler.sample()

    logger.info(f"Generating {args.n_samples} BEM runs with {args.n_workers} workers ...")
    tasks = [(i, cfg, omega, args.output_dir) for i, cfg in enumerate(configs)]

    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(run_single, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, ok = fut.result()
            if ok: n_ok += 1
            if (n_ok + 1) % 100 == 0:
                logger.info(f"  {n_ok}/{args.n_samples} completed")

    logger.info(f"Done. {n_ok}/{args.n_samples} BEM runs successful.")


if __name__ == "__main__":
    main()
