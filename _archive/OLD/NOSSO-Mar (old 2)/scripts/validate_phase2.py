#!/usr/bin/env python
"""Phase 2 validation CLI (P2-T5).

Benchmarks Phase 1 and Phase 2 surrogates on hydrodynamic coefficients and
prints a side-by-side RMSE comparison table.

Usage examples:

  # Smoke test with analytic data (no checkpoints needed)
  python scripts/validate_phase2.py --smoke

  # Benchmark against a WEC-Sim JSON file
  python scripts/validate_phase2.py --benchmark data/benchmarks/rm3_heave_synthetic.json

  # Full run with saved checkpoints
  python scripts/validate_phase2.py \\
      --checkpoint-p1 checkpoints/phase1.pt \\
      --checkpoint-p2 checkpoints/phase2.pt \\
      --benchmark data/benchmarks/rm3_heave_synthetic.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_phase2",
        description="Validate and compare Phase 1 vs Phase 2 WEC surrogates.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test with analytic data (no checkpoints required).",
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a benchmark JSON or HDF5 file (WEC-Sim format).",
    )
    p.add_argument(
        "--checkpoint-p1",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to Phase 1 model checkpoint (.pt). Falls back to fresh model if omitted.",
    )
    p.add_argument(
        "--checkpoint-p2",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to Phase 2 model checkpoint (.pt). Falls back to fresh model if omitted.",
    )
    p.add_argument(
        "--n-cases",
        type=int,
        default=100,
        metavar="N",
        help="Number of analytic cases to generate for smoke / fallback evaluation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for analytic dataset generation.",
    )
    return p


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _rel_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    scale = np.abs(target).mean()
    if scale < 1e-12:
        return float("nan")
    return _rmse(pred, target) / scale


def _peak_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).max())


def _evaluate_phase1(ds, checkpoint: Path | None) -> dict:
    import torch
    from nossomar.operators.deeponet_wec import WECDeepONet

    omega = torch.tensor(ds["omega"].values, dtype=torch.float32)
    props = torch.tensor(
        np.stack([
            ds["radius"].values, ds["draft"].values,
            ds["depth"].values, ds["bpto"].values,
        ], axis=1),
        dtype=torch.float32,
    )
    A_tgt = ds["added_mass"].values
    B_tgt = ds["damping"].values

    model = WECDeepONet(branch_dim=4, trunk_dim=1, hidden=128, n_modes=64)
    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    om = omega.unsqueeze(0).repeat(props.shape[0], 1)
    with torch.no_grad():
        A_pred, B_pred = model(props, om)

    return {
        "rmse_A": _rmse(A_pred.numpy(), A_tgt),
        "rel_rmse_A": _rel_rmse(A_pred.numpy(), A_tgt),
        "peak_A": _peak_error(A_pred.numpy(), A_tgt),
        "rmse_B": _rmse(B_pred.numpy(), B_tgt),
        "rel_rmse_B": _rel_rmse(B_pred.numpy(), B_tgt),
    }


def _evaluate_phase2(ds, checkpoint: Path | None) -> dict:
    import torch
    from nossomar.training.train_phase2 import _Phase2Model

    omega = torch.tensor(ds["omega"].values, dtype=torch.float32)
    props = torch.tensor(
        np.stack([
            ds["radius"].values, ds["draft"].values,
            ds["depth"].values, ds["bpto"].values,
        ], axis=1),
        dtype=torch.float32,
    )
    A_tgt = ds["added_mass"].values
    B_tgt = ds["damping"].values

    model = _Phase2Model(d_latent=64)
    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    om = omega.unsqueeze(0).repeat(props.shape[0], 1)
    with torch.no_grad():
        A_pred, B_pred = model(props, om)

    return {
        "rmse_A": _rmse(A_pred.numpy(), A_tgt),
        "rel_rmse_A": _rel_rmse(A_pred.numpy(), A_tgt),
        "peak_A": _peak_error(A_pred.numpy(), A_tgt),
        "rmse_B": _rmse(B_pred.numpy(), B_tgt),
        "rel_rmse_B": _rel_rmse(B_pred.numpy(), B_tgt),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(p1: dict, p2: dict) -> None:
    # ASCII column names: safe on every terminal encoding (CP1252, UTF-8, etc.)
    header = (
        f"{'Model':<12} {'RMSE A(w)':>14} {'Rel. RMSE A':>12}"
        f" {'Peak error A':>14} {'RMSE B(w)':>14} {'Rel. RMSE B':>12}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for label, m in [("Phase 1", p1), ("Phase 2", p2)]:
        print(
            f"{label:<12}"
            f" {m['rmse_A']:>14.4e}"
            f" {m['rel_rmse_A']:>12.4f}"
            f" {m['peak_A']:>14.4e}"
            f" {m['rmse_B']:>14.4e}"
            f" {m['rel_rmse_B']:>12.4f}"
        )
    print(sep)
    print()
    threshold = 0.15
    status = "PASS" if p2["rel_rmse_A"] < threshold else "FAIL"
    print(f"Phase 2 validation criterion (rel. RMSE A < {threshold:.0%}): {status}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.benchmark is not None:
        if not args.benchmark.exists():
            print(f"ERROR: benchmark file not found: {args.benchmark}", file=sys.stderr)
            return 1
        from nossomar.data.wec_sim_adapter import load_wecsim_dataset
        print(f"Loading benchmark: {args.benchmark}")
        ds = load_wecsim_dataset(args.benchmark)
    else:
        from nossomar.data.wec_dataset import generate_analytic_dataset
        print(f"Generating analytic dataset: n_cases={args.n_cases}, seed={args.seed}")
        ds = generate_analytic_dataset(n_cases=args.n_cases, seed=args.seed)

    print("Evaluating Phase 1 surrogate...")
    p1_metrics = _evaluate_phase1(ds, args.checkpoint_p1)

    print("Evaluating Phase 2 surrogate...")
    p2_metrics = _evaluate_phase2(ds, args.checkpoint_p2)

    _print_table(p1_metrics, p2_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
