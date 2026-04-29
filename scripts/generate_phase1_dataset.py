"""Generate the local analytic Phase 1 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nossomar.data.analytic_wec import make_frequency_grid
from nossomar.data.wec_dataset import generate_phase1_records, split_records, write_dataset_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the local NOSSO-MAR Phase 1 dataset.")
    parser.add_argument("--config", default="configs/lhs_phase1.yaml", help="Path to dataset YAML config.")
    args = parser.parse_args(argv)

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    freq = make_frequency_grid(
        start=float(config["frequencies"]["start"]),
        stop=float(config["frequencies"]["stop"]),
        count=int(config["frequencies"]["count"]),
    )
    records = generate_phase1_records(
        n_samples=int(config["n_samples"]),
        freq=freq,
        seed=int(config.get("seed", 0)),
    )
    splits = split_records(
        records,
        train_frac=float(config.get("train_frac", 0.7)),
        val_frac=float(config.get("val_frac", 0.15)),
        seed=int(config.get("seed", 0)),
    )
    metadata = {
        "generator": "analytic_phase1_baseline",
        "n_samples": int(config["n_samples"]),
        "freq_start_hz": float(config["frequencies"]["start"]),
        "freq_stop_hz": float(config["frequencies"]["stop"]),
        "freq_count": int(config["frequencies"]["count"]),
        "seed": int(config.get("seed", 0)),
    }
    output_path = ROOT / config["output_path"]
    write_dataset_json(output_path, splits, metadata=metadata)
    print(f"Wrote dataset to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
