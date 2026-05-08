"""Generate the Phase 1 F1A single-WEC dataset."""

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
from nossomar.data.capytaine_runner import CapytaineRunner
from nossomar.data.wec_dataset import write_dataset_json, write_dataset_zarr


def _bounds_from_config(config: dict) -> dict[str, tuple[float, float]]:
    params = config["dataset_f1a1"]["device_params"]
    bounds: dict[str, tuple[float, float]] = {}
    mapping = {
        "radius_m": "radius_m",
        "draft_m": "draft_m",
        "depth_m": "depth_m",
        "bpto_kNsm": "bpto_kNsm",
    }
    for source_key, target_key in mapping.items():
        value = params.get(source_key)
        if value is not None:
            bounds[target_key] = (float(value[0]), float(value[1]))
    return bounds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the NOSSO-MAR F1A WEC dataset.")
    parser.add_argument("--config", default="configs/scenarios/phase1_full_f1a.yaml", help="Path to F1A YAML config.")
    parser.add_argument("--output", default=None, help="Output dataset path, usually data/phase1_wec_f1a1.zarr.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sample count for smoke runs.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker count.")
    parser.add_argument(
        "--analytic-fallback",
        action="store_true",
        help="Force the analytic fallback instead of importing Capytaine.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = config["dataset_f1a1"]
    freq_cfg = dataset_cfg["wave_conditions"]["freq_hz"]
    freq = make_frequency_grid(
        start=float(freq_cfg["start"]),
        stop=float(freq_cfg["stop"]),
        count=int(freq_cfg["count"]),
    )
    n_samples = int(args.n_samples or dataset_cfg["n_samples"])
    split_cfg = dataset_cfg.get("split", {})
    output = Path(args.output or dataset_cfg["output_path"])
    if not output.is_absolute():
        output = ROOT / output

    runner = CapytaineRunner(use_capytaine=not args.analytic_fallback, max_workers=args.max_workers)
    splits = runner.run_lhs_dataset(
        n_samples=n_samples,
        param_bounds=_bounds_from_config(config),
        freq_array=freq,
        seed=int(dataset_cfg.get("seed", 0)),
        train_frac=float(split_cfg.get("train", 0.7)),
        val_frac=float(split_cfg.get("val", 0.15)),
    )
    metadata = {
        "generator": dataset_cfg.get("generator", "capytaine_lhs"),
        "backend": runner.backend,
        "n_samples": n_samples,
        "seed": int(dataset_cfg.get("seed", 0)),
        "freq_start_hz": float(freq[0]),
        "freq_stop_hz": float(freq[-1]),
        "freq_count": int(len(freq)),
    }
    if output.suffix == ".zarr":
        write_dataset_zarr(output, splits=splits, metadata=metadata)
    else:
        write_dataset_json(output, splits=splits, metadata=metadata)
    print(f"Wrote F1A dataset to {output} using backend={runner.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
