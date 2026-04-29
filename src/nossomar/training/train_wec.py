"""Training helpers for the local Phase 1 WEC operator baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nossomar.data.analytic_wec import make_frequency_grid
from nossomar.data.wec_dataset import WECDataset, generate_phase1_records, split_records, write_dataset_json
from nossomar.operators.deeponet_wec import DeepONetWECRegressor


def ensure_dataset(config: dict[str, Any]) -> Path:
    """Materialize the dataset if it does not yet exist."""

    dataset_path = Path(config["data"]["path"])
    if dataset_path.exists():
        return dataset_path

    generator_cfg = config["data"].get("generate_if_missing")
    if generator_cfg is None:
        raise FileNotFoundError(f"Dataset not found at {dataset_path} and no generator config was provided.")

    freq = make_frequency_grid(
        start=float(generator_cfg["frequencies"]["start"]),
        stop=float(generator_cfg["frequencies"]["stop"]),
        count=int(generator_cfg["frequencies"]["count"]),
    )
    records = generate_phase1_records(
        n_samples=int(generator_cfg["n_samples"]),
        freq=freq,
        seed=int(generator_cfg.get("seed", 0)),
    )
    splits = split_records(
        records,
        train_frac=float(generator_cfg.get("train_frac", 0.7)),
        val_frac=float(generator_cfg.get("val_frac", 0.15)),
        seed=int(generator_cfg.get("seed", 0)),
    )
    metadata = {
        "generator": "analytic_phase1_baseline",
        "n_samples": int(generator_cfg["n_samples"]),
        "freq_start_hz": float(generator_cfg["frequencies"]["start"]),
        "freq_stop_hz": float(generator_cfg["frequencies"]["stop"]),
        "freq_count": int(generator_cfg["frequencies"]["count"]),
        "seed": int(generator_cfg.get("seed", 0)),
    }
    write_dataset_json(dataset_path, splits, metadata=metadata)
    return dataset_path


def evaluate_model(model: DeepONetWECRegressor, dataset: WECDataset) -> dict[str, float]:
    """Return absolute and relative RMSE metrics."""

    params, freq, targets = dataset.to_regression_arrays()
    predictions = model.predict(params, freq)
    rmse = model.rmse_by_channel(targets, predictions)
    scale = np.mean(np.abs(targets), axis=0) + 1.0e-12
    rel = np.sqrt(np.mean(((predictions - targets) / scale) ** 2, axis=0))
    rmse["relative_A"] = float(rel[0])
    rmse["relative_B"] = float(rel[1])
    rmse["relative_Fex_real"] = float(rel[2])
    rmse["relative_Fex_imag"] = float(rel[3])
    return rmse


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    """Train the local factorized operator and write artifacts."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset_path = ensure_dataset(config)

    train_ds = WECDataset.from_json(dataset_path, split="train")
    val_ds = WECDataset.from_json(dataset_path, split="val")

    train_params, train_freq, train_targets = train_ds.to_regression_arrays()
    model = DeepONetWECRegressor(ridge=float(config["model"].get("ridge", 1.0e-8)))
    model.fit(train_params, train_freq, train_targets)

    metrics = {
        "train": evaluate_model(model, train_ds),
        "val": evaluate_model(model, val_ds),
    }

    model_path = Path(config["artifacts"]["model_path"])
    metrics_path = Path(config["artifacts"]["metrics_path"])
    model.save_json(model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for local training."""

    parser = argparse.ArgumentParser(description="Train the local NOSSO-MAR Phase 1 WEC baseline.")
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args(argv)
    result = train_from_config(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
