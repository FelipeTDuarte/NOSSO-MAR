from __future__ import annotations

import numpy as np

from nossomar.data.analytic_wec import make_frequency_grid
from nossomar.data.wec_dataset import WECDataset, generate_phase1_records, split_records, write_dataset_json
from nossomar.operators.deeponet_wec import DeepONetWECRegressor
from nossomar.training.train_wec import train_from_config


def test_factorized_deeponet_fits_local_synthetic_dataset(tmp_path) -> None:
    freq = make_frequency_grid(count=16)
    records = generate_phase1_records(n_samples=18, freq=freq, seed=5)
    splits = split_records(records, seed=5)
    path = tmp_path / "dataset.json"
    write_dataset_json(path, splits, metadata={"n_samples": 18})

    train_ds = WECDataset.from_json(path, split="train")
    val_ds = WECDataset.from_json(path, split="val")
    train_params, train_freq, train_targets = train_ds.to_regression_arrays()
    val_params, val_freq, val_targets = val_ds.to_regression_arrays()

    model = DeepONetWECRegressor(ridge=1.0e-12)
    model.fit(train_params, train_freq, train_targets)
    val_predictions = model.predict(val_params, val_freq)

    denom = np.mean(np.abs(val_targets), axis=0) + 1.0e-12
    rel_error = np.sqrt(np.mean(((val_predictions - val_targets) / denom) ** 2, axis=0))
    assert float(np.max(rel_error)) < 1.0e-10


def test_training_pipeline_writes_artifacts(tmp_path) -> None:
    config_path = tmp_path / "training.yaml"
    dataset_path = tmp_path / "phase1_dataset.json"
    model_path = tmp_path / "checkpoints" / "model.json"
    metrics_path = tmp_path / "checkpoints" / "metrics.json"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  path: {dataset_path.as_posix()}",
                "  generate_if_missing:",
                "    n_samples: 24",
                "    seed: 9",
                "    train_frac: 0.7",
                "    val_frac: 0.15",
                "    frequencies:",
                "      start: 0.1",
                "      stop: 2.0",
                "      count: 18",
                "model:",
                "  ridge: 1.0e-12",
                "artifacts:",
                f"  model_path: {model_path.as_posix()}",
                f"  metrics_path: {metrics_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    result = train_from_config(config_path)

    assert dataset_path.exists()
    assert model_path.exists()
    assert metrics_path.exists()
    assert result["metrics"]["val"]["relative_A"] < 1.0e-10
    assert result["metrics"]["val"]["relative_B"] < 1.0e-10
