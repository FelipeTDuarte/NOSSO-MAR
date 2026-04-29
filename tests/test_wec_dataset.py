from __future__ import annotations

import numpy as np

from nossomar.data.analytic_wec import make_frequency_grid
from nossomar.data.wec_dataset import WECDataset, generate_phase1_records, split_records, write_dataset_json


def test_dataset_generation_and_loading(tmp_path) -> None:
    freq = make_frequency_grid(count=10)
    records = generate_phase1_records(n_samples=20, freq=freq, seed=3)
    splits = split_records(records, seed=3)
    path = tmp_path / "phase1_dataset.json"
    write_dataset_json(path, splits, metadata={"n_samples": 20})

    train_ds = WECDataset.from_json(path, split="train")
    item = train_ds[0]

    assert len(train_ds) == 14
    assert item["device_params"].shape == (5,)
    assert item["A"].shape == (10,)
    assert item["B"].shape == (10,)


def test_regression_arrays_have_expected_shape() -> None:
    freq = make_frequency_grid(count=8)
    records = generate_phase1_records(n_samples=6, freq=freq, seed=1)
    dataset = WECDataset(records, split="train")
    params, freq_rows, targets = dataset.to_regression_arrays()

    assert params.shape == (48, 5)
    assert freq_rows.shape == (48,)
    assert targets.shape == (48, 4)
    assert np.all(freq_rows > 0.0)
