from __future__ import annotations

import numpy as np

from nossomar.core.contracts import WECState
from nossomar.data.capytaine_runner import CapytaineRunner
from nossomar.data.wec_dataset import WECDataset, load_zarr_payload, write_dataset_zarr


def test_capytaine_runner_single_case_has_physical_bounds() -> None:
    freq = np.linspace(0.1, 1.5, 12)
    runner = CapytaineRunner(use_capytaine=False)

    state = runner.run_single(radius=5.0, draft=4.0, depth=50.0, freq_array=freq, bpto=20_000.0)

    mid = len(freq) // 2
    assert isinstance(state, WECState)
    assert state.metadata["backend"] == "analytic_fallback"
    assert np.all(state.damping >= 0.0)
    assert float(state.added_mass[mid]) > 0.0


def test_lhs_sweep_returns_requested_states() -> None:
    freq = np.linspace(0.2, 1.2, 6)
    runner = CapytaineRunner(use_capytaine=False)

    states = runner.run_lhs_sweep(
        n_samples=3,
        param_bounds={
            "radius": (3.0, 6.0),
            "draft": (2.0, 4.0),
            "depth": (30.0, 60.0),
            "bpto": (0.0, 50_000.0),
        },
        freq_array=freq,
        seed=4,
    )

    assert len(states) == 3
    assert all(np.all(state.damping >= 0.0) for state in states)
    assert all(float(state.added_mass[len(freq) // 2]) > 0.0 for state in states)


def test_wec_dataset_zarr_round_trip(tmp_path) -> None:
    freq = np.linspace(0.2, 1.2, 6)
    runner = CapytaineRunner(use_capytaine=False)
    splits = runner.run_lhs_dataset(
        n_samples=8,
        param_bounds={
            "radius": (3.0, 6.0),
            "draft": (2.0, 4.0),
            "depth": (30.0, 60.0),
            "bpto": (0.0, 50_000.0),
        },
        freq_array=freq,
        seed=8,
    )
    output = tmp_path / "phase1_wec_f1a1.zarr"

    write_dataset_zarr(output, splits=splits, metadata={"backend": runner.backend, "n_samples": 8})
    train = WECDataset.from_zarr(output, split="train")
    payload = load_zarr_payload(output)

    assert output.is_dir()
    assert (output / "dataset.json").exists()
    assert payload["metadata"]["n_samples"] == 8
    assert len(train) == len(splits["train"])
    assert train[0]["A"].shape == freq.shape
