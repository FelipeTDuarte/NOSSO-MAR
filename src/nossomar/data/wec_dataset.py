"""Dataset generation and loading utilities for the local baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nossomar.core.contracts import WECState
from nossomar.data.analytic_wec import build_analytic_wec_state

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "radius": (3.0, 12.0),
    "draft": (2.0, 10.0),
    "depth": (30.0, 100.0),
    "bpto": (0.0, 100_000.0),
    "mass_factor": (0.75, 1.25),
}


@dataclass(slots=True)
class LHSConfig:
    """Small configuration object for the local Latin-hypercube generator."""

    n_samples: int
    seed: int = 0
    bounds: dict[str, tuple[float, float]] | None = None


def latin_hypercube_samples(
    n_samples: int,
    bounds: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Generate lightweight Latin-hypercube samples without SciPy."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    active_bounds = bounds or DEFAULT_BOUNDS
    param_names = list(active_bounds.keys())
    dim = len(param_names)
    samples = np.zeros((n_samples, dim), dtype=float)

    for column, name in enumerate(param_names):
        lower, upper = active_bounds[name]
        strata = (np.arange(n_samples, dtype=float) + rng.random(n_samples)) / n_samples
        rng.shuffle(strata)
        samples[:, column] = lower + (upper - lower) * strata

    payload: list[dict[str, float]] = []
    for row in samples:
        payload.append({name: float(value) for name, value in zip(param_names, row, strict=True)})
    return payload


def latin_hypercube_sample(
    n_samples: int,
    bounds: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Compatibility alias for the local Latin-hypercube sampler."""

    return latin_hypercube_samples(n_samples=n_samples, bounds=bounds, seed=seed)


def generate_phase1_records(
    n_samples: int,
    freq: np.ndarray | list[float],
    seed: int = 0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Generate analytic WEC cases for the local Phase 1 dataset."""

    records: list[dict[str, Any]] = []
    for index, sample in enumerate(latin_hypercube_samples(n_samples=n_samples, bounds=bounds, seed=seed)):
        displaced_mass = 1025.0 * np.pi * sample["radius"] ** 2 * sample["draft"]
        mass = displaced_mass * sample["mass_factor"]
        state = build_analytic_wec_state(
            radius=sample["radius"],
            draft=sample["draft"],
            depth=sample["depth"],
            mass=mass,
            bpto=sample["bpto"],
            freq=freq,
            metadata={"case_id": f"analytic-{index:04d}"},
        )
        records.append(
            {
                "case_id": f"analytic-{index:04d}",
                "device_params": {
                    "radius": sample["radius"],
                    "draft": sample["draft"],
                    "mass": mass,
                    "bpto": sample["bpto"],
                    "depth": sample["depth"],
                },
                "wec_state": state.to_dict(),
            }
        )
    return records


def split_records(
    records: list[dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val/test sets."""

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    n_train = int(round(len(records) * train_frac))
    n_val = int(round(len(records) * val_frac))
    n_train = min(n_train, len(records))
    n_val = min(n_val, len(records) - n_train)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return {
        "train": [records[int(index)] for index in train_indices],
        "val": [records[int(index)] for index in val_indices],
        "test": [records[int(index)] for index in test_indices],
    }


def split_database(
    records: list[dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Compatibility alias for the baseline split function."""

    return split_records(records=records, train_frac=train_frac, val_frac=val_frac, seed=seed)


def write_dataset_json(
    path: str | Path,
    splits: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the dataset and a small YAML metadata sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "splits": splits,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sidecar = target.with_suffix(".metadata.yaml")
    sidecar.write_text(yaml.safe_dump(payload["metadata"], sort_keys=True), encoding="utf-8")


def load_dataset_json(path: str | Path) -> dict[str, Any]:
    """Load a dataset bundle written by write_dataset_json."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


class WECDataset:
    """Simple in-memory dataset wrapper for the local baseline."""

    def __init__(self, records: list[dict[str, Any]], split: str = "train") -> None:
        self.records = records
        self.split = split

    @classmethod
    def from_json(cls, path: str | Path, split: str = "train") -> "WECDataset":
        payload = load_dataset_json(path)
        return cls(records=payload["splits"][split], split=split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        record = self.records[index]
        state = WECState.from_dict(record["wec_state"])
        params = record["device_params"]
        device_params = np.array(
            [params["radius"], params["draft"], params["mass"], params["bpto"], params["depth"]],
            dtype=float,
        )
        return {
            "case_id": np.array(record["case_id"]),
            "device_params": device_params,
            "freq": state.freq.copy(),
            "A": state.added_mass.copy(),
            "B": state.damping.copy(),
            "Fex_real": state.excitation_real.copy(),
            "Fex_imag": state.excitation_imag.copy(),
        }

    def to_regression_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten case-wise frequency responses into row-wise regression arrays."""

        device_rows: list[np.ndarray] = []
        freq_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []

        for record in self.records:
            params = record["device_params"]
            device_params = np.array(
                [params["radius"], params["draft"], params["mass"], params["bpto"], params["depth"]],
                dtype=float,
            )
            state = WECState.from_dict(record["wec_state"])
            repeated_params = np.repeat(device_params[None, :], len(state.freq), axis=0)
            targets = np.column_stack(
                [
                    state.added_mass,
                    state.damping,
                    state.excitation_real,
                    state.excitation_imag,
                ]
            )
            device_rows.append(repeated_params)
            freq_rows.append(state.freq.copy())
            target_rows.append(targets)

        return (
            np.vstack(device_rows),
            np.concatenate(freq_rows),
            np.vstack(target_rows),
        )


WECDatabase = WECDataset


def build_wec_database(
    n_samples: int,
    freq: np.ndarray | list[float],
    seed: int = 0,
    bounds: dict[str, tuple[float, float]] | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Generate and split a local analytic database in one call."""

    records = generate_phase1_records(n_samples=n_samples, freq=freq, seed=seed, bounds=bounds)
    return split_records(records=records, train_frac=train_frac, val_frac=val_frac, seed=seed)
