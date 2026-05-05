"""PyTorch training loop for the Phase 1 WEC DeepONet operator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nossomar.data.analytic_wec import (
    BRANCH_DIM,
    TRUNK_DIM,
    branch_feature_vector_from_params,
    make_frequency_grid,
    trunk_feature_matrix,
)
from nossomar.data.wec_dataset import WECDataset, generate_phase1_records, split_records, write_dataset_json
from nossomar.operators.factory import build_operator

CHANNELS = ("A", "B", "Fex_real", "Fex_imag")
STAGE_KEYS = {"a": "stage_a", "b": "stage_b", "c": "stage_c", "d": "stage_d"}


@dataclass(slots=True)
class Normalizer:
    """Mean/std transform stored beside checkpoints for deterministic inference."""

    branch_mean: np.ndarray
    branch_std: np.ndarray
    trunk_mean: np.ndarray
    trunk_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, branch: np.ndarray, trunk: np.ndarray, target: np.ndarray) -> "Normalizer":
        return cls(
            branch_mean=branch.mean(axis=0),
            branch_std=branch.std(axis=0) + 1.0e-8,
            trunk_mean=trunk.mean(axis=0),
            trunk_std=trunk.std(axis=0) + 1.0e-8,
            target_mean=target.mean(axis=0),
            target_std=target.std(axis=0) + 1.0e-8,
        )

    def transform_inputs(self, branch: np.ndarray, trunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (branch - self.branch_mean) / self.branch_std, (trunk - self.trunk_mean) / self.trunk_std

    def transform_targets(self, target: np.ndarray) -> np.ndarray:
        return (target - self.target_mean) / self.target_std

    def inverse_targets_torch(self, target: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, dtype=target.dtype, device=target.device)
        std = torch.as_tensor(self.target_std, dtype=target.dtype, device=target.device)
        return target * std + mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "branch_mean": self.branch_mean.tolist(),
            "branch_std": self.branch_std.tolist(),
            "trunk_mean": self.trunk_mean.tolist(),
            "trunk_std": self.trunk_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Normalizer":
        return cls(**{key: np.asarray(value, dtype=float) for key, value in payload.items()})


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _freq_from_cfg(cfg: dict[str, Any] | None) -> np.ndarray:
    freq_cfg = cfg or {"start": 0.1, "stop": 2.0, "count": 32}
    return make_frequency_grid(
        start=float(freq_cfg.get("start", 0.1)),
        stop=float(freq_cfg.get("stop", 2.0)),
        count=int(freq_cfg.get("count", 32)),
    )


def ensure_dataset(config: dict[str, Any]) -> Path:
    """Materialize a JSON analytic WEC dataset when the config requests one."""

    dataset_path = _as_path(config["data"]["path"])
    if dataset_path.exists():
        return dataset_path

    generator_cfg = config["data"].get("generate_if_missing")
    if generator_cfg is None:
        raise FileNotFoundError(f"Dataset not found at {dataset_path} and no generator config was provided.")

    freq = _freq_from_cfg(generator_cfg.get("frequencies"))
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
        "freq_start_hz": float(freq[0]),
        "freq_stop_hz": float(freq[-1]),
        "freq_count": int(len(freq)),
        "seed": int(generator_cfg.get("seed", 0)),
    }
    write_dataset_json(dataset_path, splits, metadata=metadata)
    return dataset_path


def ensure_stage_dataset(stage_cfg: dict[str, Any], stage_key: str) -> Path:
    """Resolve or build the dataset for one roadmap training stage."""

    dataset_cfg = stage_cfg.get("dataset", {})
    source = dataset_cfg.get("source")
    if source == "analytic":
        freq = _freq_from_cfg(dataset_cfg.get("freq"))
        dataset_path = _as_path(dataset_cfg.get("path", f"data/{stage_key}_analytic_wec.json"))
        if dataset_path.exists():
            return dataset_path
        records = generate_phase1_records(
            n_samples=int(dataset_cfg.get("n_samples", 500)),
            freq=freq,
            seed=int(dataset_cfg.get("seed", 0)),
        )
        splits = split_records(records, seed=int(dataset_cfg.get("seed", 0)))
        write_dataset_json(
            dataset_path,
            splits,
            metadata={
                "generator": "analytic_phase1_stage",
                "stage": stage_key,
                "n_samples": int(dataset_cfg.get("n_samples", 500)),
                "freq_count": int(len(freq)),
                "seed": int(dataset_cfg.get("seed", 0)),
            },
        )
        return dataset_path

    if isinstance(source, str):
        path = _as_path(source)
        if path.suffix == ".json":
            if not path.exists():
                raise FileNotFoundError(f"Stage dataset not found: {path}")
            return path
        raise NotImplementedError(f"{stage_key} expects dataset '{path}', but only JSON analytic datasets are wired in T06.")

    raise ValueError(f"{stage_key} does not define a dataset source.")


def _dataset_to_arrays(dataset: WECDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params, freq, targets = dataset.to_regression_arrays()
    branch = np.vstack([branch_feature_vector_from_params(row) for row in params])
    trunk = trunk_feature_matrix(freq)
    return branch, trunk, targets


def _make_loader(
    dataset: WECDataset,
    normalizer: Normalizer,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    branch, trunk, targets = _dataset_to_arrays(dataset)
    branch_norm, trunk_norm = normalizer.transform_inputs(branch, trunk)
    targets_norm = normalizer.transform_targets(targets)
    tensor_dataset = TensorDataset(
        torch.as_tensor(branch_norm, dtype=torch.float32),
        torch.as_tensor(trunk_norm, dtype=torch.float32),
        torch.as_tensor(targets_norm, dtype=torch.float32),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_physical(
    model: nn.Module,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for branch, trunk, target in loader:
            branch = branch.to(device)
            trunk = trunk.to(device).unsqueeze(1)
            pred_norm = model(branch, trunk).squeeze(1)
            pred = normalizer.inverse_targets_torch(pred_norm)
            truth = normalizer.inverse_targets_torch(target.to(device))
            pred[:, 1] = torch.clamp(pred[:, 1], min=0.0)
            predictions.append(pred.cpu().numpy())
            targets.append(truth.cpu().numpy())
    return np.vstack(predictions), np.vstack(targets)


def evaluate_model(
    model: nn.Module,
    dataset: WECDataset,
    normalizer: Normalizer,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Return absolute RMSE, relative RMSE, and damping-constraint metrics."""

    active_device = torch.device(device)
    loader = _make_loader(dataset, normalizer, batch_size=batch_size, shuffle=False)
    predictions, targets = _predict_physical(model, loader, normalizer, active_device)
    errors = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    scale = np.mean(np.abs(targets), axis=0) + 1.0e-12
    rel = np.sqrt(np.mean(((predictions - targets) / scale) ** 2, axis=0))
    metrics = {name: float(value) for name, value in zip(CHANNELS, errors, strict=True)}
    metrics.update({f"relative_{name}": float(value) for name, value in zip(CHANNELS, rel, strict=True)})
    metrics["B_violations"] = int(np.sum(predictions[:, 1] < 0.0))
    metrics["loss_finite"] = bool(np.isfinite(errors).all())
    return metrics


def _scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int):
    normalized = name.strip().lower()
    if normalized == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if normalized == "constant":
        return None
    if normalized in {"plateau", "reduce_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    return None


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    for branch, trunk, target in loader:
        branch = branch.to(device)
        trunk = trunk.to(device).unsqueeze(1)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(branch, trunk).squeeze(1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * branch.shape[0]
    return total / max(1, len(loader.dataset))


def _val_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for branch, trunk, target in loader:
            branch = branch.to(device)
            trunk = trunk.to(device).unsqueeze(1)
            target = target.to(device)
            pred = model(branch, trunk).squeeze(1)
            loss = criterion(pred, target)
            total += float(loss.item()) * branch.shape[0]
    return total / max(1, len(loader.dataset))


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    normalizer: Normalizer,
    stage_key: str,
    operator: str,
    architecture: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "stage": stage_key,
            "operator": operator,
            "architecture": architecture,
            "metrics": metrics,
        },
        path,
    )


def _load_resume(model: nn.Module, resume_from: str | Path | None, device: torch.device) -> Normalizer | None:
    if not resume_from:
        return None
    path = _as_path(resume_from)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)
    if isinstance(payload, dict) and "normalizer" in payload:
        return Normalizer.from_dict(payload["normalizer"])
    return None


def _stage_from_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("branch_input_dim", BRANCH_DIM)
    model_cfg.setdefault("trunk_input_dim", TRUNK_DIM)
    model_cfg.setdefault("hidden_dim", 64)
    model_cfg.setdefault("n_hidden", 2)
    model_cfg.setdefault("output_dim", 4)
    model_cfg.setdefault("p", 32)
    model_cfg.setdefault("activation", "tanh")
    training = dict(config.get("training", {}))
    training.setdefault("epochs", 20)
    training.setdefault("batch_size", 64)
    training.setdefault("lr", 1.0e-3)
    training.setdefault("scheduler", "cosine")
    training.setdefault("early_stopping_patience", 10)
    return {
        "name": "local_pytorch_deeponet",
        "operator": config.get("operator", "deeponet"),
        "architecture": model_cfg,
        "training": training,
        "artifacts": {
            "checkpoint": config["artifacts"]["model_path"],
            "metrics": config["artifacts"]["metrics_path"],
        },
        "data": config["data"],
    }


def train_stage(
    config: dict[str, Any],
    stage_key: str = "stage_a",
    resume_from: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train one Phase 1 DeepONet stage and write checkpoint/metrics artifacts."""

    if stage_key == "legacy":
        stage_cfg = _stage_from_legacy_config(config)
        dataset_path = ensure_dataset(stage_cfg)
    else:
        stage_cfg = config[stage_key]
        dataset_path = ensure_stage_dataset(stage_cfg, stage_key)

    seed = int(config.get("reproducibility", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = WECDataset.from_json(dataset_path, split="train")
    val_ds = WECDataset.from_json(dataset_path, split="val")
    branch, trunk, targets = _dataset_to_arrays(train_ds)
    normalizer = Normalizer.fit(branch, trunk, targets)

    training_cfg = stage_cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1.0e-3))
    patience = int(training_cfg.get("early_stopping_patience", epochs + 1))

    train_loader = _make_loader(train_ds, normalizer, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_ds, normalizer, batch_size=batch_size, shuffle=False)

    operator = str(stage_cfg.get("operator", "deeponet"))
    architecture = dict(stage_cfg.get("architecture", {}))
    architecture.setdefault("branch_input_dim", BRANCH_DIM)
    architecture.setdefault("trunk_input_dim", TRUNK_DIM)
    architecture.setdefault("output_dim", len(CHANNELS))

    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_operator(operator, architecture).to(active_device)
    resume_norm = _load_resume(model, resume_from or stage_cfg.get("resume_from"), active_device)
    if resume_norm is not None:
        normalizer = resume_norm
        train_loader = _make_loader(train_ds, normalizer, batch_size=batch_size, shuffle=True)
        val_loader = _make_loader(val_ds, normalizer, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = _scheduler(str(training_cfg.get("scheduler", "cosine")), optimizer, epochs)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, active_device)
        val_loss = _val_loss(model, val_loader, active_device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate_model(model, train_ds, normalizer, batch_size=batch_size, device=active_device),
        "val": evaluate_model(model, val_ds, normalizer, batch_size=batch_size, device=active_device),
        "history": history,
        "best_val_loss": best_val,
        "epochs_trained": len(history),
    }

    artifacts = stage_cfg.get("artifacts", {})
    checkpoint_path = _as_path(artifacts.get("checkpoint", f"checkpoints/{stage_key}_deeponet.pt"))
    metrics_path = _as_path(artifacts.get("metrics", f"checkpoints/{stage_key}_metrics.json"))
    _save_checkpoint(checkpoint_path, model, normalizer, stage_key, operator, architecture, metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "stage": stage_key,
        "dataset_path": str(dataset_path),
        "model_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def train_from_config(
    config_path: str | Path,
    stage: str = "a",
    resume_from: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train from either the compact local config or the full Phase 1 stage config."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    requested = stage.strip().lower()
    if not any(key in config for key in STAGE_KEYS.values()):
        return train_stage(config, stage_key="legacy", resume_from=resume_from, device=device)

    stage_names = list(STAGE_KEYS) if requested == "all" else [requested]
    if any(name not in STAGE_KEYS for name in stage_names):
        raise ValueError("stage must be one of: a, b, c, d, all")

    results = [
        train_stage(config, stage_key=STAGE_KEYS[name], resume_from=resume_from if index == 0 else None, device=device)
        for index, name in enumerate(stage_names)
    ]
    return results[-1] if len(results) == 1 else {"stages": results}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 1 WEC training."""

    parser = argparse.ArgumentParser(description="Train the NOSSO-MAR Phase 1 WEC DeepONet.")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML config.")
    parser.add_argument("--stage", default="a", choices=["a", "b", "c", "d", "all"], help="Training stage to run.")
    parser.add_argument("--resume-from", default=None, help="Optional PyTorch checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    args = parser.parse_args(argv)
    result = train_from_config(args.config, stage=args.stage, resume_from=args.resume_from, device=args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
