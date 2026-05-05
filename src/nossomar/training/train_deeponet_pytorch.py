"""Full end-to-end trainer for PyTorch DeepONet operator on WEC hydrodynamic response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nossomar.data.analytic_wec import branch_feature_vector_from_params, trunk_feature_matrix
from nossomar.data.wec_dataset import WECDataset, load_dataset_json, make_frequency_grid
from nossomar.operators.deeponet_wec import DeepONetWECRegressor


def prepare_torch_data(
    dataset: WECDataset,
    freq: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[DataLoader, int]:
    """Convert WEC dataset to PyTorch DataLoader with branch/trunk features.
    
    Returns:
        (dataloader, n_samples)
    """
    device_params_list = []
    freq_list = []
    target_list = []
    
    for record in dataset.records:
        params = record["device_params"]
        device_params = np.array(
            [params["radius"], params["draft"], params["mass"], params["bpto"], params["depth"]],
            dtype=float,
        )
        
        # Repeat device params for each frequency
        for _ in freq:
            device_params_list.append(device_params)
        
        freq_list.append(freq)
        
        # Collect targets: [A, B, Fex_real, Fex_imag] per frequency
        from nossomar.core.contracts import WECState
        state = WECState.from_dict(record["wec_state"])
        targets = np.column_stack([
            state.added_mass,
            state.damping,
            state.excitation_real,
            state.excitation_imag,
        ])
        target_list.append(targets)
    
    # Flatten
    X_params = np.vstack([device_params_list[i:i+len(freq)] for i in range(0, len(device_params_list), len(freq))])
    X_freq = np.concatenate(freq_list)
    Y = np.vstack(target_list)
    
    # Normalize
    X_params_mean, X_params_std = X_params.mean(axis=0), X_params.std(axis=0) + 1e-6
    X_freq_mean, X_freq_std = X_freq.mean(), X_freq.std() + 1e-6
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-6
    
    X_params_norm = (X_params - X_params_mean) / X_params_std
    X_freq_norm = (X_freq - X_freq_mean) / X_freq_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # Convert to torch
    X_params_torch = torch.from_numpy(X_params_norm).float().to(device)
    X_freq_torch = torch.from_numpy(X_freq_norm).float().to(device).unsqueeze(1)
    Y_torch = torch.from_numpy(Y_norm).float().to(device)
    
    dataset_torch = TensorDataset(X_params_torch, X_freq_torch, Y_torch)
    loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True)
    
    return loader, len(dataset), {
        "X_params_mean": X_params_mean,
        "X_params_std": X_params_std,
        "X_freq_mean": X_freq_mean,
        "X_freq_std": X_freq_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
    }


class DeepONetPyTorchTrainer:
    """Full trainer for PyTorch DeepONet on WEC response."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        for X_params, X_freq, Y in train_loader:
            self.optimizer.zero_grad()
            Y_pred = self.model(X_params, X_freq)
            loss = self.criterion(Y_pred, Y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_params, X_freq, Y in val_loader:
                Y_pred = self.model(X_params, X_freq)
                loss = self.criterion(Y_pred, Y)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> dict[str, Any]:
        """Train the model."""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history


def train_deeponet_pytorch(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs/deeponet_pytorch",
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """End-to-end training of PyTorch DeepONet.
    
    Args:
        dataset_path: Path to viana_50.json or phase1_wec_database.json
        output_dir: Directory to save model and metrics
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: "cpu" or "cuda"
    
    Returns:
        dict with model path, metrics, and training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    data = load_dataset_json(dataset_path)
    train_records = data["splits"]["train"]
    val_records = data["splits"]["val"]
    test_records = data["splits"]["test"]
    
    train_ds = WECDataset(train_records, split="train")
    val_ds = WECDataset(val_records, split="val")
    test_ds = WECDataset(test_records, split="test")
    
    freq = make_frequency_grid()
    
    # Prepare dataloaders
    train_loader, n_train, norm_params = prepare_torch_data(train_ds, freq, device, batch_size)
    val_loader, _, _ = prepare_torch_data(val_ds, freq, device, batch_size)
    test_loader, n_test, _ = prepare_torch_data(test_ds, freq, device, batch_size)
    
    print(f"Dataset: {n_train} train, {len(val_ds)} val, {n_test} test samples")
    
    # Create model
    model = DeepONetWECRegressor(branch_dim=256, trunk_dim=256, output_dim=4)
    trainer = DeepONetPyTorchTrainer(model, device=device, lr=1e-3)
    
    # Train
    print("Starting training...")
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Save model
    model_path = output_dir / "deeponet_pytorch.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save normalization params
    norm_path = output_dir / "norm_params.json"
    norm_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in norm_params.items()}
    norm_path.write_text(json.dumps(norm_dict, indent=2))
    
    # Evaluate on test
    test_loss = trainer.validate(test_loader)
    
    result = {
        "model_path": str(model_path),
        "norm_params_path": str(norm_path),
        "n_train": n_train,
        "n_val": len(val_ds),
        "n_test": n_test,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "test_loss": test_loss,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    
    print(f"Training complete. Test loss: {test_loss:.6f}")
    print(f"Model saved to: {model_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/phase1_wec_database.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = train_deeponet_pytorch(dataset_path, device=device, epochs=50, batch_size=32)
    print(f"Final metrics: {result['final_val_loss']:.6f}")
