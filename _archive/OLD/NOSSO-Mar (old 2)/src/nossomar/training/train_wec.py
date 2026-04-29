import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..operators.deeponet_wec import WECDeepONet


def train_wec_surrogate(ds, epochs=100, lr=3e-4, batch_size=64, device="cpu"):
    props = torch.tensor(
        np.stack(
            [
                ds["radius"].values,
                ds["draft"].values,
                ds["depth"].values,
                ds["bpto"].values,
            ],
            axis=1,
        ),
        dtype=torch.float32,
    )
    omega = torch.tensor(ds["omega"].values, dtype=torch.float32)
    A_tgt = torch.tensor(ds["added_mass"].values, dtype=torch.float32)
    B_tgt = torch.tensor(ds["damping"].values, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(props, A_tgt, B_tgt),
        batch_size=batch_size,
        shuffle=True,
    )
    model = WECDeepONet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []

    for _ in range(epochs):
        losses = []
        for p, a, b in loader:
            p, a, b = p.to(device), a.to(device), b.to(device)
            om = omega.unsqueeze(0).repeat(p.shape[0], 1).to(device)
            A_pred, B_pred = model(p, om)
            loss = ((A_pred - a) ** 2).mean() + ((B_pred - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history
