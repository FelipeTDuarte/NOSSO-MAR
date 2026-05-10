import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WECDeepONet(nn.Module):
    def __init__(self, branch_dim=4, trunk_dim=1, hidden=128, n_modes=64):
        super().__init__()
        self.branch_A = _MLP([branch_dim, hidden, hidden, n_modes])
        self.branch_B = _MLP([branch_dim, hidden, hidden, n_modes])
        self.trunk = _MLP([trunk_dim, hidden, hidden, n_modes])
        self.bias_A = nn.Parameter(torch.zeros(1))
        self.bias_B = nn.Parameter(torch.zeros(1))

    def forward(self, props, omega):
        b_A = self.branch_A(props)
        b_B = self.branch_B(props)
        t = self.trunk(omega.unsqueeze(-1))
        A = (b_A.unsqueeze(1) * t).sum(-1) + self.bias_A
        B = (b_B.unsqueeze(1) * t).sum(-1) + self.bias_B
        return A, torch.relu(B)
