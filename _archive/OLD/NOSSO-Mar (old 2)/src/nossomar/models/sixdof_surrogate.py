"""6-DOF hydrodynamic coefficient surrogate model.

Outputs added-mass (A) and radiation-damping (B) matrices
for a floating body. Both are real symmetric 6x6 matrices
per the Haskind relations from linear potential flow theory.

DOF ordering (WAMIT/OpenFOAM convention):
    0: surge  (x)   3: roll  (Rx)
    1: sway   (y)   4: pitch (Ry)
    2: heave  (z)   5: yaw   (Rz)

Symmetry is enforced structurally:
    L = lower-triangular output from MLP
    M = (L + L^T) / 2  => symmetric by construction

Non-negativity on diagonals is enforced via softplus activation
before placing values on the diagonal of L.

Note: a second symmetrisation pass is applied after the frequency
averaging step to eliminate any float32 accumulation asymmetry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _SymmetricHead(nn.Module):
    """Maps a feature vector to a symmetric (6,6) matrix.

    Strategy:
        1. MLP -> 21 values  (lower-triangular entries of a 6x6 matrix)
        2. Diagonal entries passed through softplus  (non-negative)
        3. Build lower-triangular L, then M = (L + L^T) / 2
    """

    N_DOF = 6
    N_LOWER = N_DOF * (N_DOF + 1) // 2  # 21

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.SiLU(),
            nn.Linear(d_in, self.N_LOWER),
        )
        # indices for lower-triangular fill
        idx = torch.tril_indices(self.N_DOF, self.N_DOF)
        self.register_buffer("row_idx", idx[0])
        self.register_buffer("col_idx", idx[1])
        diag_mask = idx[0] == idx[1]
        self.register_buffer("diag_mask", diag_mask)

    def forward(self, x: Tensor) -> Tensor:  # x: (N, d_in)
        vals = self.mlp(x)                   # (N, 21)

        # softplus on diagonal entries -> non-negative
        vals = vals.clone()
        vals[:, self.diag_mask] = torch.nn.functional.softplus(
            vals[:, self.diag_mask]
        )

        N = x.shape[0]
        L = torch.zeros(N, self.N_DOF, self.N_DOF,
                        device=x.device, dtype=x.dtype)
        L[:, self.row_idx, self.col_idx] = vals

        # symmetrise: M = (L + L^T) / 2
        M = (L + L.transpose(-2, -1)) / 2.0
        return M                             # (N, 6, 6)


def _symmetrise(M: Tensor) -> Tensor:
    """Enforce exact symmetry: M = (M + M^T) / 2."""
    return (M + M.transpose(-2, -1)) / 2.0


class SixDOFSurrogate(nn.Module):
    """Surrogate for 6-DOF hydrodynamic coefficient matrices.

    Args:
        d_hidden: Width of the shared encoder hidden layers.

    Inputs:
        props : Tensor (N, 4)   WEC property vector [radius, draft, depth, Bpto]
        omega : Tensor (N, K)   Frequency query points (rad/s)

    Returns:
        A_mat : Tensor (N, 6, 6)  Added-mass matrix
        B_mat : Tensor (N, 6, 6)  Radiation-damping matrix
    """

    def __init__(self, d_hidden: int = 64) -> None:
        super().__init__()
        d_props = 4
        d_omega = 1
        d_in = d_props + d_omega

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        self.head_A = _SymmetricHead(d_hidden)
        self.head_B = _SymmetricHead(d_hidden)

    def forward(self, props: Tensor, omega: Tensor) -> tuple[Tensor, Tensor]:
        # props: (N, 4),  omega: (N, K)
        N, K = omega.shape

        # repeat props K times -> (N*K, 4)
        props_rep = props.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)
        omega_rep = omega.reshape(N * K, 1)

        x = torch.cat([props_rep, omega_rep], dim=-1)  # (N*K, 5)
        feat = self.encoder(x)                          # (N*K, d_hidden)

        A_full = self.head_A(feat)   # (N*K, 6, 6)
        B_full = self.head_B(feat)   # (N*K, 6, 6)

        # average over K frequency samples -> (N, 6, 6)
        # re-symmetrise after averaging to remove float32 accumulation drift
        A_mat = _symmetrise(A_full.view(N, K, 6, 6).mean(dim=1))
        B_mat = _symmetrise(B_full.view(N, K, 6, 6).mean(dim=1))

        return A_mat, B_mat
