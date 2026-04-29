from __future__ import annotations

import torch

from nossomar.experiments.architecture_sweep import run_smoke_sweep
from nossomar.operators.factory import build_operator


def test_operator_smoke_sweep_covers_requested_families() -> None:
    results = run_smoke_sweep()
    assert {item.name for item in results} == {"fno2d", "wno", "gno", "deeponet", "rino2d"}
    assert all(item.parameter_count > 0 for item in results)


def test_build_operator_fno3d_forward_shape() -> None:
    model = build_operator(
        "fno3d",
        {
            "in_channels": 4,
            "out_channels": 2,
            "width": 12,
            "modes_x": 4,
            "modes_y": 4,
            "modes_t": 4,
            "n_layers": 2,
        },
    )
    output = model(torch.randn(2, 4, 8, 8, 6))
    assert output.shape == (2, 2, 8, 8, 6)
