from __future__ import annotations

import torch

from nossomar.experiments.architecture_sweep import run_smoke_sweep
from nossomar.operators.factory import build_operator


def test_operator_smoke_sweep_covers_requested_families() -> None:
    results = run_smoke_sweep()
    assert {item.name for item in results} == {"fno2d", "ffno2d", "wno", "gno", "deeponet", "rino2d"}
    assert all(item.parameter_count > 0 for item in results)

