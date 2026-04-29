"""Lightweight architecture sweep for NOSSO-MAR neural operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from nossomar.operators.factory import build_operator
from nossomar.operators.gno.mesh_utils import build_knn_graph, compute_edge_features


@dataclass(slots=True)
class SweepResult:
    """Summary of one smoke-tested operator configuration."""

    name: str
    output_shape: tuple[int, ...]
    parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def operator_sweep_configs() -> dict[str, dict[str, int | str]]:
    """Compact configurations for quick architecture validation."""

    return {
        "fno2d": {
            "in_channels": 5,
            "out_channels": 3,
            "width": 24,
            "modes_x": 8,
            "modes_y": 8,
            "n_layers": 2,
        },
        "wno": {
            "in_channels": 5,
            "out_channels": 3,
            "width": 24,
            "levels": 2,
            "n_layers": 2,
        },
        "gno": {
            "node_in_dim": 6,
            "node_out_dim": 3,
            "hidden": 32,
            "n_layers": 2,
        },
        "deeponet": {
            "branch_input_dim": 8,
            "trunk_input_dim": 2,
            "hidden_dim": 32,
            "n_hidden": 2,
            "output_dim": 3,
            "p": 24,
        },
        "rino2d": {
            "in_channels": 5,
            "out_channels": 3,
            "latent_channels": 24,
            "n_layers": 3,
            "decoder_hidden": 48,
        },
    }


def run_smoke_sweep() -> list[SweepResult]:
    """Instantiate key operator families and run one forward pass each."""

    results: list[SweepResult] = []
    cfgs = operator_sweep_configs()

    grid_input = torch.randn(2, 5, 16, 16)
    query_points = torch.rand(2, 12, 2) * 2.0 - 1.0

    fno2d = build_operator("fno2d", cfgs["fno2d"])
    fno_out = fno2d(grid_input)
    results.append(SweepResult("fno2d", tuple(fno_out.shape), fno2d.count_parameters()))

    wno = build_operator("wno", cfgs["wno"])
    wno_out = wno(grid_input)
    results.append(SweepResult("wno", tuple(wno_out.shape), wno.count_parameters()))

    node_pos = torch.rand(20, 2)
    edge_index = build_knn_graph(node_pos, k=4)
    edge_attr = compute_edge_features(node_pos, edge_index)
    node_features = torch.randn(20, 6)
    gno = build_operator("gno", cfgs["gno"])
    gno_out = gno(node_features, edge_index=edge_index, edge_attr=edge_attr)
    results.append(SweepResult("gno", tuple(gno_out.shape), gno.count_parameters()))

    deeponet = build_operator("deeponet", cfgs["deeponet"])
    branch_input = torch.randn(2, 8)
    deep_out = deeponet(branch_input, query_points=query_points)
    results.append(
        SweepResult("deeponet", tuple(deep_out.shape), deeponet.count_parameters())
    )

    rino = build_operator("rino2d", cfgs["rino2d"])
    rino_out = rino(grid_input, query_points=query_points)
    results.append(SweepResult("rino2d", tuple(rino_out.shape), rino.count_parameters()))

    return results
