import torch
from src.nosso_mar.operators.gno.graph_neural_operator import GraphNeuralOperator
from src.nosso_mar.operators.gno.mesh_utils import build_radius_graph, build_knn_graph


def test_gno_forward():
    cfg = {"node_in_dim":8,"node_out_dim":4,"edge_dim":7,"hidden":32,"n_layers":2}
    N, E = 10, 20
    out = GraphNeuralOperator(cfg)(
        torch.randn(N,8), edge_index=torch.randint(0,N,(2,E)), edge_attr=torch.randn(E,7))
    assert out.shape == (N, 4)

def test_radius_graph():
    pos = torch.randn(20, 2) * 100
    ei  = build_radius_graph(pos, radius=300.0)
    assert ei.shape[0] == 2

def test_knn_graph():
    pos = torch.randn(15, 2)
    ei  = build_knn_graph(pos, k=4)
    assert ei.shape == (2, 15 * 4)
