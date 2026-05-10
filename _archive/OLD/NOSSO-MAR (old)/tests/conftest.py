import pytest, torch, numpy as np


@pytest.fixture
def small_fno_cfg():
    return {"in_channels":3,"out_channels":2,"width":16,"modes_x":4,"modes_y":4,"n_layers":2}

@pytest.fixture
def small_wno_cfg():
    return {"in_channels":3,"out_channels":2,"width":16,"levels":2,"n_layers":2}

@pytest.fixture
def small_deeponet_cfg():
    return {"branch_input_dim":7,"trunk_input_dim":1,"hidden_dim":32,"n_hidden":2,"output_dim":4,"p":32}

@pytest.fixture
def dummy_batch_2d():
    return torch.randn(2, 3, 32, 32)

@pytest.fixture
def dummy_omega():
    return np.linspace(0.2, 3.0, 32)
