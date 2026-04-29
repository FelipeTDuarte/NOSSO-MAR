import pytest
import torch
from nossomar.core.contracts import WECState, WaveField, validate_wec_state


def test_wec_state_shape():
    s = WECState(pos=torch.zeros(3, 2), vel=torch.zeros(3, 6), force=torch.zeros(3, 6))
    assert s.pos.shape == (3, 2)


def test_wave_field_shape():
    wf = WaveField(
        eta=torch.zeros(2, 64, 64, 10),
        u=torch.zeros(2, 64, 64, 10),
        v=torch.zeros(2, 64, 64, 10),
    )
    assert wf.eta.shape == (2, 64, 64, 10)


def test_validate_rejects_nan():
    s = WECState(
        pos=torch.tensor([[float("nan"), 0.0]]),
        vel=torch.zeros(1, 6),
        force=torch.zeros(1, 6),
    )
    with pytest.raises(ValueError, match="NaN"):
        validate_wec_state(s)
