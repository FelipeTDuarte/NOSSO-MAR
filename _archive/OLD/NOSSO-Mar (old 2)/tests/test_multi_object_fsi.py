import torch
from nossomar.modules.multi_object_fsi import MultiObjectFSI
from nossomar.core.contracts import WECState, WaveField


def test_multi_wec_no_nan():
    fsi = MultiObjectFSI(n_max_objects=5, hidden=32)
    wf = WaveField(
        eta=torch.randn(2, 64, 64, 8),
        u=torch.randn(2, 64, 64, 8),
        v=torch.randn(2, 64, 64, 8),
    )
    state = WECState(
        pos=torch.rand(3, 2) * 100,
        vel=torch.zeros(3, 6),
        force=torch.zeros(3, 6),
    )
    out_state, force_field = fsi(wf, state)
    assert not torch.isnan(out_state.force).any()
    assert force_field.shape == (2, 3, 64, 64)
