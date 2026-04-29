import pytest
import torch
import torch.nn.functional as F
from nossomar.operators.rino_encoder import RINOEncoder


@pytest.fixture
def encoder():
    return RINOEncoder(in_features=4, d_latent=64)


def test_output_shape(encoder):
    pts = torch.randn(4, 128, 3)
    feat = torch.randn(4, 128, 4)
    z = encoder(pts, feat)
    assert z.shape == (4, 64), f"Expected (4, 64), got {z.shape}"


def test_no_nan(encoder):
    pts = torch.randn(2, 100, 3)
    feat = torch.randn(2, 100, 4)
    z = encoder(pts, feat)
    assert not torch.isnan(z).any(), "NaN detected in encoder output"


def test_resolution_invariance():
    """Same geometry at N=64 and N=256 should give cosine similarity > 0.5."""
    torch.manual_seed(42)
    enc = RINOEncoder(in_features=4, d_latent=64)
    enc.eval()
    # Build a low-res cloud and a high-res cloud from the same base points
    pts_lo = torch.randn(1, 64, 3)
    feat_lo = torch.randn(1, 64, 4)
    # High-res: tile and add small jitter to simulate denser sampling
    pts_hi = pts_lo.repeat(1, 4, 1) + torch.randn(1, 256, 3) * 0.01
    feat_hi = feat_lo.repeat(1, 4, 1) + torch.randn(1, 256, 4) * 0.01
    with torch.no_grad():
        z_lo = enc(pts_lo, feat_lo)
        z_hi = enc(pts_hi, feat_hi)
    sim = F.cosine_similarity(z_lo, z_hi).item()
    assert sim > 0.5, f"Resolution invariance failed: cosine_sim={sim:.4f}"


def test_batch_independence(encoder):
    """Each item in a batch should be processed independently."""
    torch.manual_seed(0)
    pts = torch.randn(3, 64, 3)
    feat = torch.randn(3, 64, 4)
    z_batch = encoder(pts, feat)  # (3, 64)
    z_single = encoder(pts[1:2], feat[1:2])  # (1, 64)
    assert torch.allclose(z_batch[1], z_single[0], atol=1e-5), \
        "Batch item differs from single-item forward pass"


def test_variable_point_count(encoder):
    """Encoder must accept any N_pts without architecture changes."""
    for n in [16, 64, 512]:
        pts = torch.randn(2, n, 3)
        feat = torch.randn(2, n, 4)
        z = encoder(pts, feat)
        assert z.shape == (2, 64), f"Failed for N_pts={n}"
