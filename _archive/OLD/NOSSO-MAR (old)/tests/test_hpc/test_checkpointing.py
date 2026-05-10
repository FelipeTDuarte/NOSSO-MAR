import torch, tempfile
from src.nosso_mar.hpc.checkpointing import HybridCheckpointer


def test_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        ck = HybridCheckpointer({"local_dir":tmp,"keep_last_n":3})
        ck.save({"epoch":5,"w":torch.randn(4)}, epoch=5)
        loaded = ck.load_latest()
        assert loaded["epoch"] == 5
