import pickle, torch
from pathlib import Path


def save_checkpoint(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")
