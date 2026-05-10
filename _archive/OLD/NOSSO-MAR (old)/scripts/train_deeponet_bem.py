"""Train DeepONet for Module 2 (BEM surrogate / WEC FSI).

Usage:
    python scripts/train_deeponet_bem.py --config configs/training/deeponet_bem.yaml
"""
import argparse, yaml, logging, torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.nosso_mar.operators.deeponet.deeponet import DeepONet
    from src.nosso_mar.training import NOTrainer
    from src.nosso_mar.training.losses.bem_loss import BEMLoss
    from src.nosso_mar.data import CapytaineDataset
    from torch.utils.data import DataLoader, random_split

    model   = DeepONet(cfg.get("model_cfg", {}))
    dataset = CapytaineDataset(cfg["data_dir"])
    n_val   = max(1, int(0.1 * len(dataset)))
    tr, val = random_split(dataset, [len(dataset) - n_val, n_val])

    def collate(batch):
        import torch
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    tr_ldr  = DataLoader(tr,  batch_size=cfg.get("batch_size",32), shuffle=True,  collate_fn=collate)
    val_ldr = DataLoader(val, batch_size=cfg.get("batch_size",32), shuffle=False, collate_fn=collate)
    bem_loss = BEMLoss()

    def loss_fn(m, b):
        out = m(b["branch_input"], b["omega"].unsqueeze(-1))  # (B, Q, 4)
        pred = {"added_mass": out[...,0], "radiation_damping": out[...,1],
                "excitation_force": out[...,2], "rao": out[...,3]}
        tgt  = {k: b[k] for k in ["added_mass","radiation_damping","excitation_force","rao"]}
        return bem_loss(pred, tgt)

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.get("lr",1e-3))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    from src.nosso_mar.training.callbacks import EarlyStopping
    trainer = NOTrainer(model, loss_fn, opt, cfg=cfg)
    trainer.train(tr_ldr, val_ldr, n_epochs=cfg.get("epochs",300),
                  callbacks=[EarlyStopping(patience=40)])

    Path(cfg.get("output_dir","outputs")).mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"{cfg.get('output_dir','outputs')}/bem_surrogate_final.pt")
    logger.info("BEM surrogate training complete.")


if __name__ == "__main__":
    main()
