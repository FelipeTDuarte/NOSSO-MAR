"""Train FNO2d / WNO for Module 1 (wave propagation).

Usage:
    python scripts/train_fno.py --config configs/training/fno_wave.yaml
    # or distributed:
    torchrun --nproc_per_node=4 scripts/train_fno.py --config configs/training/fno_wave.yaml --distributed
"""
import argparse
import logging
import yaml
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.nosso_mar.operators import WaveletNeuralOperator, FNO2d
    from src.nosso_mar.training import NOTrainer
    from src.nosso_mar.training.losses.wave_loss import WaveLoss
    from src.nosso_mar.data import OceanWave3DDataset
    from torch.utils.data import DataLoader, random_split

    op_type = cfg.get("operator_type", "wno")
    op_cfg  = cfg.get("operator_cfg",  {})
    model   = WaveletNeuralOperator(op_cfg) if op_type == "wno" else FNO2d(op_cfg)
    logger.info(f"Model: {model.summary()}")

    dataset  = OceanWave3DDataset(cfg["data_dir"])
    n_val    = max(1, int(0.1 * len(dataset)))
    tr, val  = random_split(dataset, [len(dataset) - n_val, n_val])
    tr_ldr   = DataLoader(tr,  batch_size=cfg.get("batch_size", 8), shuffle=True,  num_workers=4)
    val_ldr  = DataLoader(val, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=4)

    loss_fn  = lambda m, b: WaveLoss()(m(b["input"]), b["target"])
    opt      = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("wd", 1e-4))
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs", 200))

    from src.nosso_mar.training.callbacks import EarlyStopping, WandBLogger
    callbacks = [EarlyStopping(patience=30), WandBLogger(project="nosso-mar-wave")]

    trainer = NOTrainer(model, loss_fn, opt, sched, cfg)
    trainer.train(tr_ldr, val_ldr, n_epochs=cfg.get("epochs", 200), callbacks=callbacks)

    Path(cfg.get("output_dir", "outputs")).mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"{cfg.get('output_dir','outputs')}/wave_no_final.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
