from __future__ import annotations

import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.datamodule import MILDataModule
from src.data.dataset import BagPTDataset
from src.model.factory import create_model
from src.model.lightning import LitABMIL
from src.utils.config import load_config


def read_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def infer_in_dim(bag_dir: str, slide_id: str) -> int:
    item = torch.load(os.path.join(bag_dir, f"{slide_id}.pt"), map_location="cpu")
    return int(item["h"].shape[1])


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    L.seed_everything(int(cfg.train.seed), workers=True)

    bag_dir = os.path.join(cfg.paths.out_dir, "bags_pt")
    train_ids = read_ids(cfg.splits.train_ids)
    val_ids = read_ids(cfg.splits.val_ids)

    dm = MILDataModule(
        train_ds=BagPTDataset(bag_dir, train_ids),
        val_ds=BagPTDataset(bag_dir, val_ids),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
    )

    in_dim = (
        infer_in_dim(bag_dir, train_ids[0])
        if str(cfg.model.in_dim).lower() == "auto"
        else int(cfg.model.in_dim)
    )
    model = create_model(cfg.model, in_dim=in_dim)
    lit = LitABMIL(model=model, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    logger = TensorBoardLogger(save_dir=str(cfg.paths.log_dir), name=str(cfg.model.name))
    ckpt = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=1, filename="{epoch:02d}-{val_loss:.4f}"
    )

    trainer = L.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train a MIL model using bag tensors.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
