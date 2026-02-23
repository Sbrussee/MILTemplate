from __future__ import annotations

import argparse
import os

import lightning as L
import torch

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


def main(config_path: str = "config.yaml", ckpt_path: str | None = None, split: str = "val") -> list[dict]:
    cfg = load_config(config_path)
    bag_dir = os.path.join(cfg.paths.out_dir, "bags_pt")

    ids_file = cfg.splits.val_ids if split == "val" else cfg.splits.train_ids
    ds = BagPTDataset(bag_dir, read_ids(ids_file))
    dm = MILDataModule(
        train_ds=ds, val_ds=ds, batch_size=int(cfg.train.batch_size), num_workers=int(cfg.train.num_workers)
    )

    in_dim = (
        infer_in_dim(bag_dir, ds.slide_ids[0])
        if str(cfg.model.in_dim).lower() == "auto"
        else int(cfg.model.in_dim)
    )
    model = create_model(cfg.model, in_dim=in_dim)
    lit = LitABMIL(model=model)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        lit.load_state_dict(state["state_dict"], strict=False)

    trainer = L.Trainer(accelerator=str(cfg.train.accelerator), devices=cfg.train.devices)
    return trainer.validate(lit, dataloaders=dm.val_dataloader())


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MIL model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None, help="Path to Lightning checkpoint.")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    args = parser.parse_args()
    main(config_path=args.config, ckpt_path=args.ckpt, split=args.split)


if __name__ == "__main__":
    _cli()
