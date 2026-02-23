import os
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.config import load_config
from src.data.bag_dataset import BagPTDataset
from src.data.datamodule import MILDataModule
from src.lightning.lit_abmil import LitABMIL

# import your ABMIL
from src.models.abmil import ABMIL

def read_ids(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def infer_in_dim(bag_dir: str, slide_id: str) -> int:
    item = torch.load(os.path.join(bag_dir, f"{slide_id}.pt"), map_location="cpu")
    return int(item["h"].shape[1])

def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    L.seed_everything(int(cfg.train.seed), workers=True)

    bag_dir = os.path.join(cfg.paths.out_dir, "bags_pt")

    train_ids = read_ids(cfg.splits.train_ids)
    val_ids = read_ids(cfg.splits.val_ids)

    train_ds = BagPTDataset(bag_dir, train_ids)
    val_ds = BagPTDataset(bag_dir, val_ids)
    dm = MILDataModule(train_ds, val_ds, batch_size=int(cfg.train.batch_size), num_workers=int(cfg.train.num_workers))

    in_dim = cfg.model.in_dim
    if str(in_dim).lower() == "auto":
        in_dim = infer_in_dim(bag_dir, train_ids[0])

    model = ABMIL(
        in_dim=int(in_dim),
        embed_dim=int(cfg.model.embed_dim),
        num_fc_layers=int(cfg.model.num_fc_layers),
        dropout=float(cfg.model.dropout),
        attn_dim=int(cfg.model.attn_dim),
        gate=bool(cfg.model.gate),
        num_classes=int(cfg.model.num_classes),
    )

    lit = LitABMIL(model=model, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name="abmil")

    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="abmil-{epoch:02d}-{val_loss:.4f}")

    trainer = L.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
    )

    trainer.fit(lit, datamodule=dm)
    print("Best checkpoint:", ckpt.best_model_path)

if __name__ == "__main__":
    main()