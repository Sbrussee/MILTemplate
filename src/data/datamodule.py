from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from src.utils.collate import mil_collate


class MILDataModule(L.LightningDataModule):
    """Lightning datamodule for variable-length MIL bag training."""

    def __init__(self, train_ds, val_ds, batch_size: int = 4, num_workers: int = 4) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=mil_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=mil_collate,
            pin_memory=True,
        )
