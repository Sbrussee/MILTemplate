from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from src.model.mil import MIL


class LitABMIL(L.LightningModule):
    """Lightning wrapper for slide-level MIL classification."""

    def __init__(self, model: MIL, lr: float = 1e-4, weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        h, y, attn_mask = batch
        results, _ = self.model(h=h, loss_fn=self.loss_fn, label=y, attn_mask=attn_mask)
        loss = results["loss"]
        logits = results["logits"]
        assert loss is not None

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=h.size(0))
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=h.size(0))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        h, y, attn_mask = batch
        results, _ = self.model(h=h, loss_fn=self.loss_fn, label=y, attn_mask=attn_mask)
        loss = results["loss"]
        logits = results["logits"]
        assert loss is not None

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=h.size(0))
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, batch_size=h.size(0))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
