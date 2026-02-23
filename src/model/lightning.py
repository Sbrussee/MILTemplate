import torch
import torch.nn as nn
import lightning as L

class LitABMIL(L.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        h, y, attn_mask = batch
        results, _ = self.model(h=h, loss_fn=self.loss_fn, label=y, attn_mask=attn_mask, return_attention=False)
        loss = results["loss"]
        logits = results["logits"]

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=h.size(0))
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=h.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        h, y, attn_mask = batch
        results, _ = self.model(h=h, loss_fn=self.loss_fn, label=y, attn_mask=attn_mask, return_attention=False)
        loss = results["loss"]
        logits = results["logits"]

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=h.size(0))
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, batch_size=h.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)