from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mil import MIL
from src.model.utils import GlobalAttention, GlobalGatedAttention, create_mlp


class ABMIL(MIL):
    """Attention-based MIL (Ilse et al.).

    Input tensor contract:
        * ``h`` shape: ``(B, M, D)`` float32/float16.
        * ``attn_mask`` shape: ``(B, M)`` with values in ``{0, 1}``.

    Output:
        * logits shape: ``(B, C)``.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        attn_dim: int = 384,
        gate: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * max(0, num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_cls = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_cls(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_only: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert h.ndim == 3, f"Expected h as (B,M,D), got {tuple(h.shape)}"
        h_embed = self.patch_embed(h)
        attn_logits = self.global_attn(h_embed).transpose(-2, -1)

        if attn_mask is not None:
            assert attn_mask.shape == h.shape[:2], "attn_mask must have shape (B, M)"
            attn_logits = (
                attn_logits
                + (1.0 - attn_mask.to(attn_logits.dtype)).unsqueeze(1) * torch.finfo(attn_logits.dtype).min
            )

        if attn_only:
            return attn_logits
        return h_embed, attn_logits

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        h_embed, attn_logits = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        attn = F.softmax(attn_logits, dim=-1)
        slide_feats = torch.bmm(attn, h_embed).squeeze(1)
        return slide_feats, {"attention": attn_logits if return_attention else None}

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        assert h.ndim == 2, "Expected slide feature tensor with shape (B, E)"
        return self.classifier(h)

    def forward(
        self,
        h: torch.Tensor,
        loss_fn: nn.Module | None = None,
        label: torch.LongTensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
        slide_feats, log_dict = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention
        )
        logits = self.forward_head(slide_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        log_dict["loss"] = float(cls_loss.item()) if cls_loss is not None else -1.0
        if return_slide_feats:
            log_dict["slide_feats"] = slide_feats
        return {"logits": logits, "loss": cls_loss}, log_dict
