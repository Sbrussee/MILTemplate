from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class MIL(ABC, nn.Module):
    """Abstract base class for MIL models.

    Args:
        in_dim: Input feature dimensionality ``D`` for each instance.
        embed_dim: Embedding dimensionality after instance projection.
        num_classes: Number of output classes.
    """

    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

    @abstractmethod
    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_only: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute attention logits.

        Args:
            h: Tensor of shape ``(B, M, D)``.
            attn_mask: Optional mask of shape ``(B, M)`` where ``1`` denotes valid instances.
            attn_only: Return only attention logits when ``True``.
        """

    @abstractmethod
    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute slide-level pooled representation."""

    @abstractmethod
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """Compute classification logits from slide-level embeddings."""

    @staticmethod
    def compute_loss(
        loss_fn: nn.Module | None,
        logits: torch.Tensor | None,
        label: torch.LongTensor | None,
    ) -> torch.Tensor | None:
        """Compute optional classification loss."""
        if loss_fn is None or logits is None or label is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self) -> None:
        """Initialize common layer types with robust defaults."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
