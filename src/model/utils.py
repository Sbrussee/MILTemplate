from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def create_mlp(
    in_dim: int,
    hid_dims: list[int],
    out_dim: int,
    act: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
    end_with_fc: bool = True,
    end_with_dropout: bool = False,
    bias: bool = True,
) -> nn.Module:
    """
    Minimal MLP builder used for patch embedding in ABMIL.
    """
    layers: list[nn.Module] = []

    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim

    layers.append(nn.Linear(in_dim, out_dim, bias=bias))

    if not end_with_fc:
        layers.append(act)
    if end_with_dropout:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# -----------------------------
# Attention modules
# -----------------------------
class GlobalAttention(nn.Module):
    """
    Attention Network without gating (2-layer MLP -> score).
    Input:  x: (B, M, L)
    Output: A: (B, M, K) where K = num_classes (usually 1 head)
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    """
    Gated attention as in Ilse et al.
    Input:  x: (B, M, L)
    Output: A: (B, M, K)
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        return self.attention_c(A)