from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_ACTIVATION = nn.ReLU()


def create_mlp(
    in_dim: int,
    hid_dims: list[int],
    out_dim: int,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
    end_with_dropout: bool = False,
    bias: bool = True,
) -> nn.Module:
    """Create a simple MLP stack used in ABMIL patch embedding."""
    layers: list[nn.Module] = []
    activation = act if act is not None else DEFAULT_ACTIVATION

    for hid_dim in hid_dims:
        layers.extend([nn.Linear(in_dim, hid_dim, bias=bias), activation, nn.Dropout(dropout)])
        in_dim = hid_dim

    layers.append(nn.Linear(in_dim, out_dim, bias=bias))
    if not end_with_fc:
        layers.append(activation)
    if end_with_dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GlobalAttention(nn.Module):
    """Non-gated attention network.

    Input ``x`` has shape ``(B, M, L)`` and output has shape ``(B, M, K)``.
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1) -> None:
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
    """Gated attention network from Ilse et al."""

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_c(self.attention_a(x) * self.attention_b(x))
