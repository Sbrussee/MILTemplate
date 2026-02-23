from __future__ import annotations

import torch

from src.utils.collate import mil_collate


def test_mil_collate_pads_variable_instances() -> None:
    batch = [
        {"h": torch.randn(3, 4), "y": 0, "attn_mask": torch.ones(3)},
        {"h": torch.randn(5, 4), "y": 1, "attn_mask": torch.ones(5)},
    ]
    h, y, mask = mil_collate(batch)
    assert h.shape == (2, 5, 4)
    assert y.shape == (2,)
    assert mask.shape == (2, 5)
    assert torch.all(mask[0, 3:] == 0)
