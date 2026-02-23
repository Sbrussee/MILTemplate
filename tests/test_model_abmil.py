from __future__ import annotations

import torch

from src.model.abmil import ABMIL


def test_abmil_forward_shapes() -> None:
    model = ABMIL(in_dim=8, embed_dim=4, attn_dim=3, num_classes=2)
    h = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)

    results, log_dict = model(h=h, attn_mask=mask, return_attention=True, return_slide_feats=True)

    assert results["logits"].shape == (2, 2)
    assert log_dict["attention"].shape == (2, 1, 5)
    assert log_dict["slide_feats"].shape == (2, 4)
    assert torch.isfinite(results["logits"]).all()
