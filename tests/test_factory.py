from __future__ import annotations

from src.model.abmil import ABMIL
from src.model.factory import create_model


def test_factory_builds_default_abmil() -> None:
    cfg = {
        "name": "abmil",
        "embed_dim": 16,
        "attn_dim": 8,
        "num_fc_layers": 1,
        "dropout": 0.1,
        "gate": True,
        "num_classes": 2,
    }
    model = create_model(cfg, in_dim=32)
    assert isinstance(model, ABMIL)
    assert model.in_dim == 32
