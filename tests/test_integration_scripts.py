from __future__ import annotations

from pathlib import Path

import torch
import yaml

from scripts.eval import main as eval_main
from scripts.inference import run_inference
from scripts.save_model import save_packaged_model
from src.model.abmil import ABMIL
from src.model.lightning import LitABMIL


def _write_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train_ids.txt").write_text("slide_a\n", encoding="utf-8")
    (data_dir / "val_ids.txt").write_text("slide_a\n", encoding="utf-8")

    cfg = {
        "paths": {
            "slides_csv": "",
            "out_dir": str(tmp_path / "artifacts"),
            "log_dir": str(tmp_path / "logs"),
        },
        "preprocess": {"feature_model": "HOptimus0"},
        "bags": {},
        "train": {
            "seed": 42,
            "batch_size": 1,
            "num_workers": 0,
            "max_epochs": 1,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "accelerator": "cpu",
            "devices": 1,
        },
        "model": {
            "name": "abmil",
            "in_dim": "auto",
            "embed_dim": 8,
            "attn_dim": 4,
            "num_fc_layers": 1,
            "dropout": 0.1,
            "gate": True,
            "num_classes": 2,
        },
        "splits": {"train_ids": str(data_dir / "train_ids.txt"), "val_ids": str(data_dir / "val_ids.txt")},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_bag(tmp_path: Path) -> Path:
    bag_dir = tmp_path / "artifacts" / "bags_pt"
    bag_dir.mkdir(parents=True)
    bag_path = bag_dir / "slide_a.pt"
    torch.save({"slide_id": "slide_a", "h": torch.randn(6, 12), "y": 1, "attn_mask": torch.ones(6)}, bag_path)
    return bag_path


def _write_checkpoint(path: Path) -> None:
    model = ABMIL(in_dim=12, embed_dim=8, attn_dim=4, num_classes=2)
    lit = LitABMIL(model=model)
    torch.save({"state_dict": lit.state_dict()}, path)


def test_inference_eval_and_save_model(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    bag_path = _write_bag(tmp_path)
    ckpt_path = tmp_path / "model.ckpt"
    _write_checkpoint(ckpt_path)

    metrics = eval_main(str(cfg_path), str(ckpt_path), split="val")
    assert isinstance(metrics, list)

    pred = run_inference(str(cfg_path), str(bag_path), str(ckpt_path))
    assert 0 <= pred["confidence"] <= 1
    assert pred["pred_class"] in {0, 1}

    out_path = tmp_path / "package.pt"
    save_packaged_model(str(cfg_path), str(ckpt_path), str(out_path))
    assert out_path.exists()
    assert out_path.with_suffix(".json").exists()
