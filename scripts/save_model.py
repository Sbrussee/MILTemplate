from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.utils.config import load_config


def save_packaged_model(config_path: str, checkpoint_path: str, output_path: str) -> None:
    """Save a portable model package with metadata.

    The package is a single ``.pt`` file containing model weights and config.
    """

    cfg = load_config(config_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = {
        k.replace("model.", ""): v for k, v in state["state_dict"].items() if k.startswith("model.")
    }

    package = {
        "format_version": 1,
        "model_name": str(cfg.model.name),
        "model_config": dict(cfg.model),
        "state_dict": model_state,
        "source_checkpoint": checkpoint_path,
    }
    torch.save(package, output_path)

    metadata_path = Path(output_path).with_suffix(".json")
    metadata_path.write_text(
        json.dumps({k: v for k, v in package.items() if k != "state_dict"}, indent=2), encoding="utf-8"
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Create a portable packaged model artifact.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    save_packaged_model(args.config, args.ckpt, args.out)


if __name__ == "__main__":
    _cli()
