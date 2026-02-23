from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model.factory import create_model
from src.utils.config import load_config


def run_inference(config_path: str, bag_path: str, checkpoint_path: str) -> dict[str, float | int]:
    """Run single-bag inference.

    Args:
        config_path: Path to config YAML.
        bag_path: Path to serialized bag ``.pt`` containing ``h`` and optional ``attn_mask``.
        checkpoint_path: Lightning checkpoint path.

    Returns:
        Dict with predicted class id and confidence.
    """

    cfg = load_config(config_path)
    payload = torch.load(bag_path, map_location="cpu")
    h = payload["h"].unsqueeze(0)
    attn_mask = payload.get("attn_mask", torch.ones(h.shape[:2], dtype=torch.float32)).unsqueeze(0)

    model = create_model(cfg.model, in_dim=int(h.shape[-1]))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k.replace("model.", ""): v for k, v in state["state_dict"].items()}, strict=False)
    model.eval()

    with torch.no_grad():
        results, _ = model(h=h, attn_mask=attn_mask)
        probs = torch.softmax(results["logits"], dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

    return {"pred_class": pred, "confidence": conf}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a single MIL bag.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    result = run_inference(args.config, args.bag, args.ckpt)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
