import os

import anndata as ad
import pandas as pd
import torch

from src.utils.config import load_config


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    slides = pd.read_csv(cfg.paths.slides_csv)

    feat_dir = os.path.join(cfg.paths.out_dir, "features_anndata")
    bag_dir = os.path.join(cfg.paths.out_dir, "bags_pt")
    os.makedirs(bag_dir, exist_ok=True)

    max_M = cfg.bags.max_instances
    max_M = None if max_M in [None, "null"] else int(max_M)

    for _, row in slides.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"]) if "label" in row and not pd.isna(row["label"]) else None

        h5ad_path = os.path.join(feat_dir, f"{slide_id}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"[skip] missing features: {h5ad_path}")
            continue

        out_path = os.path.join(bag_dir, f"{slide_id}.pt")
        if os.path.exists(out_path):
            continue

        adata = ad.read_h5ad(h5ad_path)

        # Features typically in adata.X (could be sparse)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        h = torch.tensor(X, dtype=torch.float32)  # (M, D)

        M = h.shape[0]
        if max_M is not None and M > max_M:
            # Simple truncation; you can swap to random sample if preferred
            h = h[:max_M]
            M = max_M

        attn_mask = torch.ones(M, dtype=torch.float32)

        # Try to grab coords if LazySlide stored them (common keys: 'x', 'y' in obs)
        coords = None
        if cfg.bags.store_coords and "x" in adata.obs.columns and "y" in adata.obs.columns:
            coords = torch.tensor(adata.obs[["x", "y"]].values[:M], dtype=torch.float32)

        payload = {
            "slide_id": slide_id,
            "h": h,  # (M, D)
            "y": label,  # int or None
            "attn_mask": attn_mask,  # (M,)
        }
        if coords is not None:
            payload["coords"] = coords

        torch.save(payload, out_path)
        print(f"[bag] {slide_id}: M={M} D={h.shape[1]} -> {out_path}")


if __name__ == "__main__":
    main()
