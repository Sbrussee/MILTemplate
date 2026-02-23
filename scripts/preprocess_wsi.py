import os
import pandas as pd
import torch

from omegaconf import OmegaConf
from src.utils.config import load_config

from wsidata import open_wsi
import lazyslide as zs

def is_mps() -> bool:
    return torch.backends.mps.is_available()

def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    slides = pd.read_csv(cfg.paths.slides_csv)

    feat_dir = os.path.join(cfg.paths.out_dir, "features_anndata")
    os.makedirs(feat_dir, exist_ok=True)

    for _, row in slides.iterrows():
        slide_id = str(row["slide_id"])
        slide_path = str(row["slide_path"])

        out_path = os.path.join(feat_dir, f"{slide_id}.h5ad")
        if os.path.exists(out_path) and not cfg.preprocess.overwrite:
            print(f"[skip] {slide_id} exists: {out_path}")
            continue

        print(f"[wsi] opening {slide_id}: {slide_path}")
        wsi = open_wsi(slide_path)

        # Optional: save tissue overview figure
        if cfg.preprocess.thumbnail_plot:
            try:
                fig = zs.pl.tissue(wsi)
                fig_path = os.path.join(cfg.paths.out_dir, "tissue_plots")
                os.makedirs(fig_path, exist_ok=True)
                fig.savefig(os.path.join(fig_path, f"{slide_id}.png"), dpi=150, bbox_inches="tight")
            except Exception as e:
                print(f"[warn] tissue plot failed for {slide_id}: {e}")

        if cfg.preprocess.find_tissues:
            zs.pp.find_tissues(wsi)

        zs.pp.tile_tissues(wsi, int(cfg.preprocess.tile_size_px))

        # AMP handling
        amp = bool(cfg.preprocess.amp)
        if cfg.preprocess.force_no_amp_on_mps and is_mps():
            amp = False

        print(f"[feat] extracting features model={cfg.preprocess.feature_model} amp={amp}")
        zs.tl.feature_extraction(wsi, cfg.preprocess.feature_model, amp=amp)

        # Convention: "{model}_{tiles key}" -> e.g. "plip_tiles"
        features_key = f"{cfg.preprocess.feature_model}_{cfg.preprocess.tiles_key}"

        # Export to .h5ad for deterministic downstream processing
        # LazySlide stores features as AnnData; exact accessor depends on version.
        # Common pattern: wsi.tables[features_key] or wsi[features_key] or wsi.sdata.tables[...]
        adata = None
        for attr_path in [
            ("tables", features_key),
            ("sdata", "tables", features_key),
        ]:
            try:
                obj = wsi
                for a in attr_path:
                    obj = obj[a] if isinstance(obj, dict) else getattr(obj, a)
                adata = obj
                break
            except Exception:
                pass

        if adata is None:
            raise RuntimeError(
                f"Could not locate AnnData features '{features_key}' in WSI object. "
                f"Inspect wsi to find where LazySlide stored it in your environment."
            )

        print(f"[save] {slide_id} -> {out_path}")
        adata.write_h5ad(out_path)

if __name__ == "__main__":
    main()