# Minimal WSI → MIL Template

A clean and extensible computational pathology pipeline for:

1. WSI preprocessing + tiling with **LazySlide**
2. Tile feature extraction (default: **HOptimus0**)
3. MIL bag creation
4. Model training/evaluation/inference (default: **ABMIL**)

For richer feature/model options, see:
- LazySlide: https://github.com/rendeirolab/lazyslide
- MIL-Lab: https://github.com/mahmoodlab/MIL-Lab

## Installation

### 1) Create environment
```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2) Prepare metadata
Create `data/slides.csv`:

```csv
slide_id,slide_path,label
SLIDE_001,/path/to/slide1.svs,0
SLIDE_002,/path/to/slide2.svs,1
```

Create split files:
- `data/train_ids.txt`
- `data/val_ids.txt`

(one `slide_id` per line)

## Step-by-step usage

### Step 1: preprocess WSIs
```bash
python scripts/preprocess_wsi.py --config config.yaml
```
Outputs: `artifacts/features_anndata/{slide_id}.h5ad`

### Step 2: build MIL bags
```bash
python scripts/make_bags.py --config config.yaml
```
Outputs: `artifacts/bags_pt/{slide_id}.pt`

Expected bag schema:
- `slide_id: str`
- `h: Tensor[M, D]` (instance features)
- `y: int`
- `attn_mask: Tensor[M]` (1=valid)
- optional `coords: Tensor[M, 2]`

### Step 3: train
```bash
python scripts/train.py --config config.yaml
```

### Step 4: evaluate
```bash
python scripts/eval.py --config config.yaml --ckpt path/to/model.ckpt --split val
```

### Step 5: inference
```bash
python scripts/inference.py --config config.yaml --bag artifacts/bags_pt/SLIDE_001.pt --ckpt path/to/model.ckpt
```

### Step 6: package model for sharing
```bash
python scripts/save_model.py --config config.yaml --ckpt path/to/model.ckpt --out artifacts/model_package.pt
```
This creates:
- packaged weights/config `.pt`
- companion metadata `.json`

## Configuration guide

All behavior is controlled in `config.yaml`.

### Model selection
- Default model is ABMIL:
  ```yaml
  model:
    name: abmil
  ```
- You can also specify any importable model class:
  ```yaml
  model:
    name: my_package.my_module:MyMILModel
  ```
  It must implement the same MIL interface.

### Feature extractor selection
- Default extractor is:
  ```yaml
  preprocess:
    feature_model: HOptimus0
  ```
- You can set any LazySlide-supported extractor name.

## Extension instructions

### Add a new MIL model
1. Implement a class under `src/model/` inheriting `MIL`.
2. Register short-name in `src/model/factory.py` or use full import path in config.
3. Add tests in `tests/test_model_*.py`.

### Add custom preprocessing/feature extraction
1. Keep I/O schema (`.h5ad` → `.pt`) unchanged for compatibility.
2. Add extractor option to `config.yaml`.
3. Verify with integration tests (`pytest -q`).

## Testing
```bash
ruff check . --fix
ruff format .
ruff check .
pytest -q
```
