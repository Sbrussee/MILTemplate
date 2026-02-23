## Annotation Format
slide_id,slide_path,label
GTEX-11DXX-1626, data/.../GTEX-11DXX-1626.svs, 0

Minimal, modular Whole Slide Image (WSI) → MIL → ABMIL training pipeline.

This repository provides a clean end-to-end workflow for:

Processing raw WSIs (.svs, etc.)

Tissue detection and tiling (LazySlide + wsidata)

Feature extraction (e.g. PLIP)

Converting slide features to MIL-ready bags

Training an Attention-based MIL (ABMIL) model in PyTorch Lightning

Evaluating and saving trained models

The design goal is:

Single configuration file (config.yaml)

Reproducible

Modular

Research-friendly

Scales to multi-GPU training

Overview of the Pipeline
Raw WSI (.svs)
    ↓
Tissue detection + tiling
    ↓
Tile-level feature extraction (PLIP, etc.)
    ↓
AnnData feature storage (.h5ad)
    ↓
Convert to MIL bag format (.pt)
    ↓
Train ABMIL (PyTorch Lightning)
    ↓
Evaluate / Save model
Repository Structure
mil-lightning/
│
├── config.yaml
├── pyproject.toml
├── README.md
│
├── preprocess_wsi.py        # WSI → tiles → features (.h5ad)
├── make_bags.py             # .h5ad → MIL bags (.pt)
├── train.py                 # Train ABMIL
│
└── src/mil_lightning/
    ├── models/
    │   └── abmil.py
    ├── lightning/
    │   └── lit_abmil.py
    ├── data/
    │   ├── bag_dataset.py
    │   └── datamodule.py
    └── utils/
Installation (Using uv)

This project uses modern Python packaging via pyproject.toml.

1️⃣ Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

Restart your shell if needed.

2️⃣ Create environment and install dependencies

From the project root:

uv venv
uv sync

This will:

Create .venv/

Install PyTorch, Lightning, LazySlide, wsidata, etc.

Install the project in editable mode

Activate the environment:

source .venv/bin/activate
Configuration

All behavior is controlled from:

config.yaml

This file defines:

Input slide paths

Tiling parameters

Feature extraction model

MIL bag settings

Model hyperparameters

Training configuration

You should only modify this file.

Step 1 — Process Raw WSIs

This step performs:

Open WSI

Tissue detection

Tiling

Feature extraction (e.g. PLIP)

Save features as .h5ad

Prepare slide metadata

Create:

data/slides.csv

Format:

slide_id,slide_path,label
SLIDE_001,/path/to/slide1.svs,0
SLIDE_002,/path/to/slide2.svs,1
Run preprocessing
python preprocess_wsi.py --config config.yaml

Outputs:

artifacts/features_anndata/{slide_id}.h5ad

Each file contains tile-level embeddings.

Step 2 — Convert Features to MIL Bags

This step:

Loads .h5ad

Extracts feature matrix

Converts to torch tensors

Saves as .pt bags

python make_bags.py --config config.yaml

Outputs:

artifacts/bags_pt/{slide_id}.pt

Each file contains:

{
  "slide_id": str,
  "h": Tensor[M, D],
  "y": int,
  "attn_mask": Tensor[M]
}

These are ready for MIL training.

Step 3 — Train ABMIL

Train using PyTorch Lightning:

python train.py --config config.yaml

This will:

Load MIL bags

Automatically infer feature dimension (if in_dim: auto)

Train ABMIL

Log to TensorBoard

Save best checkpoint

Monitor Training

Start TensorBoard:

tensorboard --logdir tb_logs

Open:

http://localhost:6006

You will see:

train/loss

train/acc

val/loss

val/acc

Step 4 — Evaluate a Trained Model

The best checkpoint is printed at the end of training:

Best checkpoint: tb_logs/abmil/version_x/checkpoints/abmil-xx.ckpt

You can load it manually:

import torch
from mil_lightning.models.abmil import ABMIL

ckpt = torch.load("path/to/checkpoint.ckpt")

model = ABMIL(...)
model.load_state_dict(ckpt["state_dict"])
model.eval()

Or extend train.py to add a test stage via Lightning.

Step 5 — Save / Export Model

You can save the model weights in multiple ways.

Save full Lightning checkpoint (default)

Already saved during training.

Save only model weights
torch.save(model.state_dict(), "abmil_weights.pt")
TorchScript export (for inference deployment)
model.eval()
example = torch.randn(1, 100, model.in_dim)
traced = torch.jit.trace(model.forward_head, example)
traced.save("abmil_head.pt")

For full model export, wrap forward appropriately.

Reproducibility

Set seed in config.yaml:

train:
  seed: 42

Lightning will handle deterministic behavior where possible.

Advanced Notes
Attention Visualization

If tile coordinates are stored in bags (coords field), you can:

Extract attention weights

Map weights back to slide space

Render heatmaps

This requires minimal extension to the validation step.

Distributed Training

Set in config.yaml:

train:
  accelerator: gpu
  devices: 4

Lightning will automatically handle DDP.

Large Slides

For very large slides:

Increase max_instances limit

Or implement random tile sampling per epoch

Consider gradient accumulation

Common Issues
AMP on MPS gives NaNs

Disable in config:

preprocess:
  amp: false
openslide errors

Install system dependency:

sudo apt-get install openslide-tools

(macOS users may need brew install openslide)

Minimal Example Workflow
uv sync
python preprocess_wsi.py --config config.yaml
python make_bags.py --config config.yaml
python train.py --config config.yaml
tensorboard --logdir tb_logs
What This Repository Is Not

Not a full digital pathology framework

Not a slide viewer

Not a feature extraction research library

It is a clean, minimal MIL research pipeline designed for:

Slide-level classification

Experimentation with attention-based pooling

Fast iteration