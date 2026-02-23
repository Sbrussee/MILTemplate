from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class BagPTDataset(Dataset):
    """Dataset over serialized bag tensors.

    Each ``.pt`` file must provide keys: ``h``, ``y``, and optional ``attn_mask``.
    """

    def __init__(self, bag_dir: str, slide_ids: list[str]) -> None:
        self.bag_dir = bag_dir
        self.slide_ids = slide_ids

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        slide_id = self.slide_ids[idx]
        path = os.path.join(self.bag_dir, f"{slide_id}.pt")
        item = torch.load(path, map_location="cpu")
        if item["y"] is None:
            raise ValueError(f"Missing label for slide_id={slide_id}. Populate slides.csv label column.")
        return item
