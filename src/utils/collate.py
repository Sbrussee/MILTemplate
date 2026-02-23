from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def mil_collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length bags into batched tensors.

    Args:
        batch: Items with keys
            - ``h``: ``(M, D)`` float tensor
            - ``y``: int class label
            - ``attn_mask``: optional ``(M,)`` float tensor (1 valid / 0 pad)

    Returns:
        h_padded: ``(B, M_max, D)``
        y: ``(B,)`` int64
        attn_mask: ``(B, M_max)`` float32
    """

    hs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for item in batch:
        h = item["h"]
        y = item["y"]
        mask = item.get("attn_mask", torch.ones(h.shape[0], dtype=torch.float32))

        hs.append(h)
        ys.append(torch.tensor(y, dtype=torch.long))
        masks.append(mask.to(torch.float32))

    return pad_sequence(hs, batch_first=True), torch.stack(ys, dim=0), pad_sequence(masks, batch_first=True)
