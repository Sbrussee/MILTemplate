import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

def mil_collate(batch):
    """
    batch: list of dicts with keys: h (M,D), y (int), attn_mask (M,)
    returns:
      h_padded: (B, Mmax, D)
      y: (B,)
      attn_mask: (B, Mmax)
    """
    hs, ys, masks = [], [], []
    for item in batch:
        h = item["h"]
        y = item["y"]
        mask = item.get("attn_mask", torch.ones(h.shape[0], dtype=torch.float32))

        hs.append(h)
        ys.append(torch.tensor(y, dtype=torch.long))
        masks.append(mask.to(torch.float32))

    h_padded = pad_sequence(hs, batch_first=True)
    mask_padded = pad_sequence(masks, batch_first=True)

    y = torch.stack(ys, dim=0)
    return h_padded, y, mask_padded