"""S branch: which of the graspable places does this instruction refer to?

Input: frozen dense CLIP patch features (B, 512, 28, 28) and CLIP text features
(B, K, 512), both L2-normalised. Output: selection logits (B, K, 28, 28).

The zero-shot score is the cosine between each patch and the text. Two residual
adapters (zero-initialised, so training starts exactly from zero-shot CLIP) and a
learned temperature/bias turn it into a calibrated per-patch probability.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Selector(nn.Module):
    def __init__(self, dim=512, adapters=True):
        super().__init__()
        self.adapters = adapters
        if adapters:
            self.vis = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
            self.txt = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for m in (self.vis[-1], self.txt[-1]):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(100.0)))
        self.bias = nn.Parameter(torch.tensor(-19.0))       # zero-shot cosines sit around 0.15-0.25

    def forward(self, patches, text):
        if self.adapters:
            patches = F.normalize(patches + self.vis(patches), dim=1)
            text = F.normalize(text + self.txt(text), dim=-1)
        cos = torch.einsum("bkd,bdhw->bkhw", text, patches)
        return self.log_scale.exp() * cos + self.bias


def s_loss(logits, inst_masks, cand_mask, valid, candidates_only=True):
    """Balanced BCE over patches.

    Positives: this instruction's grasp region. Negatives: the other instructions'
    regions (candidates_only=True, our method) or every other patch (False).
    Background is ignored in the first case -- G already handles it.
    """
    pos = inst_masks & valid[:, :, None, None]
    if candidates_only:
        neg = cand_mask[:, None] & ~inst_masks & valid[:, :, None, None]
    else:
        neg = ~inst_masks & valid[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, pos.float(), reduction="none")
    lp = (bce * pos).sum() / pos.sum().clamp(min=1)
    ln = (bce * neg).sum() / neg.sum().clamp(min=1)
    return lp + ln, {"s_pos": lp, "s_neg": ln}
