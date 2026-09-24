import json
import os

import numpy as np
import torch


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_cache(clip_dir):
    """Text features (N, 512) + dense patch features memmap (S, 512, 28, 28)."""
    t = torch.load(os.path.join(clip_dir, "text_feats.pt"))
    text_index = {s: i for i, s in enumerate(t["texts"])}
    with open(os.path.join(clip_dir, "clip_index.json")) as f:
        meta = json.load(f)
    mm = np.memmap(os.path.join(clip_dir, "patches.f16"), dtype=np.float16, mode="r",
                   shape=tuple(meta["shape"]))
    return text_index, t["feats"].float(), meta["rows"], mm


def gather_patches(mm, rows):
    idx = rows.numpy()
    order = np.argsort(idx)
    x = np.empty((len(idx),) + mm.shape[1:], dtype=np.float16)
    x[order] = mm[idx[order]]
    return torch.from_numpy(x).float()
