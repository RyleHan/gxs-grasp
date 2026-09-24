"""Datasets over the subset produced by build_subset.py.

Two views of the same data:
  SceneDataset  one item per image, carrying *all* its instructions. This is what
                lets us supervise G with the union of sibling grasps and S with
                sibling contrast.
  SampleDataset one item per (image, instruction): the standard LGD protocol,
                used for evaluation and for the entangled baseline.
"""
import json
import os
import pickle
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .grasp import SRC_SIZE, draw_maps, region_mask, scale


def load_annotations(root):
    with open(os.path.join(root, "annotations.pkl"), "rb") as f:
        return pickle.load(f)


_IMAGE_CACHE = {}


def _cached(root, size):
    """Memmap written by scripts/cache_images.py, if present (opened lazily per worker)."""
    key = (root, size)
    if key not in _IMAGE_CACHE:
        path = os.path.join(root, f"images{size}.u8")
        if os.path.exists(path + ".json"):
            with open(path + ".json") as f:
                meta = json.load(f)
            _IMAGE_CACHE[key] = (np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(meta["shape"])),
                                 meta["rows"])
        else:
            _IMAGE_CACHE[key] = None
    return _IMAGE_CACHE[key]


def load_image(root, scene, size):
    cache = _cached(root, size)
    if cache is not None:
        x = cache[0][cache[1][scene]].astype(np.float32) / 255.0
    else:
        img = Image.open(os.path.join(root, "images", f"{scene}.jpg")).convert("RGB")
        x = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = x - x.mean()                                # per-image centring, as in GR-ConvNet
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def instructions(scene_ann):
    """Flat list of (obj, part, text, grasps) for one scene."""
    return [(o, p, v["text"], v["grasps"])
            for o, obj in sorted(scene_ann["objects"].items())
            for p, v in sorted(obj["parts"].items())]


class SceneDataset(Dataset):
    """Items: image + union grasp maps + every instruction with its region mask.

    need_image=False skips JPEG decoding (S is trained from cached CLIP features).
    pick_one=True additionally returns the maps of one random instruction, which
    is what the entangled (additive) baseline is trained on.
    """

    def __init__(self, root, split, text_index, clip_index=None, size=224, grid=28,
                 max_inst=16, need_image=True, pick_one=False, need_masks=True):
        self.ann = load_annotations(root)
        self.root, self.size, self.grid = root, size, grid
        self.scenes = self.ann["splits"][split]
        self.text_index, self.clip_index = text_index, clip_index
        self.max_inst, self.need_image, self.pick_one = max_inst, need_image, pick_one
        self.need_masks = need_masks                    # S-branch region masks (skip when training G)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        sc = self.scenes[i]
        items = instructions(self.ann["scenes"][sc])
        if len(items) > self.max_inst:
            items = random.sample(items, self.max_inst)
        s = self.size / SRC_SIZE
        all_rects = np.concatenate([g for *_, g in items])
        out = {
            "scene": sc,
            "clip_row": -1 if self.clip_index is None else self.clip_index[sc],
            "text_ids": torch.tensor([self.text_index[t] for _, _, t, _ in items]),
            "inst_masks": torch.from_numpy(np.stack([region_mask(g, self.grid) for *_, g in items]))
            if self.need_masks else torch.zeros(len(items), self.grid, self.grid, dtype=torch.bool),
            "cand_mask": torch.from_numpy(region_mask(all_rects, self.grid))
            if self.need_masks else torch.zeros(self.grid, self.grid, dtype=torch.bool),
            "obj_ids": torch.tensor([o for o, *_ in items]),
        }
        if self.need_image:
            out["image"] = load_image(self.root, sc, self.size)
            out["g_maps"] = torch.from_numpy(draw_maps(scale(all_rects, s), self.size))
        if self.pick_one:
            k = random.randrange(len(items))
            out["one_text_id"] = out["text_ids"][k]
            out["one_maps"] = torch.from_numpy(draw_maps(scale(items[k][3], s), self.size))
        return out


def collate_scenes(batch):
    """Pad the variable number of instructions per scene."""
    K = max(len(b["text_ids"]) for b in batch)
    G = batch[0]["inst_masks"].shape[-1]
    out = {"scene": [b["scene"] for b in batch],
           "clip_row": torch.tensor([b["clip_row"] for b in batch]),
           "cand_mask": torch.stack([b["cand_mask"] for b in batch]),
           "text_ids": torch.full((len(batch), K), -1, dtype=torch.long),
           "obj_ids": torch.full((len(batch), K), -1, dtype=torch.long),
           "inst_masks": torch.zeros(len(batch), K, G, G, dtype=torch.bool)}
    for i, b in enumerate(batch):
        k = len(b["text_ids"])
        out["text_ids"][i, :k] = b["text_ids"]
        out["obj_ids"][i, :k] = b["obj_ids"]
        out["inst_masks"][i, :k] = b["inst_masks"]
    out["valid"] = out["text_ids"] >= 0
    for key in ("image", "g_maps", "one_maps", "one_text_id"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out


class SampleDataset(Dataset):
    """Items: one (image, instruction) pair with its ground-truth rectangles."""

    def __init__(self, root, split, text_index, clip_index=None, size=224):
        self.ann = load_annotations(root)
        self.root, self.size = root, size
        self.text_index, self.clip_index = text_index, clip_index
        self.items = [(sc, o, p, t, g) for sc in self.ann["splits"][split]
                      for o, p, t, g in instructions(self.ann["scenes"][sc])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sc, o, p, t, g = self.items[i]
        return {"scene": sc, "obj": o, "part": p, "text": t,
                "text_id": self.text_index[t],
                "clip_row": -1 if self.clip_index is None else self.clip_index[sc],
                "image": load_image(self.root, sc, self.size),
                "gt": scale(g, self.size / SRC_SIZE)}


def collate_samples(batch):
    out = {k: [b[k] for b in batch] for k in ("scene", "obj", "part", "text", "gt")}
    out["text_id"] = torch.tensor([b["text_id"] for b in batch])
    out["clip_row"] = torch.tensor([b["clip_row"] for b in batch])
    out["image"] = torch.stack([b["image"] for b in batch])
    return out
