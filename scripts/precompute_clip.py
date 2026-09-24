"""Cache frozen-CLIP text features for every instruction and dense patch features
for every image, so neither training nor evaluation has to run CLIP again."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.dataset import instructions, load_annotations
from gxs.models.clip_dense import DenseCLIP
from gxs.utils import device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ann = load_annotations(args.data)
    clip = DenseCLIP(device=device())

    texts = sorted({t for sc in ann["scenes"].values() for _, _, t, _ in instructions(sc)})
    torch.save({"texts": texts, "feats": clip.text(texts).half()}, os.path.join(args.out, "text_feats.pt"))
    print(f"{len(texts):,} unique instructions encoded")

    scenes = sorted(ann["scenes"])
    shape = (len(scenes), 512, clip.grid, clip.grid)
    mm = np.memmap(os.path.join(args.out, "patches.f16"), dtype=np.float16, mode="w+", shape=shape)
    load = lambda sc: Image.open(os.path.join(args.data, "images", f"{sc}.jpg")).convert("RGB")
    with ThreadPoolExecutor(8) as ex:
        for i in tqdm(range(0, len(scenes), args.batch), desc="patches"):
            chunk = scenes[i:i + args.batch]
            mm[i:i + len(chunk)] = clip.patches(list(ex.map(load, chunk))).half().numpy()
    mm.flush()
    with open(os.path.join(args.out, "clip_index.json"), "w") as f:
        json.dump({"shape": shape, "rows": {sc: i for i, sc in enumerate(scenes)},
                   "model": "openai/clip-vit-base-patch16", "res": clip.res, "mode": "value-only"}, f)
    print(f"patch cache: {shape}, {os.path.getsize(os.path.join(args.out, 'patches.f16')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
