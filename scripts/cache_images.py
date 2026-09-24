"""Decode every image once to a 224x224 uint8 memmap (~3 GB for 21k images).

Colab's free tier has two CPU cores; JPEG decoding + resizing every epoch made
training input-bound (GPU at 0%)."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.dataset import load_annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    ann = load_annotations(args.data)
    scenes = sorted(ann["scenes"])
    path = os.path.join(args.data, f"images{args.size}.u8")
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(scenes), args.size, args.size, 3))

    def put(i):
        img = Image.open(os.path.join(args.data, "images", f"{scenes[i]}.jpg")).convert("RGB")
        mm[i] = np.asarray(img.resize((args.size, args.size), Image.BILINEAR))

    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(put, range(len(scenes))), total=len(scenes), desc="images"))
    mm.flush()
    with open(path + ".json", "w") as f:
        json.dump({"shape": [len(scenes), args.size, args.size, 3], "rows": {s: i for i, s in enumerate(scenes)}}, f)
    print(f"{path}: {os.path.getsize(path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
