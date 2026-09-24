"""Single image + instruction -> grasp rectangle (x, y, w, h, theta) in image pixels.

  python scripts/demo.py --image cup.jpg --text "grasp the cup by its handle" \
      --g runs/g/best.pt --s runs/s/best.pt --out pred.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import gaussian

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.grasp import corners, decode
from gxs.models.clip_dense import DenseCLIP
from gxs.utils import device
from evaluate import load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--text", required=True)
    ap.add_argument("--g", required=True); ap.add_argument("--s", required=True)
    ap.add_argument("--out", default="pred.png")
    args = ap.parse_args()
    dev = device()
    pil = Image.open(args.image).convert("RGB")
    W, H = pil.size
    x = np.asarray(pil.resize((224, 224)), dtype=np.float32) / 255.0
    x = torch.from_numpy((x - x.mean()).transpose(2, 0, 1).copy())[None].to(dev)
    g, s = load(args.g, dev), load(args.s, dev)
    maps = {k: v[0].float().cpu().numpy() for k, v in g(x).items()}
    clip = DenseCLIP(device=dev)
    P = clip.patches([pil]).to(dev)
    T = clip.text([args.text])[None].to(dev)
    S = torch.sigmoid(F.interpolate(s(P, T), size=224, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
    G = 1 / (1 + np.exp(-maps["pos"]))
    Q = gaussian(G * S, 2.0, preserve_range=True)
    r = decode(Q, maps["cos"], maps["sin"], maps["w"], maps["h"])
    sx, sy = W / 224, H / 224
    out = [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy, r[4]]
    print("grasp (x, y, w, h, theta_deg):", [round(float(v), 1) for v in out])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pil.resize((224, 224)))
    p = corners(r)
    for i in range(4):
        q = p[[i, (i + 1) % 4]]
        ax.plot(q[:, 1], q[:, 0], color="#0a84ff" if i % 2 else "#ff3b30", lw=2.5 if i % 2 else 1.2)
    ax.set_title(args.text, fontsize=9); ax.axis("off")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
