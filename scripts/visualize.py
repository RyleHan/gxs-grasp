"""Figure: one image, two instructions -> G (shared), S per instruction, Q, grasp."""
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
from gxs.data.dataset import instructions, load_annotations, load_image
from gxs.data.grasp import corners, decode, scale
from gxs.utils import device, gather_patches, load_clip_cache
from evaluate import load


def draw_rect(ax, r, color):
    p = corners(r)
    for i in range(4):
        q = p[[i, (i + 1) % 4]]
        ax.plot(q[:, 1], q[:, 0], color=color if i % 2 else "#ff3b30", lw=2.4 if i % 2 else 1.2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--g", required=True); ap.add_argument("--s", required=True)
    ap.add_argument("--scene", required=True); ap.add_argument("--objs", default="0,1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = device()
    ann = load_annotations(args.data)
    text_index, text_feats, rows, mm = load_clip_cache(args.clip_dir)
    g, s = load(args.g, dev), load(args.s, dev)
    sc = args.scene
    items = {o: (t, gr) for o, p, t, gr in instructions(ann["scenes"][sc]) if p == min(
        ann["scenes"][sc]["objects"][o]["parts"])}
    objs = [int(o) for o in args.objs.split(",")]
    x = load_image(args.data, sc, 224)[None].to(dev)
    maps = {k: v[0].float().cpu().numpy() for k, v in g(x).items()}
    G = 1 / (1 + np.exp(-maps["pos"]))
    P = gather_patches(mm, torch.tensor([rows[sc]])).to(dev)
    img = np.asarray(Image.open(os.path.join(args.data, "images", f"{sc}.jpg")).convert("RGB").resize((224, 224)))
    fig, ax = plt.subplots(len(objs), 4, figsize=(13, 3.4 * len(objs)))
    for r, o in enumerate(objs):
        text, gt = items[o]
        T = text_feats[text_index[text]][None, None].to(dev)
        S = torch.sigmoid(F.interpolate(s(P, T), size=224, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
        Q = gaussian(G * S, 2.0, preserve_range=True)
        pred = decode(Q, maps["cos"], maps["sin"], maps["w"], maps["h"])
        for c, (m, title) in enumerate([(None, f'"{text}"'), (G, "G: graspability"), (S, "S: selection"), (Q, "Q = G x S")]):
            ax[r, c].imshow(img)
            if m is not None:
                ax[r, c].imshow(m, cmap="jet", alpha=0.5, vmin=0, vmax=max(m.max(), 1e-6))
            ax[r, c].set_title(title, fontsize=10); ax[r, c].axis("off")
        for gr in scale(gt, 224 / 416)[:3]:
            draw_rect(ax[r, 0], gr, "#34c759")
        draw_rect(ax[r, 3], pred, "#0a84ff")
    fig.tight_layout(); fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
