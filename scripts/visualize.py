"""Paper figure: one image, two instructions. G is shared; S and Q change with the text.

Panels: input (+ground truth of both instructions) | G | S(l1) | Q(l1)+pred | S(l2) | Q(l2)+pred
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
from gxs.data.dataset import load_annotations, load_image
from gxs.data.grasp import corners, decode, scale
from gxs.utils import device, gather_patches, load_clip_cache
from evaluate import load

COLORS = ["#34c759", "#ff9f0a"]


def draw_rect(ax, r, color, lw=2.0):
    p = corners(r)
    for i in range(4):
        q = p[[i, (i + 1) % 4]]
        ax.plot(q[:, 1], q[:, 0], color=color, lw=lw if i % 2 else lw / 2)


@torch.no_grad()
def render(ax_row, ann, sc, objs, g, s, text_index, text_feats, rows, mm, data, dev):
    x = load_image(data, sc, 224)[None].to(dev)
    maps = {k: v[0].float().cpu().numpy() for k, v in g(x).items()}
    G = 1 / (1 + np.exp(-maps["pos"]))
    P = gather_patches(mm, torch.tensor([rows[sc]])).to(dev)
    img = np.asarray(Image.open(os.path.join(data, "images", f"{sc}.jpg")).convert("RGB").resize((224, 224)))
    show = lambda ax, m=None: (ax.imshow(img), m is not None and ax.imshow(m, cmap="jet", alpha=0.55, vmin=0, vmax=1),
                               ax.axis("off"))
    show(ax_row[0]); show(ax_row[1], G)
    texts = []
    for j, o in enumerate(objs):
        obj = ann["scenes"][sc]["objects"][o]
        part = obj["parts"][min(obj["parts"])]
        texts.append(part["text"])
        T = text_feats[text_index[part["text"]]][None, None].to(dev)
        S = torch.sigmoid(F.interpolate(s(P, T), size=224, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
        Q = gaussian(G * S, 2.0, preserve_range=True)
        pred = decode(Q, maps["cos"], maps["sin"], maps["w"], maps["h"])
        for gr in scale(part["grasps"], 224 / 416)[:2]:
            draw_rect(ax_row[0], gr, COLORS[j], 1.6)
        show(ax_row[2 + 2 * j], S); show(ax_row[3 + 2 * j], Q / max(Q.max(), 1e-6))
        draw_rect(ax_row[3 + 2 * j], pred, "white", 2.4)
        draw_rect(ax_row[3 + 2 * j], pred, COLORS[j], 1.6)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--g", required=True); ap.add_argument("--s", required=True)
    ap.add_argument("--scenes", required=True, help="comma-separated scene ids")
    ap.add_argument("--objs", default="0,1", help="two object ids per scene, ';'-separated per scene")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = device()
    ann = load_annotations(args.data)
    text_index, text_feats, rows, mm = load_clip_cache(args.clip_dir)
    g, s = load(args.g, dev), load(args.s, dev)
    scenes = args.scenes.split(",")
    objs = [[int(o) for o in grp.split(",")] for grp in args.objs.split(";")]
    objs += [objs[-1]] * (len(scenes) - len(objs))
    fig, axes = plt.subplots(len(scenes), 6, figsize=(12, 2.15 * len(scenes)), squeeze=False)
    heads = ["input + GT", "$G$ (shared)", r"$S(\ell_1)$", r"$Q(\ell_1)$", r"$S(\ell_2)$", r"$Q(\ell_2)$"]
    for r, (sc, ob) in enumerate(zip(scenes, objs)):
        texts = render(axes[r], ann, sc, ob, g, s, text_index, text_feats, rows, mm, args.data, dev)
        for c, h in enumerate(heads):
            if r == 0:
                axes[r, c].set_title(h, fontsize=10)
        axes[r, 0].text(0, 238, r"$\ell_1$: " + texts[0], fontsize=7, color=COLORS[0], va="top")
        axes[r, 0].text(0, 256, r"$\ell_2$: " + texts[1], fontsize=7, color=COLORS[1], va="top")
    fig.subplots_adjust(wspace=0.03, hspace=0.35, left=0.005, right=0.995, top=0.9, bottom=0.12)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
