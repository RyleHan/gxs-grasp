"""Evaluate on an official test split (or val) with the LGD protocol plus our paired metric.

  --method gxs       Q = sigmoid(G) * sigmoid(S)          (--g and --s)
  --method g_only    Q = G, i.e. ignore the instruction  (--g)
  --method additive  entangled GR-ConvNet + CLIP         (--additive)
  --method oracle    Q = sigmoid(G) * [target region]    (--g) upper bound with perfect selection

Metrics
  success        best grasp has IoU > 0.25 and |dtheta| < 30 deg with any ground-truth
                 rectangle of that instruction (LGD / GR-ConvNet protocol)
  success_multi  the same, only on scenes with >= 2 listed objects
  paired         each pair of objects in a multi-object scene, first instruction of
                 each: both grasps must succeed (a language-blind model scores ~0)
  select         predicted centre lies in the target object's rectangles rather than
                 in another object's (multi-object scenes)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.draw import polygon
from skimage.filters import gaussian
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.dataset import SampleDataset, collate_samples
from gxs.data.grasp import SRC_SIZE, corners, decode, scale, success
from gxs.models.grconvnet import GRConvNet
from gxs.models.selector import Selector
from gxs.utils import device, gather_patches, load_clip_cache


def load(path, dev):
    ck = torch.load(path, map_location="cpu")
    kind = ck["model"]
    net = GRConvNet() if kind == "g" else GRConvNet(text_dim=512) if kind == "additive" \
        else Selector(adapters=kind != "s_zs")
    net.load_state_dict(ck["state"])
    return net.to(dev).eval()


def object_regions(ann, scene, size):
    out = {}
    for o, obj in ann["scenes"][scene]["objects"].items():
        m = np.zeros((size, size), dtype=bool)
        for p in obj["parts"].values():
            for r in scale(p["grasps"], size / SRC_SIZE):
                m[polygon(*corners(r).T, shape=(size, size))] = True
        out[o] = m
    return out


@torch.no_grad()
def predict(args, dev):
    text_index, text_feats, clip_rows, mm = load_clip_cache(args.clip_dir)
    ds = SampleDataset(args.data, args.split, text_index, clip_rows)
    dl = DataLoader(ds, args.bs, num_workers=args.workers, collate_fn=collate_samples)
    g = load(args.g, dev) if args.g else None
    s = load(args.s, dev) if args.s else None
    add = load(args.additive, dev) if args.additive else None
    recs = []
    for b in tqdm(dl, desc=f"{args.method} / {args.split}"):
        x = b["image"].to(dev)
        if args.method == "additive":
            maps = add(x, text_feats[b["text_id"]].to(dev))
        else:
            maps = g(x)
        maps = {k: v.float().cpu().numpy() for k, v in maps.items()}
        q = 1 / (1 + np.exp(-maps["pos"]))              # pos head outputs logits
        if args.method == "gxs":
            P = gather_patches(mm, b["clip_row"]).to(dev)
            T = text_feats[b["text_id"]][:, None].to(dev)
            sel = torch.sigmoid(F.interpolate(s(P, T), size=q.shape[-1], mode="bilinear",
                                              align_corners=False))[:, 0].cpu().numpy()
            q = q * sel
        elif args.method == "oracle":                    # perfect selection: the instruction's own rectangles
            for i, gt in enumerate(b["gt"]):
                m = np.zeros(q.shape[1:], dtype=np.float32)
                for r in gt:
                    m[polygon(*corners(r).T, shape=m.shape)] = 1.0
                q[i] = q[i] * m
        for i in range(len(b["scene"])):
            qi = gaussian(q[i], 2.0, preserve_range=True)        # as in GR-ConvNet post-processing
            pred = decode(qi, maps["cos"][i], maps["sin"][i], maps["w"][i], maps["h"][i])
            recs.append({"scene": b["scene"][i], "obj": b["obj"][i], "part": b["part"][i],
                         "text": b["text"][i], "pred": pred.tolist(),
                         "success": bool(success(pred, b["gt"][i]))})
    return ds.ann, recs


def summarise(ann, recs, size=224):
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    multi = {sc for sc, rs in by_scene.items() if len({r["obj"] for r in rs}) >= 2}
    out = {"n": len(recs), "success": np.mean([r["success"] for r in recs])}
    rm = [r for r in recs if r["scene"] in multi]
    out["n_multi"] = len(rm)
    out["success_multi"] = float(np.mean([r["success"] for r in rm])) if rm else float("nan")
    pairs, sel = [], []
    for sc in multi:
        first = {}
        for r in sorted(by_scene[sc], key=lambda r: (r["obj"], r["part"])):
            first.setdefault(r["obj"], r)
        for a, b in combinations(sorted(first), 2):
            pairs.append(first[a]["success"] and first[b]["success"])
        regions = object_regions(ann, sc, size)
        for r in by_scene[sc]:
            c, rr = int(round(r["pred"][0])), int(round(r["pred"][1]))
            c, rr = min(max(c, 0), size - 1), min(max(rr, 0), size - 1)
            sel.append(bool(regions[r["obj"]][rr, c]))
    out["n_pairs"] = len(pairs)
    out["paired"] = float(np.mean(pairs)) if pairs else float("nan")
    out["select"] = float(np.mean(sel)) if sel else float("nan")
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["gxs", "g_only", "additive", "oracle"])
    ap.add_argument("--split", default="test_seen")
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--g"); ap.add_argument("--s"); ap.add_argument("--additive")
    ap.add_argument("--out", required=True, help="json file for metrics + per-sample records")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    ann, recs = predict(args, device())
    summary = summarise(ann, recs)
    summary.update(method=args.method, split=args.split, g=args.g, s=args.s, additive=args.additive)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": recs}, f)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
