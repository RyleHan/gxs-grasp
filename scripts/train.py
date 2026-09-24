"""Train one component.

  --model g          G branch: GR-ConvNet on the union of all grasps in the image
  --model s          S branch, our loss (negatives = sibling regions only)
  --model s_full     S branch, negatives = every non-target patch (ablation)
  --model s_zs       S branch without adapters: only temperature + bias (zero-shot CLIP)
  --model additive   entangled baseline: GR-ConvNet + CLIP text broadcast-added
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.dataset import SceneDataset, collate_scenes
from gxs.models.grconvnet import GRConvNet, g_loss
from gxs.models.selector import Selector, s_loss
from gxs.utils import device, gather_patches, load_clip_cache


def selection_accuracy(logits, batch):
    """Per instruction: does the target object's region get the highest mean logit
    among the objects of the scene? (only scenes with >= 2 objects count)"""
    hit = tot = 0
    for b in range(logits.shape[0]):
        valid = batch["valid"][b]
        objs = batch["obj_ids"][b][valid]
        if len(set(objs.tolist())) < 2:
            continue
        masks = batch["inst_masks"][b][valid]
        regions = {o: masks[objs == o].any(0) for o in set(objs.tolist())}
        for k in range(len(objs)):
            scores = {o: logits[b, k][r].mean().item() for o, r in regions.items() if r.any()}
            hit += max(scores, key=scores.get) == objs[k].item()
            tot += 1
    return hit, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["g", "s", "s_full", "s_zs", "additive"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = device()
    torch.manual_seed(0)

    text_index, text_feats, clip_rows, mm = load_clip_cache(args.clip_dir)
    is_s = args.model.startswith("s")
    kw = dict(text_index=text_index, clip_index=clip_rows, need_image=not is_s,
              pick_one=args.model == "additive", need_masks=is_s)
    tr = SceneDataset(args.data, "train", **kw)
    va = SceneDataset(args.data, "val", **kw)
    mk = lambda ds, sh: DataLoader(ds, args.bs, shuffle=sh, num_workers=args.workers,
                                   collate_fn=collate_scenes, drop_last=sh,
                                   persistent_workers=args.workers > 0)
    tl, vl = mk(tr, True), mk(va, False)

    if args.model == "g":
        net = GRConvNet()
    elif args.model == "additive":
        net = GRConvNet(text_dim=512)
    else:
        net = Selector(adapters=args.model != "s_zs")
    net.to(dev)
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True          # fixed 224x224 inputs
        if not is_s:
            net = net.to(memory_format=torch.channels_last)
    lr = args.lr if args.model != "s_zs" else 1e-2
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(tl))
    amp = dev == "cuda" and not is_s
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step(batch, train):
        if is_s:
            P = gather_patches(mm, batch["clip_row"]).to(dev)
            ids = batch["text_ids"].clamp(min=0)
            T = text_feats[ids].to(dev)
            logits = net(P, T)
            loss, parts = s_loss(logits, batch["inst_masks"].to(dev), batch["cand_mask"].to(dev),
                                 batch["valid"].to(dev), candidates_only=args.model != "s_full")
            return loss, parts, logits
        with torch.autocast("cuda", enabled=amp):
            x = batch["image"].to(dev, non_blocking=True)
            if dev == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            if args.model == "g":
                pred = net(x)
                target = batch["g_maps"].to(dev, non_blocking=True)
            else:
                pred = net(x, text_feats[batch["one_text_id"]].to(dev))
                target = batch["one_maps"].to(dev, non_blocking=True)
            loss, parts = g_loss({k: v.float() for k, v in pred.items()}, target)
        return loss, parts, None

    log, best, t0, steps = [], None, time.time(), 0
    for ep in range(args.epochs):
        net.train()
        tot = n = 0
        for batch in tl:
            loss, _, _ = step(batch, True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item(); n += 1; steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        net.eval()
        vtot = vn = hit = cnt = 0
        with torch.no_grad():
            for batch in vl:
                loss, _, logits = step(batch, False)
                vtot += loss.item(); vn += 1
                if logits is not None:
                    h, c = selection_accuracy(logits.cpu(), batch)
                    hit += h; cnt += c
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vtot / max(vn, 1),
               "minutes": (time.time() - t0) / 60}
        if is_s:
            rec["val_select_acc"] = hit / max(cnt, 1)
            rec["scale"] = net.log_scale.exp().item()
        score = rec.get("val_select_acc", -rec["val_loss"])
        if best is None or score > best:
            best = score
            torch.save({"model": args.model, "state": net.state_dict(), "epoch": ep, "args": vars(args)},
                       os.path.join(args.out, "best.pt"))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if args.max_steps and steps >= args.max_steps:
            break
    torch.save({"model": args.model, "state": net.state_dict(), "epoch": ep, "args": vars(args)},
               os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
