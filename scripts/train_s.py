"""Train the three S variants in a single pass over the data.

They consume exactly the same batches (frozen CLIP patches, text features, region
masks) and differ only in their loss / capacity, so sharing the input pipeline
cuts the disk reads and mask rasterisation of S training by a factor of three.

  s       adapters, negatives = sibling regions only (ours)
  s_full  adapters, negatives = every non-target patch
  s_zs    no adapters: temperature + bias on zero-shot CLIP similarities
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gxs.data.dataset import SceneDataset, collate_scenes
from gxs.models.selector import Selector, s_loss
from gxs.utils import device, gather_patches, load_clip_cache
from train import selection_accuracy

VARIANTS = {"s": (True, True), "s_full": (True, False), "s_zs": (False, True)}  # adapters, candidates_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out", required=True, help="runs dir; writes <out>/s, s_full, s_zs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    dev = device()
    torch.manual_seed(0)

    text_index, text_feats, clip_rows, mm = load_clip_cache(args.clip_dir)
    kw = dict(text_index=text_index, clip_index=clip_rows, need_image=False, need_masks=True)
    mk = lambda split, sh: DataLoader(SceneDataset(args.data, split, **kw), args.bs, shuffle=sh,
                                      num_workers=args.workers, collate_fn=collate_scenes,
                                      drop_last=sh, persistent_workers=args.workers > 0)
    tl, vl = mk("train", True), mk("val", False)

    nets, opts, scheds, logs, best = {}, {}, {}, {}, {}
    for name, (adapters, _) in VARIANTS.items():
        nets[name] = Selector(adapters=adapters).to(dev)
        opts[name] = torch.optim.Adam(nets[name].parameters(), lr=args.lr if adapters else 1e-2)
        scheds[name] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[name], args.epochs * len(tl))
        logs[name], best[name] = [], None
        os.makedirs(os.path.join(args.out, name), exist_ok=True)

    def inputs(batch):
        P = gather_patches(mm, batch["clip_row"]).to(dev)
        T = text_feats[batch["text_ids"].clamp(min=0)].to(dev)
        return P, T, batch["inst_masks"].to(dev), batch["cand_mask"].to(dev), batch["valid"].to(dev)

    t0 = time.time()
    for ep in range(args.epochs):
        tot = {n: 0.0 for n in nets}
        for net in nets.values():
            net.train()
        for i, batch in enumerate(tl):
            P, T, M, C, V = inputs(batch)
            for name, net in nets.items():
                loss, _ = s_loss(net(P, T), M, C, V, candidates_only=VARIANTS[name][1])
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                opts[name].step()
                scheds[name].step()
                tot[name] += loss.item()
            if i % 100 == 0:
                print(f"  epoch {ep} step {i}/{len(tl)}  {(time.time() - t0) / 60:.1f} min", flush=True)
        stats = {n: {"loss": 0.0, "hit": 0, "cnt": 0} for n in nets}
        with torch.no_grad():
            for net in nets.values():
                net.eval()
            for batch in vl:
                P, T, M, C, V = inputs(batch)
                for name, net in nets.items():
                    logits = net(P, T)
                    stats[name]["loss"] += s_loss(logits, M, C, V, candidates_only=VARIANTS[name][1])[0].item()
                    h, c = selection_accuracy(logits.cpu(), batch)
                    stats[name]["hit"] += h
                    stats[name]["cnt"] += c
        for name, net in nets.items():
            acc = stats[name]["hit"] / max(stats[name]["cnt"], 1)
            rec = {"model": name, "epoch": ep, "train_loss": tot[name] / len(tl),
                   "val_loss": stats[name]["loss"] / len(vl), "val_select_acc": acc,
                   "scale": net.log_scale.exp().item(), "minutes": (time.time() - t0) / 60}
            logs[name].append(rec)
            print(json.dumps(rec), flush=True)
            ck = {"model": name, "state": net.state_dict(), "epoch": ep, "args": vars(args)}
            if best[name] is None or acc > best[name]:
                best[name] = acc
                torch.save(ck, os.path.join(args.out, name, "best.pt"))
            torch.save(ck, os.path.join(args.out, name, "last.pt") + ".tmp")
            with open(os.path.join(args.out, name, "log.json"), "w") as f:
                json.dump(logs[name], f, indent=1)
    for name in nets:                                   # last.pt marks a finished run for run_all.sh
        os.replace(os.path.join(args.out, name, "last.pt") + ".tmp", os.path.join(args.out, name, "last.pt"))


if __name__ == "__main__":
    main()
