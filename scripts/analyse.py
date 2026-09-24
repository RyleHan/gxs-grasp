"""Break down per-sample results and pick scenes for the qualitative figure.

  python scripts/analyse.py RESULTS_DIR [--split test_seen]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load(d, tag, split):
    path = os.path.join(d, f"{tag}_{split}.json")
    return json.load(open(path))["records"] if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--split", default="test_seen")
    args = ap.parse_args()
    tags = ["g_only", "additive", "gxs_zs", "gxs_full", "gxs", "oracle"]
    recs = {t: load(args.results, t, args.split) for t in tags}
    recs = {t: r for t, r in recs.items() if r}
    ref = next(iter(recs.values()))
    n_obj = defaultdict(set)
    for r in ref:
        n_obj[r["scene"]].add(r["obj"])
    print(f"split={args.split}")
    print(f"{'method':10s} {'single-obj':>11s} {'multi-obj':>10s}")
    for t, rs in recs.items():
        single = [r["success"] for r in rs if len(n_obj[r["scene"]]) == 1]
        multi = [r["success"] for r in rs if len(n_obj[r["scene"]]) >= 2]
        print(f"{t:10s} {100*np.mean(single):10.1f}% {100*np.mean(multi):9.1f}%   (n={len(single)}/{len(multi)})")

    if "gxs" in recs and "g_only" in recs:
        key = lambda r: (r["scene"], r["obj"], r["part"])
        ok = {key(r): r["success"] for r in recs["gxs"]}
        base = {key(r): r["success"] for r in recs["g_only"]}
        good = []
        for sc, objs in n_obj.items():
            if len(objs) < 2:
                continue
            first = {}
            for k in sorted(ok):
                if k[0] == sc:
                    first.setdefault(k[1], k)
            ks = list(first.values())[:2]
            if len(ks) == 2 and all(ok[k] for k in ks) and not all(base[k] for k in ks):
                good.append((sc, ks[0][1], ks[1][1]))
        print(f"\n{len(good)} scenes where G x S gets both objects right and G-only does not:")
        for sc, a, b in good[:12]:
            print(f"  --scenes {sc} --objs {a},{b}")


if __name__ == "__main__":
    main()
