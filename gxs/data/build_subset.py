"""Build the train / val / test subsets of Grasp-Anything++ used in this project.

Test sets are the official LGD splits, untouched. Training data are official
train scenes plus extra multi-object scenes from the full release, subject to:

  R1  the scene does not appear in any official test split
  R2  every annotated object name in the scene belongs to the official-train
      vocabulary (so no unseen category leaks into training)
  R3  no two annotated objects in the scene share a name (language cannot
      disambiguate them)

Output:
  <out>/images/<scene>.jpg
  <out>/annotations.pkl   scenes -> objects -> parts -> {text, grasps (N,5)}
  <splits_dir>/*.txt, stats.json   (small, meant to be committed)
"""
import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from .zipstore import Archive

GA = "https://huggingface.co/datasets/airvlab/Grasp-Anything/resolve/main/"
PP = "https://huggingface.co/datasets/airvlab/Grasp-Anything-pp/resolve/main/"


def sources(local_dir):
    """Small archives can be downloaded whole (fast on Colab); images stay remote."""
    def pick(name, base):
        p = os.path.join(local_dir, name) if local_dir else None
        return [p] if p and os.path.exists(p) else [base + name]
    return {
        "image": [GA + "image_part_aa", GA + "image_part_ab"],
        "prompt": pick("scene_description.zip", GA),
        "instruction": pick("grasp_instructions.zip", PP),
        "grasp": pick("grasp_label_positive.zip", PP),
    }


def load_official(split_dir):
    out = {}
    for name in ["train_seen", "test_seen", "test_unseen"]:
        with open(os.path.join(split_dir, f"{name}.obj"), "rb") as f:
            out[name] = list(pickle.load(f))
    return out


def pmap(fn, items, workers, desc):
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc, leave=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache", help="zip central-directory caches")
    ap.add_argument("--local-zips", default=None, help="dir holding downloaded small zips")
    ap.add_argument("--lgd-splits", default="splits/lgd")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--n-extra", type=int, default=12000, help="extra multi-object train scenes")
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--max-official-train", type=int, default=0, help="debug: cap official train scenes")
    ap.add_argument("--max-test", type=int, default=0, help="debug: cap test scenes per split")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("indexing archives (cached after first run) ...")
    src = sources(args.local_zips)
    arc = {k: Archive(v, os.path.join(args.cache, f"{k}_index.pkl")) for k, v in src.items()}

    # scene -> obj -> [part ids], from the instruction archive
    tree = defaultdict(lambda: defaultdict(list))
    for name in arc["instruction"].index:
        sc, o, p = name.rsplit("/", 1)[-1][:-4].rsplit("_", 2)
        tree[sc][int(o)].append(int(p))

    off = load_official(args.lgd_splits)
    split_scenes = {k: sorted({key.rsplit("_", 1)[0] for key in v} & tree.keys()) for k, v in off.items()}
    listed = defaultdict(set)                                 # scene -> officially listed objects
    for k in ("train_seen", "test_seen", "test_unseen"):
        for key in off[k]:
            sc, o = key.rsplit("_", 1)
            if sc in tree and int(o) in tree[sc]:
                listed[(k, sc)].add(int(o))
    test_scenes = set(split_scenes["test_seen"]) | set(split_scenes["test_unseen"])

    def names_of(sc):
        _, names = pickle.loads(arc["prompt"].get(f"scene_description/{sc}.pkl"))
        return [n.lower().strip() for n in names]

    names = {}
    need = split_scenes["train_seen"] + sorted(test_scenes)
    names.update(zip(need, pmap(names_of, need, args.workers, "descriptions")))

    vocab = {names[sc][o] for sc in split_scenes["train_seen"]
             for o in listed[("train_seen", sc)] if o < len(names[sc])}

    def check(sc):
        """Return None if the scene passes R1-R3, else the name of the failed rule."""
        if sc in test_scenes:
            return "R1"
        nm = [names[sc][o] if o < len(names[sc]) else None for o in tree[sc]]
        if None in nm or any(n not in vocab for n in nm):
            return "R2"
        if len(set(nm)) < len(nm):
            return "R3"
        return None

    drop = defaultdict(int)
    official = []
    for sc in split_scenes["train_seen"]:
        rule = check(sc)
        if rule:
            drop[f"official_{rule}"] += 1
        else:
            official.append(sc)
    if args.max_official_train:
        rng.shuffle(official)
        official = sorted(official[:args.max_official_train])

    # extra multi-object scenes, drawn at random from the rest of the release
    taken = set(split_scenes["train_seen"]) | test_scenes
    pool = sorted(sc for sc, objs in tree.items() if len(objs) >= 2 and sc not in taken)
    rng.shuffle(pool)
    extra, want, i = [], args.n_extra + args.n_val, 0
    while len(extra) < want and i < len(pool):
        batch = pool[i:i + 4000]
        i += len(batch)
        names.update(zip(batch, pmap(names_of, batch, args.workers, "extra descriptions")))
        for sc in batch:
            rule = check(sc)
            if rule:
                drop[f"extra_{rule}"] += 1
            elif len(extra) < want:
                extra.append(sc)
    val, extra_train = sorted(extra[:args.n_val]), sorted(extra[args.n_val:])

    tests = {}
    for k in ("test_seen", "test_unseen"):
        scs = split_scenes[k]
        if args.max_test:
            scs = sorted(rng.sample(scs, min(args.max_test, len(scs))))
        tests[k] = scs

    splits = {"train": sorted(official + extra_train), "val": val, **tests}

    # objects to materialise: every annotated object for train/val, listed ones for test
    jobs = []
    for split, scs in splits.items():
        for sc in scs:
            objs = sorted(tree[sc]) if split in ("train", "val") else sorted(listed[(split, sc)])
            jobs.append((split, sc, objs))

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    def fetch(job):
        split, sc, objs = job
        img_path = os.path.join(args.out, "images", f"{sc}.jpg")
        if not os.path.exists(img_path):
            blob = arc["image"].get(f"image/{sc}.jpg")
            with open(img_path + ".tmp", "wb") as f:
                f.write(blob)
            os.replace(img_path + ".tmp", img_path)
        entry = {"split": split, "names": names[sc], "objects": {}}
        for o in objs:
            parts = {}
            for p in sorted(tree[sc][o]):
                stem = f"{sc}_{o}_{p}"
                text = pickle.loads(arc["instruction"].get(f"grasp_instructions/{stem}.pkl"))
                g = torch.load(io_bytes(arc["grasp"].get(f"grasp_label_positive/{stem}.pt")))
                parts[p] = {"text": str(text).strip(),
                            "grasps": g.numpy().astype(np.float32)[:, 1:6]}   # x, y, w, h, theta(deg)
            entry["objects"][o] = {"name": names[sc][o] if o < len(names[sc]) else "?", "parts": parts}
        return sc, entry

    scenes = dict(pmap(fetch, jobs, args.workers, "materialising"))

    ann = {"meta": {"rules": ["R1 not in official test", "R2 names in official-train vocab",
                              "R3 no duplicate names"],
                    "seed": args.seed, "vocab": sorted(vocab)},
           "splits": splits, "scenes": scenes}
    with open(os.path.join(args.out, "annotations.pkl"), "wb") as f:
        pickle.dump(ann, f, protocol=4)

    stats = summarise(ann)
    stats["dropped"] = dict(drop)
    stats["official_train_scenes_kept"] = len(official)
    stats["extra_train_scenes"] = len(extra_train)
    os.makedirs(args.splits_dir, exist_ok=True)
    for k, v in splits.items():
        with open(os.path.join(args.splits_dir, f"{k}.txt"), "w") as f:
            f.write("\n".join(v) + "\n")
    with open(os.path.join(args.splits_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def io_bytes(b):
    import io
    return io.BytesIO(b)


def summarise(ann):
    """Per-split counts, plus how often sibling parts of one object share labels."""
    out = {}
    for split, scs in ann["splits"].items():
        n_obj = n_inst = n_multi = 0
        for sc in scs:
            objs = ann["scenes"][sc]["objects"]
            n_obj += len(objs)
            n_inst += sum(len(o["parts"]) for o in objs.values())
            n_multi += len(objs) >= 2
        out[split] = {"scenes": len(scs), "objects": n_obj, "instructions": n_inst,
                      "multi_object_scenes": n_multi}
    same = pairs = 0
    for sc in ann["splits"]["train"]:
        for o in ann["scenes"][sc]["objects"].values():
            gs = [p["grasps"] for p in o["parts"].values()]
            for a in range(len(gs)):
                for b in range(a + 1, len(gs)):
                    pairs += 1
                    same += gs[a].shape == gs[b].shape and np.allclose(gs[a], gs[b], atol=1e-2)
    out["train_part_pairs_identical_labels"] = {"pairs": pairs, "identical": int(same),
                                                "ratio": round(same / max(pairs, 1), 4)}
    return out


if __name__ == "__main__":
    main()
