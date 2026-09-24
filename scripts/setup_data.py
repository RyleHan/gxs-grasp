"""One-shot data preparation (meant for Colab).

1. download the three small annotation archives of Grasp-Anything(++) (~5.8 GB)
2. build the subset (images are range-read from the 65 GB remote archive)
"""
import argparse
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download

FILES = [("airvlab/Grasp-Anything", "scene_description.zip"),
         ("airvlab/Grasp-Anything-pp", "grasp_instructions.zip"),
         ("airvlab/Grasp-Anything-pp", "grasp_label_positive.zip")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/content/raw")
    ap.add_argument("--out", default="/content/data")
    ap.add_argument("--cache", default="/content/cache")
    ap.add_argument("--n-extra", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=500)
    args, rest = ap.parse_known_args()
    os.makedirs(args.raw, exist_ok=True)
    for repo, name in FILES:
        if not os.path.exists(os.path.join(args.raw, name)):
            print("downloading", repo, name, flush=True)
            hf_hub_download(repo, name, repo_type="dataset", local_dir=args.raw)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "gxs.data.build_subset", "--out", args.out, "--cache", args.cache,
           "--local-zips", args.raw, "--lgd-splits", os.path.join(root, "splits/lgd"),
           "--splits-dir", os.path.join(root, "splits"),
           "--n-extra", str(args.n_extra), "--n-val", str(args.n_val)] + rest
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
