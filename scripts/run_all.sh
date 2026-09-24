#!/usr/bin/env bash
# Train every component and evaluate every row of the ablation table.
# Usage: bash scripts/run_all.sh DATA_DIR CLIP_DIR RUNS_DIR [G_EPOCHS] [S_EPOCHS]
set -euo pipefail
DATA=$1; CLIP=$2; RUNS=$3; GE=${4:-30}; SE=${5:-10}
W=${WORKERS:-2}
mkdir -p "$RUNS/results"
[ -f "$DATA/images224.u8.json" ] || python scripts/cache_images.py --data "$DATA"

train() { # name model epochs
  [ -f "$RUNS/$1/last.pt" ] && { echo "skip $1 (done)"; return; }
  python scripts/train.py --model "$2" --data "$DATA" --clip-dir "$CLIP" --out "$RUNS/$1" \
         --epochs "$3" --workers "$W" | tee "$RUNS/$1.log"
}
train g        g        "$GE"
train additive additive "$GE"
train s        s        "$SE"
train s_full   s_full   "$SE"
train s_zs     s_zs     "$SE"

evaluate() { # tag method split extra-args...
  local tag=$1 m=$2 split=$3; shift 3
  python scripts/evaluate.py --method "$m" --split "$split" --data "$DATA" --clip-dir "$CLIP" \
         --out "$RUNS/results/${tag}_${split}.json" --workers "$W" "$@" | tail -n 20
}
for split in test_seen test_unseen; do
  evaluate g_only   g_only   $split --g "$RUNS/g/best.pt"
  evaluate additive additive $split --additive "$RUNS/additive/best.pt"
  evaluate gxs_zs   gxs      $split --g "$RUNS/g/best.pt" --s "$RUNS/s_zs/best.pt"
  evaluate gxs_full gxs      $split --g "$RUNS/g/best.pt" --s "$RUNS/s_full/best.pt"
  evaluate gxs      gxs      $split --g "$RUNS/g/best.pt" --s "$RUNS/s/best.pt"
done
python scripts/make_table.py "$RUNS/results"
