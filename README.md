# G×S: Decoupling *how* to grasp from *what* to grasp for language-driven grasp detection

Given an RGB image and a grasp instruction (e.g. *"Grip keychain on its ring."*), predict one
rectangle grasp `(x, y, w, h, θ)` — the task of [Grasp-Anything++](https://airvlab.github.io/grasp-anything/).

We factorise the grasp-quality map into

```
Q(x) = G(x) · S(x | instruction)
```

* **G — graspability.** A GR-ConvNet that never sees the instruction. It predicts where a
  parallel gripper can grasp, and the grasp geometry there (angle, opening `w`, jaw size `h`).
  The angle and width of a grasp at a pixel are properties of the object's shape, not of the
  sentence, so all geometry heads are language-free. G is supervised with the **union of the
  grasps of every instruction that shares the image**.
* **S — selection.** Which of the graspable places does the instruction refer to? Frozen CLIP
  ViT-B/16 dense features (value-only last layer, MaskCLIP-style) are compared with the CLIP
  text embedding; a small zero-initialised adapter and a learned temperature are trained with a
  **sibling-contrast loss**: positives are this instruction's grasp region, negatives are the
  regions of the *other* instructions of the same image, background is ignored (G handles it).

Both supervision signals come for free from the fact that each Grasp-Anything++ image carries
several instructions — no masks or boxes are needed. At test time the backbone runs once per
image; a new instruction only costs one dot product.

## Results

_Filled in from `runs/results/table.md` after training._

## Setup

```bash
pip install -r requirements.txt
```

## Data

The official test splits of LGD (`splits/lgd/`) are used unchanged. Training scenes are the
official training scenes plus extra multi-object scenes from the full release, filtered by

* **R1** scene not in any official test split,
* **R2** every annotated object name is in the official-train vocabulary (no unseen-category
  leakage),
* **R3** no two objects with the same name.

```bash
python scripts/setup_data.py --raw /content/raw --out /content/data --cache /content/cache
```

This downloads the three small annotation archives (~5.8 GB) and range-reads only the needed
images from the 65 GB image archive on Hugging Face. Scene lists and statistics are written to
`splits/`.

## Train and evaluate

```bash
python scripts/precompute_clip.py --data /content/data --out /content/clip   # frozen CLIP features
bash scripts/run_all.sh /content/data /content/clip runs                    # all rows of the table
```

Single image:

```bash
python scripts/demo.py --image img.jpg --text "grasp the mug by its handle" \
    --g runs/g/best.pt --s runs/s/best.pt --out pred.png
```

The Colab notebook `notebooks/colab.ipynb` runs the whole pipeline on a free T4.

## Evaluation protocol

A prediction is correct if its IoU with any ground-truth rectangle of that instruction exceeds
0.25 and the angle differs by less than 30° (LGD / GR-ConvNet protocol). We additionally report

* **paired success** — for every pair of objects in a multi-object test image, both grasps must
  be correct; a model that ignores the instruction scores near zero here;
* **selection** — the predicted centre lies on the target object rather than another object.

## Acknowledgements

GR-ConvNet code adapted from [LGD](https://github.com/Fsoft-AIC/LGD) (MIT). Data:
[Grasp-Anything](https://huggingface.co/datasets/airvlab/Grasp-Anything) and
[Grasp-Anything++](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp).
