# Climbing Holds & Volumes Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an instance-segmentation training pipeline to the WongKinYiu/YOLO repo for climbing hold/volume detection, deliver two mobile-targeted variants (`v9-t-seg`, `v9-s-seg`), and train them on the 19-image hand-labeled dataset to produce visible mask overlays.

**Architecture:** YOLACT-style head (`MultiheadSegmentation` already scaffolded; rewire its forward to emit detection + mask outputs in a structured dict). Detection outputs flow through the existing `Vec2Box` decoder; mask outputs (`coefs`, `proto`) bypass `Vec2Box` and are consumed directly by a new `MaskLoss` (BCE + Dice + bbox crop). Phase A delivers a detection-only MVP on the same data to validate the pipeline; Phase B layers segmentation on top.

**Tech Stack:** PyTorch + Lightning, Hydra configs, `pycocotools` for COCO eval, OpenCV (`cv2`) for polygon rasterization. Local CPU dev on Intel Mac; real training on Colab Free T4 (Kaggle as backup).

**Spec:** `docs/superpowers/specs/2026-05-06-climbing-holds-segmentation-design.md`

---

## File Structure

### New files (Phase A)

| Path | Responsibility |
|---|---|
| `tools/data_prep/__init__.py` | package init |
| `tools/data_prep/via_to_coco.py` | VIA JSON → COCO instance-seg JSON converter |
| `tools/data_prep/sanity_check.py` | Visualizes converted polygons on sample images |
| `yolo/config/dataset/climbing_holds.yaml` | Dataset config for the climbing dataset |
| `notebooks/colab/phase_a_detection.ipynb` | Colab notebook for Phase A training |
| `tests/test_tools/test_via_to_coco.py` | Unit tests for converter |

### New files (Phase B)

| Path | Responsibility |
|---|---|
| `yolo/config/model/v9-t-seg.yaml` | New tiny seg model config |
| `yolo/config/model/v9-s-seg.yaml` | New small seg model config |
| `tools/weight_transfer/v9_to_seg.py` | Detection ckpt → seg ckpt transform |
| `notebooks/colab/phase_b_segmentation.ipynb` | Colab notebook for Phase B training |
| `tests/test_model/test_multihead_segmentation.py` | Tests for rewired forward |
| `tests/test_utils/test_mask_rasterize.py` | Tests for polygon→mask raster |
| `tests/test_tools/test_polygon_augment.py` | Tests for polygon-aware augmentation |
| `tests/test_tools/test_mosaic.py` | Tests for Mosaic augmentation |
| `tests/test_tools/test_mask_loss.py` | Tests for `MaskLoss` |
| `tests/test_tools/test_weight_transfer.py` | Tests for weight transfer |

### Modified files (Phase B)

| Path | Change |
|---|---|
| `yolo/model/module.py` | Rewire `MultiheadSegmentation.forward` to emit `{detect, mask_coefs, proto}` dict |
| `yolo/utils/dataset_utils.py` | Add `rasterize_masks()` helper |
| `yolo/tools/data_augmentation.py` | Make existing transforms polygon-aware; add `Mosaic` |
| `yolo/tools/data_loader.py` | Extend `load_valid_labels` and `__getitem__` to retain polygons; extend `collate_fn` to pad masks |
| `yolo/tools/loss_functions.py` | Add `MaskLoss`; extend `DualLoss` to route mask path when `task_type == "segmentation"` |
| `yolo/tools/solver.py` | Extend `training_step` to unpack and forward `gt_masks` |
| `yolo/tools/drawer.py` | Add `draw_masks()` overlay helper |
| `yolo/config/config.py` | Add `task_type: str = "detection"` field on `ModelConfig` (or equivalent) |
| `.gitignore` | Add `data/climbing_holds/`, `weights/`, `runs/` |

---

## Phase 0 — Setup

### Task 1: Create feature branch and gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Verify clean working tree and create branch**

```bash
git status   # expect: clean (one design+plan doc commit ahead)
git checkout -b feature/climbing-seg
git branch --show-current   # expect: feature/climbing-seg
```

- [ ] **Step 2: Append project-specific entries to `.gitignore`**

Append at the end of `/Users/htjo/workspace/YOLO/.gitignore`:

```
# climbing-seg project artifacts
data/climbing_holds/
weights/
runs/
.tensorboard/
sanity/
```

- [ ] **Step 3: Verify and commit**

```bash
git diff .gitignore
git add .gitignore
git commit -m "🔧 [Add] gitignore entries for climbing-seg artifacts"
```

---

## Phase A — Detection MVP

### Task 2: VIA → COCO converter (TDD)

**Files:**
- Create: `tools/data_prep/__init__.py`
- Create: `tools/data_prep/via_to_coco.py`
- Create: `tests/test_tools/test_via_to_coco.py`

- [ ] **Step 1: Create `tools/data_prep/__init__.py`** (empty file)

```bash
mkdir -p /Users/htjo/workspace/YOLO/tools/data_prep
touch /Users/htjo/workspace/YOLO/tools/data_prep/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_tools/test_via_to_coco.py`:

```python
import json
from pathlib import Path

import pytest

from tools.data_prep.via_to_coco import convert_via_to_coco


@pytest.fixture
def via_json(tmp_path: Path) -> Path:
    """Minimal VIA JSON with two regions: one hold polygon, one volume polygon."""
    via = {
        "_via_img_metadata": {
            "img1.jpg100": {
                "filename": "img1.jpg",
                "size": 100,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [10, 20, 20, 10],
                            "all_points_y": [10, 10, 20, 20],
                        },
                        "region_attributes": {"hold_type": "hold"},
                    },
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [30, 40, 40, 30],
                            "all_points_y": [30, 30, 40, 40],
                        },
                        "region_attributes": {"hold_type": "volume"},
                    },
                ],
            },
            "img2.jpg200": {
                "filename": "img2.jpg",
                "size": 200,
                "regions": [],  # no labels — must be skipped
            },
        }
    }
    path = tmp_path / "via.json"
    path.write_text(json.dumps(via))
    return path


def test_convert_via_to_coco_writes_coco_with_two_categories(via_json: Path, tmp_path: Path):
    """Converter emits standard COCO instance-seg JSON with hold(0)/volume(1)."""
    out_path = tmp_path / "coco.json"
    # image_size_lookup maps filename -> (width, height) so the test does not need real images
    convert_via_to_coco(
        via_json=via_json,
        output_json=out_path,
        image_size_lookup={"img1.jpg": (100, 100)},
    )
    coco = json.loads(out_path.read_text())
    assert {c["name"] for c in coco["categories"]} == {"hold", "volume"}
    assert {c["id"] for c in coco["categories"]} == {0, 1}
    assert len(coco["images"]) == 1   # img2 dropped (zero regions)
    assert len(coco["annotations"]) == 2
    # check polygon flattened correctly
    ann = coco["annotations"][0]
    assert ann["segmentation"] == [[10, 10, 20, 10, 20, 20, 10, 20]]
    assert ann["bbox"] == [10, 10, 10, 10]   # x, y, w, h
    assert ann["iscrowd"] == 0


def test_convert_via_to_coco_skips_polygons_with_fewer_than_three_points(tmp_path: Path):
    via = {
        "_via_img_metadata": {
            "a.jpg10": {
                "filename": "a.jpg",
                "size": 10,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [1, 2],   # only 2 points — invalid
                            "all_points_y": [1, 2],
                        },
                        "region_attributes": {"hold_type": "hold"},
                    }
                ],
            }
        }
    }
    via_path = tmp_path / "via.json"
    via_path.write_text(json.dumps(via))
    out_path = tmp_path / "coco.json"
    convert_via_to_coco(
        via_json=via_path,
        output_json=out_path,
        image_size_lookup={"a.jpg": (50, 50)},
    )
    coco = json.loads(out_path.read_text())
    assert coco["annotations"] == []
    assert coco["images"] == []   # no valid annotations means image is dropped
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/htjo/workspace/YOLO && pytest tests/test_tools/test_via_to_coco.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.data_prep.via_to_coco'`.

- [ ] **Step 4: Implement converter**

Create `/Users/htjo/workspace/YOLO/tools/data_prep/via_to_coco.py`:

```python
"""Convert VIA (VGG Image Annotator) JSON to COCO instance-segmentation JSON.

Used to convert hand-labelled climbing wall photos into the format consumed
by yolo/tools/data_loader.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

CATEGORY_MAP = {"hold": 0, "volume": 1}


def convert_via_to_coco(
    via_json: Path,
    output_json: Path,
    image_size_lookup: Optional[Dict[str, Tuple[int, int]]] = None,
    images_dir: Optional[Path] = None,
) -> None:
    """Read a VIA JSON file and write a COCO instance-seg JSON.

    Args:
        via_json: Path to VIA JSON file.
        output_json: Where to write the COCO JSON.
        image_size_lookup: Optional pre-computed {filename: (width, height)}.
            If absent and `images_dir` is given, sizes are read with PIL.
        images_dir: Directory containing the image files; used only if
            `image_size_lookup` is None.
    """
    via = json.loads(Path(via_json).read_text())
    metadata = via["_via_img_metadata"]

    images, annotations = [], []
    image_id = 0
    ann_id = 0

    for entry in metadata.values():
        filename = entry["filename"]
        regions = entry.get("regions", [])
        if not regions:
            continue

        valid_anns = []
        for r in regions:
            shape = r.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) < 3 or len(ys) < 3:
                continue
            hold_type = r.get("region_attributes", {}).get("hold_type")
            if hold_type not in CATEGORY_MAP:
                continue
            valid_anns.append((xs, ys, hold_type))

        if not valid_anns:
            continue

        # resolve image size
        if image_size_lookup and filename in image_size_lookup:
            width, height = image_size_lookup[filename]
        elif images_dir is not None:
            from PIL import Image  # local import: only needed if no lookup
            with Image.open(Path(images_dir) / filename) as im:
                width, height = im.size
        else:
            raise ValueError(
                f"image_size for {filename} unresolved (provide image_size_lookup or images_dir)"
            )

        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )

        for xs, ys, hold_type in valid_anns:
            seg_flat = [v for pair in zip(xs, ys) for v in pair]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_MAP[hold_type],
                    "segmentation": [seg_flat],
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "climbing"}
            for name, cid in CATEGORY_MAP.items()
        ],
    }
    Path(output_json).write_text(json.dumps(coco))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--via", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    args = parser.parse_args()
    convert_via_to_coco(args.via, args.out, images_dir=args.images_dir)
    print(f"Wrote {args.out}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tools/test_via_to_coco.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/data_prep/__init__.py tools/data_prep/via_to_coco.py tests/test_tools/test_via_to_coco.py
git commit -m "✨ [Add] VIA → COCO instance-seg converter for climbing dataset"
```

---

### Task 3: Run conversion + sanity check

**Files:**
- Create: `tools/data_prep/sanity_check.py`
- Generates: `data/climbing_holds/annotations/{train,val}.json`, `data/climbing_holds/images/{train,val}/`, `data/sanity/*.png`

- [ ] **Step 1: Create the train/val image directories with symlinks**

```bash
ARCHIVE=/Users/htjo/workspace/kaggle/hold-segmentation/archive
TARGET=/Users/htjo/workspace/YOLO/data/climbing_holds
mkdir -p $TARGET/images/train $TARGET/images/val $TARGET/annotations

# Train: bh labeled images. The labeled image set is determined from the VIA JSON
#        in the next step; for now symlink ALL bh images and let the converter pick.
for f in $ARCHIVE/bh/*.jpg;       do ln -sf "$f" "$TARGET/images/train/$(basename "$f")"; done
for f in $ARCHIVE/bh-phone/*.jpg; do ln -sf "$f" "$TARGET/images/val/$(basename bh-phone-$(basename "$f"))"; done
for f in $ARCHIVE/sm/*.jpg;       do ln -sf "$f" "$TARGET/images/val/$(basename sm-$(basename "$f"))"; done
ls $TARGET/images/train | wc -l   # expect: 1061
ls $TARGET/images/val   | wc -l   # expect: 240 + 972 = 1212
```

(Filenames in `bh-phone` and `sm` collide — `000.jpg` exists in both — so the symlink target name is prefixed with the source dir.)

- [ ] **Step 2: Run conversion for train (bh)**

```bash
cd /Users/htjo/workspace/YOLO
python -m tools.data_prep.via_to_coco \
    --via /Users/htjo/workspace/kaggle/hold-segmentation/archive/bh-annotation.json \
    --out data/climbing_holds/annotations/train.json \
    --images-dir data/climbing_holds/images/train
```

Expected stdout: `Wrote data/climbing_holds/annotations/train.json`

- [ ] **Step 3: Run conversion for val (bh-phone + sm) into a single JSON**

Because the val JSON must contain images from two different sources (and the symlinks were prefixed), use a small inline Python to merge:

```bash
python - <<'PY'
import json
from pathlib import Path
from tools.data_prep.via_to_coco import convert_via_to_coco

ARCHIVE = Path("/Users/htjo/workspace/kaggle/hold-segmentation/archive")
TGT = Path("data/climbing_holds")

# Convert each, then merge with prefixed filenames matching the symlinks.
for src, prefix in [("bh-phone", "bh-phone-"), ("sm", "sm-")]:
    convert_via_to_coco(
        via_json=ARCHIVE / f"{src}-annotation.json",
        output_json=TGT / f"_tmp_{src}.json",
        images_dir=ARCHIVE / src,
    )

merged = {"images": [], "annotations": [], "categories": None}
img_id = 0
ann_id = 0
for src, prefix in [("bh-phone", "bh-phone-"), ("sm", "sm-")]:
    coco = json.loads((TGT / f"_tmp_{src}.json").read_text())
    if merged["categories"] is None:
        merged["categories"] = coco["categories"]
    id_remap = {}
    for img in coco["images"]:
        new = dict(img)
        new["file_name"] = prefix + img["file_name"]
        new["id"] = img_id
        id_remap[img["id"]] = img_id
        img_id += 1
        merged["images"].append(new)
    for a in coco["annotations"]:
        new = dict(a)
        new["id"] = ann_id
        new["image_id"] = id_remap[a["image_id"]]
        ann_id += 1
        merged["annotations"].append(new)
    (TGT / f"_tmp_{src}.json").unlink()

(TGT / "annotations" / "val.json").write_text(json.dumps(merged))
print(f"Wrote val.json: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
PY
```

Expected stdout includes: `Wrote val.json: 4 images, ~201 annotations`.

- [ ] **Step 4: Verify instance counts match the spec**

```bash
python - <<'PY'
import json
for split in ["train", "val"]:
    coco = json.loads(open(f"data/climbing_holds/annotations/{split}.json").read())
    counts = {0: 0, 1: 0}
    for a in coco["annotations"]:
        counts[a["category_id"]] += 1
    print(f"{split}: images={len(coco['images'])} hold={counts[0]} volume={counts[1]}")
PY
```

Expected: `train: images=15 hold=1408 volume=79` and `val: images=4 hold=189 volume=12` (totals 1597 hold, 91 volume).

- [ ] **Step 5: Write sanity-check visualizer**

Create `/Users/htjo/workspace/YOLO/tools/data_prep/sanity_check.py`:

```python
"""Render polygons from a COCO instance-seg JSON onto images for visual QA."""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np


def render_sanity(
    coco_json: Path, images_dir: Path, output_dir: Path, sample: int = 4
) -> None:
    coco = json.loads(Path(coco_json).read_text())
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    color = {0: (0, 255, 0), 1: (0, 0, 255)}  # hold green, volume red
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled = random.sample(list(images_by_id.values()), min(sample, len(images_by_id)))
    for img in sampled:
        path = images_dir / img["file_name"]
        canvas = cv2.imread(str(path))
        if canvas is None:
            continue
        for a in anns_by_image.get(img["id"], []):
            poly = np.array(a["segmentation"][0], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(canvas, [poly], isClosed=True, color=color[a["category_id"]], thickness=2)
        out = output_dir / img["file_name"]
        cv2.imwrite(str(out), canvas)
        print(f"  {out.name}  ({len(anns_by_image.get(img['id'], []))} polys, {cats})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=4)
    args = p.parse_args()
    render_sanity(args.coco, args.images, args.out, args.sample)
```

- [ ] **Step 6: Run sanity check on both splits**

```bash
mkdir -p sanity
python -m tools.data_prep.sanity_check --coco data/climbing_holds/annotations/train.json \
    --images data/climbing_holds/images/train --out sanity/train --sample 3
python -m tools.data_prep.sanity_check --coco data/climbing_holds/annotations/val.json \
    --images data/climbing_holds/images/val --out sanity/val --sample 4
open sanity/train sanity/val   # macOS: open in Finder for visual review
```

Visually verify polygons land on actual holds/volumes.

- [ ] **Step 7: Commit the sanity helper (annotations are gitignored)**

```bash
git add tools/data_prep/sanity_check.py
git commit -m "✨ [Add] sanity-check visualizer for converted COCO annotations"
```

---

### Task 4: Climbing dataset config

**Files:**
- Create: `yolo/config/dataset/climbing_holds.yaml`

- [ ] **Step 1: Author the dataset config**

Create `/Users/htjo/workspace/YOLO/yolo/config/dataset/climbing_holds.yaml`:

```yaml
path: data/climbing_holds
train: images/train
validation: images/val

class_num: 2
class_list: ["hold", "volume"]
```

- [ ] **Step 2: Verify Hydra can compose with this config**

```bash
python - <<'PY'
from hydra import compose, initialize
with initialize(config_path="yolo/config", version_base=None):
    cfg = compose(config_name="config", overrides=["task=train", "dataset=climbing_holds", "model=v9-t"])
print("class_num:", cfg.dataset.class_num)
print("class_list:", cfg.dataset.class_list)
PY
```

Expected: `class_num: 2`, `class_list: ['hold', 'volume']`.

- [ ] **Step 3: Commit**

```bash
git add yolo/config/dataset/climbing_holds.yaml
git commit -m "✨ [Add] climbing_holds dataset config (2 classes: hold, volume)"
```

---

### Task 5: Phase A smoke test on laptop CPU

**Files:** none new — exercises the existing detection training pipeline with the new dataset.

- [ ] **Step 1: Confirm pretrained v9-t.ckpt is available locally**

```bash
ls weights/v9-t.ckpt 2>&1 || echo "NOT FOUND — will train from scratch (acceptable for smoke)"
```

If absent, the smoke test runs with random init — that's fine for smoke. Real training (Task 7) downloads the official ckpt.

- [ ] **Step 2: Run a 2-epoch smoke training on CPU**

```bash
cd /Users/htjo/workspace/YOLO
python yolo/lazy.py task=train \
    model=v9-t \
    dataset=climbing_holds \
    task.data.batch_size=2 \
    task.data.image_size=320 \
    task.epoch=2 \
    device=cpu \
    weight=False
```

Expected: training starts, loss components are logged each step, two epochs complete, val PNG written under `runs/<...>/`. Total wall-clock ≤ 10 min.

- [ ] **Step 3: Verify smoke pass criteria**

- [ ] Loss decreased between epoch 1 and 2 (check logs).
- [ ] No NaN in any loss component.
- [ ] At least one validation PNG saved with bbox overlays.

Failures here block Phase A — investigate before continuing.

- [ ] **Step 4: No commit needed** (no source changes; this is operational verification).

---

### Task 6: Sync data to Google Drive + push fork

**Files:** none. Operational task.

- [ ] **Step 1: Zip the converted dataset**

```bash
cd /Users/htjo/workspace/YOLO
# follow symlinks so the zip contains real files
zip -ry climbing_holds.zip data/climbing_holds
ls -lh climbing_holds.zip
```

- [ ] **Step 2: Upload to Google Drive**

Manual: drag `climbing_holds.zip` into `My Drive/climbing-holds/` via the Drive web UI (or `rclone copy` if configured). Verify it's visible.

- [ ] **Step 3: Push branch to fork**

```bash
git push -u origin feature/climbing-seg
```

Expected: branch created on `https://github.com/ob-choco/YOLO`. Open a draft PR (optional) for review tracking.

---

### Task 7: Phase A Colab training — `v9-t`

**Files:**
- Create: `notebooks/colab/phase_a_detection.ipynb`

- [ ] **Step 1: Author the notebook**

Create `/Users/htjo/workspace/YOLO/notebooks/colab/phase_a_detection.ipynb` with the following cells. (Use `jupyter nbconvert` or VSCode/Cursor's notebook editor; below is the cell content.)

**Cell 1 — Mount Drive and clone fork:**

```python
from google.colab import drive
drive.mount("/content/drive")

import os, subprocess
os.chdir("/content")
if not os.path.exists("YOLO"):
    subprocess.check_call(["git", "clone", "-b", "feature/climbing-seg", "https://github.com/ob-choco/YOLO.git"])
os.chdir("/content/YOLO")
subprocess.check_call(["git", "pull"])
print(subprocess.check_output(["git", "log", "--oneline", "-3"]).decode())
```

**Cell 2 — Install dependencies:**

```python
!pip install -q -r requirements.txt
!pip install -q pycocotools
```

**Cell 3 — Unzip dataset:**

```python
!mkdir -p /content/YOLO/data
!unzip -q -o /content/drive/MyDrive/climbing-holds/climbing_holds.zip -d /content/YOLO/
!ls /content/YOLO/data/climbing_holds/images/train | wc -l   # expect 1061
!ls /content/YOLO/data/climbing_holds/images/val   | wc -l   # expect 1212
```

**Cell 4 — Download pretrained `v9-t.ckpt`:**

```python
import os, urllib.request
os.makedirs("weights", exist_ok=True)
url = "https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/v9-t.ckpt"
out = "weights/v9-t.ckpt"
if not os.path.exists(out):
    urllib.request.urlretrieve(url, out)
print(os.path.getsize(out), "bytes")
```

**Cell 5 — Train:**

```python
import datetime, subprocess
run_name = f"phaseA-v9-t-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
subprocess.check_call([
    "python", "yolo/lazy.py",
    "task=train",
    "model=v9-t",
    "dataset=climbing_holds",
    "task.data.batch_size=8",
    "task.data.image_size=640",
    "task.epoch=50",
    "device=cuda",
    "weight=weights/v9-t.ckpt",
    f"name={run_name}",
])
```

**Cell 6 — Save artifacts to Drive:**

```python
import shutil
src = f"/content/YOLO/runs/{run_name}"
dst = f"/content/drive/MyDrive/climbing-holds/runs/{run_name}"
shutil.copytree(src, dst, dirs_exist_ok=True)
print("Saved to:", dst)
```

- [ ] **Step 2: Run the notebook on Colab Free (T4)**

Open in Colab via "Open in Colab" badge (or upload the .ipynb). Run all cells. Total time on T4: ~1–3 hours.

- [ ] **Step 3: Confirm artifacts in Drive**

`runs/phaseA-v9-t-<datetime>/` contains: `last.ckpt`, `best.ckpt`, `tensorboard/`, `metrics.csv`, `visualizations/epoch_*/`.

- [ ] **Step 4: Commit the notebook**

```bash
git add notebooks/colab/phase_a_detection.ipynb
git commit -m "✨ [Add] Colab notebook for Phase A detection training"
git push origin feature/climbing-seg
```

---

### Task 8: Phase A Colab training — `v9-s`

**Files:** Reuse `phase_a_detection.ipynb`.

- [ ] **Step 1: In the same Colab notebook, change `model=v9-t` → `model=v9-s` and `weight=weights/v9-s.ckpt`**

The pretrained URL is the same release: `https://github.com/MultimediaTechLab/YOLO/releases/download/v1.0-alpha/v9-s.ckpt`. Update Cell 4 and Cell 5 accordingly. Re-run cells 4–6.

- [ ] **Step 2: Confirm artifacts in Drive**

`runs/phaseA-v9-s-<datetime>/` populated.

- [ ] **Step 3: No source-code change to commit**

---

### Task 9: Phase A gate

**Files:** none.

- [ ] **Step 1: Pull artifacts to laptop for analysis**

```bash
mkdir -p runs/_pulled
# Drive desktop client or rclone — manual is fine
rsync -av "$HOME/Library/CloudStorage/GoogleDrive-*/My Drive/climbing-holds/runs/" runs/_pulled/
```

- [ ] **Step 2: Open visualization PNGs and metrics**

```bash
open runs/_pulled/phaseA-v9-t-*/visualizations/epoch_50/
cat runs/_pulled/phaseA-v9-t-*/metrics.csv | head -20
```

Verify per spec §5.6:
- [ ] Bbox overlays visually capture hold positions on all 4 val images.
- [ ] `mAP@bbox 0.5 > 0` for at least one model size.
- [ ] No training NaN.

If gate fails: investigate (training divergence? annotations broken? hyperparameter issue?) before Phase B.

---

## Phase B — Segmentation Pipeline

### Task 10: `v9-t-seg.yaml` config

**Files:**
- Create: `yolo/config/model/v9-t-seg.yaml`

- [ ] **Step 1: Author the config**

Create `/Users/htjo/workspace/YOLO/yolo/config/model/v9-t-seg.yaml`. Take `yolo/config/model/v9-t.yaml` as starting point (133 lines) and replace **only the `detection:` block**.

The full file (lines 1–93 + auxiliary lines 99–133 from `v9-t.yaml` are copied verbatim; only lines 94–98 below replace the existing `detection:` block):

```yaml
name: v9-t-seg

anchor:
  reg_max: 16

model:
  backbone:
    - Conv:
        args: {out_channels: 16, kernel_size: 3, stride: 2}
        source: 0
    - Conv:
        args: {out_channels: 32, kernel_size: 3, stride: 2}
    - ELAN:
        args: {out_channels: 32, part_channels: 32}

    - AConv:
        args: {out_channels: 64}
    - RepNCSPELAN:
        args:
            out_channels: 64
            part_channels: 64
            csp_args: {repeat_num: 3}
        tags: B3

    - AConv:
        args: {out_channels: 96}
    - RepNCSPELAN:
        args:
            out_channels: 96
            part_channels: 96
            csp_args: {repeat_num: 3}
        tags: B4

    - AConv:
        args: {out_channels: 128}
    - RepNCSPELAN:
        args:
            out_channels: 128
            part_channels: 128
            csp_args: {repeat_num: 3}
        tags: B5

  neck:
    - SPPELAN:
        args: {out_channels: 128}
        tags: N3

    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Concat:
        source: [-1, B4]
    - RepNCSPELAN:
        args:
            out_channels: 96
            part_channels: 96
            csp_args: {repeat_num: 3}
        tags: N4

  head:
    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Concat:
        source: [-1, B3]

    - RepNCSPELAN:
        args:
            out_channels: 64
            part_channels: 64
            csp_args: {repeat_num: 3}
        tags: P3
    - AConv:
        args: {out_channels: 48}
    - Concat:
        source: [-1, N4]

    - RepNCSPELAN:
        args:
            out_channels: 96
            part_channels: 96
            csp_args: {repeat_num: 3}
        tags: P4
    - AConv:
        args: {out_channels: 64}
    - Concat:
        source: [-1, N3]

    - RepNCSPELAN:
        args:
            out_channels: 128
            part_channels: 128
            csp_args: {repeat_num: 3}
        tags: P5

  detection:
    - RepNCSPELAN:
        source: P3
        args: {out_channels: 128, part_channels: 128, csp_args: {repeat_num: 2}}
    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Conv:
        args: {out_channels: 128, kernel_size: 3}

    - MultiheadSegmentation:
        source: [P3, P4, P5, -1]
        args: {num_maskes: 32}
        tags: Main
        output: True

  auxiliary:
    - SPPELAN:
        source: B5
        args: {out_channels: 128}
        tags: A5

    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Concat:
        source: [-1, B4]

    - RepNCSPELAN:
        args:
            out_channels: 96
            part_channels: 96
            csp_args: {repeat_num: 3}
        tags: A4

    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Concat:
        source: [-1, B3]

    - RepNCSPELAN:
        args:
            out_channels: 64
            part_channels: 64
            csp_args: {repeat_num: 3}
        tags: A3

    - MultiheadDetection:
        source: [A3, A4, A5]
        tags: AUX
        output: True
```

- [ ] **Step 2: Verify Hydra can compose**

```bash
python - <<'PY'
from hydra import compose, initialize
with initialize(config_path="yolo/config", version_base=None):
    cfg = compose(config_name="config", overrides=["task=train", "dataset=climbing_holds", "model=v9-t-seg"])
print(cfg.model.name)
PY
```

Expected: `v9-t-seg`.

- [ ] **Step 3: Commit**

```bash
git add yolo/config/model/v9-t-seg.yaml
git commit -m "✨ [Add] v9-t-seg model config (tiny YOLOv9 with MultiheadSegmentation)"
```

---

### Task 11: `v9-s-seg.yaml` config

**Files:**
- Create: `yolo/config/model/v9-s-seg.yaml`

- [ ] **Step 1: Author the config**

Create `/Users/htjo/workspace/YOLO/yolo/config/model/v9-s-seg.yaml` identical to `v9-s.yaml` except for the `detection:` block:

```yaml
  detection:
    - RepNCSPELAN:
        source: P3
        args: {out_channels: 256, part_channels: 256, csp_args: {repeat_num: 2}}
    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Conv:
        args: {out_channels: 256, kernel_size: 3}

    - MultiheadSegmentation:
        source: [P3, P4, P5, -1]
        args: {num_maskes: 32}
        tags: Main
        output: True
```

(All other `backbone`, `neck`, `head`, `auxiliary` sections copied verbatim from `v9-s.yaml`. Top of file: `name: v9-s-seg`.)

- [ ] **Step 2: Verify and commit**

```bash
python -c "from hydra import compose, initialize; \
import sys; \
[None for _ in [initialize(config_path='yolo/config', version_base=None)]]; \
print(compose(config_name='config', overrides=['task=train','dataset=climbing_holds','model=v9-s-seg']).model.name)"
git add yolo/config/model/v9-s-seg.yaml
git commit -m "✨ [Add] v9-s-seg model config"
```

---

### Task 12: Rewire `MultiheadSegmentation.forward` (TDD)

**Files:**
- Modify: `yolo/model/module.py:149-163`
- Create: `tests/test_model/test_multihead_segmentation.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_model/test_multihead_segmentation.py`:

```python
import torch

from yolo.model.module import MultiheadSegmentation


def test_forward_returns_dict_with_detect_mask_coefs_and_proto():
    """Forward emits structured outputs: detection per FPN, mask coef per FPN, single proto."""
    in_channels = [64, 96, 128, 128]   # P3, P4, P5, proto-source channels (matches v9-t-seg)
    head = MultiheadSegmentation(in_channels=in_channels, num_classes=2, num_maskes=32, reg_max=16)
    head.eval()

    # Build dummy feature pyramid + proto source matching the channel widths.
    feats = [
        torch.randn(2, 64, 80, 80),    # P3 stride 8 at 640 input
        torch.randn(2, 96, 40, 40),    # P4 stride 16
        torch.randn(2, 128, 20, 20),   # P5 stride 32
        torch.randn(2, 128, 160, 160), # prototype source — already upsampled to stride 4
    ]
    out = head(feats)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"detect", "mask_coefs", "proto"}

    # detect: list of 3 tuples, each (cls, anc, box) per FPN level
    assert len(out["detect"]) == 3

    # mask_coefs: list of 3 tensors, (B, 32, h, w) per FPN level
    assert len(out["mask_coefs"]) == 3
    assert out["mask_coefs"][0].shape == (2, 32, 80, 80)
    assert out["mask_coefs"][1].shape == (2, 32, 40, 40)
    assert out["mask_coefs"][2].shape == (2, 32, 20, 20)

    # proto: single tensor (B, 32, H/4, W/4)
    assert out["proto"].shape == (2, 32, 160, 160)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model/test_multihead_segmentation.py -v
```

Expected: FAIL — current forward returns a list, not a dict.

- [ ] **Step 3: Rewire forward**

In `/Users/htjo/workspace/YOLO/yolo/model/module.py` lines 149-163, replace the entire `MultiheadSegmentation` class body with:

```python
class MultiheadSegmentation(nn.Module):
    """Multihead segmentation: detection (per FPN) + mask coef (per FPN) + shared prototype.

    Inputs to forward:
        x_list — [P3_feat, P4_feat, P5_feat, proto_source_feat]

    Output:
        dict with keys:
          "detect"      → list[tuple]; one (cls, anc, box) tuple per FPN level (3 levels)
          "mask_coefs"  → list[Tensor]; one (B, num_maskes, h, w) tensor per FPN level
          "proto"       → Tensor (B, num_maskes, H/4, W/4)
    """

    def __init__(self, in_channels: List[int], num_classes: int, num_maskes: int, **head_kwargs):
        super().__init__()
        mask_channels, proto_channels = in_channels[:-1], in_channels[-1]

        self.detect = MultiheadDetection(mask_channels, num_classes, **head_kwargs)
        self.mask_heads = nn.ModuleList(
            [Segmentation((mask_channels[0], in_channel), num_maskes) for in_channel in mask_channels]
        )
        self.proto_head = Conv(proto_channels, num_maskes, 1)

    def forward(self, x_list: List[torch.Tensor]) -> dict:
        feats = x_list[:-1]
        proto_src = x_list[-1]
        detect_outputs = self.detect(feats)
        mask_coefs = [head(f) for head, f in zip(self.mask_heads, feats)]
        proto = self.proto_head(proto_src)
        return {"detect": detect_outputs, "mask_coefs": mask_coefs, "proto": proto}
```

(Note the rename: previous `self.heads` is split into `self.mask_heads` (the per-FPN coefficient heads) and `self.proto_head` (the prototype Conv). This makes weight transfer in Task 13 unambiguous.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_model/test_multihead_segmentation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add yolo/model/module.py tests/test_model/test_multihead_segmentation.py
git commit -m "🔨 [Fix] MultiheadSegmentation.forward emits {detect, mask_coefs, proto}"
```

---

### Task 13: Weight transfer script — detection ckpt → seg ckpt (TDD)

**Files:**
- Create: `tools/weight_transfer/__init__.py`
- Create: `tools/weight_transfer/v9_to_seg.py`
- Create: `tests/test_tools/test_weight_transfer.py`

- [ ] **Step 1: Create package init**

```bash
mkdir -p /Users/htjo/workspace/YOLO/tools/weight_transfer
touch /Users/htjo/workspace/YOLO/tools/weight_transfer/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_tools/test_weight_transfer.py`:

```python
import torch

from tools.weight_transfer.v9_to_seg import transfer_detection_to_seg


def _fake_detection_state(prefix: str = "model.") -> dict:
    """Synthesize a minimal dict resembling a v9 detection ckpt state_dict."""
    return {
        f"{prefix}backbone.0.weight": torch.randn(16, 3, 3, 3),
        f"{prefix}neck.0.weight": torch.randn(64, 16, 3, 3),
        f"{prefix}head.detection.0.weight": torch.randn(8, 64, 1, 1),  # detection head conv
    }


def test_transfer_keeps_backbone_neck_head_weights_and_inserts_seg_init():
    det = _fake_detection_state()
    out = transfer_detection_to_seg(det, num_maskes=32, prototype_channels=128)
    # Backbone/neck/head detection weights are preserved, possibly under renamed keys.
    assert any(k.endswith("backbone.0.weight") for k in out)
    # New mask coef + prototype keys exist with random init shapes.
    proto_keys = [k for k in out if "proto_head" in k]
    assert proto_keys, "expected proto_head weights to be present after transfer"
    mask_head_keys = [k for k in out if "mask_heads" in k]
    assert mask_head_keys, "expected mask_heads weights to be present after transfer"
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_tools/test_weight_transfer.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the transfer**

Create `/Users/htjo/workspace/YOLO/tools/weight_transfer/v9_to_seg.py`:

```python
"""Transfer a detection (`v9-t.ckpt` / `v9-s.ckpt`) state_dict into the
parameter layout expected by `v9-t-seg` / `v9-s-seg`.

Strategy:
  * Backbone, neck, P3/P4/P5 head weights → copied verbatim.
  * `MultiheadSegmentation.detect` (the internal `MultiheadDetection`) →
    receives the source detection-head weights (renamed key prefix).
  * `mask_heads` and `proto_head` parameters → randomly initialized
    (the caller supplies the target model and we use that model's
    fresh-init params for the new keys).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def transfer_detection_to_seg(
    detection_state: dict,
    num_maskes: int,
    prototype_channels: int,
    target_model: Optional[nn.Module] = None,
) -> dict:
    """Build a state_dict suitable for loading into a v9-*-seg model.

    Args:
        detection_state: state_dict from a v9-t / v9-s detection ckpt.
        num_maskes: number of prototype masks (32 by default).
        prototype_channels: prototype branch output channels (128 for v9-t-seg, 256 for v9-s-seg).
        target_model: optional target seg model — if given, its current
            parameter shapes seed random initialization for new keys.

    Returns:
        A state_dict with the union of: copied backbone/neck/head detection
        weights and freshly-initialized mask_heads/proto_head weights.
    """
    out: dict = {}

    # 1) copy any keys whose path is in {backbone, neck, head} as-is.
    for k, v in detection_state.items():
        if any(seg in k for seg in (".backbone.", ".neck.", ".head.")):
            out[k] = v

    # 2) re-route detection-head keys to seg model's MultiheadSegmentation.detect path.
    for k, v in detection_state.items():
        if ".detection." in k:
            new_k = k.replace(".detection.", ".detection.detect.")
            out[new_k] = v

    # 3) seed mask_heads / proto_head from target model if provided, else leave keys absent
    #    (PyTorch `load_state_dict(strict=False)` will keep their fresh init).
    if target_model is not None:
        for k, v in target_model.state_dict().items():
            if "mask_heads" in k or "proto_head" in k:
                out[k] = v.clone()

    # `num_maskes` and `prototype_channels` are accepted for API future-proofing
    # (e.g. a future variant might need them to size the random init manually).
    _ = num_maskes, prototype_channels
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--detection-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prototype-channels", type=int, default=128)
    p.add_argument("--num-maskes", type=int, default=32)
    args = p.parse_args()

    ckpt = torch.load(args.detection_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = transfer_detection_to_seg(state, args.num_maskes, args.prototype_channels)
    torch.save({"state_dict": new_state}, args.out)
    print(f"Wrote {args.out}: {len(new_state)} keys")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_tools/test_weight_transfer.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/weight_transfer/ tests/test_tools/test_weight_transfer.py
git commit -m "✨ [Add] weight-transfer script v9 detection → v9-seg"
```

---

### Task 14: `rasterize_masks` utility (TDD)

**Files:**
- Modify: `yolo/utils/dataset_utils.py` (append function)
- Create: `tests/test_utils/test_mask_rasterize.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_utils/test_mask_rasterize.py`:

```python
import numpy as np
import torch

from yolo.utils.dataset_utils import rasterize_masks


def test_rasterize_single_square_polygon():
    # square polygon covering [0.25, 0.5] in normalized coords on a 640x640 image,
    # rasterized at stride 4 → mask shape (160, 160)
    poly = np.array([[0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]])
    masks = rasterize_masks([poly], image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (1, 160, 160)
    assert masks.dtype == torch.uint8
    # Pixels inside [40, 80] x [40, 80] should be 1
    assert masks[0, 60, 60] == 1
    assert masks[0, 30, 30] == 0
    assert masks[0, 90, 90] == 0


def test_rasterize_empty_returns_zero_tensor():
    masks = rasterize_masks([], image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (0, 160, 160)


def test_rasterize_multiple_polygons():
    polys = [
        np.array([[0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]]),
        np.array([[0.9, 0.9, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0]]),
    ]
    masks = rasterize_masks(polys, image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (2, 160, 160)
    assert masks[0].sum() > 0   # top-left corner has pixels
    assert masks[1].sum() > 0   # bottom-right corner has pixels
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_utils/test_mask_rasterize.py -v
```

Expected: FAIL with `ImportError: cannot import name 'rasterize_masks'`.

- [ ] **Step 3: Implement the function**

Append to `/Users/htjo/workspace/YOLO/yolo/utils/dataset_utils.py`:

```python
import cv2  # add at top of file if missing


def rasterize_masks(polygons, image_size, mask_ratio: int = 4):
    """Rasterize a list of normalized polygons to binary masks at stride-`mask_ratio`.

    Args:
        polygons: List[np.ndarray]; each entry shape (1, 2K) flat normalized [0,1] coords.
        image_size: (W, H) of the source image space.
        mask_ratio: downsample ratio for mask resolution (4 → masks at stride 4).

    Returns:
        torch.uint8 tensor of shape (N, H/mask_ratio, W/mask_ratio).
    """
    import torch

    W, H = image_size
    h, w = H // mask_ratio, W // mask_ratio
    if not polygons:
        return torch.zeros((0, h, w), dtype=torch.uint8)
    masks = np.zeros((len(polygons), h, w), dtype=np.uint8)
    for i, poly in enumerate(polygons):
        flat = np.asarray(poly).reshape(-1, 2)
        pts = (flat * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(masks[i], [pts], 1)
    return torch.from_numpy(masks)
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_utils/test_mask_rasterize.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add yolo/utils/dataset_utils.py tests/test_utils/test_mask_rasterize.py
git commit -m "✨ [Add] rasterize_masks utility for polygon → stride-4 binary mask"
```

---

### Task 15: Polygon-aware augmentation extensions (TDD)

**Files:**
- Modify: `yolo/tools/data_augmentation.py` (extend `HorizontalFlip`, `RandomAffine`)
- Create: `tests/test_tools/test_polygon_augment.py`

- [ ] **Step 1: Read current augmentation classes**

```bash
grep -n "class HorizontalFlip\|class RandomAffine\|class HSV\|def __call__" yolo/tools/data_augmentation.py
```

Note signatures so the polygon parameter is added without breaking existing detection-only callers (default `polygons=None`).

- [ ] **Step 2: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_tools/test_polygon_augment.py`:

```python
import numpy as np
import torch
from PIL import Image

from yolo.tools.data_augmentation import HorizontalFlip, RandomAffine


def _make_image(w=64, h=64):
    return Image.new("RGB", (w, h), color=(128, 128, 128))


def test_horizontal_flip_mirrors_polygon_x():
    img = _make_image()
    bbox = torch.tensor([[0, 0.2, 0.3, 0.4, 0.5]])  # cls + xyxy normalized
    polygons = [np.array([[0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.2, 0.5]])]
    flip = HorizontalFlip(prob=1.0)
    img2, bbox2, polygons2, _ = flip(img, bbox, polygons=polygons)
    pts = polygons2[0].reshape(-1, 2)
    # x' = 1 - x for each point
    np.testing.assert_allclose(pts[:, 0], 1 - np.array([0.2, 0.4, 0.4, 0.2]), atol=1e-6)


def test_random_affine_with_identity_returns_input_polygons():
    img = _make_image()
    bbox = torch.tensor([[0, 0.2, 0.3, 0.4, 0.5]])
    polygons = [np.array([[0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.2, 0.5]])]
    # Force identity transform via 0 degrees, 1.0 scale, 0 translate
    aff = RandomAffine(degrees=0, translate=0, scale=(1.0, 1.0))
    img2, bbox2, polygons2, _ = aff(img, bbox, polygons=polygons)
    np.testing.assert_allclose(polygons2[0], polygons[0], atol=1e-5)
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_tools/test_polygon_augment.py -v
```

Expected: FAIL — current `__call__` doesn't accept `polygons`.

- [ ] **Step 4: Implement polygon support**

In `yolo/tools/data_augmentation.py`, locate `HorizontalFlip.__call__` and `RandomAffine.__call__`. Edit both to accept and transform polygons:

```python
class HorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, bboxes, polygons=None, rev_tensor=None):
        if torch.rand(1).item() < self.prob:
            image = TF.hflip(image)
            if bboxes.numel():
                bboxes[:, [1, 3]] = 1 - bboxes[:, [3, 1]]   # mirror xyxy
            if polygons is not None:
                polygons = [
                    np.concatenate(
                        [(1.0 - p.reshape(-1, 2)[:, 0:1]), p.reshape(-1, 2)[:, 1:2]],
                        axis=1,
                    ).reshape(1, -1)
                    for p in polygons
                ]
        return image, bboxes, polygons, rev_tensor
```

(Apply the same `polygons=None` parameter and `affine` matrix application to `RandomAffine`. The exact body depends on the existing implementation — preserve current image/bbox handling, then apply the same affine matrix to each polygon's normalized points using `cv2.transform` after de-normalizing to pixel coords, then re-normalizing.)

For `RandomAffine`, the polygon transform uses the same 2x3 matrix `M` that is applied to the image:

```python
if polygons is not None:
    W, H = image.size
    transformed = []
    for p in polygons:
        pts = p.reshape(-1, 2) * np.array([W, H], dtype=np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)        # (N, 3)
        warped = pts_h @ M.T                                # (N, 2)
        warped /= np.array([W, H], dtype=np.float32)
        transformed.append(warped.reshape(1, -1))
    polygons = transformed
```

Ensure `HSV.__call__` simply passes `polygons` through unchanged.

- [ ] **Step 5: Run to verify tests pass**

```bash
pytest tests/test_tools/test_polygon_augment.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add yolo/tools/data_augmentation.py tests/test_tools/test_polygon_augment.py
git commit -m "✨ [Add] polygon-aware HorizontalFlip and RandomAffine"
```

---

### Task 16: `Mosaic` augmentation (TDD)

**Files:**
- Modify: `yolo/tools/data_augmentation.py` (add `Mosaic` class)
- Create: `tests/test_tools/test_mosaic.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_tools/test_mosaic.py`:

```python
import numpy as np
import torch
from PIL import Image

from yolo.tools.data_augmentation import Mosaic


def _sample(idx):
    img = Image.new("RGB", (64, 64), color=(idx * 60 % 256, 0, 0))
    bbox = torch.tensor([[0, 0.1, 0.1, 0.5, 0.5]], dtype=torch.float32)
    polygons = [np.array([[0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]])]
    return img, bbox, polygons


def test_mosaic_combines_four_samples():
    samples = [_sample(i) for i in range(4)]
    mosaic = Mosaic(image_size=(128, 128))
    out_img, out_bbox, out_polys, _ = mosaic(samples)
    assert out_img.size == (128, 128)
    # Four input bboxes — but some may have been clipped to zero area; expect ≤ 4 ≥ 1
    assert 1 <= out_bbox.shape[0] <= 4
    assert len(out_polys) == out_bbox.shape[0]


def test_mosaic_clips_polygons_to_canvas():
    samples = [_sample(i) for i in range(4)]
    mosaic = Mosaic(image_size=(128, 128))
    _, out_bbox, out_polys, _ = mosaic(samples)
    for bbox, poly in zip(out_bbox, out_polys):
        # all coords in [0, 1]
        assert (bbox[1:] >= 0).all() and (bbox[1:] <= 1).all()
        pts = poly.reshape(-1, 2)
        assert (pts >= 0).all() and (pts <= 1).all()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_tools/test_mosaic.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `Mosaic`**

Append to `/Users/htjo/workspace/YOLO/yolo/tools/data_augmentation.py`:

```python
class Mosaic:
    """Standard YOLO 4-image mosaic augmentation.

    Takes four (image, bbox, polygons) samples and composes them into a single
    canvas of `image_size` with a randomized split point. Bboxes and polygons
    are translated into their respective quadrant and clipped to canvas.
    """

    def __init__(self, image_size, center_jitter: float = 0.25):
        self.W, self.H = image_size
        self.center_jitter = center_jitter

    def __call__(self, samples):
        # Pick a random center near the middle.
        cx = int(self.W * (0.5 + (np.random.rand() - 0.5) * self.center_jitter))
        cy = int(self.H * (0.5 + (np.random.rand() - 0.5) * self.center_jitter))

        canvas = Image.new("RGB", (self.W, self.H), color=(114, 114, 114))
        boxes_out, polys_out = [], []

        # Quadrant target (x1,y1,x2,y2) on the canvas.
        quads = [
            (0, 0, cx, cy),
            (cx, 0, self.W, cy),
            (0, cy, cx, self.H),
            (cx, cy, self.W, self.H),
        ]

        for sample, (qx1, qy1, qx2, qy2) in zip(samples, quads):
            img, bbox, polygons = sample[0], sample[1], sample[2]
            qw, qh = qx2 - qx1, qy2 - qy1
            if qw <= 0 or qh <= 0:
                continue
            img_resized = img.resize((qw, qh))
            canvas.paste(img_resized, (qx1, qy1))

            for i in range(bbox.shape[0]):
                cls, x1, y1, x2, y2 = bbox[i].tolist()
                # original normalized coords → quadrant pixel space → canvas normalized
                px1 = qx1 + x1 * qw
                py1 = qy1 + y1 * qh
                px2 = qx1 + x2 * qw
                py2 = qy1 + y2 * qh
                # clip
                px1c = max(px1, qx1); py1c = max(py1, qy1)
                px2c = min(px2, qx2); py2c = min(py2, qy2)
                if px2c - px1c < 1 or py2c - py1c < 1:
                    continue   # degenerate after clip
                boxes_out.append([
                    cls,
                    px1c / self.W, py1c / self.H,
                    px2c / self.W, py2c / self.H,
                ])

                # polygon for this instance: same translation; clip with quadrant rect.
                if i < len(polygons):
                    pts = polygons[i].reshape(-1, 2)
                    pts_canvas = np.column_stack([
                        qx1 + pts[:, 0] * qw,
                        qy1 + pts[:, 1] * qh,
                    ])
                    pts_canvas[:, 0] = np.clip(pts_canvas[:, 0], qx1, qx2)
                    pts_canvas[:, 1] = np.clip(pts_canvas[:, 1], qy1, qy2)
                    pts_norm = pts_canvas / np.array([self.W, self.H], dtype=np.float32)
                    polys_out.append(pts_norm.reshape(1, -1))

        boxes_out = (
            torch.tensor(boxes_out, dtype=torch.float32)
            if boxes_out else torch.zeros((0, 5), dtype=torch.float32)
        )
        return canvas, boxes_out, polys_out, None
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_tools/test_mosaic.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add yolo/tools/data_augmentation.py tests/test_tools/test_mosaic.py
git commit -m "✨ [Add] Mosaic augmentation with polygon support"
```

---

### Task 17: Data loader polygon retention (TDD)

**Files:**
- Modify: `yolo/tools/data_loader.py` (extend `load_valid_labels`, `__getitem__`)

- [ ] **Step 1: Write the failing test**

Append to `/Users/htjo/workspace/YOLO/tests/test_tools/test_data_loader.py` (file already exists):

```python
import numpy as np
import torch
from yolo.tools.data_loader import YoloDataset


def test_load_valid_labels_returns_polygons_when_segmentation_task(monkeypatch):
    """When task_type=='segmentation', __getitem__ returns (image, bboxes, polygons, rev, path)."""
    # synthesize a single-image dataset with one polygon annotation
    dataset = YoloDataset.__new__(YoloDataset)   # bypass full init
    dataset.task_type = "segmentation"
    seg_data = [[0, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]]
    bboxes, polygons = dataset.load_valid_labels("img1", seg_data)
    assert bboxes.shape == (1, 5)
    assert isinstance(polygons, list)
    assert polygons[0].shape == (1, 8)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_tools/test_data_loader.py::test_load_valid_labels_returns_polygons_when_segmentation_task -v
```

Expected: FAIL — `task_type` and polygon return are absent.

- [ ] **Step 3: Modify `YoloDataset.__init__` and `load_valid_labels`**

In `yolo/tools/data_loader.py`:

1. Add `self.task_type = getattr(dataset_cfg, "task_type", "detection")` to `__init__`.
2. Change `load_valid_labels` to also collect normalized polygon flat arrays. Replace existing implementation:

```python
def load_valid_labels(self, label_path: str, seg_data_one_img: list):
    """Return (bboxes_tensor, polygons_or_None).

    bboxes_tensor: (N, 5) cls + xyxy normalized — same format as before.
    polygons:      list of np.ndarray (1, 2K) normalized [0,1], parallel to bboxes,
                   only when self.task_type == 'segmentation'; else None.
    """
    import numpy as np
    bboxes, polygons = [], []
    for seg_data in seg_data_one_img:
        cls = seg_data[0]
        points = np.array(seg_data[1:]).reshape(-1, 2).clip(0, 1)
        valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
        if valid_points.size > 1:
            bbox = torch.tensor(
                [cls, *valid_points.min(axis=0), *valid_points.max(axis=0)]
            )
            bboxes.append(bbox)
            polygons.append(valid_points.reshape(1, -1).astype(np.float32))
    if not bboxes:
        empty_bbox = torch.zeros((0, 5))
        return (empty_bbox, [] if self.task_type == "segmentation" else None)
    return (
        torch.stack(bboxes),
        polygons if self.task_type == "segmentation" else None,
    )
```

3. Update all internal call sites to unpack a 2-tuple:

```python
# in _filter_data:
labels, polygons = self.load_valid_labels(image_id, image_seg_annotations)
data.append((img_path, labels, polygons, width / height))   # add polygons to tuple
```

(also update the constructor's storage: change `self.bboxes` to `(self.bboxes, self.polygons)` aligned with the new tuple.)

4. Update `__getitem__` to thread polygons through augmentation and return them:

```python
def __getitem__(self, idx):
    img, bboxes, img_path = self.get_data(idx)
    polygons = self.polygons[idx] if self.task_type == "segmentation" else None

    if self.dynamic_shape:
        self._update_image_size(idx)

    img, bboxes, polygons, rev_tensor = self.transform(img, bboxes, polygons=polygons)
    bboxes[:, [1, 3]] *= self.image_size[0]
    bboxes[:, [2, 4]] *= self.image_size[1]

    if self.task_type == "segmentation":
        return img, bboxes, polygons, rev_tensor, img_path
    return img, bboxes, rev_tensor, img_path
```

- [ ] **Step 4: Run all data_loader tests**

```bash
pytest tests/test_tools/test_data_loader.py -v
```

Expected: existing detection tests still pass; the new test passes.

- [ ] **Step 5: Commit**

```bash
git add yolo/tools/data_loader.py tests/test_tools/test_data_loader.py
git commit -m "✨ [Add] data loader polygon retention for segmentation task"
```

---

### Task 18: Collate function for masks (TDD)

**Files:**
- Modify: `yolo/tools/data_loader.py:201-228` (extend `collate_fn`)

- [ ] **Step 1: Add failing test**

Append to `tests/test_tools/test_data_loader.py`:

```python
def test_collate_fn_pads_masks_for_segmentation():
    from yolo.tools.data_loader import collate_fn_seg
    import torch
    # two samples, different number of instances
    img = torch.zeros(3, 64, 64)
    rev = torch.zeros(4)
    s1 = (img, torch.tensor([[0, 0.1, 0.1, 0.5, 0.5]]), torch.ones(1, 16, 16, dtype=torch.uint8), rev, "a")
    s2 = (img, torch.tensor([[1, 0.2, 0.2, 0.4, 0.4],
                             [0, 0.3, 0.3, 0.6, 0.6]]),
          torch.ones(2, 16, 16, dtype=torch.uint8), rev, "b")
    bs, imgs, targets, masks, rev_t, paths = collate_fn_seg([s1, s2])
    assert bs == 2
    assert imgs.shape == (2, 3, 64, 64)
    assert targets.shape == (2, 2, 5)            # padded to max instances=2
    assert masks.shape == (2, 2, 16, 16)         # padded similarly
    assert paths == ("a", "b")
```

- [ ] **Step 2: Run to fail**

```bash
pytest tests/test_tools/test_data_loader.py::test_collate_fn_pads_masks_for_segmentation -v
```

- [ ] **Step 3: Implement `collate_fn_seg`**

Append to `yolo/tools/data_loader.py`:

```python
def collate_fn_seg(batch):
    """Variant of `collate_fn` for segmentation: also stacks padded GT masks.

    Each sample is (image, bboxes (N,5), masks (N, h, w), rev, path).
    Padded outputs:
      batch_targets: (B, N_max, 5), pad cls=-1
      batch_masks:   (B, N_max, h, w) zeros for padding
    """
    import torch
    batch_size = len(batch)
    target_sizes = [item[1].size(0) for item in batch]
    n_max = min(max(target_sizes), 100)

    batch_targets = torch.full((batch_size, n_max, 5), -1.0)
    sample_masks = batch[0][2]
    h, w = sample_masks.shape[-2:]
    batch_masks = torch.zeros((batch_size, n_max, h, w), dtype=torch.uint8)

    for idx, n in enumerate(target_sizes):
        n_use = min(n, n_max)
        batch_targets[idx, :n_use] = batch[idx][1][:n_use]
        batch_masks[idx, :n_use] = batch[idx][2][:n_use]

    images = torch.stack([item[0] for item in batch])
    revs = torch.stack([item[3] for item in batch])
    paths = tuple(item[4] for item in batch)
    return batch_size, images, batch_targets, batch_masks, revs, paths
```

Then in `create_dataloader`, route to this collate_fn when `dataset_cfg.task_type == "segmentation"`:

```python
collate = collate_fn_seg if getattr(dataset_cfg, "task_type", "detection") == "segmentation" else collate_fn
```

- [ ] **Step 4: Pass tests**

```bash
pytest tests/test_tools/test_data_loader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add yolo/tools/data_loader.py tests/test_tools/test_data_loader.py
git commit -m "✨ [Add] collate_fn_seg padding masks for segmentation batching"
```

---

### Task 19: `MaskLoss` (TDD)

**Files:**
- Modify: `yolo/tools/loss_functions.py` (append `MaskLoss`)
- Create: `tests/test_tools/test_mask_loss.py`

- [ ] **Step 1: Write failing test**

Create `/Users/htjo/workspace/YOLO/tests/test_tools/test_mask_loss.py`:

```python
import torch

from yolo.tools.loss_functions import MaskLoss


def test_mask_loss_returns_finite_scalar_with_grad():
    nm = 32
    B, A, H, W = 2, 16, 64, 64           # 16 anchors total, 64x64 mask res
    coefs = torch.randn(B, A, nm, requires_grad=True)
    proto = torch.randn(B, nm, H, W, requires_grad=True)
    gt_masks = torch.randint(0, 2, (B, 4, H, W), dtype=torch.uint8)   # 4 GT per image
    target_bbox = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.8, 0.8]],
        [[0.0, 0.0, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]],
    ])
    # match table: for each anchor, which GT (or -1 for negative)
    match_idx = torch.full((B, A), -1, dtype=torch.long)
    match_idx[:, :4] = torch.arange(4)   # first 4 anchors in each batch matched to GT 0..3

    loss_fn = MaskLoss(bce_weight=0.5, dice_weight=0.5)
    loss = loss_fn(coefs, proto, gt_masks, target_bbox, match_idx)
    assert torch.isfinite(loss)
    loss.backward()
    assert coefs.grad is not None and torch.isfinite(coefs.grad).all()
    assert proto.grad is not None and torch.isfinite(proto.grad).all()


def test_mask_loss_zero_when_no_positives():
    nm = 32
    B, A, H, W = 1, 8, 32, 32
    coefs = torch.randn(B, A, nm)
    proto = torch.randn(B, nm, H, W)
    gt_masks = torch.zeros(B, 0, H, W, dtype=torch.uint8)
    target_bbox = torch.zeros(B, 0, 4)
    match_idx = torch.full((B, A), -1, dtype=torch.long)
    loss = MaskLoss()(coefs, proto, gt_masks, target_bbox, match_idx)
    assert loss.item() == 0.0
```

- [ ] **Step 2: Run to fail**

```bash
pytest tests/test_tools/test_mask_loss.py -v
```

- [ ] **Step 3: Implement `MaskLoss`**

Append to `/Users/htjo/workspace/YOLO/yolo/tools/loss_functions.py`:

```python
class MaskLoss(nn.Module):
    """YOLACT-style mask loss: BCE + Dice with bbox crop, averaged over positives.

    Args (forward):
        coefs:        (B, A, nm) per-anchor mask coefficient predictions
        proto:        (B, nm, H, W) shared prototype masks per image
        gt_masks:     (B, N_max, H, W) GT masks (uint8 or float)
        target_bbox:  (B, N_max, 4) GT bbox xyxy in normalized [0,1] coords
        match_idx:    (B, A) GT index each anchor is matched to (-1 = negative)

    Returns:
        scalar loss tensor.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    @staticmethod
    def _crop(mask: Tensor, bbox: Tensor) -> Tensor:
        """Zero out everything outside the bbox region. mask (H, W); bbox xyxy in [0,1]."""
        H, W = mask.shape[-2:]
        x1 = (bbox[0] * W).clamp(0, W).long()
        y1 = (bbox[1] * H).clamp(0, H).long()
        x2 = (bbox[2] * W).clamp(0, W).long()
        y2 = (bbox[3] * H).clamp(0, H).long()
        cropped = torch.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return cropped

    def forward(self, coefs, proto, gt_masks, target_bbox, match_idx):
        device = coefs.device
        positives = match_idx >= 0   # (B, A)
        if not positives.any():
            return torch.zeros((), device=device)

        # gather GT-side data per positive
        b_idx, a_idx = positives.nonzero(as_tuple=True)
        gt_idx = match_idx[b_idx, a_idx]   # (P,)
        coef = coefs[b_idx, a_idx]          # (P, nm)
        proto_b = proto[b_idx]              # (P, nm, H, W)
        gt = gt_masks[b_idx, gt_idx].float()       # (P, H, W)
        bbox = target_bbox[b_idx, gt_idx]          # (P, 4)

        # synthesize predicted masks
        # einsum("pn,pnhw->phw")
        pred_logit = torch.einsum("pn,pnhw->phw", coef, proto_b)   # (P, H, W)

        # crop both pred and gt to bbox region
        cropped_pred, cropped_gt = [], []
        for p in range(pred_logit.shape[0]):
            cp = self._crop(pred_logit[p], bbox[p])
            cg = self._crop(gt[p], bbox[p])
            cropped_pred.append(cp)
            cropped_gt.append(cg)
        cropped_pred = torch.stack(cropped_pred)
        cropped_gt = torch.stack(cropped_gt)

        # BCE on logits, mean over (H, W)
        bce_per = self.bce(cropped_pred, cropped_gt).mean(dim=(-1, -2))

        # Dice on sigmoid
        prob = cropped_pred.sigmoid()
        intersection = (prob * cropped_gt).sum(dim=(-1, -2))
        union = prob.sum(dim=(-1, -2)) + cropped_gt.sum(dim=(-1, -2))
        dice = 1 - (2 * intersection + 1) / (union + 1)

        return (self.bce_w * bce_per + self.dice_w * dice).mean()
```

- [ ] **Step 4: Pass test**

```bash
pytest tests/test_tools/test_mask_loss.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add yolo/tools/loss_functions.py tests/test_tools/test_mask_loss.py
git commit -m "✨ [Add] MaskLoss (BCE + Dice + bbox crop) for instance segmentation"
```

---

### Task 20: `DualLoss` extension for segmentation

**Files:**
- Modify: `yolo/tools/loss_functions.py:109-142`

- [ ] **Step 1: Write a smoke test that constructs DualLoss in seg mode**

Append to `tests/test_tools/test_loss_functions.py`:

```python
def test_dualloss_constructs_in_segmentation_mode(cfg, monkeypatch):
    cfg.model.task_type = "segmentation"
    cfg.task.loss.objective = {"BoxLoss": 7.5, "DFLoss": 1.5, "BCELoss": 0.5, "MaskLoss": 7.5}
    from yolo.utils.bounding_box_utils import generate_anchors, Vec2Box
    from yolo.tools.loss_functions import DualLoss
    # We don't need full Vec2Box state; just construct DualLoss and ensure mask_loss exists.
    class DummyVec2Box:
        anchor_grid = None; scaler = None
    dl = DualLoss(cfg, DummyVec2Box())
    assert hasattr(dl, "mask_loss")
```

- [ ] **Step 2: Run to fail**

```bash
pytest tests/test_tools/test_loss_functions.py::test_dualloss_constructs_in_segmentation_mode -v
```

- [ ] **Step 3: Modify `DualLoss`**

Replace `DualLoss` in `yolo/tools/loss_functions.py:109-135`:

```python
class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

        self.task_type = getattr(cfg.model, "task_type", "detection")
        if self.task_type == "segmentation":
            mask_cfg = getattr(loss_cfg, "mask", None)
            bce_w = getattr(mask_cfg, "bce_weight", 0.5) if mask_cfg else 0.5
            dice_w = getattr(mask_cfg, "dice_weight", 0.5) if mask_cfg else 0.5
            self.mask_loss = MaskLoss(bce_weight=bce_w, dice_weight=dice_w)
            self.mask_rate = loss_cfg.objective.get("MaskLoss", 7.5)

    def __call__(self, aux_predicts, main_predicts, targets, mask_inputs=None):
        # detection (always)
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        total = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        ]
        loss_dict = {f"Loss/{n}Loss": v.detach().item() for n, v in zip(["Box", "DFL", "BCE"], total)}

        # segmentation (main only)
        if self.task_type == "segmentation" and mask_inputs is not None:
            coefs, proto, gt_masks, target_bbox, match_idx = mask_inputs
            ml = self.mask_loss(coefs, proto, gt_masks, target_bbox, match_idx)
            total.append(self.mask_rate * ml)
            loss_dict["Loss/MaskLoss"] = ml.detach().item()

        return sum(total), loss_dict
```

- [ ] **Step 4: Pass test**

```bash
pytest tests/test_tools/test_loss_functions.py -v
```

- [ ] **Step 5: Commit**

```bash
git add yolo/tools/loss_functions.py tests/test_tools/test_loss_functions.py
git commit -m "🔨 [Update] DualLoss routes mask path when task_type=segmentation"
```

---

### Task 21: `task_type` config plumbing

**Files:**
- Modify: `yolo/config/config.py` (`ModelConfig` dataclass)
- Modify: `yolo/config/model/v9-t-seg.yaml`, `v9-s-seg.yaml`, `v9-c-seg.yaml`

- [ ] **Step 1: Add field to ModelConfig**

In `yolo/config/config.py`, locate the `ModelConfig` dataclass and add:

```python
task_type: str = "detection"   # "detection" | "segmentation" | "classification"
```

- [ ] **Step 2: Set field in seg model configs**

Append to top of `yolo/config/model/v9-t-seg.yaml`, `v9-s-seg.yaml`, `v9-c-seg.yaml` (under `name:`):

```yaml
task_type: segmentation
```

- [ ] **Step 3: Verify Hydra parses correctly**

```bash
python -c "from hydra import compose, initialize; \
  initialize(config_path='yolo/config', version_base=None); \
  print(compose('config', overrides=['model=v9-t-seg','task=train','dataset=climbing_holds']).model.task_type)"
```

Expected: `segmentation`.

- [ ] **Step 4: Commit**

```bash
git add yolo/config/config.py yolo/config/model/v9-t-seg.yaml yolo/config/model/v9-s-seg.yaml yolo/config/model/v9-c-seg.yaml
git commit -m "✨ [Add] ModelConfig.task_type field; set segmentation on seg configs"
```

---

### Task 22: Solver `training_step` extension for segmentation

**Files:**
- Modify: `yolo/tools/solver.py:87-102`
- Modify: `yolo/tools/solver.py` constructor (handle `task_type`)

- [ ] **Step 1: Read current `training_step`**

```bash
sed -n '60,110p' yolo/tools/solver.py
```

- [ ] **Step 2: Modify `training_step` to thread mask data**

Replace the body of `training_step` in `yolo/tools/solver.py` (lines 87-102):

```python
def training_step(self, batch, batch_idx):
    lr_dict = self.trainer.optimizers[0].next_batch()

    if self.task_type == "segmentation":
        batch_size, images, targets, gt_masks, *_ = batch
    else:
        batch_size, images, targets, *_ = batch
        gt_masks = None

    raw = self(images)

    if self.task_type == "segmentation":
        # raw['Main'] is now a dict {detect, mask_coefs, proto}; raw['AUX'] is the detection list
        aux_predicts = self.vec2box(raw["AUX"])
        main_detect = self.vec2box(raw["Main"]["detect"])

        # gather mask coefs across FPN levels into (B, A, nm)
        coefs = torch.cat(
            [c.flatten(2).transpose(1, 2) for c in raw["Main"]["mask_coefs"]],
            dim=1,
        )
        proto = raw["Main"]["proto"]

        # BoxMatcher's matched index (added in Task 20 expectation):
        # we obtain it by re-running the matcher on main_detect to get (match_idx, valid).
        # This avoids changing the matcher signature; small extra cost is acceptable.
        from yolo.utils.bounding_box_utils import BoxMatcher  # local import to avoid cycles
        # Already used inside YOLOLoss; we re-extract index for mask.
        match_idx = self._compute_match_idx(main_detect, targets)
        target_bbox = targets[..., 1:5]

        loss, loss_item = self.loss_fn(
            aux_predicts, main_detect, targets,
            mask_inputs=(coefs, proto, gt_masks, target_bbox, match_idx),
        )
    else:
        aux_predicts = self.vec2box(raw["AUX"])
        main_predicts = self.vec2box(raw["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)

    self.log_dict(loss_item, prog_bar=True, on_epoch=True, batch_size=batch_size, rank_zero_only=True)
    self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
    return loss * batch_size
```

- [ ] **Step 3: Add `_compute_match_idx` helper to the same TrainModel class**

Append a method:

```python
def _compute_match_idx(self, main_predicts, targets):
    """Returns (B, A) tensor with GT index per anchor (-1 negative).
    Reuses BoxMatcher logic exposed via YOLOLoss internals."""
    preds_cls, _, preds_box = main_predicts
    matcher = self.loss_fn.loss.matcher
    align_targets, valid = matcher(targets, (preds_cls.detach(), preds_box.detach()))
    # `align_targets` shape is (B, A, 5+) and the first column is the GT index inside the
    # batch; valid is (B, A) bool. Convert to (-1) for invalid.
    match = align_targets[..., 0].long()
    match[~valid] = -1
    return match
```

(If `align_targets` does not contain a target-index column in your repo's BoxMatcher, extend `BoxMatcher.__call__` to return the matched index as a third return value — see Risk 2 in spec §7.)

- [ ] **Step 4: Add `task_type` to TrainModel constructor**

In `yolo/tools/solver.py`, in the `TrainModel.__init__` (or equivalent base class init), add:

```python
self.task_type = getattr(cfg.model, "task_type", "detection")
```

- [ ] **Step 5: Run Phase A regression tests**

```bash
pytest tests/test_tools/test_solver.py -v
```

Expected: detection-only tests still pass (segmentation path is gated).

- [ ] **Step 6: Commit**

```bash
git add yolo/tools/solver.py
git commit -m "🔨 [Add] training_step segmentation path with mask loss inputs"
```

---

### Task 23: Inference visualization — `draw_masks`

**Files:**
- Modify: `yolo/tools/drawer.py` (add `draw_masks`)

- [ ] **Step 1: Author `draw_masks`**

Append to `/Users/htjo/workspace/YOLO/yolo/tools/drawer.py`:

```python
def draw_masks(image, masks, bboxes, idx2label, alpha: float = 0.5):
    """Overlay per-instance masks (uint8 0/1) on an image with bbox + class label.

    image:   PIL.Image RGB
    masks:   list of (H, W) uint8 tensors per instance
    bboxes:  (N, 6) xyxy + cls + score (or (N, 5) without score)
    """
    import numpy as np
    from PIL import Image, ImageDraw

    np_img = np.array(image).astype(np.float32)
    palette = np.array(
        [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
        dtype=np.float32,
    )
    overlay = np_img.copy()
    for i, m in enumerate(masks):
        col = palette[i % len(palette)]
        if m.shape != np_img.shape[:2]:
            m_resized = np.array(Image.fromarray(m.numpy() * 255).resize((np_img.shape[1], np_img.shape[0]))) > 127
        else:
            m_resized = m.numpy() > 0
        overlay[m_resized] = (1 - alpha) * np_img[m_resized] + alpha * col
    blended = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(blended)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, cls = bbox[:5]
        col = tuple(int(c) for c in palette[i % len(palette)])
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        draw.text((x1, max(0, y1 - 10)), idx2label[int(cls)], fill=col)
    return blended
```

- [ ] **Step 2: Sanity test the helper**

```bash
python - <<'PY'
import torch
from PIL import Image
from yolo.tools.drawer import draw_masks
img = Image.new("RGB", (64, 64), (200, 200, 200))
m1 = torch.zeros(64, 64, dtype=torch.uint8); m1[10:30, 10:30] = 1
out = draw_masks(img, [m1], [[10, 10, 30, 30, 0]], idx2label=["hold", "volume"])
out.save("/tmp/draw_masks_smoke.png")
print("OK")
PY
```

- [ ] **Step 3: Commit**

```bash
git add yolo/tools/drawer.py
git commit -m "✨ [Add] draw_masks helper for segmentation result visualization"
```

---

### Task 24: Phase B smoke test on laptop CPU

**Files:** none new.

- [ ] **Step 1: Generate seg-init weights from Phase A best detection ckpt**

```bash
mkdir -p weights
python -m tools.weight_transfer.v9_to_seg \
    --detection-ckpt runs/_pulled/phaseA-v9-t-*/best.ckpt \
    --out weights/v9-t-seg-init.pt \
    --prototype-channels 128
```

- [ ] **Step 2: Run a 2-epoch seg smoke**

```bash
python yolo/lazy.py task=train \
    model=v9-t-seg \
    dataset=climbing_holds \
    task.data.batch_size=2 \
    task.data.image_size=320 \
    task.epoch=2 \
    device=cpu \
    weight=weights/v9-t-seg-init.pt
```

Expected: `Loss/MaskLoss` appears in logs, finite, decreases (or at least non-NaN). Mask overlay PNG written under `runs/`. Total wall-clock ≤ 15 min.

- [ ] **Step 3: Verify smoke pass criteria**

- [ ] All 4 loss components (Box/DFL/BCE/Mask) finite
- [ ] Mask loss decreased between epoch 1 and 2
- [ ] Mask overlay PNG contains ≥1 colored region

If failed: investigate (most likely the `_compute_match_idx` integration; debug with a tiny synthetic batch).

---

### Task 25: Phase B Colab training — `v9-t-seg`

**Files:**
- Create: `notebooks/colab/phase_b_segmentation.ipynb`

- [ ] **Step 1: Author the notebook**

Same structure as `phase_a_detection.ipynb` (Task 7) with these differences:

- Cell 4 also downloads `v9-t.ckpt` and runs the weight transfer:

  ```python
  !python -m tools.weight_transfer.v9_to_seg \
      --detection-ckpt weights/v9-t.ckpt \
      --out weights/v9-t-seg-init.pt \
      --prototype-channels 128
  ```

- Cell 5 trains:

  ```python
  subprocess.check_call([
      "python", "yolo/lazy.py",
      "task=train",
      "model=v9-t-seg",
      "dataset=climbing_holds",
      "task.data.batch_size=8",
      "task.data.image_size=640",
      "task.epoch=100",
      "device=cuda",
      "weight=weights/v9-t-seg-init.pt",
      f"name={run_name}",
  ])
  ```

- Cell 6 saves `runs/phaseB-v9-t-seg-<datetime>/` to Drive.

- [ ] **Step 2: Run on Colab Free (T4)**

Total time on T4: ~3–6 hours. Auto-resume from `last.ckpt` if disconnect.

- [ ] **Step 3: Confirm artifacts in Drive** (`runs/phaseB-v9-t-seg-<datetime>/` populated).

- [ ] **Step 4: Commit notebook**

```bash
git add notebooks/colab/phase_b_segmentation.ipynb
git commit -m "✨ [Add] Colab notebook for Phase B segmentation training"
git push origin feature/climbing-seg
```

---

### Task 26: Phase B Colab training — `v9-s-seg`

**Files:** Reuse `phase_b_segmentation.ipynb`.

- [ ] **Step 1: Edit notebook to use `v9-s.ckpt` and `--prototype-channels 256`**

- [ ] **Step 2: Run cells 4–6**

- [ ] **Step 3: Confirm artifacts in Drive** (`runs/phaseB-v9-s-seg-<datetime>/`).

---

### Task 27: Phase B gate

**Files:** none.

- [ ] **Step 1: Pull artifacts**

```bash
rsync -av "$HOME/Library/CloudStorage/GoogleDrive-*/My Drive/climbing-holds/runs/phaseB-*" runs/_pulled/
```

- [ ] **Step 2: Visual + metric review per spec §5.6 Phase B exit criteria**

- [ ] Mask overlays on the 4 val images visually capture hold/volume shapes
- [ ] `mAP@mask 0.5 > 0` for at least one model size
- [ ] Per-domain breakdown logged (`val/bh-phone`, `val/sm`)
- [ ] Per-class breakdown logged (`hold`, `volume`)

If gate passes — **Phase B complete; deliverable is the seg model + visualizations.**

If fails — see spec §7 Risks; most likely candidates are mask threshold (sweep needed), insufficient training (more epochs), or matching index extraction bugs.

- [ ] **Step 3: (Optional) Open final results PR**

```bash
gh pr create --title "Climbing holds & volumes segmentation — Phase B complete" \
  --body "Closes the climbing-seg implementation. Spec: docs/superpowers/specs/2026-05-06-climbing-holds-segmentation-design.md. See attached visualizations and metrics.csv."
```

---

## Self-Review

### Spec coverage

| Spec section | Implementing task(s) |
|---|---|
| §2 Data Pipeline | Tasks 2, 3, 4 |
| §3 Model Architecture (configs + transfer) | Tasks 10, 11, 13 |
| §3 `MultiheadSegmentation` rewire | Task 12 |
| §4.1 Phase A | Tasks 4, 5, 6, 7, 8, 9 |
| §4.2.1 Data loader extension | Task 17 |
| §4.2.2 Mask rasterization | Task 14 |
| §4.2.3 Augmentation (polygon + Mosaic) | Tasks 15, 16 |
| §4.2.4 MaskLoss + DualLoss | Tasks 19, 20 |
| §4.2.5 Solver | Tasks 21, 22 |
| §4.2.6 Collate | Task 18 |
| §5.4 Visualization | Task 23 |
| §5.6 Phase gates | Tasks 9, 27 |
| §6 Workflow (Colab + Drive) | Tasks 6, 7, 8, 25, 26 |
| §6.4 Smoke tests | Tasks 5, 24 |

All sections have at least one task. No gaps.

### Placeholder scan

No "TBD" / "TODO" in plan. All code blocks contain runnable code; all commands are exact. Risk 2 of the spec ("BoxMatcher may need extension") is folded into Task 22 Step 3 with explicit fallback instructions.

### Type consistency

- `MultiheadSegmentation.forward` returns dict `{"detect", "mask_coefs", "proto"}` — referenced consistently in Tasks 12, 22, 25.
- `task_type` field added in Task 21, consumed in Tasks 17, 18, 20, 22.
- `MaskLoss` inputs `(coefs, proto, gt_masks, target_bbox, match_idx)` — same tuple ordering in Tasks 19, 20, 22.
- `collate_fn_seg` returns `(batch_size, images, targets, masks, revs, paths)` — destructured the same way in Task 22.

Naming and types consistent across tasks.
