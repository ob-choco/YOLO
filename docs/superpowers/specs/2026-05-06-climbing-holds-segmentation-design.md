# Climbing Holds & Volumes Segmentation — Design

**Date**: 2026-05-06
**Status**: Draft (pending user review)
**Owner**: htjo

## 1. Overview

Train an instance segmentation model that detects climbing holds and volumes in indoor climbing wall photos. Two model variants (`v9-t-seg`, `v9-s-seg`) are produced for mobile deployment.

The work is implemented as a series of additions to the existing [WongKinYiu/YOLO](https://github.com/WongKinYiu/YOLO) repository (MIT licensed). The repo currently scaffolds segmentation (model head module + `v9-c-seg.yaml` config) but the **training pipeline (mask target loading, mask loss, solver integration) is not implemented** — this work fills that gap and adds two lighter-weight (`v9-t-seg`, `v9-s-seg`) variants.

The dataset at `/Users/htjo/workspace/kaggle/hold-segmentation/archive/` is in VIA (VGG Image Annotator) format with 19 hand-labeled images across three sources, totalling ~1,688 polygon instances of two classes (`hold`, `volume`).

### 1.1 Goals
- Implement segmentation training in WongKinYiu/YOLO for the `MultiheadSegmentation` head.
- Add `v9-t-seg` and `v9-s-seg` model configurations.
- Train on the 19-image climbing dataset.
- Produce mask-overlay visualizations on held-out validation images that visibly capture hold and volume shapes.

### 1.2 Non-goals
- SOTA mAP. The dataset is too small for meaningful absolute-value benchmarking; functional correctness and visual quality are the primary success signals.
- Mobile export tooling itself (ONNX → CoreML/TFLite). The model architecture is sized for mobile deployment, but on-device export is out of scope here.
- Auxiliary-branch segmentation supervision. The aux branch remains detection-only.
- Copy-paste augmentation, focal loss, class-rebalanced sampling. Deferred to follow-up experiments after initial results.

### 1.3 License Note
The Ultralytics codebase (AGPL-3.0) was consulted for *architectural ideas only* — channel widths, prototype design conventions, training hyperparameters, and loss formulation principles. **No code was copied.** All implementation is written from scratch in the MIT-licensed WongKinYiu/YOLO fork.

## 2. Data Pipeline

### 2.1 Data Inventory

| Source | Images | Labeled images | Polygon instances |
|---|---|---|---|
| `bh` (main camera, ~4K) | 1,061 | 15 | 1,487 |
| `bh-phone` (same gym, phone camera) | 240 | 2 | 73 |
| `sm` (different gym) | 972 | 2 | 128 |
| **Total** | **2,273** | **19** | **1,688** |

Class distribution: `hold` 1,597 instances (94.6%), `volume` 91 instances (5.4%).

### 2.2 Train / Validation Split

| Split | Source | Images | Purpose |
|---|---|---|---|
| `train` | `bh` (15 labeled) | 15 | Training |
| `val/bh-phone` | `bh-phone` (2) | 2 | Camera-domain generalization |
| `val/sm` | `sm` (2) | 2 | Gym-domain generalization |
| `val/all` | combined OOD | 4 | Aggregate metric |

The validation set is intentionally OOD (different camera and different gym) so the metric reflects whether the model has learned the *concept* of a hold/volume rather than memorized the training images. Per-domain breakdown is the primary evaluation signal.

### 2.3 Directory Structure

Original archive at `/Users/htjo/workspace/kaggle/hold-segmentation/archive/` is **read-only**.

Within the YOLO repo:
```
data/climbing_holds/                   ← gitignored
├── images/
│   ├── train/                         (symlinks or copies of 15 bh images)
│   ├── val/                           (4 images: bh-phone + sm)
│   └── train_resized/                 (longest-side 1024 pre-resize, for CPU smoke tests)
└── annotations/
    ├── train.json                     (COCO instance-seg)
    └── val.json
tools/data_prep/
└── via_to_coco.py                     ← VIA → COCO converter (this work)
```

### 2.4 VIA → COCO Converter

`tools/data_prep/via_to_coco.py` reads a VIA JSON file and emits COCO instance-seg JSON.

**Extracts**: `regions[].shape_attributes` (polygon `all_points_x`, `all_points_y`); `regions[].region_attributes.hold_type` (`hold` | `volume`); image dimensions (resolved by opening the image file).

**Emits**: `images` (id, file_name, width, height); `categories` (`id:0 hold`, `id:1 volume`); `annotations` (id, image_id, category_id, `segmentation: [[x1,y1,...]]` COCO polygon, `bbox: [x,y,w,h]` derived from polygon min/max, `area`, `iscrowd: 0`).

**Skip rules**: empty `regions`, polygons with fewer than 3 points, images with zero annotations (excluded from `train.json`).

### 2.5 Sanity Checks

After conversion:
1. Load via `pycocotools.COCO()` without errors.
2. Render polygons on a sampled image to `data/sanity/<filename>.png` and visually verify alignment.
3. Verify instance counts match the original CSV: `hold` 1,597, `volume` 91.

## 3. Model Architecture

### 3.1 Two New Configs

`yolo/config/model/v9-t-seg.yaml` and `yolo/config/model/v9-s-seg.yaml`. Each is derived from the corresponding base detection config (`v9-t.yaml`, `v9-s.yaml`) by replacing the `detection:` block with a prototype branch + `MultiheadSegmentation` head. The `auxiliary:` branch (`MultiheadDetection`) is unchanged.

### 3.2 Channel Widths

Prototype branch widths follow the Ultralytics `npr=256` convention (wider prototype channels for mask quality, even on smaller models):

| | P3 ch | P4 ch | P5 ch | Prototype branch | num_maskes | Estimated params |
|---|---|---|---|---|---|---|
| `v9-c-seg` (existing) | 256 | 512 | 512 | RepNCSPELAN(256) → Conv(256) | 32 | ~25M |
| `v9-s-seg` (new) | 128 | 192 | 256 | RepNCSPELAN(256) → Conv(256) | 32 | ~9–10M |
| `v9-t-seg` (new) | 64 | 96 | 128 | RepNCSPELAN(128) → Conv(128) | 32 | ~2.5–3M |

`num_maskes=32` is held constant across all three models, matching `v9-c-seg` and Ultralytics defaults.

### 3.3 Detection Block (example: `v9-t-seg`)

```yaml
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
```

The `-1` source is the upsampled + conv'd prototype feature map (stride 4, since P3 is stride 8 and `UpSample 2x` halves the stride).

### 3.4 Auxiliary Branch

Kept as `MultiheadDetection` (detection-only). Rationale: minimizes divergence from the upstream `v9-t/v9-s` design, keeps detection-pretrained weight transfer clean, and avoids adding mask supervision in a low-data regime where it would likely over-fit.

### 3.5 Pretrained Weight Transfer

Source: official upstream `v9-t.ckpt` / `v9-s.ckpt` (detection variants).

A weight-transfer script generalizes the existing `f1585d3` v9-c → v9-c-seg transform to the smaller variants:
- Backbone, neck, P3–P5 head: copied directly from detection ckpt.
- `MultiheadSegmentation.detect` (internal `MultiheadDetection`): copied from detection ckpt.
- Prototype branch (RepNCSPELAN + Conv): randomly initialized.
- `MultiheadSegmentation` mask coefficient layers and prototype Conv: randomly initialized.

Outputs: `weights/v9-t-seg-init.pt`, `weights/v9-s-seg-init.pt` — used as the starting point for Phase B training.

## 4. Training Pipeline

The work is organized into two phases. **Phase A delivers a working detection model** on this dataset and validates the data pipeline. **Phase B adds segmentation on top.**

### 4.1 Phase A — Detection MVP

Phase A requires no new training-pipeline code; the existing `load_valid_labels` already collapses polygon annotations into bboxes, so the COCO seg JSON is consumed as detection input as-is.

**Deliverables**:
- VIA → COCO converter (Section 2.4) and sanity check (Section 2.5).
- New dataset config `yolo/config/dataset/climbing_holds.yaml` (`class_num: 2`, `class_list: ["hold", "volume"]`, paths to `data/climbing_holds/`).
- Detection training runs on `v9-t` and `v9-s`, producing bbox visualizations on the OOD validation set.

### 4.2 Phase B — Segmentation Pipeline

#### 4.2.1 Data Loader Extension (`yolo/tools/data_loader.py`)

`load_valid_labels` is extended so that, when the task is segmentation:
- It returns `(bboxes (N, 5), polygons (List[ndarray]))` instead of just bboxes.
- Polygons are kept normalized to `[0, 1]`.

For detection tasks, the second value is `None` and the existing call sites are unchanged.

#### 4.2.2 Mask Rasterization (`yolo/utils/dataset_utils.py`)

A new `rasterize_masks(polygons, image_size, mask_ratio=4)` function converts polygons to binary masks at stride-4 resolution (e.g., 160×160 for a 640-input image), matching Ultralytics' `mask_ratio=4` convention. It is invoked **after** augmentation, just before the forward pass — so augmentation operates on polygon coordinates (no resampling artifacts) and rasterization happens once on the final geometry.

#### 4.2.3 Augmentation (`yolo/tools/data_augmentation.py`)

Existing transforms become polygon-aware:

| Transform | Image | Polygon |
|---|---|---|
| HSV | applied | unchanged |
| HorizontalFlip | mirror | `x' = 1 - x` |
| RandomAffine (scale, rotate, translate) | affine warp | same affine matrix applied to polygon points |

**New**: `Mosaic` (4-image, standard YOLO). Polygons are translated into the corresponding mosaic cell and clipped to the mosaic boundary; instances clipped to zero area are dropped. Mosaic is critical given only 19 training images. Default: **on** (`mosaic: 1.0`); toggleable via dataset config. Disabled automatically for the last `mosaic_close_epochs: 10` epochs of training to let the model fine-tune on un-mosaicked images (standard YOLO recipe).

**Copy-paste**: not implemented in v1 (deferred — risk of memorizing specific hold shapes when all training images are from the same gym).

#### 4.2.4 Loss (`yolo/tools/loss_functions.py`)

**New `MaskLoss`** — BCE + Dice with bbox crop, matching Ultralytics' formulation:
- Input: predicted mask coefficients `(B, N_anchors, 32)`, prototype masks `(B, 32, H/4, W/4)`, GT masks `(B, N_max, H/4, W/4)`, target bboxes `(B, N_max, 4)`, valid-anchor mask.
- Per matched positive, compute `pred_mask = sigmoid(coef ⊙ proto)` via `einsum`, crop to target bbox, then average `0.5 * BCE + 0.5 * Dice` across positives, normalized by bbox area.
- Returns scalar loss.

**`DualLoss` extension** — when `model.task_type == "segmentation"`, the main branch's loss adds a mask term; the aux branch is unchanged (detection-only, per Section 3.4).

**Hyperparameters** (initial values from Ultralytics defaults):
- `BoxLoss: 7.5`, `DFLoss: 1.5`, `BCELoss: 0.5`, `MaskLoss: 7.5`
- Mask sub-weights: `bce_weight: 0.5`, `dice_weight: 0.5`, `mask_ratio: 4`
- Optimizer: SGD `lr=0.01`, `momentum=0.937`, `weight_decay=0.0005`

#### 4.2.5 Solver (`yolo/tools/solver.py`)

`training_step` is extended to unpack `gt_masks` from the batch and forward them to `loss_fn`. The `Vec2Box` pipeline must be reviewed carefully: it currently assumes detection-only output, and the seg head's mask-coefficient and prototype outputs need a clear path through (or around) it. **This is the highest-risk change** and is called out in §7.

#### 4.2.6 Collate Function

`gt_masks` are variable-length per image. The collate function pads to `(B, N_max, H/4, W/4)` with a `valid_target` mask used downstream to ignore padded entries.

### 4.3 Class Imbalance

`volume` represents only ~5% of instances. In v1, no rebalancing is applied; per-class metrics are logged so the impact is measurable. Rebalancing options (class-weighted BCE, focal loss, oversampling) are deferred to follow-up.

## 5. Evaluation & Visualization

### 5.1 Metrics

Standard COCO eval via `pycocotools`. Both `iouType='bbox'` and (Phase B) `iouType='segm'` are computed.

| Phase | Metric | Role |
|---|---|---|
| A | `mAP@bbox 0.5` | secondary signal |
| A | `mAP@bbox 0.5:0.95` | primary |
| B | `mAP@bbox 0.5:0.95` | regression guard |
| B | `mAP@mask 0.5` | secondary signal |
| B | `mAP@mask 0.5:0.95` | **primary** |

Per-class (`hold`, `volume`) and per-domain (`val/bh-phone`, `val/sm`, `val/all`) breakdowns are logged separately.

### 5.2 Per-Domain Breakdown

This is the most informative part of evaluation given the dataset constraints:
- `val/bh-phone` — same gym, different camera. Tests camera-domain generalization (relevant to mobile deployment).
- `val/sm` — different gym. Tests gym-domain generalization (true concept learning vs. memorization).

With only 2 images per domain, absolute mAP numbers are noisy; **trends over training and visual results are co-signals**.

### 5.3 Logging

Loss components, learning rate, epoch metrics, val mAP (per class, per domain), GPU memory and iteration speed are written to TensorBoard and a CSV log every step / epoch / N-epochs as appropriate.

### 5.4 Inference Visualization

For every validation pass, prediction overlays for all 4 val images are saved as PNG to `runs/<run_id>/visualizations/epoch_<n>/<image>.png`. Phase A draws bboxes; Phase B draws bboxes + semi-transparent masks. A new `draw_masks` helper is added to `yolo/tools/drawer.py`.

### 5.5 Checkpoint Selection

`last.ckpt` is overwritten every epoch. `best.ckpt` is tracked by the primary metric for the phase:
- Phase A: `mAP@bbox 0.5:0.95` on `val/all`.
- Phase B: `mAP@mask 0.5:0.95` on `val/all`.

Per-domain metrics are logged but not used for checkpoint selection (avoids over-fitting to one domain's 2 images).

### 5.6 Phase Gates (Acceptance Criteria)

Both phase exits are defined by **functional correctness**, not absolute mAP, given the dataset size.

**Phase A exit**:
- VIA → COCO converter passes sanity checks.
- 1-epoch CPU smoke test on laptop (image_size=320, 4 images, batch=2) trains end-to-end with decreasing loss.
- 50-epoch Colab run completes for both `v9-t` and `v9-s`.
- Bbox visualizations on the 4 val images visually capture hold positions ("approximately right").
- `mAP@bbox 0.5 > 0` (model has learned *something*).

**Phase B exit**:
- Two new model configs and weight-transfer script produce loadable initial checkpoints.
- Unit tests pass for: `MaskLoss`, polygon augmentation, mask rasterization, mosaic, VIA→COCO conversion.
- 1-epoch CPU smoke test on laptop runs end-to-end with decreasing mask loss (no NaN).
- 100-epoch Colab run completes for both `v9-t-seg` and `v9-s-seg`.
- Mask overlays on the 4 val images visually capture hold/volume shapes.
- `mAP@mask 0.5 > 0`.

## 6. Execution Workflow

### 6.1 Environment Split

| Environment | Role |
|---|---|
| Local laptop (Intel Mac, CPU-only) | Code authoring, unit tests, smoke runs (≤ 5 min cycle), result analysis |
| Colab Free (T4 16GB) | Real training, evaluation, checkpoint generation |
| Kaggle (P100 / T4×2) | Backup environment if Colab is unavailable |
| GitHub fork | Code sync between laptop and cloud |
| Google Drive | Persistent dataset, pretrained weights, run artifacts |

### 6.2 Repo Strategy

All work happens on a `feature/climbing-seg` branch of the user's fork (`htjo/YOLO`). Upstream `main` is not modified.

### 6.3 Data Sync

The original archive is never copied. The converted `data/climbing_holds/` directory is gitignored, generated locally, zipped, and uploaded once to `My Drive/climbing-holds/climbing_holds.zip`. Colab notebooks mount Drive and unzip on session start.

### 6.4 Smoke Test Definition

Phase A (CPU, ~5 min):
```bash
python yolo/lazy.py task=train model=v9-t dataset=climbing_holds \
    task.data.batch_size=2 task.data.image_size=320 \
    task.epoch=2 device=cpu
```
Smoke test runs on the full 15-image train set at small resolution; 2 epochs at 320px batch 2 completes within ~5 minutes on the laptop CPU. Pass criteria: loss decreases, val PNG produced.

Phase B: as above with `model=v9-t-seg` and `MaskLoss` enabled. Additional pass criteria: `Loss/MaskLoss` is finite and decreases; mask overlay PNG contains *something*.

Unit tests (`pytest`, all CPU, < 30 s total):
- `tests/test_via_conversion.py`
- `tests/test_polygon_augment.py`
- `tests/test_mask_loss.py`
- `tests/test_mosaic.py`
- `tests/test_mask_rasterize.py`

### 6.5 Colab Notebooks

Two notebooks at `notebooks/colab/`:
- `phase_a_detection.ipynb` — env setup, Drive mount, data unzip, pretrained weight download, train v9-t (50 epochs) + v9-s (50 epochs), evaluate, save artifacts to Drive.
- `phase_b_segmentation.ipynb` — same flow plus weight-transfer step, train v9-t-seg + v9-s-seg (100 epochs each), save mask visualizations and COCO-format prediction JSON (with RLE-encoded masks).

Both notebooks save `last.ckpt` to Drive every N epochs and auto-resume on session restart.

### 6.6 Drive Artifact Layout

```
My Drive/climbing-holds/
├── data/climbing_holds.zip
├── pretrained/
│   ├── v9-t.ckpt
│   ├── v9-s.ckpt
│   ├── v9-t-seg-init.pt
│   └── v9-s-seg-init.pt
└── runs/
    ├── phaseA-v9-t-<datetime>/
    ├── phaseA-v9-s-<datetime>/
    ├── phaseB-v9-t-seg-<datetime>/
    └── phaseB-v9-s-seg-<datetime>/
        ├── config.yaml
        ├── last.ckpt
        ├── best.ckpt
        ├── tensorboard/
        ├── metrics.csv
        └── visualizations/epoch_<n>/*.png
```

### 6.7 Step-by-Step Sequence

| # | Step | Environment |
|---|---|---|
| 1 | Implement VIA → COCO converter and run sanity checks | Local |
| 2 | Add dataset config; pass Phase A smoke test | Local |
| 3 | Upload data zip to Drive; push fork to GitHub | Local |
| 4 | Phase A training (`v9-t`) — 50 epochs | Colab |
| 5 | Phase A training (`v9-s`) — 50 epochs | Colab |
| 6 | **Phase A gate** — analyze results, go/no-go | Local |
| 7 | Author `v9-t-seg.yaml`, `v9-s-seg.yaml` | Local |
| 8 | Implement weight-transfer script and verify | Local |
| 9 | Extend data loader for polygon retention; add unit tests | Local |
| 10 | Implement Mosaic and polygon-aware augmentation; add unit tests | Local |
| 11 | Implement `MaskLoss` and extend `DualLoss`; add unit tests | Local |
| 12 | Extend solver `training_step`; pass Phase B smoke test | Local |
| 13 | Phase B training (`v9-t-seg`) — 100 epochs | Colab |
| 14 | Phase B training (`v9-s-seg`) — 100 epochs | Colab |
| 15 | **Phase B gate** — analyze final results | Local |

Each step is a separate commit. Steps 6 and 12 are explicit go/no-go gates.

## 7. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| `Vec2Box` pipeline currently assumes detection-only outputs; routing mask coefficients and prototypes through (or around) it is the most architecturally invasive change. | Explicitly addressed first in implementation. Likely add a `task_type`-aware branch that extracts mask outputs before `Vec2Box` is invoked. |
| `BoxMatcher` output format may need extension to surface the matched-target index for mask supervision. | Reuse existing matching result; add a thin index-extraction utility. |
| Mosaic ↔ polygon clipping correctness at corner cases. | Dedicated unit tests, including degenerate cases (polygon entirely outside, polygon spanning two cells). |
| Colab Free GPU disconnect / preemption. | Auto-resume from `last.ckpt`; Kaggle backup notebook prepared in advance. |
| Pretrained `v9-t.ckpt` / `v9-s.ckpt` URL changes on upstream. | One-time download archived to `My Drive/climbing-holds/pretrained/`. |
| Mask threshold sensitivity affects mAP measurement. | One-time threshold sweep after training to log best-threshold mAP alongside default-0.5 mAP. |
| Validation set is 4 images total — metric numbers are noisy. | Always interpret metrics together with visualizations (Section 5.4). |
| Volume class has only 79 training instances — likely poor recall. | Per-class metrics logged; rebalancing deferred to follow-up after measurement. |

## 8. Out of Scope (explicit)

- Auxiliary-branch segmentation supervision (`MultiheadSegmentation` on AUX).
- Architecture changes from upstream (e.g., introducing C2PSA from YOLO11).
- Copy-paste augmentation, focal/class-weighted loss, oversampling.
- ONNX → CoreML / TFLite export tooling.
- Active learning or pseudo-labeling on the unlabeled images (1,061 + 238 + 970 ≈ 2,269 unlabeled).
- Hyperparameter sweeps. Initial values come from Ultralytics defaults; tuning is a follow-up if results justify it.
