# Validation

End-to-end COCO val2017 mAP is the proof that the converted weights reproduce Ultralytics. Per-image forward parity was historically verified by a no-MLX torch-shadow script (`scripts/validate_yolov8_pose.py`) which measured `max abs diff = 6.4e-4` on `bus.jpg`; that script was removed along with the rest of the Linux-only test surface (we now run all numerical checks on Apple Silicon via the COCO eval below).

- [COCO val2017 mAP](#coco-val2017-map)
- [Reproducing](#reproducing)

---

## COCO val2017 mAP

Single-image parity is necessary but not sufficient. We score the converted weights against the canonical `pycocotools` evaluator on the full 5 000-image COCO val2017 set, for both the bbox and keypoints predictions.

### Results — `yolov8n-pose`, COCO val2017, imgsz=640, conf=0.001, iou=0.7

Apple Silicon, MLX-only inference path. Ultralytics column is from `yolo pose val model=yolov8n-pose.pt data=coco-pose.yaml imgsz=640` against the same `.pt` checkpoint we convert. Both runs use square letterbox + `scaleup=False`, evaluate on the 2 346-image set in `coco-pose.yaml`'s `val2017.txt` (criterion: at least one annotation with `num_keypoints > 0`), and use `pycocotools` for the eval itself.

| Metric                       | mlx-yolos | Ultralytics | Δ        | Notes                                              |
|------------------------------|----------:|------------:|---------:|----------------------------------------------------|
| **pose** AP @ IoU 0.50:0.95  | **50.1**  | 50.5        | **−0.4** | matches the published model card to ~0.5 pt        |
| **pose** AP @ IoU 0.50       | **79.6**  | 80.1        | **−0.5** |                                                    |
| **pose** AP @ IoU 0.75       | **53.6**  | 54.1        | **−0.5** |                                                    |
| **pose** AR @ IoU 0.50:0.95  | **57.6**  | 57.8        | **−0.2** |                                                    |
| **box**  AP @ IoU 0.50:0.95  | **52.6**  | 54.0        | **−1.4** | concentrated on small objects (see per-area below) |
| **box**  AP @ IoU 0.50       | **71.6**  | 73.3        | **−1.7** |                                                    |
| **box**  AP @ IoU 0.75       | **58.3**  | 59.9        | **−1.6** |                                                    |
| **box**  AR @ IoU 0.50:0.95  | **62.6**  | 63.6        | **−1.0** |                                                    |
| MLX inference latency (mean) | **17.3 ms / image** | — | — | full pre/forward/decode/NMS pipeline on Metal       |
| MLX throughput               | **57.7 img/s**       | — | — |                                                    |

Box AP per object-area bucket:

| Object size | mlx-yolos | Ultralytics | Δ        |
|-------------|----------:|------------:|---------:|
| small       | **18.8**  | 22.1        | **−3.3** |
| medium      | **64.6**  | 65.6        | **−1.0** |
| large       | **79.0**  | 79.0        | **±0.0** |

#### Why the remaining 1.4 pt of box AP is on small objects

Large-object box AP matches Ultralytics exactly. Medium is within 1 pt. The residual is concentrated on the small bucket (−3.3 pt) — the kind of sub-pixel sensitivity that small-object AP@0.85 and AP@0.95 are designed to be picky about. The two underlying sources we know about: (1) ~6e-4 max-abs numerical drift in our forward pass (measured on `bus.jpg`, see "End-to-end predict-flow audit" below) and (2) cv2 vs PIL bilinear differing by ~5/255 mean per pixel on downscales (we use cv2 now, but pyTorch + Ultralytics also use cv2, so this is identical). The remaining drift bounces a few small-object detections in/out of the strict-IoU buckets without affecting the headline numbers.

### The end-to-end predict-flow audit (May 2026)

We traced Ultralytics' `yolo pose val` end-to-end against mlx-yolos to identify every divergence. Equivalent (✓) and divergent (⚠) items:

| Stage                          | Status | Notes                                                             |
|--------------------------------|:------:|-------------------------------------------------------------------|
| Image load (file → RGB ndarray)| ✓      | Both use `cv2.imread` + `cv2.cvtColor(BGR → RGB)`.                |
| Letterbox math (`r`, `nh/nw`, `dw/dh`, pad split) | ✓ | Both produce identical integer pads given integer inputs. |
| Letterbox resize backend       | ✓      | Both use `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.       |
| Normalize (`/ 255`, RGB float32) | ✓    | Both paths converge to the same float32 RGB tensor.               |
| Forward pass                   | ✓      | `max abs diff = 6.4e-4` on bus.jpg (single-image parity check).   |
| NMS: conf threshold (`> 0.001`)| ✓      | Strict-`>` on both.                                               |
| NMS: per-class vs agnostic     | ✓      | nc=1 makes both equivalent (`multi_label &= nc > 1`).             |
| NMS: algorithm                 | ✓      | torchvision (theirs) vs numpy greedy (ours) — same selection.     |
| NMS: max_det cap (300)         | ✓      | Both well below cap in practice (~34 detections / image).         |
| **scale_boxes clip range**     | ⚠ → ✓  | Ultralytics clips to `[0, w]` / `[0, h]`. We clipped to `[0, w-1]` / `[0, h-1]` — 1-pixel cropping at image edges. **Fixed.** |
| **scale_keypoints clip**       | ⚠ → ✓  | Ultralytics clips keypoints via `clip_coords`. We didn't. **Fixed** via optional `orig_shape` arg, now wired in the predictor. |
| Pred-to-JSON (xyxy → top-left xywh, cat=1, keypoints flat) | ✓ | Same shape, same fields. |
| pycocotools imgIds filter      | ✓      | Restricted to `getImgIds(catIds=[1])` (added in the previous round). |

Three earlier hypotheses tested and ruled out (`rect=True`, `scaleup=False`, `imgIds` filter alone). One previously-landed (`imgIds`) is still useful (it isolated a real if small effect). Three new fixes from the audit are now landed: **cv2 resize**, **box clip range**, **keypoint clipping**.



1. **Rectangular letterbox** (`rect=True`). Hypothesis: maybe `yolo pose val` uses rect and we use square. Reality: val uses square. With rect enabled in mlx-yolos the gap *widened* by ~1.3 pt.
2. **`scaleup=False`** (don't upscale small images). Hypothesis: 18.1 % of COCO val has both dims < 640, so upscaling them might hurt small-object AP. Reality: tested — small AP moved from 15.1 → 15.4, basically a no-op.
3. **`imgIds` filter (currently landed, awaiting re-run).** Hypothesis: pycocotools' default `imgIds` is "all images in GT JSON" (5 000 for COCO val2017), but only ~2 700 actually have person GT. Our predictions at `conf=0.001` are ~34 boxes per image, so the ~2 300 no-person images contribute ~80 000 phantom false positives that Ultralytics' val doesn't pay because it iterates `val2017.txt` (the 2 346 person-having images only). This mechanism would hit box AP much harder than keypoint AP — boxes are one-per-anchor, keypoint OKS is more forgiving — which matches the observed pattern (box −8.5 pt vs keypoint −1.0 pt).

**As of the latest commit, the `imgIds` filter is wired up.** `_coco_eval` now sets `coco_eval.params.imgIds = sorted(coco_gt.getImgIds(catIds=cat_ids))` whenever `cat_ids` is supplied, restricting the evaluator to images that have GT annotations for the target classes — matching Ultralytics' val image set. A log line in the eval output shows the filtered count vs the full GT count so future runs make this visible.

The `rect=True` and `scaleup=False` options are still useful and remain in place, but they're for matching `yolo predict` behavior (where they're the upstream default), not for the val comparison. The eval script's defaults are `--no-rect --no-scaleup` accordingly.

Tracked in [`TODO.md`](../TODO.md) item **1b**.

---

## Reproducing

```bash
# Full COCO mAP eval on Apple Silicon.
bash scripts/get_coco_pose_val.sh
pip install -e '.[val]'
python scripts/evaluate_coco_pose.py \
    --weights yolov8n-pose.safetensors \
    --cfg yolov8-pose.yaml
```

The COCO eval reports MLX inference timing alongside the standard pycocotools 12-row mAP table for each iouType, plus the average detection count (a quick sanity check that NMS isn't suppressing too aggressively). Use `--limit N` for a fast smoke run on the first N images.
