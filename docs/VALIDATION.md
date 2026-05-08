# Validation

How we know the MLX port reproduces Ultralytics. Two layers:

1. **Per-image numerical parity** — does the forward output of mlx-yolos match Ultralytics on the same input, tensor-for-tensor? Cheap, deterministic, runs without MLX.
2. **COCO val2017 mAP** — does the converted weight set hit the same accuracy the Ultralytics docs publish? End-to-end, the real proof.

- [Per-image numerical parity](#per-image-numerical-parity)
- [COCO val2017 mAP](#coco-val2017-map)
- [Reproducing](#reproducing)

---

## Per-image numerical parity

[`scripts/validate_yolov8_pose.py`](../scripts/validate_yolov8_pose.py) is a no-MLX-required parity check: it builds a PyTorch model that mirrors mlx-yolos's class layout exactly (same NHWC kernel layout, same BN `eps=1e-3`), loads the converted safetensors into it, and compares its forward output against the official Ultralytics `yolov8n-pose.pt` on `bus.jpg`.

On the released `yolov8n-pose` checkpoint:

| metric                      | result                                |
|-----------------------------|---------------------------------------|
| missing keys / extra keys   | 0 / 0                                 |
| max abs diff vs Ultralytics | 6.4e-4                                |
| detections > 0.25 conf      | 30 (matches Ultralytics exactly)      |

```bash
python scripts/validate_yolov8_pose.py
```

The script is also what the CI parity job runs ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)), with a hard-fail at `max abs diff > 5e-3` — generous enough to absorb cross-BLAS reduction-order noise (the CPU-only torch wheel from pytorch.org uses OpenBLAS; MKL-backed torch on dev machines uses different reduction order, ~6e-4 vs ~1e-3 noise floor) but tight enough to catch real architecture / weight regressions, which during the port stabilization were several orders of magnitude larger.

---

## COCO val2017 mAP

Single-image parity is necessary but not sufficient. We score the converted weights against the canonical `pycocotools` evaluator on the full 5 000-image COCO val2017 set, for both the bbox and keypoints predictions.

### Results — `yolov8n-pose`, COCO val2017, imgsz=640, conf=0.001, iou=0.7

Apple Silicon, MLX-only inference path. Ultralytics column is from running `yolo pose val model=yolov8n-pose.pt data=coco-pose.yaml imgsz=640` against the same `.pt` checkpoint we convert.

| Metric                       | mlx-yolos | Ultralytics | Δ        | Notes                                              |
|------------------------------|----------:|------------:|---------:|----------------------------------------------------|
| **pose** AP @ IoU 0.50:0.95  | **49.9**  | 50.5        | **−0.6** | matches the published model card to ~1 pt          |
| **pose** AP @ IoU 0.50       | **78.7**  | 80.1        | **−1.4** |                                                    |
| **pose** AP @ IoU 0.75       | **53.6**  | 54.1        | **−0.5** |                                                    |
| **pose** AR @ IoU 0.50:0.95  | **57.9**  | 57.8        | **+0.1** |                                                    |
| **box**  AP @ IoU 0.50:0.95  | **45.5**  | 54.0        | **−8.5** | most of the gap is letterbox shape; see below      |
| **box**  AP @ IoU 0.50       | **61.9**  | 73.3        | **−11.4**|                                                    |
| **box**  AP @ IoU 0.75       | **50.4**  | 59.9        | **−9.5** |                                                    |
| **box**  AR @ IoU 0.50:0.95  | **56.6**  | 63.6        | **−7.0** |                                                    |
| MLX inference latency (mean) | **17.67 ms / image** | — | — | full pre/forward/decode/NMS pipeline on Metal      |
| MLX throughput               | **56.6 img/s**       | — | — |                                                    |

#### Why pose AP matches but box AP doesn't (yet)

The pose-AP gaps are inside the cross-framework reproduction noise floor (typical Ultralytics → other-framework ports report 0.2–1.0 pt; the small numerical drift in our forward — max abs diff ~6e-4 vs Ultralytics — moves a handful of detections in/out of the top-20 per image, which moves AP@50 a touch more than AP@50:95).

The box-AP gap is much larger and is **not** noise. The most likely cause is a **letterbox shape mismatch**:

- **Ultralytics' `val` pipeline uses `rect=True`** by default — each image is letterboxed to the smallest stride-aligned size (e.g. 640×384 for landscape, 384×640 for portrait), preserving more pixels for the actual content and adding less zero-padding.
- **mlx-yolos always pads to a 640×640 square** today — simpler letterbox, but for a non-square image it spends a meaningful fraction of the input grid on padding pixels the model has to ignore.

This affects bbox accuracy more than keypoint accuracy because:
- **Box AP** integrates IoU at every threshold from 0.5 to 0.95; small spatial-resolution losses at object edges drop the strict-IoU rows hard.
- **Keypoint AP** integrates per-keypoint OKS, which is normalized by object scale — losing a few pixels of effective resolution is mostly absorbed.

Implementing rectangular letterbox in our predictor is straightforward and is the next step for box AP parity. Tracked separately.

---

## Reproducing

```bash
# Per-image parity (fast, no MLX needed):
python scripts/validate_yolov8_pose.py

# Full COCO mAP:
bash scripts/get_coco_pose_val.sh
pip install -e '.[val]'
python scripts/evaluate_coco_pose.py \
    --weights yolov8n-pose.safetensors \
    --cfg yolov8-pose.yaml
```

The COCO eval reports MLX inference timing alongside the standard pycocotools 12-row mAP table for each iouType, plus the average detection count (a quick sanity check that NMS isn't suppressing too aggressively). Use `--limit N` for a fast smoke run on the first N images.
