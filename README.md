# mlx-yolos

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org)
[![MLX](https://img.shields.io/badge/MLX-0.20%2B-FF6F00.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-000000.svg?logo=apple)](https://support.apple.com/en-us/116943)

Pure [MLX](https://github.com/ml-explore/mlx) inference for [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO models on Apple Silicon. The package is structured for **easy extension** — new models slot in as a YAML config + (optionally) a head class + a post-processor.

> **Currently shipping**
> - `yolov8-pose` (n / s / m / l / x scales — all share the same architecture; the bundled config defaults to `n`).
>
> **Architecture-ready (not wired to a CLI yet)**
> - `yolov8` plain detection — drop in a `yolov8.yaml` and reuse `DetectV8` + the `detect` post-processor.

<details>
<summary><b>Table of contents</b></summary>

- [Attribution & License](#attribution--license-read-first) — start here
- [Install](#install)
- [Quick start (yolov8-pose)](#quick-start-yolov8-pose)
  - [Convert weights](#1-convert-weights-one-time)
  - [Predict (CLI)](#2-predict)
  - [Predict (library)](#or-as-a-library)
- [Headline results](#headline-results)
- [Deeper docs](#deeper-docs)
- [Caveats](#caveats)
- [Acknowledgements](#acknowledgements)
- [License](#license)

</details>

---

## Attribution & License (READ FIRST)

This project is a derivative work of **[Ultralytics](https://github.com/ultralytics/ultralytics)**: the model architectures, YAML configurations, and the `.pt` weights this package converts are all originally Ultralytics' work, distributed under the **[GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)** license.

To stay in compliance:

1. **mlx-yolos is licensed under AGPL-3.0** (see [LICENSE](LICENSE)) — same as upstream.
2. **Source files include the upstream attribution comment**:
   ```
   # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
   ```
   Please keep these headers when redistributing.
3. **If you serve this code as part of a network service** (web app, API, hosted inference) you must make the complete corresponding source available to your users — this is the "A" (Affero) clause that distinguishes AGPL from plain GPL. If that obligation isn't acceptable for your use case, contact Ultralytics about an [Enterprise license](https://www.ultralytics.com/license).
4. **The weights you convert** (e.g. `yolov8n-pose.pt` → `yolov8n-pose.safetensors`) inherit Ultralytics' license terms — the conversion doesn't relicense them.

If you find any case where attribution is missing or unclear, please file an issue.

---

## Install

```bash
# Inference on Apple Silicon
pip install -e .

# Plus weight conversion (PyTorch + Ultralytics) — only on the machine
# where you do the conversion; not needed at inference time.
pip install -e '.[convert]'
```

Inference works on macOS / Apple Silicon (where MLX has Metal acceleration). The conversion step works anywhere — it doesn't import `mlx`.

Optional extras: `[val]` (pycocotools, for COCO mAP eval) and `[benchmark]` (torch + ultralytics + matplotlib, for the cross-backend timing comparison). Both are documented in [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

---

## Quick start (yolov8-pose)

### 1. Convert weights (one-time)

```bash
# Download the Ultralytics weights (or use ones you already have)
curl -L -o yolov8n-pose.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt

mlx-yolos convert --pt yolov8n-pose.pt --out yolov8n-pose.safetensors
# converted 333 tensors (72 conv weights transposed); skipped 64 bookkeeping keys
```

### 2. Predict

```bash
mlx-yolos predict \
    --cfg yolov8-pose.yaml \
    --weights yolov8n-pose.safetensors \
    --source bus.jpg \
    --save out/
```

### Or as a library

```python
from mlxyolos import YOLO

model = YOLO(
    "yolov8-pose.yaml",
    scale="n",
    weights="yolov8n-pose.safetensors",
)

results = model.predict("bus.jpg", conf=0.25, iou=0.45)
for r in results:
    print(r.boxes.xyxy)            # (N, 4)
    print(r.boxes.conf)            # (N,)
    print(r.keypoints.data.shape)  # (N, 17, 3)  -> (x, y, visibility)
```

For a richer per-detection log:

```python
print(results[0].verbose())
# /path/to/bus.jpg  810x1080  3 detections
#   [0] person     0.883  box=(48.5, 396.4, 244.2, 905.5)  kpts=16/17
#   [1] person     0.875  box=(222.1, 407.0, 345.4, 858.0)  kpts=17/17
#   [2] person     0.868  box=(668.7, 397.7, 809.0, 876.4)  kpts=8/17
```

To draw boxes + class badges + skeleton:

```python
results[0].plot().save("annotated.jpg")
```

`Results.plot()` automatically picks the right drawing path: boxes + class badges always, plus the COCO skeleton when keypoints are available. If you need the lower-level helpers directly:

```python
from mlxyolos.utils import draw_boxes, draw_pose

r = results[0]
img = draw_pose(
    r.orig_img,
    r.boxes.xyxy, r.boxes.conf, r.boxes.cls,
    r.keypoints.data,
    names={0: "person"},
)
img.save("annotated.jpg")
```

---

## Headline results

Apple Silicon, `yolov8-pose`. Methodology and full per-scale reproduction steps in [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

**Accuracy** — `yolov8n-pose` on COCO val2017 (5 000 images, imgsz=640, conf=0.001, iou=0.7). Ultralytics column from running `yolo pose val model=yolov8n-pose.pt data=coco-pose.yaml` on the same checkpoint.

| Metric                  | mlx-yolos | Ultralytics | Δ        |
|-------------------------|----------:|------------:|---------:|
| pose AP @ IoU 0.50:0.95 | **49.9**  | 50.5        | **−0.6** |
| pose AP @ IoU 0.50      | **78.7**  | 80.1        | **−1.4** |
| box  AP @ IoU 0.50:0.95 | **45.5**  | 54.0        | **−8.5** |

Pose AP matches Ultralytics within 0.6 pt — that's a faithful inference port. The box AP gap is bigger than noise and is **not** a forward-pass issue; it's because Ultralytics' `val` pipeline uses **rectangular letterbox** (`rect=True`, padding to stride-aligned non-square dims) while mlx-yolos always pads to a 640×640 square today. Per-image numerical parity is `max abs diff = 6.4e-4` against Ultralytics on `bus.jpg`.

**Speed** — all five `yolov8{n,s,m,l,x}-pose` scales, 200 random COCO val images per cell, imgsz=640, conf=0.25, iou=0.45, warmup=40.

![Cross-backend inference benchmark across 5 scales](docs/benchmark.png)

Mean latency, ms / image (lower is better; **bold** = fastest backend in column):

| Backend     | yolov8n   | yolov8s   | yolov8m   | yolov8l   | yolov8x    |
|-------------|----------:|----------:|----------:|----------:|-----------:|
| **mlx**     | **16.52** | **27.93** | **53.63** | 84.43     | 134.52     |
| torch-cpu   | 35.60     | 53.88     | 90.63     | 143.84    | 195.92     |
| torch-mps   | 33.12     | 35.59     | 51.21     | **69.89** | **98.91**  |

There's a **crossover at yolov8m / yolov8l**: mlx wins at the small-to-medium end (the entire pipeline stays on Metal with one eval boundary, no torchvision-NMS CPU fallback), torch-mps catches up once compute dominates over per-op launch overhead. Pick the backend by model size — see [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full per-scale table and discussion.

---

## Deeper docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — repository layout, how conversion works, and the recipe for adding a new model family.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — per-image numerical parity + COCO val2017 mAP methodology, full results tables, the `catIds` gotcha for bbox eval, and reproduction steps.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — cross-backend timing methodology, the small-model / large-model crossover discussion, and full per-scale tables.

---

## Caveats

* **No training / fine-tuning yet** — inference only.
* **Apple Silicon only at runtime** — MLX's Linux pip wheel is not always installable; conversion is designed to work without MLX so it can run on any box.
* **Pose post-processing uses class-agnostic NMS** (matches Ultralytics' default for the single-class person model). For multi-class pose, the predictor will need a per-class NMS path.

---

## Acknowledgements

* **[Ultralytics](https://github.com/ultralytics/ultralytics)** — original YOLOv8 implementation, training pipeline, weights, and config schema. mlx-yolos is a translation, not a re-design; their work is the substrate.
* **[Apple MLX](https://github.com/ml-explore/mlx)** — the array framework powering this package on Apple Silicon.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) for the full text and the [Attribution](#attribution--license-read-first) section above for what that means in practice.
