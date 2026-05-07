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

`Results.plot()` automatically picks the right drawing path: boxes + class
badges always, plus the COCO skeleton when keypoints are available. If you
need the lower-level helpers directly:

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

## Repository layout

```
mlx-yolos/
├── pyproject.toml
├── README.md
├── LICENSE                  # AGPL-3.0 (matches Ultralytics)
├── src/mlxyolos/
│   ├── __init__.py          # exposes YOLO
│   ├── cli.py               # `mlx-yolos convert | predict`
│   ├── cfg/models/v8/
│   │   └── yolov8-pose.yaml
│   ├── converters/
│   │   └── ultralytics_pt.py  # PT → MLX safetensors (pure CPU, no MLX)
│   ├── engine/
│   │   ├── model.py         # YOLO façade
│   │   ├── predictor.py     # task-dispatched post-processing
│   │   └── results.py       # Boxes / Keypoints / Results
│   ├── nn/
│   │   ├── tasks.py         # YAML parser → BaseModel
│   │   └── modules/
│   │       ├── conv.py      # Conv / DWConv / Concat (NHWC)
│   │       ├── block.py     # Bottleneck / C2f / SPPF
│   │       └── head.py      # DetectV8 / PoseV8
│   └── utils/
│       ├── ops.py           # numpy: letterbox / NMS / scale_coords
│       ├── ops_mlx.py       # MLX-native: xywh_to_xyxy / scale_* / NMS-with-on-device-IoU
│       └── plotting.py      # draw_boxes / draw_pose
├── scripts/
│   └── validate_yolov8_pose.py  # numerical parity check vs Ultralytics
└── tests/
    └── test_converter.py
```

Adding a new model family is a three-step recipe:

1. Drop a YAML in `src/mlxyolos/cfg/models/<family>/`.
2. If the head differs from `DetectV8`/`PoseV8`, add a class in `nn/modules/head.py` and register it in `nn/tasks.py::MODULE_MAP` and `nn/tasks.py::HEAD_BUILDERS`.
3. Add a post-processor in `engine/predictor.py::POSTPROCESSORS` and a row in `engine/model.py::_HEAD_TO_TASK`.

The converter does **not** need to know about new families — module containers are plain Python lists, so state-dict keys match Ultralytics 1:1 and the only operations are conv-weight transpose + dropping bookkeeping.

---

## Numerical validation

`scripts/validate_yolov8_pose.py` is a no-MLX-required parity check: it builds a PyTorch model that mirrors mlx-yolos's class layout exactly (same NHWC kernel layout, same BN `eps=1e-3`), loads the converted safetensors into it, and compares its forward output against Ultralytics on `bus.jpg`. On the released yolov8n-pose checkpoint it produces:

| metric                     | result      |
|----------------------------|-------------|
| missing keys / extra keys  | 0 / 0       |
| max abs diff vs Ultralytics| 6.4e-4      |
| detections > 0.25 conf     | 30 (matches Ultralytics exactly) |

```bash
python scripts/validate_yolov8_pose.py
```

---

## Where MLX runs (and where it doesn't)

The hot path stays on the Apple-Silicon GPU end to end:

| Stage                              | Backend     | Notes                                                                                       |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| Image read                         | PIL         | File I/O is CPU-bound; MLX has no image decoder.                                            |
| Letterbox resize                   | PIL         | One small bilinear resize on uint8; not worth a Metal kernel.                               |
| Normalize (`/255` + NHWC float32)  | **MLX**     | Done on device after upload — the host only ships the uint8 buffer.                        |
| Forward pass                       | **MLX**     | Lazy graph; no eval forced yet.                                                             |
| Per-anchor decode (`max` / `argmax` over class scores, `xywh→xyxy` over ~8400 anchors) | **MLX** | Fused into the same eval as the forward pass. |
| Confidence filter                  | NumPy       | Variable-shape gather is awkward in MLX; runs on the small post-eval array.                |
| NMS — pairwise IoU matrix          | **MLX**     | Computed on device for the (typically <few hundred) post-conf detections.                   |
| NMS — greedy keep loop             | NumPy       | Inherently serial; runs on the (already small) IoU matrix.                                  |
| Scale boxes / keypoints back to original image | NumPy | Operates on the post-NMS set (~tens of detections); MLX overhead would dominate. |
| Drawing                            | Pillow      | CPU-only by definition.                                                                     |

The post-processors auto-dispatch by input type: `mx.array` triggers the
on-device path, `np.ndarray` runs pure NumPy. That keeps the test suite
backend-agnostic — the parity tests in `tests/test_ops_mlx.py` confirm
the two paths agree on synthetic boxes when MLX is available, and the
end-to-end numerical check in `scripts/validate_yolov8_pose.py` runs the
full numpy path on Linux.

### Switching from MLX to NumPy

There is exactly **one** MLX → NumPy boundary per inference call, by
design. MLX is lazy: every op (forward pass, slice, `max`/`argmax`,
`xywh→xyxy`) builds a graph and returns immediately without doing any
work. Compute is only triggered when something forces evaluation —
either an explicit `mx.eval(...)` or a host-side read like
`np.asarray(mx_array)` / `mx_array.item()`.

The predictor exploits this by stacking up *all* the per-anchor decode
work on the lazy graph and forcing **one** `mx.eval` over the
combined result (boxes + scores + cls + flat keypoints). The single
boundary crossing pays for the forward pass *and* the decode together,
so we don't round-trip per intermediate. Concretely, in
`engine/predictor.py::_decode_anchors_mlx`:

```python
score    = mx.max(cls_scores,    axis=-1)   # lazy
cls_idx  = mx.argmax(cls_scores, axis=-1)   # lazy
box_xyxy = ops_mlx.xywh_to_xyxy(box_xywh)   # lazy
mx.eval(box_xyxy, score, cls_idx, kpts_flat)   # ← one boundary, fused
return np.asarray(box_xyxy), np.asarray(score), ...
```

NMS does the same trick: the IoU matrix is built on device and then
materialized once with `np.asarray(iou)` so the greedy keep-loop has
the (small, already-resident) matrix to scan. After that we stay in
NumPy — the data is already tiny (≤a few hundred boxes), and pushing
it back to the GPU would cost more than it saves.

Rule of thumb: **MLX for shape-stable batched ops over thousands of
items, NumPy once the data shrinks to "a few".** Crossing the boundary
is cheap once, expensive in a loop — `np.asarray` triggers eval and
flushes the graph.

---

## How conversion works

The converter at `src/mlxyolos/converters/ultralytics_pt.py` does just three things:

1. Drops `*.num_batches_tracked` (MLX BatchNorm doesn't have it).
2. Drops `*.dfl.conv.weight` — that's a constant `arange` projection, reconstructed at runtime in `DetectV8`.
3. Transposes 4-D conv weights from PyTorch `(Cout, Cin, kH, kW)` to MLX `(Cout, kH, kW, Cin)`.

Everything else is copied verbatim. There are **no model-specific regex remappers** because mlx-yolos's `BaseModel` exposes its layer list as plain `self.model = [...]`, matching Ultralytics' attribute name; submodule containers (`m`, `cv2`, `cv3`, `cv4`) are also plain lists, so MLX's `tree_flatten` produces parameter keys that line up with the Ultralytics state dict directly.

That property is what keeps the converter from growing as new families are added.

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

AGPL-3.0-only. See [LICENSE](LICENSE) for the full text and the Attribution section above for what that means in practice.
