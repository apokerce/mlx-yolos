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

To draw the skeleton:

```python
from mlxyolos.utils import draw_pose

r = results[0]
img = draw_pose(r.orig_img, r.boxes.xyxy, r.boxes.conf, r.keypoints.data)
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
│       ├── ops.py           # letterbox / NMS / scale_coords
│       └── plotting.py      # draw_pose
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
