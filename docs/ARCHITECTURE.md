# Architecture

How mlx-yolos is laid out, why it's laid out that way, and how to extend it for a new model family.

- [Repository layout](#repository-layout)
- [How conversion works](#how-conversion-works)
- [Adding a new model family](#adding-a-new-model-family)
- [Why plain Python lists for module containers](#why-plain-python-lists-for-module-containers)

---

## Repository layout

```
mlx-yolos/
├── pyproject.toml            # package metadata + extras ([convert] / [val] / [benchmark] / [dev])
├── README.md
├── LICENSE                   # AGPL-3.0 (matches Ultralytics)
├── .github/
│   └── workflows/
│       └── ci.yml            # ubuntu-latest: ruff + pytest + numerical-parity gate
├── src/mlxyolos/
│   ├── __init__.py           # exposes YOLO
│   ├── cli.py                # `mlx-yolos convert | predict`
│   ├── cfg/models/v8/
│   │   └── yolov8-pose.yaml
│   ├── converters/
│   │   └── ultralytics_pt.py # PT → MLX safetensors (no MLX import — runs anywhere)
│   ├── engine/
│   │   ├── model.py          # YOLO façade (load weights, dispatch task)
│   │   ├── predictor.py      # task-dispatched post-processing, MLX/numpy backends
│   │   └── results.py        # Boxes / Keypoints / Results (with .verbose() + .plot())
│   ├── nn/
│   │   ├── tasks.py          # YAML parser → BaseModel
│   │   └── modules/
│   │       ├── conv.py       # Conv / DWConv / Concat (NHWC)
│   │       ├── block.py      # Bottleneck / C2f / SPPF
│   │       └── head.py       # DetectV8 / PoseV8
│   └── utils/
│       ├── ops.py            # numpy: letterbox / NMS / scale_coords
│       ├── ops_mlx.py        # MLX-native: xywh_to_xyxy / scale_* / NMS-with-on-device-IoU
│       └── plotting.py       # draw_boxes / draw_pose
├── scripts/
│   ├── validate_yolov8_pose.py    # per-image numerical parity check (no MLX needed)
│   ├── get_coco_pose_val.sh       # download COCO val2017 + annotations
│   ├── evaluate_coco_pose.py      # full COCO mAP via pycocotools + MLX timing
│   └── benchmark_inference.py     # MLX vs torch-cpu vs torch-mps + bar chart
├── docs/
│   ├── ARCHITECTURE.md       # this file
│   ├── VALIDATION.md         # numerical parity + COCO mAP results
│   ├── BENCHMARK.md          # cross-backend timing
│   └── benchmark.png         # rendered by scripts/benchmark_inference.py
└── tests/
    ├── test_converter.py     # name-mapping unit tests (no MLX)
    └── test_ops_mlx.py       # numpy/MLX parity tests (auto-skip without MLX)
```

---

## How conversion works

The converter at [`src/mlxyolos/converters/ultralytics_pt.py`](../src/mlxyolos/converters/ultralytics_pt.py) does just three things:

1. Drops `*.num_batches_tracked` (MLX BatchNorm doesn't have it).
2. Drops `*.dfl.conv.weight` — that's a constant `arange` projection, reconstructed at runtime in `DetectV8`.
3. Transposes 4-D conv weights from PyTorch `(C_out, C_in, k_H, k_W)` to MLX `(C_out, k_H, k_W, C_in)`.

Everything else is copied verbatim. There are **no model-specific regex remappers**.

---

## Adding a new model family

A new model family slots in as **a YAML config + (optionally) a head class + a post-processor**.

1. Drop a YAML in `src/mlxyolos/cfg/models/<family>/`.
2. If the head differs from `DetectV8` / `PoseV8`, add a class in `nn/modules/head.py` and register it in `nn/tasks.py::MODULE_MAP` and `nn/tasks.py::HEAD_BUILDERS`.
3. Add a post-processor in `engine/predictor.py::POSTPROCESSORS` and a row in `engine/model.py::_HEAD_TASKS` (subclass tasks must come **before** their parents in the tuple — `PoseV8` extends `DetectV8`, so dispatching to `"detect"` would silently swallow the keypoint columns; this is the exact bug the `_HEAD_TASKS` ordering fixes).

The converter does **not** need to know about new families — see [Why plain Python lists for module containers](#why-plain-python-lists-for-module-containers) below.

For a worked example of adding `yolov8-seg`, the recipe in detail (with the `Proto` block, `ConvTranspose2d` weight transpose rule, mask post-processing edge cases, etc.) is part of the open scope when training/segmentation lands.

---

## Why plain Python lists for module containers

mlx-yolos's `BaseModel` exposes its layer list as plain `self.model = [...]`, matching Ultralytics' attribute name; submodule containers (`m`, `cv2`, `cv3`, `cv4`) are also plain Python lists. MLX's `tree_flatten` walks lists during parameter collection, producing keys like `model.4.m.0.cv1.conv.weight` that line up **1:1 with the Ultralytics state dict**.

That's the property that keeps the converter from growing as new families are added — the only operations are conv-weight transpose + dropping bookkeeping.

The alternative we considered (and that the closely-related `yolo-mlx` repo uses) is wrapping containers in a `Sequential` / `ModuleList` class with `self.layers = [...]`. That puts a `.layers.` segment in every parameter path, which means each new model family needs a regex-remapper to translate Ultralytics keys into MLX keys. Faster to wrap, slower to extend.
