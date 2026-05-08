# Cross-backend benchmark

Apples-to-apples per-image latency on the same `yolov8n-pose` weights, same images, same `imgsz`, across:

- **mlx** — `mlxyolos.YOLO`, full pipeline (pre / forward / decode / NMS) on Metal.
- **torch-cpu** — `ultralytics.YOLO(...).to("cpu")`.
- **torch-mps** — same, `.to("mps")`. We call `torch.mps.synchronize()` after every forward so the timer captures the actual MPS work, not just queueing.

Backends absent from the host (no MLX runtime, no MPS device, no torch installed) are auto-skipped.

- [Headline results](#headline-results)
- [How to read the numbers](#how-to-read-the-numbers)
- [Reproducing](#reproducing)
- [Methodology notes](#methodology-notes)

---

## Headline results

`yolov8{n,s,m,l,x}-pose`, 200 random COCO val2017 images per cell, imgsz=640, conf=0.25, iou=0.45, warmup=40.

![Cross-backend inference benchmark — mlx vs torch-cpu vs torch-mps across 5 scales](benchmark.png)

### Mean latency per backend × scale (ms / image, lower is better)

| Backend     | yolov8n  | yolov8s  | yolov8m  | yolov8l  | yolov8x   |
|-------------|---------:|---------:|---------:|---------:|----------:|
| **mlx**     | **16.52** | **27.93** | **53.63** | 84.43    | 134.52    |
| torch-cpu   | 35.60    | 53.88    | 90.63    | 143.84   | 195.92    |
| torch-mps   | 33.12    | 35.59    | 51.21    | **69.89** | **98.91** |

Bold = fastest backend in the column.

### Per-scale detail

| Scale  | Backend     | mean (ms) | median (ms) | p95 (ms) | throughput     | avg obj/img | wall (s) |
|--------|-------------|----------:|------------:|---------:|---------------:|------------:|---------:|
| n      | mlx         | **16.52** | 16.40       | 18.07    | **60.6 img/s** | 1.18        | 3.3      |
| n      | torch-cpu   | 35.60     | 35.76       | 41.87    | 28.1 img/s     | 1.15        | 7.1      |
| n      | torch-mps   | 33.12     | 21.19       | 55.31    | 30.2 img/s     | 1.15        | 6.6      |
| s      | mlx         | **27.93** | 27.85       | 29.66    | **35.8 img/s** | 1.19        | 5.6      |
| s      | torch-cpu   | 53.88     | 53.51       | 64.30    | 18.6 img/s     | 1.21        | 10.8     |
| s      | torch-mps   | 35.59     | 23.99       | 69.16    | 28.1 img/s     | 1.21        | 7.1      |
| m      | mlx         | **53.63** | 53.54       | 55.42    | **18.6 img/s** | 1.18        | 10.7     |
| m      | torch-cpu   | 90.63     | 90.39       | 113.10   | 11.0 img/s     | 1.23        | 18.1     |
| m      | torch-mps   | 51.21     | 38.14       | 84.05    | 19.5 img/s     | 1.23        | 10.2     |
| l      | mlx         | 84.43     | 84.07       | 88.38    | 11.8 img/s     | 1.19        | 16.9     |
| l      | torch-cpu   | 143.84    | 143.24      | 178.40   | 7.0 img/s      | 1.20        | 28.8     |
| l      | torch-mps   | **69.89** | 56.85       | 113.15   | **14.3 img/s** | 1.20        | 14.0     |
| x      | mlx         | 134.52    | 133.89      | 142.72   | 7.4 img/s      | 1.19        | 26.9     |
| x      | torch-cpu   | 195.92    | 192.52      | 251.19   | 5.1 img/s      | 1.18        | 39.2     |
| x      | torch-mps   | **98.91** | 75.38       | 133.67   | **10.1 img/s** | 1.18        | 19.8     |

---

## How to read the numbers

There's a **crossover at the m/l boundary** that's worth understanding before you pick a backend for production:

- **Small models (n, s, m): mlx wins.** yolov8n / yolov8s / yolov8m are small enough that *kernel launch overhead and host↔device sync cost* dominate the wall-clock. MLX runs the entire pipeline (pre-process, forward, decode, NMS IoU matrix) on Metal with lazy graph fusion and a single eval boundary; PyTorch on MPS pays per-op launch overhead and bounces back to CPU for the torchvision NMS, which adds up. mlx is **2.1× faster** at yolov8n, **1.5× faster** at yolov8s, ~tied with torch-mps at yolov8m.
- **Large models (l, x): torch-mps wins.** Now compute is the dominant cost (yolov8x is ~21× more FLOPs than yolov8n), and PyTorch MPS' tuned matmul kernels pay for the per-op overhead they couldn't amortize before. torch-mps is **1.21× faster than mlx at yolov8l** and **1.36× faster at yolov8x**.
- **torch-cpu loses everywhere.** Predictably — even for the smallest model the CPU path is ~2× slower than the GPU paths and the gap widens with FLOPs (~1.5× → ~2× behind torch-mps from n to x).
- **`avg obj/img` agrees across all three** at every scale (1.15–1.23). Same NMS shape, no post-processing divergence — the latency comparison is apples-to-apples.

So the practical rule is: **use mlx for n/s/m, use torch-mps for l/x** if you're running yolov8-pose at imgsz=640 today. The crossover point will move depending on imgsz and image complexity, but the underlying physics — overhead-bound on small graphs, compute-bound on big ones — is the same.

The MPS p95↔median ratio stays high across all scales (e.g. 55/21 = 2.6× at yolov8n, 134/75 = 1.8× at yolov8x). That's structural, not warmup-related: `torchvision.ops.nms` falls back to CPU on MPS, and the round-trip cost varies per image with the number of candidate boxes. More warmup won't fix it.

---

## Reproducing

```bash
# Same dataset the COCO mAP eval uses:
bash scripts/get_coco_pose_val.sh

# Download all five Ultralytics .pt files and convert each to MLX (one-time):
bash scripts/download_convert_v8.sh

# All five scales, all three backends, default --n-images 200 --warmup 40:
pip install -e '.[benchmark]'
python scripts/benchmark_inference.py
```

The script writes a vertical grouped-bar chart (latency on Y, scale on X, one bar per backend per scale, p95 whisker on each) to `benchmark.png` next to the run. We commit the chart at [`docs/benchmark.png`](benchmark.png) so the README stays viewable on hosts without the COCO dataset or the `[benchmark]` extras installed — pass `--save-plot docs/benchmark.png` to write directly there.

Common variations:

```bash
# Subset of scales while iterating:
python scripts/benchmark_inference.py --scales n,s

# Only one backend (faster smoke run):
python scripts/benchmark_inference.py --only mlx

# Different sample size or warmup:
python scripts/benchmark_inference.py --n-images 500 --warmup 60

# Weights stored elsewhere:
python scripts/benchmark_inference.py \
    --pt-pattern  '/weights/yolov8{scale}-pose.pt' \
    --mlx-pattern '/weights/yolov8{scale}-pose.safetensors'
```

For sample count: at N=200 with a CV (std-dev / mean) of ~5–10%, the standard error of the mean is already 0.4–0.7%. Going to 1000 cuts that to 0.2–0.3%, but the trend is statistically settled at 200. Save the wall time.

For warmup: 40 covers MPS shader compilation across all five scales. Lower values (e.g. 10) start to undercount MPS' effective speed because the v8-pose graph touches enough distinct ops that the first ~20 calls per scale are still paying compilation cost.

---

## Methodology notes

- **Same `.pt` source on both torch backends.** We load the original Ultralytics `yolov8n-pose.pt` for torch-cpu / torch-mps so the underlying weights are identical to what `mlx-yolos convert` produced for the MLX backend.
- **Per-call sync on MPS.** `torch.mps.synchronize()` is invoked after every forward; without it the timer captures only enqueue time and reports artificially low MPS latency.
- **MPS p95 spread is structural.** `torchvision.ops.nms` falls back to CPU on MPS, and the MPS→CPU→MPS round-trip cost varies per image with the number of candidate boxes. More warmup won't fix it; that's a PyTorch-side engineering issue.
- **Image read + letterbox stay on the host on every backend.** They're CPU-bound regardless and would only confuse the cross-backend comparison.
