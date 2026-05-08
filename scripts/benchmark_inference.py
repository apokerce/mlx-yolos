# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-scale inference benchmark: yolov8{n,s,m,l,x}-pose × {mlx, torch-cpu, torch-mps}.

Same images, same imgsz, same conf/iou per call, three backends per scale
(auto-skipped when not available). Output:

  * a per-scale + summary table on stdout, and
  * a vertical grouped bar chart (latency on Y axis, scale on X axis) saved
    next to the run.

Setup:
    bash scripts/get_coco_pose_val.sh
    pip install -e '.[benchmark]'

    # Download all five Ultralytics .pt files and convert each to MLX:
    for s in n s m l x; do
        curl -L -o yolov8${s}-pose.pt \\
            https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8${s}-pose.pt
        mlx-yolos convert \\
            --pt yolov8${s}-pose.pt \\
            --out yolov8${s}-pose.safetensors
    done

Run:
    # All five scales, all three backends:
    python scripts/benchmark_inference.py

    # Subset of scales (matches dir layout: yolov8{n,s}-pose.{pt,safetensors}):
    python scripts/benchmark_inference.py --scales n,s

    # Override file naming if your weights live elsewhere:
    python scripts/benchmark_inference.py \\
        --pt-pattern  '/weights/yolov8{scale}-pose.pt' \\
        --mlx-pattern '/weights/yolov8{scale}-pose.safetensors'

For each (scale, backend) cell we report mean / median / p95 latency,
throughput, average detections per image (sanity), and total wall time.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _has_mps() -> bool:
    if not _has_torch():
        return False
    import torch

    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# Backend wrappers — each builder returns (predict_fn, n_objects_fn).
# ---------------------------------------------------------------------------


def _build_mlx_backend(weights: str, cfg: str, scale: str):
    from mlxyolos import YOLO

    model = YOLO(cfg, scale=scale, weights=weights)

    def predict(path: str, *, imgsz: int, conf: float, iou: float):
        return model.predict(path, imgsz=imgsz, conf=conf, iou=iou)

    def n_objects(results) -> int:
        return 0 if results[0].boxes is None else len(results[0].boxes)

    return predict, n_objects


def _build_torch_backend(weights: str, device: str):
    """Run yolov8-pose .pt through Ultralytics' own pipeline on the requested
    torch device. We sync after each call so the measured latency includes
    the GPU/MPS work, not just enqueue time."""
    import torch
    from ultralytics import YOLO as TorchYOLO

    model = TorchYOLO(weights)
    model.to(device)

    def _sync():
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        # CPU is synchronous already.

    def predict(path: str, *, imgsz: int, conf: float, iou: float):
        results = model.predict(
            source=path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False
        )
        _sync()
        return results

    def n_objects(results) -> int:
        r = results[0]
        return 0 if r.boxes is None else len(r.boxes)

    return predict, n_objects


# ---------------------------------------------------------------------------
# Result type + benchmark loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    scale: str
    backend: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_img_per_s: float
    avg_objects: float
    n_images: int
    wall_s: float


def _bench(
    scale: str,
    backend_name: str,
    predict_fn: Callable,
    n_objects_fn: Callable,
    paths: list[Path],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    warmup: int,
) -> BenchResult:
    print(f"  warmup ({warmup} runs)...", flush=True)
    for i in range(warmup):
        _ = predict_fn(str(paths[i % len(paths)]), imgsz=imgsz, conf=conf, iou=iou)

    times_ms: list[float] = []
    n_objs: list[int] = []
    print(f"  timing {len(paths)} images...", flush=True)
    t_total = time.perf_counter()
    for path in paths:
        t0 = time.perf_counter()
        results = predict_fn(str(path), imgsz=imgsz, conf=conf, iou=iou)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        n_objs.append(n_objects_fn(results))
    wall_s = time.perf_counter() - t_total

    arr = np.asarray(times_ms)
    return BenchResult(
        scale=scale,
        backend=backend_name,
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        throughput_img_per_s=1000.0 / float(arr.mean()),
        avg_objects=float(np.mean(n_objs)),
        n_images=len(paths),
        wall_s=wall_s,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_per_scale_table(results: list[BenchResult], scale: str) -> None:
    rows = [r for r in results if r.scale == scale]
    if not rows:
        return
    print(f"\n--- yolov8{scale}-pose ---")
    print(
        f"{'backend':<12} {'mean':>9} {'median':>9} {'p95':>9} "
        f"{'throughput':>13} {'avg obj/img':>13} {'wall':>9}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r.backend:<12} "
            f"{r.mean_ms:>7.2f} ms "
            f"{r.median_ms:>7.2f} ms "
            f"{r.p95_ms:>7.2f} ms "
            f"{r.throughput_img_per_s:>9.1f} img/s "
            f"{r.avg_objects:>13.2f} "
            f"{r.wall_s:>7.1f} s"
        )


def _print_summary(results: list[BenchResult]) -> None:
    if not results:
        return
    scales = sorted({r.scale for r in results}, key=lambda s: "nsmlx".index(s))
    backends = sorted({r.backend for r in results})

    print()
    print("=" * (12 + 11 * len(scales)))
    print(f"Summary — mean latency per backend × scale (ms / image)")
    print("=" * (12 + 11 * len(scales)))
    header = f"{'backend':<12}" + "".join(f"{('yolov8' + s):>11}" for s in scales)
    print(header)
    print("-" * len(header))
    for b in backends:
        row = f"{b:<12}"
        for s in scales:
            cell = next((r for r in results if r.scale == s and r.backend == b), None)
            row += f"{(cell.mean_ms if cell else float('nan')):>11.2f}"
        print(row)


def _save_grouped_bar_chart(
    results: list[BenchResult],
    path: Path,
    *,
    n_images: int,
    imgsz: int,
) -> None:
    """Vertical grouped-bar chart: scales on X axis, latency on Y axis, one
    bar per backend per scale. A second subplot shows throughput. Skips
    silently if matplotlib isn't importable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"\n[plot] matplotlib unavailable ({e}); skipping bar chart")
        return

    if not results:
        print("[plot] no results to plot")
        return

    palette = {
        "mlx": "#4c8cff",
        "torch-cpu": "#ff6f6f",
        "torch-mps": "#7fc97f",
    }
    scales = sorted({r.scale for r in results}, key=lambda s: "nsmlx".index(s))
    backends = sorted({r.backend for r in results})

    def _cell(scale: str, backend: str) -> BenchResult | None:
        return next((r for r in results if r.scale == scale and r.backend == backend), None)

    x = np.arange(len(scales))
    n_bk = len(backends)
    bar_w = 0.8 / max(n_bk, 1)

    fig, axes = plt.subplots(2, 1, figsize=(max(7, 1.2 * len(scales) * n_bk), 8.5))
    fig.suptitle(
        f"yolov8-pose inference — COCO val2017 sample (N={n_images}, imgsz={imgsz})",
        fontsize=12,
    )

    # --- (1) Mean latency vs scale -----------------------------------------
    ax = axes[0]
    for i, b in enumerate(backends):
        means = [(_cell(s, b).mean_ms if _cell(s, b) else np.nan) for s in scales]
        p95s = [(_cell(s, b).p95_ms if _cell(s, b) else np.nan) for s in scales]
        offset = (i - (n_bk - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            means,
            bar_w,
            label=b,
            color=palette.get(b, "#888888"),
            edgecolor="black",
            linewidth=0.5,
        )
        # p95 whisker on top of each bar.
        for j, p95 in enumerate(p95s):
            if not np.isnan(p95):
                ax.plot(
                    [x[j] + offset, x[j] + offset],
                    [means[j], p95],
                    color="black",
                    linewidth=1.0,
                )
        # value labels above each bar.
        for j, bar in enumerate(bars):
            if not np.isnan(means[j]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{means[j]:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([f"yolov8{s}" for s in scales])
    ax.set_ylabel("mean latency (ms / image) — lower is better")
    ax.set_title("inference latency (whisker = p95)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # --- (2) Throughput vs scale ------------------------------------------
    ax = axes[1]
    for i, b in enumerate(backends):
        tps = [(_cell(s, b).throughput_img_per_s if _cell(s, b) else np.nan) for s in scales]
        offset = (i - (n_bk - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            tps,
            bar_w,
            label=b,
            color=palette.get(b, "#888888"),
            edgecolor="black",
            linewidth=0.5,
        )
        for j, bar in enumerate(bars):
            if not np.isnan(tps[j]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{tps[j]:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([f"yolov8{s}" for s in scales])
    ax.set_ylabel("throughput (images / second) — higher is better")
    ax.set_title("throughput")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sample_images(image_dir: Path, n: int, seed: int) -> list[Path]:
    all_paths = sorted(image_dir.glob("*.jpg"))
    if not all_paths:
        raise SystemExit(f"no .jpg under {image_dir} — run scripts/get_coco_pose_val.sh")
    rng = random.Random(seed)
    return rng.sample(all_paths, min(n, len(all_paths)))


def _resolve_path(pattern: str, scale: str) -> Path:
    return Path(pattern.format(scale=scale))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scales",
        default="n,s,m,l,x",
        help="comma-separated list of yolov8-pose scales to benchmark",
    )
    p.add_argument(
        "--pt-pattern",
        default="yolov8{scale}-pose.pt",
        help="path template for Ultralytics .pt weights; {scale} substituted",
    )
    p.add_argument(
        "--mlx-pattern",
        default="yolov8{scale}-pose.safetensors",
        help="path template for MLX safetensors; {scale} substituted",
    )
    p.add_argument("--cfg", default="yolov8-pose.yaml", help="MLX model YAML")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("COCO_POSE_DIR", "./datasets/coco-pose"),
        help="COCO root containing images/val2017/",
    )
    p.add_argument("--n-images", type=int, default=200, help="how many val images to time on")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--seed", type=int, default=0, help="image-sampling seed for reproducibility")
    p.add_argument(
        "--only",
        choices=["mlx", "torch-cpu", "torch-mps"],
        action="append",
        default=None,
        help="restrict which backends run; repeat flag to allow multiple",
    )
    p.add_argument(
        "--save-plot",
        default="benchmark.png",
        help="path to save the comparison bar chart (PNG); use '' to disable",
    )
    args = p.parse_args()

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    bad = [s for s in scales if s not in {"n", "s", "m", "l", "x"}]
    if bad:
        print(f"unknown scale(s): {bad}; expected subset of n,s,m,l,x", file=sys.stderr)
        return 2

    image_dir = Path(args.data_dir) / "images" / "val2017"
    paths = _sample_images(image_dir, args.n_images, args.seed)
    print(f"sampled {len(paths)} images from {image_dir}")
    print(
        f"scales={scales}  imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}  "
        f"warmup={args.warmup}"
    )

    selected = set(args.only) if args.only else {"mlx", "torch-cpu", "torch-mps"}
    results: list[BenchResult] = []

    for scale in scales:
        mlx_w = _resolve_path(args.mlx_pattern, scale)
        pt_w = _resolve_path(args.pt_pattern, scale)
        print(f"\n##### yolov8{scale}-pose #####")

        # ---- mlx ----
        if "mlx" in selected:
            if not _has_mlx():
                print("[skip] mlx — `mlx.core` not importable on this host")
            elif not mlx_w.exists():
                print(f"[skip] mlx — weights not found at {mlx_w}")
            else:
                print(f"loading mlx {mlx_w}")
                predict_fn, n_obj_fn = _build_mlx_backend(str(mlx_w), args.cfg, scale)
                results.append(
                    _bench(
                        scale,
                        "mlx",
                        predict_fn,
                        n_obj_fn,
                        paths,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        warmup=args.warmup,
                    )
                )

        # ---- torch-cpu ----
        if "torch-cpu" in selected:
            if not _has_torch():
                print("[skip] torch-cpu — `torch` not installed (`pip install '.[benchmark]'`)")
            elif not pt_w.exists():
                print(f"[skip] torch-cpu — weights not found at {pt_w}")
            else:
                print(f"loading torch-cpu {pt_w}")
                predict_fn, n_obj_fn = _build_torch_backend(str(pt_w), "cpu")
                results.append(
                    _bench(
                        scale,
                        "torch-cpu",
                        predict_fn,
                        n_obj_fn,
                        paths,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        warmup=args.warmup,
                    )
                )

        # ---- torch-mps ----
        if "torch-mps" in selected:
            if not _has_mps():
                print("[skip] torch-mps — not on Apple Silicon, or torch built without MPS")
            elif not pt_w.exists():
                print(f"[skip] torch-mps — weights not found at {pt_w}")
            else:
                print(f"loading torch-mps {pt_w}")
                predict_fn, n_obj_fn = _build_torch_backend(str(pt_w), "mps")
                results.append(
                    _bench(
                        scale,
                        "torch-mps",
                        predict_fn,
                        n_obj_fn,
                        paths,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        warmup=args.warmup,
                    )
                )

        _print_per_scale_table(results, scale)

    if not results:
        print("\nno backends ran for any scale — install at least one of mlx / torch", file=sys.stderr)
        return 1

    _print_summary(results)
    if args.save_plot:
        _save_grouped_bar_chart(
            results,
            Path(args.save_plot),
            n_images=len(paths),
            imgsz=args.imgsz,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
