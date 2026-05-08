# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate mlx-yolos yolov8-pose against COCO val2017.

Computes the standard pycocotools mAP tables for both the person-bbox
and the keypoints predictions, and reports MLX timing alongside (mean /
median / p95 per-image inference latency, throughput, and average
detection count for sanity).

Setup:
    bash scripts/get_coco_pose_val.sh
    pip install -e '.[val]'

Run:
    python scripts/evaluate_coco_pose.py \\
        --weights yolov8n-pose.safetensors \\
        --cfg yolov8-pose.yaml

The script intentionally uses ``conf=0.001``, the Ultralytics evaluation
default — anything higher truncates the AP curve and underestimates
mAP. NMS uses ``iou=0.7`` to match upstream's val behavior (vs. 0.45 for
predict — the looser threshold lets pycocotools' AP integration
accumulate more candidates before its own deduplication).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# COCO category id for "person" (class 0 in yolov8-pose).
COCO_PERSON_CATEGORY = 1


def _format_ms_stats(ms: list[float]) -> str:
    """Compact mean/median/p95/throughput summary for a per-image timing list."""
    arr = np.asarray(ms)
    return (
        f"mean={arr.mean():7.2f} ms  "
        f"median={np.median(arr):7.2f} ms  "
        f"p95={np.percentile(arr, 95):7.2f} ms  "
        f"throughput={1000.0 / arr.mean():6.1f} img/s"
    )


def _build_predictions(
    yolo,
    image_dir: Path,
    image_ids: list[int],
    id_to_filename: dict[int, str],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    limit: int | None,
    verbose_every: int,
) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    """Predict over all val images, return COCO-formatted predictions + timings."""
    predictions: list[dict[str, Any]] = []
    timings_ms: list[float] = []
    detection_counts: list[int] = []

    if limit is not None:
        image_ids = image_ids[:limit]

    n = len(image_ids)
    print(f"running inference on {n} images...")
    t_total_start = time.perf_counter()

    for idx, image_id in enumerate(image_ids):
        path = image_dir / id_to_filename[image_id]
        if not path.exists():
            # Some COCO image_ids may be absent in val2017 if user trimmed —
            # skip silently to keep the eval robust to partial datasets.
            continue

        t0 = time.perf_counter()
        results = yolo.predict(str(path), imgsz=imgsz, conf=conf, iou=iou)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

        r = results[0]
        boxes = r.boxes
        kpts = r.keypoints
        n_det = 0 if boxes is None else len(boxes)
        detection_counts.append(n_det)

        if boxes is not None and kpts is not None and kpts.data is not None:
            xywh = boxes.xywh
            conf_arr = boxes.conf
            kp_arr = kpts.data  # (N, 17, 3)
            for i in range(n_det):
                cx, cy, w, h = xywh[i].tolist()
                # COCO bbox is [x_top, y_top, w, h].
                bbox = [cx - w / 2, cy - h / 2, w, h]
                # Keypoints flat [x0, y0, v0, x1, y1, v1, ...].
                kp_flat = kp_arr[i].astype(np.float32).reshape(-1).tolist()
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": COCO_PERSON_CATEGORY,
                        "bbox": [float(v) for v in bbox],
                        "score": float(conf_arr[i]),
                        "keypoints": kp_flat,
                    }
                )

        if verbose_every and (idx + 1) % verbose_every == 0:
            partial = np.asarray(timings_ms)
            print(
                f"  [{idx + 1:>5}/{n}]  "
                f"{partial.mean():.1f} ms/img  "
                f"avg objects/img={np.mean(detection_counts):.2f}"
            )

    t_total = time.perf_counter() - t_total_start
    print(f"done. total wall: {t_total:.1f} s")
    return predictions, timings_ms, detection_counts


def _coco_eval(
    gt_path: Path,
    predictions: list[dict[str, Any]],
    iou_type: str,
    *,
    max_dets: list[int],
    cat_ids: list[int] | None = None,
):
    """Run pycocotools mAP eval and print its standard summary.

    If ``cat_ids`` is given, restrict the AP averaging to those categories.
    Critical for bbox mAP on yolov8-pose: the instances GT carries all 80
    COCO classes, but we only predict ``person`` (id 1). Without this
    filter pycocotools averages our person AP across 80 categories with
    79 zeros, dragging the overall mAP to ~1/80 of the true value.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        print(f"\n[{iou_type}] no predictions — skipping eval")
        return None

    coco_gt = COCO(str(gt_path))
    # loadRes wants either a path or a list of dicts; we pass a tmp file so
    # large prediction sets stream through pycocotools' own parser.
    tmp = gt_path.parent / f"_mlxyolos_{iou_type}_predictions.json"
    with open(tmp, "w") as f:
        json.dump(predictions, f)
    try:
        coco_dt = coco_gt.loadRes(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if max_dets:
        coco_eval.params.maxDets = list(max_dets)
    if cat_ids:
        coco_eval.params.catIds = list(cat_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    print(f"\n[{iou_type} mAP]")
    coco_eval.summarize()
    return coco_eval


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="MLX safetensors weights")
    p.add_argument("--cfg", default="yolov8-pose.yaml", help="model YAML")
    p.add_argument("--scale", default=None, help="model scale (n/s/m/l/x)")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("COCO_POSE_DIR", "./datasets/coco-pose"),
        help="COCO root containing images/val2017 + annotations/",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001, help="Ultralytics eval default; do not raise")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU; matches Ultralytics val")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--verbose-every", type=int, default=500, help="progress print interval")
    p.add_argument(
        "--skip-bbox", action="store_true", help="run only keypoints eval (saves ~1 min)"
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images" / "val2017"
    keypoints_gt = data_dir / "annotations" / "person_keypoints_val2017.json"
    bbox_gt = data_dir / "annotations" / "instances_val2017.json"

    for required in (image_dir, keypoints_gt, bbox_gt):
        if not required.exists():
            print(f"missing {required} — run scripts/get_coco_pose_val.sh first", file=sys.stderr)
            return 1

    # Load image_id → file_name from the keypoints annotations (it lists every
    # val image, since person_keypoints uses the same image set as instances).
    print(f"loading annotations from {keypoints_gt}")
    with open(keypoints_gt) as f:
        ann = json.load(f)
    id_to_filename = {im["id"]: im["file_name"] for im in ann["images"]}
    image_ids = sorted(id_to_filename)

    from mlxyolos import YOLO

    print(f"loading model: cfg={args.cfg} scale={args.scale} weights={args.weights}")
    yolo = YOLO(args.cfg, scale=args.scale, weights=args.weights)

    # Warmup — first MLX call pays for graph compilation; don't let it
    # contaminate the timing average.
    warmup_path = image_dir / id_to_filename[image_ids[0]]
    print(f"warming up on {warmup_path.name}")
    for _ in range(3):
        yolo.predict(str(warmup_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    predictions, timings, det_counts = _build_predictions(
        yolo,
        image_dir,
        image_ids,
        id_to_filename,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        limit=args.limit,
        verbose_every=args.verbose_every,
    )

    # ---- Timing summary ----------------------------------------------------
    print()
    print("=" * 72)
    print("MLX inference timing")
    print("=" * 72)
    print(_format_ms_stats(timings))
    print(f"avg detections / image: {np.mean(det_counts):.2f}")
    print(f"images processed:       {len(timings)}")

    # ---- mAP ---------------------------------------------------------------
    # Both evaluations are restricted to category_id=1 (person). For
    # keypoints this is a no-op (single-category task); for bbox it's
    # critical — see _coco_eval for the gotcha it avoids.
    if not args.skip_bbox:
        _coco_eval(
            bbox_gt,
            predictions,
            iou_type="bbox",
            max_dets=[1, 10, 100],
            cat_ids=[COCO_PERSON_CATEGORY],
        )
    _coco_eval(
        keypoints_gt,
        predictions,
        iou_type="keypoints",
        max_dets=[20],
        cat_ids=[COCO_PERSON_CATEGORY],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
