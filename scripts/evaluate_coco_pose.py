# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate mlx-yolos yolov8-pose against COCO val2017.

Computes the standard pycocotools mAP tables for both the person-bbox
and the keypoints predictions, and reports MLX timing alongside (mean /
median / p95 per-image inference latency, throughput, and average
detection count for sanity).

Both bbox and keypoint mAP are evaluated against
``person_keypoints_val2017.json`` — the same ground-truth file Ultralytics'
``yolo pose val`` uses for both metrics. **Don't switch to
``instances_val2017.json`` for bbox** thinking it's "more correct" — that
file has ~4 400 extra person annotations (crowd boxes, occluded / tiny
persons without keypoint labels) that Ultralytics' pipeline never asks
the model to detect; including them tanks recall by 7 pt for no gain.

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
    rect: bool,
    scaleup: bool,
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
        results = yolo.predict(
            str(path), imgsz=imgsz, conf=conf, iou=iou, rect=rect, scaleup=scaleup
        )
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
    img_ids: list[int] | None = None,
):
    """Run pycocotools mAP eval and print its standard summary.

    Two pycocotools parameters matter for matching Ultralytics' val:

    1. ``params.catIds = cat_ids`` — restrict the AP average to the requested
       categories. For our pose eval against ``person_keypoints_val2017.json``
       this is a no-op (the GT JSON only has one category), but it's left
       explicit for documentation and for future detect-task evals.
    2. ``params.imgIds`` — restrict the eval *image set*. Defaulting
       pycocotools to "all images in the GT JSON" includes ~2 300 images
       with no person GT plus another ~350 with bbox-only persons (no
       visible keypoints) that Ultralytics' ``val2017.txt`` filters out
       (criterion: at least one annotation with ``num_keypoints > 0``).
       Pass the same 2 346-image set in here and the eval lines up with
       what ``yolo pose val`` reports.
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
    if img_ids is None and cat_ids:
        # Fallback: any image with a GT in the target categories.
        img_ids = sorted(coco_gt.getImgIds(catIds=cat_ids))
    if img_ids is not None:
        coco_eval.params.imgIds = list(img_ids)
        print(
            f"[{iou_type}] evaluating on {len(img_ids)} images "
            f"(full GT JSON has {len(coco_gt.getImgIds())})"
        )
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
    p.add_argument(
        "--rect",
        dest="rect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rectangular letterbox (pads each dim to a stride-multiple). "
            "Default OFF for this script — Ultralytics' `yolo pose val` uses "
            "square letterbox, so the apples-to-apples val comparison goes "
            "through square mode."
        ),
    )
    p.add_argument(
        "--scaleup",
        dest="scaleup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow upscaling images smaller than --imgsz. Default OFF for this "
            "script — Ultralytics' `yolo pose val` does NOT upscale, so val "
            "comparison goes through scaleup=False. About 18 %% of COCO val2017 "
            "has both dims < 640, and that's where most small-object GTs live, "
            "so this flag materially affects box AP @ small."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--verbose-every", type=int, default=500, help="progress print interval")
    p.add_argument(
        "--skip-bbox", action="store_true", help="run only keypoints eval (saves ~1 min)"
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images" / "val2017"
    # Use person_keypoints_val2017.json for BOTH bbox and keypoint eval —
    # Ultralytics' pose-val pipeline does the same (see
    # `ultralytics/models/yolo/pose/val.py::eval_json`). person_keypoints
    # annotations carry both `bbox` and `keypoints` per object, but cover
    # only the ~2 346 "well-keypointed" person instances. Evaluating bbox
    # against `instances_val2017.json` (cat=1) adds ~4 400 extra person
    # GTs that *don't* have keypoints — crowd boxes, occluded / tiny
    # persons, etc. — which Ultralytics' val pipeline never asks the
    # model to find. The recall denominator difference is the missing
    # 7 pt of box AP from earlier rounds.
    gt_path = data_dir / "annotations" / "person_keypoints_val2017.json"

    for required in (image_dir, gt_path):
        if not required.exists():
            print(f"missing {required} — run scripts/get_coco_pose_val.sh first", file=sys.stderr)
            return 1

    # Load image_id → file_name (the annotations file lists every val2017
    # image even though it only has annotations for the person-keypoint subset).
    print(f"loading annotations from {gt_path}")
    with open(gt_path) as f:
        ann = json.load(f)
    id_to_filename = {im["id"]: im["file_name"] for im in ann["images"]}
    image_ids = sorted(id_to_filename)

    # Compute the eval image set the way Ultralytics' pose-val pipeline does:
    # only images with at least one annotation that has ``num_keypoints > 0``.
    # `coco-pose.yaml`'s `val: val2017.txt` is this exact 2 346-image subset.
    # Without this filter our recall-denominator is 2 693 — the extra ~350
    # images are the "bbox-only persons" (crowd, occluded, distant) that
    # Ultralytics' val never asks the model to find.
    eval_img_ids = sorted(
        {a["image_id"] for a in ann["annotations"] if a.get("num_keypoints", 0) > 0}
    )
    print(
        f"  total val2017 images in JSON: {len(id_to_filename)}"
        f"  ({len(eval_img_ids)} have ≥1 annotation with num_keypoints > 0)"
    )

    from mlxyolos import YOLO

    print(f"loading model: cfg={args.cfg} scale={args.scale} weights={args.weights}")
    yolo = YOLO(args.cfg, scale=args.scale, weights=args.weights)

    # Warmup — first MLX call pays for graph compilation; don't let it
    # contaminate the timing average.
    warmup_path = image_dir / id_to_filename[image_ids[0]]
    print(f"warming up on {warmup_path.name}")
    for _ in range(3):
        yolo.predict(
            str(warmup_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            rect=args.rect,
            scaleup=args.scaleup,
        )

    predictions, timings, det_counts = _build_predictions(
        yolo,
        image_dir,
        image_ids,
        id_to_filename,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        rect=args.rect,
        scaleup=args.scaleup,
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
    # Both bbox and keypoint eval use the SAME person_keypoints_val2017.json
    # ground truth and the SAME ~2 346-image set (annotations with
    # num_keypoints > 0), matching Ultralytics' pose-val pipeline exactly.
    if not args.skip_bbox:
        _coco_eval(
            gt_path,
            predictions,
            iou_type="bbox",
            max_dets=[1, 10, 100],
            cat_ids=[COCO_PERSON_CATEGORY],
            img_ids=eval_img_ids,
        )
    _coco_eval(
        gt_path,
        predictions,
        iou_type="keypoints",
        max_dets=[20],
        cat_ids=[COCO_PERSON_CATEGORY],
        img_ids=eval_img_ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
