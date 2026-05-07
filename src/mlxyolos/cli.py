# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Command-line interface.

Subcommands:
    convert   Ultralytics ``.pt`` → mlx-yolos ``.safetensors`` (no MLX needed).
    predict   Run inference and optionally save the annotated image.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _cmd_convert(args: argparse.Namespace) -> int:
    from mlxyolos.converters import convert_ultralytics_checkpoint

    summary = convert_ultralytics_checkpoint(args.pt, args.out, verbose=not args.quiet)
    if not args.quiet:
        print(
            f"converted {summary['kept']} tensors "
            f"({summary['conv_weights_transposed']} conv weights transposed); "
            f"skipped {summary['skipped']} bookkeeping keys → {summary['out_path']}"
        )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    # Lazy: import MLX-dependent code only when actually running inference.
    from mlxyolos import YOLO
    from mlxyolos.utils.plotting import draw_pose

    model = YOLO(args.cfg, scale=args.scale, weights=args.weights, verbose=args.verbose)
    results = model.predict(args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"{r.path or '<array>'}: {n} detections")

    if args.save and results:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(results):
            stem = Path(r.path).stem if r.path else f"img{i}"
            out_path = out_dir / f"{stem}.jpg"
            if model.task == "pose" and r.boxes is not None and r.keypoints is not None and r.keypoints.data is not None:
                im = draw_pose(r.orig_img, r.boxes.xyxy, r.boxes.conf, r.keypoints.data)
            else:
                from PIL import Image as _PIL

                im = _PIL.fromarray(r.orig_img)
            im.save(out_path)
            print(f"  -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlx-yolos")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("convert", help="Convert Ultralytics .pt to MLX safetensors")
    pc.add_argument("--pt", required=True, help="Path to Ultralytics .pt checkpoint")
    pc.add_argument("--out", required=True, help="Output .safetensors path")
    pc.add_argument("-q", "--quiet", action="store_true")
    pc.set_defaults(func=_cmd_convert)

    pp = sub.add_parser("predict", help="Run inference")
    pp.add_argument("--cfg", required=True, help="YAML config (e.g. yolov8-pose.yaml)")
    pp.add_argument("--weights", required=True, help="MLX safetensors weights")
    pp.add_argument("--source", required=True, help="Image path")
    pp.add_argument("--scale", default=None, help="Model scale: n/s/m/l/x")
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--conf", type=float, default=0.25)
    pp.add_argument("--iou", type=float, default=0.45)
    pp.add_argument("--save", default=None, help="Directory to save annotated images")
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(func=_cmd_predict)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
