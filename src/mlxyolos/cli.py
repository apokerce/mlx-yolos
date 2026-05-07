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

    model = YOLO(args.cfg, scale=args.scale, weights=args.weights, verbose=args.verbose)
    print(f"loaded {model.task!r} head, nc={model.nc}, names={model.names}")
    print(
        f"running on {args.source!r}: imgsz={args.imgsz}, "
        f"conf={args.conf}, iou={args.iou}, kpt_thr={args.kpt_thr}"
    )
    results = model.predict(args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    out_dir: Path | None = Path(args.save) if args.save else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        # Elaborate per-detection log: file, class label, conf, box, kpt count.
        print(r.verbose(kpt_thr=args.kpt_thr))

        if out_dir is not None:
            stem = Path(r.path).stem if r.path else f"img{i}"
            out_path = out_dir / f"{stem}.jpg"
            r.plot(kpt_thr=args.kpt_thr).save(out_path)
            print(f"  → saved {out_path}")
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
    pp.add_argument(
        "--kpt-thr",
        type=float,
        default=0.5,
        dest="kpt_thr",
        help="Visibility threshold below which keypoints are skipped (pose only)",
    )
    pp.add_argument("--save", default=None, help="Directory to save annotated images")
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(func=_cmd_predict)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
