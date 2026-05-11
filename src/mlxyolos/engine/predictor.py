# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generic predictor: image → preprocess → MLX forward → MLX post-process.

Single MLX path. Pre-processing uses cv2 for image read + letterbox (matches
Ultralytics' pixel grid). Everything else — per-anchor decode, ``xywh→xyxy``,
the NMS IoU matrix, scale-back — runs on Metal via ``mlxyolos.utils.ops_mlx``.
The post-processors materialize to numpy exactly once, at the boundary, for
``Boxes`` / ``Keypoints`` container construction and JSON serialization.

Adding a new task = registering one function in ``POSTPROCESSORS``, a head
class in ``nn/modules/head.py``, and a YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlxyolos.engine.results import Boxes, Keypoints, Results
from mlxyolos.utils.ops import letterbox
from mlxyolos.utils.ops_mlx import nms as nms_mlx
from mlxyolos.utils.ops_mlx import scale_coords as scale_coords_mlx
from mlxyolos.utils.ops_mlx import scale_keypoints as scale_keypoints_mlx
from mlxyolos.utils.ops_mlx import xywh_to_xyxy as xywh_to_xyxy_mlx

__all__ = ["Predictor", "POSTPROCESSORS"]


# ---------------------------------------------------------------------------
# Decode helpers — run on device, eval once at the boundary
# ---------------------------------------------------------------------------


def _decode_anchors(pred: mx.array, nc: int, nk: int):
    """Run heavy per-anchor decode on Metal, eval once, return MLX arrays.

    ``max``/``argmax`` over class scores and ``xywh → xyxy`` happen on the
    GPU and get fused into the same eval boundary as the model's forward,
    so the host doesn't pay for individual round-trips.
    """
    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    score = mx.max(cls_scores, axis=-1)
    cls_idx = mx.argmax(cls_scores, axis=-1)
    box_xyxy = xywh_to_xyxy_mlx(box_xywh)

    if nk > 0:
        kpts_flat = pred[:, 4 + nc : 4 + nc + nk]
        mx.eval(box_xyxy, score, cls_idx, kpts_flat)
        return box_xyxy, score, cls_idx, kpts_flat
    mx.eval(box_xyxy, score, cls_idx)
    return box_xyxy, score, cls_idx, None


# ---------------------------------------------------------------------------
# Per-task post-processors
# ---------------------------------------------------------------------------


def _postprocess_detect(
    pred: mx.array,
    nc: int,
    *,
    conf: float,
    iou: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> tuple[Boxes, None]:
    box_xyxy, score, cls, _ = _decode_anchors(pred, nc, nk=0)

    score_np = np.asarray(score)
    keep_mask = score_np > conf
    if not keep_mask.any():
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None
    keep_idx_np = np.flatnonzero(keep_mask)
    keep_idx = mx.array(keep_idx_np.astype(np.int32))

    box_xyxy = box_xyxy[keep_idx]
    score = score[keep_idx]
    cls = cls[keep_idx]

    nms_keep_np = nms_mlx(box_xyxy, score, iou_thr=iou)
    if len(nms_keep_np) == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None
    nms_keep = mx.array(nms_keep_np.astype(np.int32))

    box_xyxy = scale_coords_mlx(box_xyxy[nms_keep], ratio, pad, orig_shape)
    score = score[nms_keep]
    cls = cls[nms_keep]
    mx.eval(box_xyxy, score, cls)

    out = np.column_stack(
        [
            np.asarray(box_xyxy),
            np.asarray(score),
            np.asarray(cls).astype(np.float32),
        ]
    )
    return Boxes(out, orig_shape), None


def _postprocess_pose(
    pred: mx.array,
    nc: int,
    *,
    conf: float,
    iou: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    kpt_shape: tuple[int, int],
) -> tuple[Boxes, Keypoints]:
    nk = kpt_shape[0] * kpt_shape[1]
    box_xyxy, score, cls, kpts_flat = _decode_anchors(pred, nc, nk=nk)
    assert kpts_flat is not None

    score_np = np.asarray(score)
    keep_mask = score_np > conf
    if not keep_mask.any():
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)
    keep_idx_np = np.flatnonzero(keep_mask)
    keep_idx = mx.array(keep_idx_np.astype(np.int32))

    box_xyxy = box_xyxy[keep_idx]
    score = score[keep_idx]
    cls = cls[keep_idx]
    kpts_flat = kpts_flat[keep_idx]

    nms_keep_np = nms_mlx(box_xyxy, score, iou_thr=iou)
    if len(nms_keep_np) == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)
    nms_keep = mx.array(nms_keep_np.astype(np.int32))

    box_xyxy = scale_coords_mlx(box_xyxy[nms_keep], ratio, pad, orig_shape)
    score = score[nms_keep]
    cls = cls[nms_keep]
    kpts = kpts_flat[nms_keep].reshape(-1, kpt_shape[0], kpt_shape[1])
    kpts = scale_keypoints_mlx(kpts, ratio, pad, orig_shape)
    mx.eval(box_xyxy, score, cls, kpts)

    out = np.column_stack(
        [
            np.asarray(box_xyxy),
            np.asarray(score),
            np.asarray(cls).astype(np.float32),
        ]
    )
    return Boxes(out, orig_shape), Keypoints(np.asarray(kpts), orig_shape)


PostprocessFn = Callable[..., tuple[Boxes, Any]]
POSTPROCESSORS: dict[str, PostprocessFn] = {
    "detect": _postprocess_detect,
    "pose": _postprocess_pose,
}


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """Drives one model through preprocess → forward → post-process."""

    def __init__(
        self,
        model: nn.Module,
        task: str,
        nc: int,
        names: dict[int, str] | None = None,
        kpt_shape: tuple[int, int] | None = None,
    ) -> None:
        if task not in POSTPROCESSORS:
            raise ValueError(
                f"Unsupported task {task!r}; expected one of {sorted(POSTPROCESSORS)}"
            )
        self.model = model
        self.task = task
        self.nc = nc
        self.names = names or {0: "person"}
        self.kpt_shape = kpt_shape

    def __call__(
        self,
        source: str | Path | np.ndarray | list,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        rect: bool = True,
        scaleup: bool = True,
    ) -> list[Results]:
        sources = source if isinstance(source, list) else [source]
        results: list[Results] = []
        for src in sources:
            results.append(
                self._run_one(
                    src, imgsz=imgsz, conf=conf, iou=iou, rect=rect, scaleup=scaleup
                )
            )
        return results

    # ----- internals ---------------------------------------------------------

    def _load_image(self, src: str | Path | np.ndarray) -> tuple[np.ndarray, str]:
        """Return ``(rgb_uint8_ndarray, path_str)``. cv2-based file loading."""
        if isinstance(src, (str, Path)):
            path = str(src)
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError(f"could not load image: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), path
        if isinstance(src, np.ndarray):
            if src.ndim != 3 or src.shape[-1] not in (3, 4):
                raise ValueError(f"expected (H, W, 3|4) ndarray, got shape {src.shape}")
            return src[..., :3].astype(np.uint8, copy=False), ""
        raise TypeError(f"unsupported source type: {type(src)!r}")

    def _run_one(
        self,
        src: str | Path | np.ndarray,
        *,
        imgsz: int,
        conf: float,
        iou: float,
        rect: bool = True,
        scaleup: bool = True,
    ) -> Results:
        orig, path = self._load_image(src)
        lb_img, ratio, pad = letterbox(orig, imgsz, rect=rect, scaleup=scaleup)

        # Pre-process on device: upload uint8, normalize on Metal.
        x_uint8 = lb_img[None, ...]  # (1, H, W, 3) uint8
        x = mx.array(x_uint8).astype(mx.float32) / 255.0

        pred = self.model(x)[0]  # (A, 4 + nc + nk) — lazy until the post-processor evals

        kwargs: dict[str, Any] = dict(
            nc=self.nc,
            conf=conf,
            iou=iou,
            ratio=ratio,
            pad=pad,
            orig_shape=orig.shape[:2],
        )
        if self.task == "pose":
            kwargs["kpt_shape"] = self.kpt_shape or (17, 3)

        boxes, extra = POSTPROCESSORS[self.task](pred, **kwargs)

        return Results(
            orig_img=orig,
            path=path,
            names=self.names,
            boxes=boxes,
            keypoints=extra if self.task == "pose" else None,
        )
