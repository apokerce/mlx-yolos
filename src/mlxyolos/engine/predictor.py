# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generic predictor: image → preprocess → MLX forward → per-task post-processing.

The post-processors are dispatched by task and dual-backend: when called
with an ``mx.array`` they keep the heavy per-anchor decode work
(``argmax``/``max``/``xywh→xyxy`` over thousands of anchors, plus the
NMS IoU matrix) on the Apple-Silicon GPU, evaluate once at the boundary,
and finish on the host with the small post-filter set. When called with
an ``np.ndarray`` (tests, MLX-less environments) they take the pure-numpy
path. Adding a new task = registering one function in ``POSTPROCESSORS``,
plus a head class and YAML elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover — type hints only
    import mlx.core as mx_t  # noqa: F401
    import mlx.nn as nn

from mlxyolos.engine.results import Boxes, Keypoints, Results
from mlxyolos.utils.ops import letterbox
from mlxyolos.utils.ops import nms as nms_np
from mlxyolos.utils.ops import scale_coords as scale_coords_np
from mlxyolos.utils.ops import scale_keypoints as scale_keypoints_np
from mlxyolos.utils.ops import xywh_to_xyxy as xywh_to_xyxy_np

__all__ = ["Predictor", "POSTPROCESSORS"]


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _is_mlx(x: Any) -> bool:
    """Return True iff ``x`` is an ``mlx.core.array`` instance.

    Imported lazily so this module stays usable on machines without MLX.
    """
    cls = type(x)
    return cls.__module__.startswith("mlx") and cls.__name__ == "array"


def _decode_anchors_mlx(pred, nc: int, nk: int):
    """Run heavy per-anchor decode on device, eval once, return numpy arrays.

    Inputs come straight from the model as ``(A, 4 + nc + nk)``. We do
    ``max`` / ``argmax`` over the class scores and the ``xywh → xyxy`` decode
    on the GPU; the single ``mx.eval`` brings everything across the boundary
    together so the host doesn't pay for individual round-trips.
    """
    import mlx.core as mx

    from mlxyolos.utils.ops_mlx import xywh_to_xyxy as xywh_to_xyxy_mlx

    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    score = mx.max(cls_scores, axis=-1)
    cls_idx = mx.argmax(cls_scores, axis=-1)
    box_xyxy = xywh_to_xyxy_mlx(box_xywh)

    if nk > 0:
        kpts_flat = pred[:, 4 + nc : 4 + nc + nk]
        mx.eval(box_xyxy, score, cls_idx, kpts_flat)
        return (
            np.asarray(box_xyxy),
            np.asarray(score),
            np.asarray(cls_idx).astype(np.float32),
            np.asarray(kpts_flat),
        )
    mx.eval(box_xyxy, score, cls_idx)
    return (
        np.asarray(box_xyxy),
        np.asarray(score),
        np.asarray(cls_idx).astype(np.float32),
        None,
    )


def _decode_anchors_np(pred: np.ndarray, nc: int, nk: int):
    """Pure-numpy decode (used in tests / MLX-less environments)."""
    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    score = cls_scores.max(axis=-1)
    cls_idx = cls_scores.argmax(axis=-1).astype(np.float32)
    box_xyxy = xywh_to_xyxy_np(box_xywh)
    kpts_flat = pred[:, 4 + nc : 4 + nc + nk] if nk > 0 else None
    return box_xyxy, score, cls_idx, kpts_flat


def _nms_dispatch(box_xyxy: np.ndarray, score: np.ndarray, iou_thr: float, *, on_device: bool) -> np.ndarray:
    """NMS that puts the IoU matrix on device when MLX is available."""
    if not on_device:
        return nms_np(box_xyxy, score, iou_thr=iou_thr)
    import mlx.core as mx

    from mlxyolos.utils.ops_mlx import nms as nms_mlx

    return nms_mlx(mx.array(box_xyxy), mx.array(score), iou_thr=iou_thr)


# ---------------------------------------------------------------------------
# Per-task post-processors (dual backend)
# ---------------------------------------------------------------------------


def _postprocess_detect(
    pred,
    nc: int,
    *,
    conf: float,
    iou: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> tuple[Boxes, None]:
    on_device = _is_mlx(pred)
    if on_device:
        box_xyxy, score, cls, _ = _decode_anchors_mlx(pred, nc, nk=0)
    else:
        if pred.size == 0:
            return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None
        box_xyxy, score, cls, _ = _decode_anchors_np(pred, nc, nk=0)

    keep_mask = score > conf
    if not keep_mask.any():
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None
    box_xyxy, score, cls = box_xyxy[keep_mask], score[keep_mask], cls[keep_mask]

    keep = _nms_dispatch(box_xyxy, score, iou_thr=iou, on_device=on_device)
    if len(keep) == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None
    box_xyxy, score, cls = box_xyxy[keep], score[keep], cls[keep]

    box_xyxy = scale_coords_np(box_xyxy, ratio, pad, orig_shape)
    out = np.column_stack([box_xyxy, score, cls])
    return Boxes(out, orig_shape), None


def _postprocess_pose(
    pred,
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
    on_device = _is_mlx(pred)
    if on_device:
        box_xyxy, score, cls, kpts_flat = _decode_anchors_mlx(pred, nc, nk=nk)
    else:
        if pred.size == 0:
            return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)
        box_xyxy, score, cls, kpts_flat = _decode_anchors_np(pred, nc, nk=nk)

    keep_mask = score > conf
    if not keep_mask.any():
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)
    box_xyxy = box_xyxy[keep_mask]
    score = score[keep_mask]
    cls = cls[keep_mask]
    kpts_flat = kpts_flat[keep_mask]

    keep = _nms_dispatch(box_xyxy, score, iou_thr=iou, on_device=on_device)
    if len(keep) == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)
    box_xyxy = box_xyxy[keep]
    score = score[keep]
    cls = cls[keep]
    kpts_flat = kpts_flat[keep]

    box_xyxy = scale_coords_np(box_xyxy, ratio, pad, orig_shape)
    kpts = kpts_flat.reshape(-1, kpt_shape[0], kpt_shape[1])
    kpts = scale_keypoints_np(kpts, ratio, pad)

    out = np.column_stack([box_xyxy, score, cls])
    return Boxes(out, orig_shape), Keypoints(kpts, orig_shape)


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
            raise ValueError(f"Unsupported task {task!r}; expected one of {sorted(POSTPROCESSORS)}")
        self.model = model
        self.task = task
        self.nc = nc
        self.names = names or {0: "person"}
        self.kpt_shape = kpt_shape

    def __call__(
        self,
        source: str | Path | np.ndarray | Image.Image | list,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[Results]:
        sources = source if isinstance(source, list) else [source]
        results: list[Results] = []
        for src in sources:
            results.append(self._run_one(src, imgsz=imgsz, conf=conf, iou=iou))
        return results

    # ----- internals ---------------------------------------------------------

    def _load_image(self, src: str | Path | np.ndarray | Image.Image) -> tuple[np.ndarray, str]:
        if isinstance(src, (str, Path)):
            img = np.asarray(Image.open(src).convert("RGB"))
            return img, str(src)
        if isinstance(src, Image.Image):
            return np.asarray(src.convert("RGB")), ""
        if isinstance(src, np.ndarray):
            if src.ndim != 3 or src.shape[-1] not in (3, 4):
                raise ValueError(f"Expected (H, W, 3|4) ndarray, got shape {src.shape}")
            return src[..., :3], ""
        raise TypeError(f"Unsupported source type: {type(src)!r}")

    def _run_one(
        self,
        src: str | Path | np.ndarray | Image.Image,
        *,
        imgsz: int,
        conf: float,
        iou: float,
    ) -> Results:
        # MLX is imported lazily so the post-processors stay usable on
        # machines without MLX (the numpy branch in the dispatchers above).
        import mlx.core as mx

        orig, path = self._load_image(src)
        lb_img, ratio, pad = letterbox(orig, imgsz)

        # Pre-process on device: upload uint8, normalize on the GPU.
        # We deliberately do NOT eval here — the predictor's post-processor
        # forces a single eval at the boundary together with score/box decode.
        x_uint8 = lb_img[None, ...]  # (1, H, W, 3) uint8
        x = mx.array(x_uint8).astype(mx.float32) / 255.0

        pred = self.model(x)[0]  # (A, 4 + nc + nk) — still lazy

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
