# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generic predictor: image → preprocess → MLX forward → per-task post-processing.

Per-task post-processing lives in dispatch tables (``POSTPROCESSORS``).
Adding a new task = registering one function here, plus the head class
and YAML elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover — type hints only
    import mlx.nn as nn

from mlxyolos.engine.results import Boxes, Keypoints, Results
from mlxyolos.utils.ops import letterbox, nms, scale_coords, scale_keypoints, xywh_to_xyxy

__all__ = ["Predictor", "POSTPROCESSORS"]


# ---------------------------------------------------------------------------
# Per-task post-processors
# ---------------------------------------------------------------------------


def _postprocess_detect(
    pred: np.ndarray,
    nc: int,
    *,
    conf: float,
    iou: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> tuple[Boxes, None]:
    """Decode raw ``(A, 4 + nc)`` predictions to a ``Boxes`` container."""
    if pred.size == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None

    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    score = cls_scores.max(axis=-1)
    cls = cls_scores.argmax(axis=-1).astype(np.float32)

    keep_mask = score > conf
    box_xywh, score, cls = box_xywh[keep_mask], score[keep_mask], cls[keep_mask]
    if not len(score):
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), None

    box_xyxy = xywh_to_xyxy(box_xywh)
    keep = nms(box_xyxy, score, iou_thr=iou)
    box_xyxy, score, cls = box_xyxy[keep], score[keep], cls[keep]

    box_xyxy = scale_coords(box_xyxy, ratio, pad, orig_shape)
    out = np.column_stack([box_xyxy, score, cls])
    return Boxes(out, orig_shape), None


def _postprocess_pose(
    pred: np.ndarray,
    nc: int,
    *,
    conf: float,
    iou: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    kpt_shape: tuple[int, int],
) -> tuple[Boxes, Keypoints]:
    """Decode raw ``(A, 4 + nc + nk)`` predictions to ``(Boxes, Keypoints)``."""
    if pred.size == 0:
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)

    nk = kpt_shape[0] * kpt_shape[1]
    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    kpts = pred[:, 4 + nc : 4 + nc + nk].reshape(-1, kpt_shape[0], kpt_shape[1])

    score = cls_scores.max(axis=-1)
    cls = cls_scores.argmax(axis=-1).astype(np.float32)

    keep_mask = score > conf
    box_xywh, score, cls, kpts = box_xywh[keep_mask], score[keep_mask], cls[keep_mask], kpts[keep_mask]
    if not len(score):
        return Boxes(np.empty((0, 6), dtype=np.float32), orig_shape), Keypoints(None, orig_shape)

    box_xyxy = xywh_to_xyxy(box_xywh)
    keep = nms(box_xyxy, score, iou_thr=iou)
    box_xyxy, score, cls, kpts = box_xyxy[keep], score[keep], cls[keep], kpts[keep]

    box_xyxy = scale_coords(box_xyxy, ratio, pad, orig_shape)
    kpts = scale_keypoints(kpts, ratio, pad)

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
        model: "nn.Module",
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
        # Import MLX lazily so the pure-numpy post-processors stay usable
        # (e.g. for tests, or when post-processing tensors fed in by a host
        # that already ran the model elsewhere).
        import mlx.core as mx

        orig, path = self._load_image(src)
        lb_img, ratio, pad = letterbox(orig, imgsz)

        x = (lb_img.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)
        y = self.model(mx.array(x))
        mx.eval(y)
        pred = np.asarray(y[0])

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
