# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MLX-native variants of the post-processing helpers.

These mirror ``utils/ops.py`` (numpy) so the predictor can keep work on the
GPU as long as possible. The greedy NMS keep-loop is inherently serial and
runs on the host, but the heavy IoU matrix is computed on device first.

We intentionally keep this module self-contained; ``utils/ops.py`` remains
the canonical numpy implementation for tests / Linux fallback paths.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

__all__ = [
    "xywh_to_xyxy",
    "scale_coords",
    "scale_keypoints",
    "nms",
]


def xywh_to_xyxy(b: mx.array) -> mx.array:
    """``[cx, cy, w, h]`` → ``[x1, y1, x2, y2]`` for a (..., 4) MLX array."""
    cx = b[..., 0]
    cy = b[..., 1]
    w = b[..., 2]
    h = b[..., 3]
    return mx.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def scale_coords(
    xyxy: mx.array,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> mx.array:
    """Undo letterbox in MLX: input coords → original-image coords, clipped."""
    px, py = float(pad[0]), float(pad[1])
    h, w = orig_shape
    shift = mx.array([px, py, px, py], dtype=mx.float32)
    out = (xyxy - shift) / float(ratio)
    lo = mx.array([0.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    # Clip bounds match Ultralytics' clip_boxes: [0, w] / [0, h], NOT [0, w-1] / [0, h-1].
    hi = mx.array([w, h, w, h], dtype=mx.float32)
    return mx.minimum(mx.maximum(out, lo), hi)


def scale_keypoints(
    kpts: mx.array,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int] | None = None,
) -> mx.array:
    """Undo letterbox for ``(N, K, dim)`` MLX keypoints (xy only).

    When ``orig_shape`` is given, also clip xy to image bounds — matching
    Ultralytics' ``clip_coords`` ([0, w] / [0, h]). Visibility/confidence
    columns (index 2+) are left alone.
    """
    px, py = float(pad[0]), float(pad[1])
    if kpts.shape[-1] >= 3:
        xy = (kpts[..., :2] - mx.array([px, py], dtype=mx.float32)) / float(ratio)
        if orig_shape is not None:
            h, w = orig_shape
            lo = mx.array([0.0, 0.0], dtype=mx.float32)
            hi = mx.array([w, h], dtype=mx.float32)
            xy = mx.minimum(mx.maximum(xy, lo), hi)
        rest = kpts[..., 2:]
        return mx.concatenate([xy, rest], axis=-1)
    out = (kpts - mx.array([px, py], dtype=mx.float32)) / float(ratio)
    if orig_shape is not None:
        h, w = orig_shape
        lo = mx.array([0.0, 0.0], dtype=mx.float32)
        hi = mx.array([w, h], dtype=mx.float32)
        out = mx.minimum(mx.maximum(out, lo), hi)
    return out


def nms(
    boxes_xyxy: mx.array,
    scores: mx.array,
    iou_thr: float = 0.45,
) -> np.ndarray:
    """Class-agnostic NMS with IoU matrix computed on device.

    Returns a numpy ``int64`` array of kept indices into the *original*
    (unsorted) input. The greedy pick is unavoidably serial — we run that
    on the host once the (small) IoU matrix has been evaluated.
    """
    n = boxes_xyxy.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # Sort by descending score on device; permutation feeds the IoU layout.
    order = mx.argsort(-scores)
    boxes_sorted = boxes_xyxy[order]

    x1 = boxes_sorted[:, 0]
    y1 = boxes_sorted[:, 1]
    x2 = boxes_sorted[:, 2]
    y2 = boxes_sorted[:, 3]
    areas = mx.maximum(x2 - x1, 0.0) * mx.maximum(y2 - y1, 0.0)

    # Pairwise IoU — fine on device for typical post-conf-filter counts (≤few hundred).
    xx1 = mx.maximum(x1[:, None], x1[None, :])
    yy1 = mx.maximum(y1[:, None], y1[None, :])
    xx2 = mx.minimum(x2[:, None], x2[None, :])
    yy2 = mx.minimum(y2[:, None], y2[None, :])
    inter = mx.maximum(xx2 - xx1, 0.0) * mx.maximum(yy2 - yy1, 0.0)
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / mx.maximum(union, 1e-7)

    iou_np = np.asarray(iou)
    order_np = np.asarray(order)

    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(int(order_np[i]))
        if i + 1 < n:
            suppressed[i + 1 :] |= iou_np[i, i + 1 :] > iou_thr
    return np.asarray(keep, dtype=np.int64)
