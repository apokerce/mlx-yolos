# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""NumPy implementations of letterbox / NMS / coordinate scaling.

Kept off the MLX path on purpose — these run cheaply on CPU and don't
benefit from Metal acceleration; doing them in NumPy avoids forcing
`mx.eval` between operations that don't share state.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = [
    "letterbox",
    "xywh_to_xyxy",
    "scale_coords",
    "scale_keypoints",
    "nms",
]


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to ``new_shape``.

    Returns
    -------
    out: (H, W, 3) uint8 letterboxed image
    ratio: scale factor applied to the original image
    pad: (left, top) padding in pixels
    """
    if isinstance(new_shape, int):
        new_h = new_w = new_shape
    else:
        new_h, new_w = new_shape

    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_w, pad_h = new_w - nw, new_h - nh
    left = pad_w // 2
    top = pad_h // 2

    if (h, w) != (nh, nw):
        resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    else:
        resized = img
    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    """``[cx, cy, w, h]`` → ``[x1, y1, x2, y2]``."""
    out = np.empty_like(b)
    out[..., 0] = b[..., 0] - b[..., 2] / 2
    out[..., 1] = b[..., 1] - b[..., 3] / 2
    out[..., 2] = b[..., 0] + b[..., 2] / 2
    out[..., 3] = b[..., 1] + b[..., 3] / 2
    return out


def scale_coords(
    xyxy: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """Undo letterbox: input coords → original-image coords, clipped."""
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / ratio
    h, w = orig_shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
    return out


def scale_keypoints(
    kpts: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """Undo letterbox for keypoint (x, y, ...) tuples."""
    out = kpts.copy()
    out[..., 0] = (out[..., 0] - pad[0]) / ratio
    out[..., 1] = (out[..., 1] - pad[1]) / ratio
    return out


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
    """Class-agnostic NMS, returns indices ordered by descending score."""
    if boxes_xyxy.size == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou < iou_thr]
    return np.asarray(keep, dtype=np.int64)
