# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU-side image preprocessing — letterbox only.

Everything else (xywh→xyxy, scale_coords, scale_keypoints, NMS) lives on
the MLX side in :mod:`mlxyolos.utils.ops_mlx`. We resize with ``cv2`` to
match Ultralytics' `val` / `predict` pipeline pixel-for-pixel; that's
load-bearing for box-AP parity (see ``docs/VALIDATION.md``).
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["letterbox"]


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
    *,
    rect: bool = False,
    stride: int = 32,
    scaleup: bool = True,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad onto a canvas (Ultralytics-compatible).

    Modes:

    - **Square / explicit** (``rect=False``, default; or ``new_shape`` is a tuple):
      pad to a fixed ``(new_shape, new_shape)`` square — or to the explicit
      ``(h, w)`` if you pass a tuple.
    - **Rectangular** (``rect=True`` with an int ``new_shape``):
      compute the resize ratio so the longer side fits ``new_shape``, then
      pad each dim only up to the next multiple of ``stride``. Matches
      Ultralytics' ``yolo predict`` letterbox.

    ``scaleup`` controls whether images smaller than ``new_shape`` get
    upscaled. Default ``True`` matches ``yolo predict``; ``False`` matches
    ``yolo val`` (small images stay at native resolution inside the padded
    canvas).

    Returns
    -------
    out: (H, W, 3) uint8 letterboxed image
    ratio: scale factor applied to the original image
    pad: (left, top) padding in pixels
    """
    h, w = img.shape[:2]

    if isinstance(new_shape, int):
        target = new_shape
        r = min(target / h, target / w)
        if not scaleup:
            r = min(r, 1.0)
        nh, nw = int(round(h * r)), int(round(w * r))
        if rect:
            out_h = ((nh + stride - 1) // stride) * stride
            out_w = ((nw + stride - 1) // stride) * stride
        else:
            out_h = out_w = target
    else:
        out_h, out_w = new_shape
        r = min(out_h / h, out_w / w)
        if not scaleup:
            r = min(r, 1.0)
        nh, nw = int(round(h * r)), int(round(w * r))

    pad_w_total = out_w - nw
    pad_h_total = out_h - nh
    left = pad_w_total // 2
    top = pad_h_total // 2

    if (h, w) != (nh, nw):
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img
    out = np.full((out_h, out_w, 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)
