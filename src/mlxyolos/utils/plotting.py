# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Drawing helpers, cv2-only.

Both ``draw_boxes`` and ``draw_pose`` take an RGB ``np.ndarray`` (the same
layout the predictor handed back), draw on a writable copy, and return a
new ``np.ndarray`` (RGB). Persist to disk with cv2:

    >>> import cv2
    >>> cv2.imwrite("out.jpg", cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))

Or use :meth:`mlxyolos.engine.Results.save` which handles that conversion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import cv2
import numpy as np

__all__ = ["COCO_SKELETON", "draw_boxes", "draw_pose"]


# COCO 17-keypoint skeleton (0-indexed pairs).
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)

# Tableau-ish palette — RGB, cycles per class.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56), (255, 159, 56), (255, 217, 56), (139, 217, 56),
    (56, 217, 122), (56, 217, 217), (56, 122, 217), (122, 56, 217),
    (217, 56, 217), (217, 56, 139), (255, 128, 128), (128, 255, 128),
)


def _color_for_class(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(cls_id) % len(_PALETTE)]


def _draw_label(
    img: np.ndarray,
    xy: tuple[float, float],
    text: str,
    *,
    color_rgb: tuple[int, int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.5,
    thickness: int = 1,
    pad: int = 3,
) -> None:
    """Draw ``text`` on a filled badge anchored at ``xy`` (top-left of the box).

    Modifies ``img`` in place. ``color_rgb`` is the badge fill; text color is
    auto-picked (white on dark fills, black on light) for legibility.
    """
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(round(xy[0])), int(round(xy[1]))
    # Place the badge so its bottom-left lines up with the box's top-left.
    bx1, by1 = x, max(0, y - th - 2 * pad - baseline)
    bx2, by2 = x + tw + 2 * pad, by1 + th + 2 * pad + baseline
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color_rgb, thickness=-1)
    luma = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
    fg = (0, 0, 0) if luma > 160 else (255, 255, 255)
    cv2.putText(
        img,
        text,
        (bx1 + pad, by2 - pad - baseline),
        font,
        scale,
        fg,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray | None,
    scores: np.ndarray | None,
    cls: np.ndarray | None,
    *,
    names: Mapping[int, str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes + class badges. Returns a new RGB ``np.ndarray``."""
    out = np.ascontiguousarray(image).copy()
    if boxes_xyxy is None or scores is None or cls is None or len(boxes_xyxy) == 0:
        return out
    names = names or {}
    for box, score, c in zip(boxes_xyxy, scores, cls, strict=True):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.tolist())
        color = _color_for_class(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        label = f"{names.get(int(c), str(int(c)))} {float(score):.2f}"
        _draw_label(out, (x1, y1), label, color_rgb=color)
    return out


def draw_pose(
    image: np.ndarray,
    boxes_xyxy: np.ndarray | None,
    scores: np.ndarray | None,
    cls: np.ndarray | None,
    kpts: np.ndarray | None,
    *,
    names: Mapping[int, str] | None = None,
    kpt_thr: float = 0.5,
    skeleton: Iterable[tuple[int, int]] = COCO_SKELETON,
) -> np.ndarray:
    """Boxes + class badges + COCO skeleton. Returns a new RGB ``np.ndarray``."""
    out = np.ascontiguousarray(image).copy()
    if boxes_xyxy is None or len(boxes_xyxy) == 0 or kpts is None:
        return out
    names = names or {}
    skel = tuple(skeleton)

    for box, score, c, kp in zip(boxes_xyxy, scores, cls, kpts, strict=True):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.tolist())
        color = _color_for_class(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        label = f"{names.get(int(c), str(int(c)))} {float(score):.2f}"
        _draw_label(out, (x1, y1), label, color_rgb=color)

        # Skeleton edges first, keypoint dots on top.
        kp_arr = np.asarray(kp)
        has_vis = kp_arr.shape[-1] >= 3
        for a, b in skel:
            if a >= len(kp_arr) or b >= len(kp_arr):
                continue
            va = float(kp_arr[a, 2]) if has_vis else 1.0
            vb = float(kp_arr[b, 2]) if has_vis else 1.0
            if va > kpt_thr and vb > kpt_thr:
                p1 = (int(round(float(kp_arr[a, 0]))), int(round(float(kp_arr[a, 1]))))
                p2 = (int(round(float(kp_arr[b, 0]))), int(round(float(kp_arr[b, 1]))))
                cv2.line(out, p1, p2, (255, 128, 0), thickness=2, lineType=cv2.LINE_AA)
        for row in kp_arr:
            x, y = float(row[0]), float(row[1])
            v = float(row[2]) if has_vis else 1.0
            if v > kpt_thr:
                cv2.circle(
                    out,
                    (int(round(x)), int(round(y))),
                    radius=3,
                    color=(0, 200, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
                # Thin black outline for contrast on light backgrounds.
                cv2.circle(
                    out,
                    (int(round(x)), int(round(y))),
                    radius=3,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
    return out
