# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pose / detection drawing helpers (Pillow only, no extra deps)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["COCO_SKELETON", "draw_pose"]


# COCO 17-keypoint skeleton (0-indexed pairs).
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def draw_pose(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    kpts: np.ndarray,
    *,
    kpt_thr: float = 0.5,
    skeleton: tuple[tuple[int, int], ...] = COCO_SKELETON,
) -> Image.Image:
    """Draw boxes and 17-keypoint skeleton onto ``image`` (RGB ndarray)."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for box, score, kp in zip(boxes_xyxy, scores, kpts, strict=True):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"person {score:.2f}", fill=(0, 255, 0))
        for a, b in skeleton:
            xa, ya, va = kp[a]
            xb, yb, vb = kp[b]
            if va > kpt_thr and vb > kpt_thr:
                draw.line([(xa, ya), (xb, yb)], fill=(255, 128, 0), width=2)
        for x, y, v in kp:
            if v > kpt_thr:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 200, 255))
    return img
