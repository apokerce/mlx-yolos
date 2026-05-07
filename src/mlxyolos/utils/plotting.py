# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pose / detection drawing helpers (Pillow only, no extra deps).

Two public entry points:

* ``draw_boxes`` — bounding boxes + class label badges.
* ``draw_pose``  — boxes + COCO skeleton + per-keypoint dots.

Labels are drawn on a solid colored badge so they remain legible even when
the underlying image is busy. We try to load a TTF font for crisp text and
fall back to PIL's bitmap default if no system fonts are findable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["COCO_SKELETON", "draw_boxes", "draw_pose"]


# COCO 17-keypoint skeleton (0-indexed pairs).
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)

# Tableau-ish palette — cycles per class.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56), (255, 159, 56), (255, 217, 56), (139, 217, 56),
    (56, 217, 122), (56, 217, 217), (56, 122, 217), (122, 56, 217),
    (217, 56, 217), (217, 56, 139), (255, 128, 128), (128, 255, 128),
)


def _color_for_class(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(cls_id) % len(_PALETTE)]


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    """Best-effort scalable font; PIL's default if no TTF is reachable."""
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # most Linux distros
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    pad: int = 3,
) -> None:
    """Draw ``text`` on a filled badge anchored at ``xy`` (top-left)."""
    x, y = xy
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # very old PIL — best-effort fallback
        tw, th = font.getsize(text)
    bx1, by1 = x, max(0, y - th - 2 * pad)
    bx2, by2 = x + tw + 2 * pad, by1 + th + 2 * pad
    draw.rectangle([bx1, by1, bx2, by2], fill=color)
    # White or black text depending on badge brightness.
    luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    fg = (0, 0, 0) if luma > 160 else (255, 255, 255)
    draw.text((bx1 + pad, by1 + pad), text, fill=fg, font=font)


# ---------------------------------------------------------------------------
# Public draw helpers
# ---------------------------------------------------------------------------


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray | None,
    scores: np.ndarray | None,
    cls: np.ndarray | None,
    *,
    names: Mapping[int, str] | None = None,
) -> Image.Image:
    """Draw bounding-box annotations only (no skeleton)."""
    img = Image.fromarray(image)
    if boxes_xyxy is None or scores is None or cls is None or len(boxes_xyxy) == 0:
        return img
    draw = ImageDraw.Draw(img)
    font = _load_font(size=16)
    names = names or {}
    for box, score, c in zip(boxes_xyxy, scores, cls, strict=True):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        color = _color_for_class(int(c))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{names.get(int(c), str(int(c)))} {float(score):.2f}"
        _draw_label(draw, (x1, y1), label, color=color, font=font)
    return img


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
) -> Image.Image:
    """Draw boxes + class badges + COCO skeleton on top of ``image``."""
    img = Image.fromarray(image)
    if boxes_xyxy is None or len(boxes_xyxy) == 0 or kpts is None:
        return img
    draw = ImageDraw.Draw(img)
    font = _load_font(size=16)
    names = names or {}
    skel = tuple(skeleton)

    for box, score, c, kp in zip(boxes_xyxy, scores, cls, kpts, strict=True):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        color = _color_for_class(int(c))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{names.get(int(c), str(int(c)))} {float(score):.2f}"
        _draw_label(draw, (x1, y1), label, color=color, font=font)

        # Skeleton lines first, then keypoint dots on top.
        for a, b in skel:
            if a >= len(kp) or b >= len(kp):
                continue
            xa, ya, va = float(kp[a, 0]), float(kp[a, 1]), float(kp[a, 2]) if kp.shape[1] >= 3 else 1.0
            xb, yb, vb = float(kp[b, 0]), float(kp[b, 1]), float(kp[b, 2]) if kp.shape[1] >= 3 else 1.0
            if va > kpt_thr and vb > kpt_thr:
                draw.line([(xa, ya), (xb, yb)], fill=(255, 128, 0), width=2)
        for x, y, *rest in kp.tolist():
            v = rest[0] if rest else 1.0
            if v > kpt_thr:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 200, 255), outline=(0, 0, 0))
    return img
