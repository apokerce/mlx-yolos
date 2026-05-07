# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight result containers, deliberately mirroring Ultralytics' API.

Each task that's added later can attach the relevant attribute on
``Results`` (boxes / masks / keypoints / obb) — the container itself stays
generic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Boxes", "Keypoints", "Results"]


class Boxes:
    """N detections in absolute pixel space, ``[x1, y1, x2, y2, conf, cls]``."""

    def __init__(self, data: np.ndarray, orig_shape: tuple[int, int]) -> None:
        self.data = np.asarray(data, dtype=np.float32) if data is not None else np.empty((0, 6), dtype=np.float32)
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4] if len(self.data) else np.empty((0, 4), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        if not len(self.data):
            return np.empty((0, 4), dtype=np.float32)
        b = self.data[:, :4]
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        return np.stack([cx, cy, w, h], axis=-1)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4] if len(self.data) else np.empty((0,), dtype=np.float32)

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5] if len(self.data) else np.empty((0,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Boxes(n={len(self)}, orig_shape={self.orig_shape})"


class Keypoints:
    """N persons × K keypoints × 3 ``(x, y, vis)``."""

    def __init__(self, data: np.ndarray | None, orig_shape: tuple[int, int]) -> None:
        self.data = np.asarray(data, dtype=np.float32) if data is not None else None
        self.orig_shape = orig_shape

    @property
    def xy(self) -> np.ndarray | None:
        return self.data[..., :2] if self.data is not None else None

    @property
    def conf(self) -> np.ndarray | None:
        return self.data[..., 2] if self.data is not None and self.data.shape[-1] >= 3 else None

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data)

    def __repr__(self) -> str:
        shape = None if self.data is None else self.data.shape
        return f"Keypoints(n={len(self)}, shape={shape})"


class Results:
    """Per-image inference results."""

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str = "",
        names: dict[int, str] | None = None,
        boxes: Boxes | None = None,
        keypoints: Keypoints | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.path = path
        self.names = names or {}
        self.boxes = boxes
        self.keypoints = keypoints
        self.speed = speed or {}

    @property
    def orig_shape(self) -> tuple[int, int]:
        return self.orig_img.shape[:2]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.boxes is not None:
            parts.append(f"boxes={len(self.boxes)}")
        if self.keypoints is not None:
            parts.append(f"keypoints={len(self.keypoints)}")
        return f"Results({', '.join(parts) or 'empty'}, orig_shape={self.orig_shape}, path={self.path!r})"
