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

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def summary(self, kpt_thr: float = 0.5) -> list[dict[str, Any]]:
        """Per-detection structured summary, one dict per box.

        Always includes ``index, name, class, conf, box_xyxy, box_xywh``.
        For pose results it adds ``keypoints_visible`` (e.g. ``"15/17"``)
        and the keypoint array under ``keypoints``.
        """
        out: list[dict[str, Any]] = []
        if self.boxes is None or len(self.boxes) == 0:
            return out
        xyxy = self.boxes.xyxy
        xywh = self.boxes.xywh
        conf = self.boxes.conf
        cls = self.boxes.cls
        kpts = self.keypoints.data if self.keypoints is not None else None
        for i in range(len(self.boxes)):
            ci = int(cls[i])
            entry: dict[str, Any] = {
                "index": i,
                "class": ci,
                "name": self.names.get(ci, str(ci)),
                "conf": float(conf[i]),
                "box_xyxy": [float(v) for v in xyxy[i].tolist()],
                "box_xywh": [float(v) for v in xywh[i].tolist()],
            }
            if kpts is not None and i < len(kpts):
                kp = kpts[i]
                if kp.shape[-1] >= 3:
                    visible = int((kp[:, 2] > kpt_thr).sum())
                    entry["keypoints_visible"] = f"{visible}/{kp.shape[0]}"
                entry["keypoints"] = kp.tolist()
            out.append(entry)
        return out

    def verbose(self, *, kpt_thr: float = 0.5, max_rows: int | None = None) -> str:
        """Multi-line summary suitable for stdout / logs."""
        h, w = self.orig_shape
        n = 0 if self.boxes is None else len(self.boxes)
        head = f"{self.path or '<array>'}  {w}x{h}  {n} detection{'s' if n != 1 else ''}"
        rows: list[str] = [head]
        items = self.summary(kpt_thr=kpt_thr)
        if max_rows is not None:
            items = items[:max_rows]
        for d in items:
            x1, y1, x2, y2 = d["box_xyxy"]
            line = (
                f"  [{d['index']}] {d['name']:<10} {d['conf']:.3f}  "
                f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )
            if "keypoints_visible" in d:
                line += f"  kpts={d['keypoints_visible']}"
            rows.append(line)
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.verbose()

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.boxes is not None:
            parts.append(f"boxes={len(self.boxes)}")
        if self.keypoints is not None:
            parts.append(f"keypoints={len(self.keypoints)}")
        return f"Results({', '.join(parts) or 'empty'}, orig_shape={self.orig_shape}, path={self.path!r})"

    # ------------------------------------------------------------------
    # Plot — annotate the original image with whatever data is available
    # ------------------------------------------------------------------

    def plot(self, *, kpt_thr: float = 0.5):
        """Return an annotated copy of ``orig_img`` as an ``np.ndarray`` (RGB).

        Always draws boxes if any are present. Adds the COCO skeleton when
        keypoints are available (pose task). Class label + confidence go in
        the corner of every box, on a filled background for legibility.

        Persist with :meth:`save`, or directly via ``cv2``:

            cv2.imwrite("out.jpg", cv2.cvtColor(r.plot(), cv2.COLOR_RGB2BGR))
        """
        from mlxyolos.utils.plotting import draw_boxes, draw_pose

        if (
            self.keypoints is not None
            and self.keypoints.data is not None
            and self.boxes is not None
            and len(self.boxes) > 0
        ):
            return draw_pose(
                self.orig_img,
                self.boxes.xyxy,
                self.boxes.conf,
                self.boxes.cls,
                self.keypoints.data,
                names=self.names,
                kpt_thr=kpt_thr,
            )
        return draw_boxes(
            self.orig_img,
            self.boxes.xyxy if self.boxes is not None else None,
            self.boxes.conf if self.boxes is not None else None,
            self.boxes.cls if self.boxes is not None else None,
            names=self.names,
        )

    def save(self, path: str, *, kpt_thr: float = 0.5) -> str:
        """Annotate via :meth:`plot` and write to ``path`` using cv2.

        Returns the resolved path written to. cv2.imwrite picks the codec
        from the extension (``.jpg``, ``.png``, …).
        """
        import cv2

        img_rgb = self.plot(kpt_thr=kpt_thr)
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return path
