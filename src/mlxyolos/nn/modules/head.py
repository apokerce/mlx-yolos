# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detection heads.

Each task gets its own head class; everything else (backbone / neck) is
shared. Adding a new task = a new ``Head`` subclass + a new YAML.

Implemented:
    * ``DetectV8``  — YOLOv8 box-only head (DFL, 3 anchor scales, NMS).
    * ``PoseV8``    — YOLOv8 pose head (DetectV8 + per-keypoint cv4 branch).

Reference: ``ultralytics/ultralytics/nn/modules/head.py`` (legacy / non-end2end
variant — yolov8 release weights use the legacy 3×3+3×3+1×1 cls branch and
DFL with reg_max=16, no end-to-end NMS-free branch).

Output convention (inference path):
    * ``DetectV8``: ``(B, A, 4 + nc)`` where the first 4 dims are
      ``cxcywh`` in input-pixel coords and the rest are sigmoid class scores.
    * ``PoseV8``  : ``(B, A, 4 + nc + nk)`` with keypoints decoded to
      input-pixel ``(x, y, vis-sigmoid)`` triples and flattened along the
      keypoint axis.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv

__all__ = ["DetectV8", "PoseV8"]


# ---------------------------------------------------------------------------
# Helpers (pure functions kept here so the head is self-contained)
# ---------------------------------------------------------------------------


def make_anchors(
    feat_hw: list[tuple[int, int]],
    strides: list[int] | tuple[int, ...],
    offset: float = 0.5,
) -> tuple[mx.array, mx.array]:
    """Generate anchor centers and per-anchor stride for each feature level."""
    points: list[mx.array] = []
    stride_t: list[mx.array] = []
    for (h, w), s in zip(feat_hw, strides, strict=True):
        sx = mx.arange(w, dtype=mx.float32) + offset
        sy = mx.arange(h, dtype=mx.float32) + offset
        gy, gx = mx.meshgrid(sy, sx, indexing="ij")
        points.append(mx.stack([gx, gy], axis=-1).reshape(-1, 2))
        stride_t.append(mx.full((h * w, 1), float(s), dtype=mx.float32))
    return mx.concatenate(points, axis=0), mx.concatenate(stride_t, axis=0)


def dist2bbox(distance: mx.array, anchor_points: mx.array, xywh: bool = True) -> mx.array:
    """Convert predicted ``[l, t, r, b]`` distances → bounding boxes."""
    lt, rb = mx.split(distance, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        cxy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return mx.concatenate([cxy, wh], axis=-1)
    return mx.concatenate([x1y1, x2y2], axis=-1)


# ---------------------------------------------------------------------------
# DetectV8 — base class for v8-family heads
# ---------------------------------------------------------------------------


class DetectV8(nn.Module):
    """YOLOv8 detection head (legacy-cls, DFL, NMS path).

    The head outputs predictions in input-pixel ``cxcywh`` space + sigmoid
    class scores. NMS is intentionally left to post-processing so the same
    forward pass can serve both ``mx.compile``'d inference and offline
    debugging.
    """

    strides: tuple[int, ...] = (8, 16, 32)

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4

        c2 = max(16, ch[0] // 4, reg_max * 4)  # box-branch hidden channels
        c3 = max(ch[0], min(nc, 100))  # cls-branch hidden channels (legacy)

        # cv2 = box branch, cv3 = cls branch. Both are
        # ``[Conv 3×3, Conv 3×3, Conv2d 1×1]`` per feature level — exactly
        # matching the released yolov8 checkpoints (legacy=True). Storing
        # them as a list of lists keeps weight names like
        # ``cv2.0.0.conv.weight`` so the converter is a no-op rename.
        self.cv2: list[list[nn.Module]] = [
            [Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)] for x in ch
        ]
        self.cv3: list[list[nn.Module]] = [
            [Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)] for x in ch
        ]

    @staticmethod
    def _seq(stack: list[nn.Module], x: mx.array) -> mx.array:
        for layer in stack:
            x = layer(x)
        return x

    def _forward_branches(self, feats: list[mx.array]) -> tuple[mx.array, mx.array, list[tuple[int, int]]]:
        """Run cv2 / cv3 over each feature level, return concatenated box-dist
        and class logits plus the per-level (H, W)."""
        bs = feats[0].shape[0]
        feat_hw = [(f.shape[1], f.shape[2]) for f in feats]
        boxes_lvl: list[mx.array] = []
        scores_lvl: list[mx.array] = []
        for i, f in enumerate(feats):
            box = self._seq(self.cv2[i], f)
            cls = self._seq(self.cv3[i], f)
            n = box.shape[1] * box.shape[2]
            boxes_lvl.append(box.reshape(bs, n, 4 * self.reg_max))
            scores_lvl.append(cls.reshape(bs, n, self.nc))
        return mx.concatenate(boxes_lvl, axis=1), mx.concatenate(scores_lvl, axis=1), feat_hw

    def _decode_boxes(
        self, boxes: mx.array, anchors: mx.array, strides: mx.array
    ) -> mx.array:
        """Apply DFL (softmax over reg_max bins, integrate with [0..reg_max-1])
        then convert distances → cxcywh in input-pixel coords."""
        bs = boxes.shape[0]
        bx = boxes.reshape(bs, -1, 4, self.reg_max)
        bx = mx.softmax(bx, axis=-1)
        proj = mx.arange(self.reg_max, dtype=bx.dtype)
        dist = (bx * proj).sum(axis=-1)
        return dist2bbox(dist, anchors[None, ...], xywh=True) * strides

    def __call__(self, feats: list[mx.array]) -> mx.array:
        boxes, scores, feat_hw = self._forward_branches(feats)
        anchors, strides = make_anchors(feat_hw, self.strides)
        dbox = self._decode_boxes(boxes, anchors, strides)
        return mx.concatenate([dbox, mx.sigmoid(scores)], axis=-1)


# ---------------------------------------------------------------------------
# PoseV8 — DetectV8 + per-keypoint cv4 branch
# ---------------------------------------------------------------------------


class PoseV8(DetectV8):
    """YOLOv8 pose head: DetectV8 with an extra keypoint regression branch."""

    def __init__(
        self,
        nc: int = 1,
        kpt_shape: tuple[int, int] = (17, 3),
        reg_max: int = 16,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, ch=ch)
        self.kpt_shape = tuple(kpt_shape)
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]

        c4 = max(ch[0] // 4, self.nk)
        self.cv4: list[list[nn.Module]] = [
            [Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)] for x in ch
        ]

    def __call__(self, feats: list[mx.array]) -> mx.array:
        bs = feats[0].shape[0]
        boxes, scores, feat_hw = self._forward_branches(feats)

        kpts_lvl: list[mx.array] = []
        for i, f in enumerate(feats):
            kpt = self._seq(self.cv4[i], f)
            n = kpt.shape[1] * kpt.shape[2]
            kpts_lvl.append(kpt.reshape(bs, n, self.nk))
        kpts = mx.concatenate(kpts_lvl, axis=1)

        anchors, strides = make_anchors(feat_hw, self.strides)
        dbox = self._decode_boxes(boxes, anchors, strides)
        scores = mx.sigmoid(scores)

        # Decode keypoints to (x, y, vis-sigmoid) in input-pixel coords.
        ndim = self.kpt_shape[1]
        k = kpts.reshape(bs, -1, self.kpt_shape[0], ndim)
        k_xy = (k[..., :2] * 2.0 + (anchors[None, :, None, :] - 0.5)) * strides[None, :, None, :]
        if ndim == 3:
            k_v = mx.sigmoid(k[..., 2:3])
            k = mx.concatenate([k_xy, k_v], axis=-1)
        else:
            k = k_xy
        kpts_decoded = k.reshape(bs, -1, self.nk)
        return mx.concatenate([dbox, scores, kpts_decoded], axis=-1)
