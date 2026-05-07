# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Parity tests for the MLX-native post-processing helpers.

These run only when MLX is actually importable. On the validation Linux
box (where ``libmlx.so`` isn't present) they're auto-skipped, but on
Apple Silicon they catch any divergence between the numpy and MLX paths.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
from mlxyolos.utils import ops, ops_mlx  # noqa: E402


def test_xywh_to_xyxy_parity():
    rng = np.random.default_rng(0)
    box = rng.uniform(50, 500, size=(64, 4)).astype(np.float32)
    np_out = ops.xywh_to_xyxy(box)
    mlx_out = np.asarray(ops_mlx.xywh_to_xyxy(mx.array(box)))
    np.testing.assert_allclose(np_out, mlx_out, atol=1e-5)


def test_scale_coords_parity():
    rng = np.random.default_rng(1)
    box = rng.uniform(0, 640, size=(32, 4)).astype(np.float32)
    np_out = ops.scale_coords(box.copy(), ratio=0.625, pad=(40, 0), orig_shape=(720, 1280))
    mlx_out = np.asarray(
        ops_mlx.scale_coords(mx.array(box), ratio=0.625, pad=(40, 0), orig_shape=(720, 1280))
    )
    np.testing.assert_allclose(np_out, mlx_out, atol=1e-4)


def test_scale_keypoints_parity():
    rng = np.random.default_rng(2)
    kpts = rng.uniform(0, 640, size=(8, 17, 3)).astype(np.float32)
    np_out = ops.scale_keypoints(kpts.copy(), ratio=0.5, pad=(20, 80))
    mlx_out = np.asarray(ops_mlx.scale_keypoints(mx.array(kpts), ratio=0.5, pad=(20, 80)))
    # Visibility column is untouched on both paths; xy is letterbox-undone.
    np.testing.assert_allclose(np_out, mlx_out, atol=1e-4)


def test_nms_parity_on_synthetic_clusters():
    """Generate 3 clusters of overlapping boxes — both NMS paths must agree."""
    rng = np.random.default_rng(3)
    centers = np.array([[100, 100], [300, 100], [100, 300]], dtype=np.float32)
    boxes = []
    scores = []
    for c in centers:
        for _ in range(8):
            jitter = rng.normal(scale=4.0, size=2)
            cx, cy = c + jitter
            w, h = 50, 50
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(rng.uniform(0.3, 0.95))
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    np_keep = set(ops.nms(boxes_arr, scores_arr, iou_thr=0.45).tolist())
    mlx_keep = set(
        ops_mlx.nms(mx.array(boxes_arr), mx.array(scores_arr), iou_thr=0.45).tolist()
    )
    assert np_keep == mlx_keep, f"numpy={sorted(np_keep)} mlx={sorted(mlx_keep)}"
