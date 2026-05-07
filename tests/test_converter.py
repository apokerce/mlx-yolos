# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Converter unit tests that don't require MLX or Ultralytics weights.

We test the pure helpers (``_remap_key``, ``_is_conv_weight``) and the
overall expectation that conv weights — and only conv weights — get
transposed.
"""

from __future__ import annotations

from mlxyolos.converters.ultralytics_pt import _is_conv_weight, _remap_key


def test_remap_drops_num_batches_tracked():
    assert _remap_key("model.0.bn.num_batches_tracked") is None


def test_remap_drops_dfl_constant():
    assert _remap_key("model.22.dfl.conv.weight") is None


def test_remap_passthrough():
    assert _remap_key("model.4.m.0.cv1.conv.weight") == "model.4.m.0.cv1.conv.weight"
    assert _remap_key("model.22.cv4.0.2.weight") == "model.22.cv4.0.2.weight"


def test_is_conv_weight_basic():
    assert _is_conv_weight("model.0.conv.weight", 4)
    assert _is_conv_weight("model.4.m.0.cv1.conv.weight", 4)
    assert _is_conv_weight("model.22.cv2.0.0.conv.weight", 4)
    # final 1×1 in head is bare nn.Conv2d
    assert _is_conv_weight("model.22.cv2.0.2.weight", 4)
    # bn weight isn't a conv weight
    assert not _is_conv_weight("model.0.bn.weight", 1)
    # 1-D weight tensor isn't a conv kernel even if name matches
    assert not _is_conv_weight("model.0.conv.weight", 1)
