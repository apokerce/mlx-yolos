# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for letterbox math in ``mlxyolos.utils.ops``.

Two layers:

1. **Backward-compat** — default behavior (square, ``rect=False``) is
   identical to what it was before the rect option existed.
2. **Rect parity** — `rect=True` matches the math Ultralytics' LetterBox
   class uses internally (``new_unpad`` per dim, padded to the next
   stride multiple). Verified against hand-derived cases that span the
   common image shapes; no MLX or torchvision dependency.
"""

from __future__ import annotations

import numpy as np

from mlxyolos.utils.ops import letterbox


def _make_img(h: int, w: int) -> np.ndarray:
    """Solid-gray RGB image, mostly used as a shape carrier in these tests."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Square mode (default) — unchanged behavior
# ---------------------------------------------------------------------------


def test_square_pad_landscape():
    """810w × 1080h portrait → 640×640 square with horizontal padding."""
    img = _make_img(h=1080, w=810)
    out, r, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    # r = min(640/1080, 640/810) = 640/1080.
    np.testing.assert_allclose(r, 640 / 1080)
    # nw = round(810 * r) = 480; horizontal pad = (640-480) // 2 = 80.
    assert left == 80
    assert top == 0


def test_square_pad_square_input():
    img = _make_img(640, 640)
    out, r, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == 1.0
    assert (left, top) == (0, 0)


def test_explicit_tuple_shape_uses_those_dims():
    """Passing a tuple bypasses both square and rect logic — pad to exactly (h, w)."""
    img = _make_img(1080, 810)
    out, _, _ = letterbox(img, (384, 640))
    assert out.shape == (384, 640, 3)


# ---------------------------------------------------------------------------
# Rectangular mode — math must match Ultralytics' LetterBox `auto=True` path
# ---------------------------------------------------------------------------


def test_rect_portrait_810x1080_to_640x480():
    """The headline case from VALIDATION.md.

    Ultralytics rect math: r = 640/1080, nw = round(810*r) = 480, nh = round(1080*r) = 640.
    Then pad each dim up to the next multiple of stride=32 — both are already
    aligned, so zero padding either way."""
    img = _make_img(h=1080, w=810)
    out, r, (left, top) = letterbox(img, 640, rect=True, stride=32)
    assert out.shape == (640, 480, 3)
    np.testing.assert_allclose(r, 640 / 1080)
    assert (left, top) == (0, 0)


def test_rect_landscape_480x640_unchanged():
    """Already 640 wide and a stride multiple tall — rect should be a no-op."""
    img = _make_img(h=480, w=640)
    out, r, (left, top) = letterbox(img, 640, rect=True, stride=32)
    assert out.shape == (480, 640, 3)
    assert r == 1.0
    assert (left, top) == (0, 0)


def test_rect_requires_pad_for_non_stride_multiple():
    """720×480 → r = 640/720 = 0.8889; nh = 640, nw = round(480*r) = 427.
    Next stride-32 multiple of 427 is 448, so pad_w = 21, left = 10."""
    img = _make_img(h=720, w=480)
    out, r, (left, top) = letterbox(img, 640, rect=True, stride=32)
    assert out.shape == (640, 448, 3)
    np.testing.assert_allclose(r, 640 / 720)
    assert left == 10  # (448 - 427) // 2
    assert top == 0


def test_rect_square_input_matches_square_mode():
    """A square input has no aspect ratio to preserve — rect and square give
    the same canvas. Important for backward compatibility on existing
    benchmarks that test on square 640×640 inputs."""
    img = _make_img(640, 640)
    sq_out, sq_r, sq_pad = letterbox(img, 640, rect=False)
    rc_out, rc_r, rc_pad = letterbox(img, 640, rect=True, stride=32)
    assert sq_out.shape == rc_out.shape == (640, 640, 3)
    assert sq_r == rc_r == 1.0
    assert sq_pad == rc_pad == (0, 0)


# ---------------------------------------------------------------------------
# Custom stride / smaller imgsz (sanity checks for the formula)
# ---------------------------------------------------------------------------


def test_rect_stride_16_smaller_imgsz():
    """320 imgsz, stride 16, portrait 480×640 → r=320/640=0.5; nh=240, nw=160.
    240 → ceil(240/16)*16 = 240 (already aligned). 160 → 160. No pad."""
    img = _make_img(h=480, w=640)
    out, r, pad = letterbox(img, 320, rect=True, stride=16)
    # NOTE: actually h=480, w=640 means landscape (wider than tall).
    # r = min(320/480, 320/640) = 320/640 = 0.5.
    # nh = 240, nw = 320. Both stride-16 aligned. Output (240, 320).
    assert out.shape == (240, 320, 3)
    assert r == 0.5
    assert pad == (0, 0)


# ---------------------------------------------------------------------------
# scaleup=False — never upscale below imgsz (matches Ultralytics `val`)
# ---------------------------------------------------------------------------


def test_scaleup_false_small_image_kept_at_native_size_square():
    """320×240 image with imgsz=640 + scaleup=False should NOT be upscaled.
    Inside a 640×640 square canvas the original 240×320 ends up centered."""
    img = _make_img(h=240, w=320)
    out, r, (left, top) = letterbox(img, 640, scaleup=False)
    assert out.shape == (640, 640, 3)
    assert r == 1.0  # no upscale
    # nh = 240, nw = 320; (640-320)//2 = 160 left, (640-240)//2 = 200 top.
    assert (left, top) == (160, 200)


def test_scaleup_true_small_image_gets_upscaled_square():
    """Same input with scaleup=True (the default) rescales the smaller dim
    until the larger dim hits imgsz."""
    img = _make_img(h=240, w=320)
    out, r, _ = letterbox(img, 640, scaleup=True)
    assert out.shape == (640, 640, 3)
    # r = min(640/240, 640/320) = min(2.667, 2.0) = 2.0
    np.testing.assert_allclose(r, 2.0)


def test_scaleup_irrelevant_for_large_images():
    """A 1080×810 input is larger than imgsz in both dims — r ≤ 1.0 either way,
    so scaleup=True and scaleup=False must produce identical output."""
    img = _make_img(h=1080, w=810)
    o1, r1, p1 = letterbox(img, 640, scaleup=True)
    o2, r2, p2 = letterbox(img, 640, scaleup=False)
    assert o1.shape == o2.shape == (640, 640, 3)
    assert r1 == r2
    assert p1 == p2


def test_scaleup_false_combined_with_rect_small_image():
    """rect=True + scaleup=False on a small input: the canvas shrinks to the
    smallest stride-multiple that contains the (un-upscaled) image. For
    240×320 (h=240 is *not* a stride-32 multiple — 7.5×32), rect pads h up
    to 256 while w=320 stays put."""
    img = _make_img(h=240, w=320)
    out, r, (left, top) = letterbox(img, 640, rect=True, scaleup=False, stride=32)
    assert out.shape == (256, 320, 3)
    assert r == 1.0
    # pad_h = 256 - 240 = 16 (top = 8), pad_w = 0.
    assert (left, top) == (0, 8)


def test_rect_padding_only_one_dim_when_other_aligned():
    """Manufactured case to exercise the asymmetric-pad path: h=500, w=640.
    r = min(640/500, 640/640) = 1.0. nh = 500, nw = 640. Stride 32:
    nh→512 (pad_h=12, top=6), nw→640 (no pad). Output (512, 640)."""
    img = _make_img(h=500, w=640)
    out, r, (left, top) = letterbox(img, 640, rect=True, stride=32)
    assert out.shape == (512, 640, 3)
    assert r == 1.0
    assert left == 0
    assert top == 6
