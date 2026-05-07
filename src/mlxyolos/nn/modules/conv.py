# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution building blocks (NHWC) shared across YOLO families.

These are direct ports of the Ultralytics `Conv` / `DWConv` / `Concat` modules
adapted to MLX's NHWC layout. They are generic — every YOLO task in this
package (detect / pose / segment / obb …) is built on top of them.

Reference: ``ultralytics/ultralytics/nn/modules/conv.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["autopad", "Conv", "DWConv", "Concat"]


def autopad(k, p=None, d: int = 1):
    """Pad-to-same calculation, mirroring Ultralytics' implementation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d + BatchNorm + (optional) activation in NHWC."""

    default_act: nn.Module | None = None  # set lazily below to avoid import order issues

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            dilation=d,
            groups=g,
            bias=False,
        )
        # Ultralytics overrides every BN's eps to 1e-3 in `initialize_weights`,
        # and the released checkpoints carry that value. MLX's default is 1e-5,
        # which silently shifts every activation downstream — so we set it here.
        self.bn = nn.BatchNorm(num_features=c2, eps=1e-3, momentum=0.03, affine=True)

        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DWConv(Conv):
    """Depth-wise convolution (groups = c1)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__(c1, c2, k, s, g=c1, d=d, act=act)


class Concat(nn.Module):
    """Concat along the channel axis. NHWC ⇒ axis = -1.

    The YAML config uses ``dimension=1`` (PyTorch NCHW convention); we map
    that to MLX's ``axis=-1`` automatically so configs stay portable.
    """

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = -1 if dimension == 1 else dimension

    def __call__(self, x: list[mx.array]) -> mx.array:
        return mx.concatenate(x, axis=self.d)
