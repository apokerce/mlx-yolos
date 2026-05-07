# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CSP / SPPF / Bottleneck building blocks shared across YOLO families.

Reference: ``ultralytics/ultralytics/nn/modules/block.py``.

We use plain Python lists for module containers (``self.m = [...]``) — MLX
walks these natively when collecting parameters, so the resulting parameter
names mirror Ultralytics' state-dict naming verbatim. That keeps the
PyTorch → MLX converter dumb (transpose + skip), with no per-model regex
remapping, which is the property that makes adding new model families easy.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv

__all__ = ["Bottleneck", "C2f", "SPPF"]


class Bottleneck(nn.Module):
    """Standard bottleneck (cv1 → cv2) with optional residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def __call__(self, x: mx.array) -> mx.array:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions, n stacked Bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = [Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)]

    def __call__(self, x: mx.array) -> mx.array:
        y = self.cv1(x)
        a, b = mx.split(y, 2, axis=-1)
        outs: list[mx.array] = [a, b]
        for m in self.m:
            outs.append(m(outs[-1]))
        return self.cv2(mx.concatenate(outs, axis=-1))


class SPPF(nn.Module):
    """Spatial-pyramid pooling — fast (3 × MaxPool, kernel=k)."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))
