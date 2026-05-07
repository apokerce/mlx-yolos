# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YAML-driven model assembly.

The parser is a faithful subset of Ultralytics' ``parse_model`` adapted to
MLX (NHWC). It reads a YAML config, applies the chosen scale's depth/width
multipliers, instantiates each layer, and stitches them together into a
``BaseModel`` whose forward routes activations between layers based on the
``from`` indices recorded in the YAML.

Adding a new model family means: drop a YAML in ``cfg/models/<family>/`` and,
if the head differs, register a new head class in ``HEAD_BUILDERS``.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import yaml

from .modules import (
    SPPF,
    Bottleneck,
    C2f,
    Concat,
    Conv,
    DetectV8,
    DWConv,
    PoseV8,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaling helper
# ---------------------------------------------------------------------------


def make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


# ---------------------------------------------------------------------------
# Tiny MLX-friendly Upsample (nearest-2x by default).
#
# MLX has ``nn.Upsample`` but it's been added relatively recently; using
# ``mx.repeat`` keeps us compatible with older MLX builds and is exactly
# what the v8 YAML asks for (``[None, 2, "nearest"]``).
# ---------------------------------------------------------------------------


class _NearestUpsample(nn.Module):
    def __init__(self, scale_factor: float = 2.0, mode: str = "nearest") -> None:
        super().__init__()
        if mode != "nearest":
            raise NotImplementedError(f"Only nearest upsample is supported (got {mode!r})")
        if int(scale_factor) != scale_factor:
            raise NotImplementedError(f"Only integer scale factors are supported (got {scale_factor})")
        self.scale = int(scale_factor)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, self.scale, axis=1)
        x = mx.repeat(x, self.scale, axis=2)
        return x


# ---------------------------------------------------------------------------
# Module registry — used by parse_model to map YAML names → classes
# ---------------------------------------------------------------------------


MODULE_MAP: dict[str, type[nn.Module]] = {
    "Conv": Conv,
    "DWConv": DWConv,
    "Concat": Concat,
    "Bottleneck": Bottleneck,
    "C2f": C2f,
    "SPPF": SPPF,
    # Heads are referenced by their YAML names. The DetectV8 / PoseV8
    # split keeps the per-task code in one place; new heads register here.
    "Detect": DetectV8,
    "Pose": PoseV8,
}


# Modules whose first two args are ``(c1, c2)`` and that get the standard
# width-scaled output channels.
_BASE_MODULES = {Conv, DWConv, Bottleneck, C2f, SPPF}
# Modules that take an extra ``n`` (repeat count) inserted after ``c2``.
_REPEAT_MODULES = {C2f}
# Heads — their args are built by per-class builders below, not the
# ``[c1, c2, ...]`` rule.
_HEAD_TYPES: tuple[type[nn.Module], ...] = (DetectV8, PoseV8)


def _build_detect_args(args: list[Any], nc: int, reg_max: int, ch_list: list[int]) -> list[Any]:
    return [nc, reg_max, ch_list]


def _build_pose_args(args: list[Any], nc: int, reg_max: int, ch_list: list[int]) -> list[Any]:
    kpt_shape = args[1] if len(args) > 1 else (17, 3)
    return [nc, kpt_shape, reg_max, ch_list]


HEAD_BUILDERS: dict[type[nn.Module], Callable[[list[Any], int, int, list[int]], list[Any]]] = {
    DetectV8: _build_detect_args,
    PoseV8: _build_pose_args,
}


# ---------------------------------------------------------------------------
# BaseModel — a thin nn.Module that owns the layer list and routing info.
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    """Multi-output sequential model with feature routing."""

    def __init__(self, layers: list[nn.Module], save: list[int]) -> None:
        super().__init__()
        # Plain Python list — MLX walks lists during parameter collection,
        # giving us names like ``model.0.conv.weight`` that match Ultralytics
        # 1:1 (no per-layer remapping in the converter).
        self.model = layers
        self._save = sorted(set(save))

    def __call__(self, x: mx.array) -> Any:
        y: list[Any] = []
        for m in self.model:
            f = getattr(m, "f", -1)
            if f != -1:
                if isinstance(f, int):
                    x = y[f]
                else:
                    x = [y[j] if j != -1 else x for j in f]
            x = m(x)
            y.append(x if getattr(m, "i", -1) in self._save else None)
        return x


# ---------------------------------------------------------------------------
# YAML loading + parser
# ---------------------------------------------------------------------------


def load_model_config(cfg: str | Path | dict) -> dict:
    """Load a YAML model spec. Searches the package's bundled configs."""
    if isinstance(cfg, dict):
        return cfg
    cfg_path = Path(cfg)
    candidates = [
        cfg_path,
        Path(__file__).parent.parent / "cfg" / "models" / cfg_path.name,
        Path(__file__).parent.parent / "cfg" / "models" / "v8" / cfg_path.name,
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            # Pull a default scale out of the filename, e.g. ``yolov8n-pose.yaml`` → "n".
            m = re.search(r"yolov8([nsmlx])", path.stem)
            if m and "scale" not in data:
                data["scale"] = m.group(1)
            return data
    raise FileNotFoundError(f"Config not found: {cfg} (searched {[str(p) for p in candidates]})")


def parse_model(cfg: dict, ch_in: int = 3, scale: str | None = None, verbose: bool = False) -> BaseModel:
    """Build a model from a YAML config dict.

    Args:
        cfg: Already-loaded YAML dictionary (not a path).
        ch_in: Input channels (3 for RGB).
        scale: Override the scale variant ('n', 's', 'm', 'l', 'x'). If None, falls
            back to ``cfg["scale"]`` and finally to (1.0, 1.0, 1024).
        verbose: Pretty-print each layer as it's instantiated.
    """
    nc = cfg.get("nc", 80)
    reg_max = cfg.get("reg_max", 16)
    scales = cfg.get("scales", {})
    s = scale or cfg.get("scale", "n")
    if s in scales:
        depth, width, max_channels = scales[s]
    else:
        depth, width, max_channels = 1.0, 1.0, 1024

    layers: list[nn.Module] = []
    save: list[int] = []
    ch: list[int] = [ch_in]

    if verbose:
        header = f"{'idx':>3} {'from':>20} {'n':>3} {'module':<28} {'args'}"
        logger.info(header)
        logger.info("-" * len(header))

    spec = list(cfg.get("backbone", [])) + list(cfg.get("head", []))
    for i, (f, n, m, args) in enumerate(spec):
        # Resolve the module class.
        if isinstance(m, str):
            if m.startswith("nn.Upsample"):
                m_cls: type[nn.Module] = _NearestUpsample
                # YAML form: [None, 2, "nearest"] → scale_factor only.
                args = [args[1] if len(args) > 1 else 2, args[2] if len(args) > 2 else "nearest"]
            elif m.startswith("nn."):
                attr = m[3:]
                if not hasattr(nn, attr):
                    raise ValueError(f"Unknown nn module: {m!r}")
                m_cls = getattr(nn, attr)
            elif m in MODULE_MAP:
                m_cls = MODULE_MAP[m]
            else:
                raise ValueError(f"Unknown module: {m!r}")
        else:
            m_cls = m

        # Resolve string args (e.g. literal "nc" → numeric).
        args = list(args)
        for j, a in enumerate(args):
            if isinstance(a, str):
                if a == "nc":
                    args[j] = nc
                elif a == "reg_max":
                    args[j] = reg_max
                else:
                    with contextlib.suppress(ValueError, SyntaxError):
                        args[j] = ast.literal_eval(a)

        n_ = max(round(n * depth), 1) if n > 1 else n

        # Per-class arg construction.
        if m_cls in _BASE_MODULES:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m_cls in _REPEAT_MODULES:
                args.insert(2, n_)
                n_ = 1
        elif m_cls is _NearestUpsample:
            c2 = ch[f]
        elif m_cls is Concat:
            c2 = sum(ch[x] for x in f)
            args = [1]
        elif m_cls in _HEAD_TYPES:
            ch_list = [ch[x] for x in f]
            args = HEAD_BUILDERS[m_cls](args, nc, reg_max, ch_list)
            c2 = None
        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        if n_ > 1:
            mod = _Repeat([m_cls(*args) for _ in range(n_)])
        else:
            mod = m_cls(*args)

        # Tag the module with its YAML metadata so BaseModel can route inputs.
        mod.i = i  # type: ignore[attr-defined]
        mod.f = f  # type: ignore[attr-defined]
        mod.type = m_cls.__name__  # type: ignore[attr-defined]

        if verbose:
            logger.info(f"{i:>3} {str(f):>20} {n_:>3} {m_cls.__name__:<28} {args}")

        save.extend(x % (i + 1) for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(mod)
        if i == 0:
            ch = []
        ch.append(c2 if c2 is not None else 0)

    return BaseModel(layers, save)


class _Repeat(nn.Module):
    """Sequential of n identical sub-modules (used for ``n > 1`` YAML rows)."""

    def __init__(self, modules: list[nn.Module]) -> None:
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_model(
    cfg: str | Path | dict,
    ch: int = 3,
    nc: int | None = None,
    scale: str | None = None,
    verbose: bool = False,
) -> BaseModel:
    """Build an MLX YOLO model from a YAML config (or pre-loaded dict)."""
    cfg_d = load_model_config(cfg)
    if nc is not None:
        cfg_d = {**cfg_d, "nc": nc}
    return parse_model(cfg_d, ch_in=ch, scale=scale, verbose=verbose)
