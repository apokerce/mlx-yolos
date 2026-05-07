# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert Ultralytics ``.pt`` checkpoints to mlx-yolos ``.safetensors``.

Design notes
------------

* The conversion is **generic, not per-model**. Because we use plain Python
  lists as module containers in mlx-yolos (matching Ultralytics' own list
  layout), state-dict keys map 1:1 — no regex remapping is required.
* The only operations we apply are:
    1. Drop ``num_batches_tracked`` (MLX BatchNorm doesn't have it).
    2. Drop the constant DFL projection conv weight — it's reconstructed
       from ``mx.arange(reg_max)`` at runtime in ``DetectV8``.
    3. Transpose 4-D conv weights from PyTorch ``(Cout, Cin, kH, kW)`` to
       MLX ``(Cout, kH, kW, Cin)``.
* Everything else (BN affine, BN buffers, 1-D biases) is copied verbatim.

This file does **not** import ``mlx`` — that lets the converter run on
any machine (Linux, macOS, Apple Silicon, x86) regardless of whether MLX
is installable there. The ``.safetensors`` it writes is loaded later by
``YOLO`` on whichever Apple Silicon machine actually runs inference.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_torch_state_dict(pt_path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for converting Ultralytics .pt checkpoints. "
            "Install with `pip install 'mlx-yolos[convert]'`."
        ) from e

    try:
        import ultralytics  # noqa: F401  — needed so torch.load can unpickle the model object
    except ImportError as e:
        raise ImportError(
            "Ultralytics is required for converting Ultralytics .pt checkpoints. "
            "Install with `pip install 'mlx-yolos[convert]'`."
        ) from e

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            return ckpt["model"].float().state_dict()
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    if hasattr(ckpt, "state_dict"):
        return ckpt.float().state_dict()
    return ckpt


# Heuristic for "this 4-D tensor is an nn.Conv2d weight" — the patterns
# listed cover the v8 family and any future model that follows the same
# ``Conv``/``DWConv`` wrapper or bare ``nn.Conv2d`` naming convention.
_CONV_WEIGHT_RE = re.compile(
    r"(?:"
    r"\.conv\.weight$"            # standard Conv wrapper
    r"|\.cv\d+\.\d+\.\d+\.weight$"  # head 1×1 final convs (cv2/cv3/cv4.<lvl>.<idx>.weight)
    r"|\.dfl\.conv\.weight$"      # DFL constant — we'll drop it later
    r"|\.m\.\d+\.weight$"         # bare Conv2d in a list-typed sub-module
    r")"
)


def _is_conv_weight(name: str, ndim: int) -> bool:
    return ndim == 4 and bool(_CONV_WEIGHT_RE.search(name))


def _is_dfl_constant(name: str) -> bool:
    """Detect the single DFL projection conv (``model.<head>.dfl.conv.weight``)."""
    return name.endswith(".dfl.conv.weight")


def _remap_key(pt_key: str) -> str | None:
    """Translate a PyTorch state-dict key into the mlx-yolos parameter path.

    The mlx-yolos ``BaseModel`` exposes its layer list as ``self.model``,
    matching Ultralytics' attribute name, so the keys are essentially
    identical except for things we drop entirely:

    * ``num_batches_tracked``: BN bookkeeping not used in MLX.
    * ``*.dfl.conv.weight``  : reconstructed at inference time.
    """
    if pt_key.endswith("num_batches_tracked"):
        return None
    if _is_dfl_constant(pt_key):
        return None
    return pt_key


def convert_ultralytics_checkpoint(
    pt_path: str | Path,
    out_path: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert an Ultralytics ``.pt`` to mlx-yolos ``.safetensors``.

    Args:
        pt_path: Source ``.pt`` checkpoint path.
        out_path: Destination ``.safetensors`` path.
        verbose: Log conversion summary.

    Returns:
        Summary dict ``{kept, skipped, conv_weights, out_path}``.
    """
    try:
        import torch  # noqa: F401  — ensure import error is clear
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            "Conversion requires `torch` + `safetensors`. "
            "Install with `pip install 'mlx-yolos[convert]'`."
        ) from e

    pt_path = Path(pt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = _load_torch_state_dict(pt_path)

    converted: dict[str, Any] = {}
    skipped: list[str] = []
    n_conv = 0
    for k, v in state.items():
        new_k = _remap_key(k)
        if new_k is None:
            skipped.append(k)
            continue
        t = v.float() if hasattr(v, "float") else v
        if _is_conv_weight(new_k, t.dim()):
            t = t.permute(0, 2, 3, 1).contiguous()
            n_conv += 1
        else:
            t = t.contiguous()
        converted[new_k] = t

    save_file(converted, str(out_path))

    summary = {
        "kept": len(converted),
        "skipped": len(skipped),
        "conv_weights_transposed": n_conv,
        "out_path": str(out_path),
    }
    if verbose:
        logger.info(
            "converted %d tensors (%d conv weights transposed); "
            "skipped %d bookkeeping keys → %s",
            summary["kept"],
            summary["conv_weights_transposed"],
            summary["skipped"],
            summary["out_path"],
        )
    return summary
