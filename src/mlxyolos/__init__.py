# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""mlx-yolos — pure-MLX inference for Ultralytics YOLO models.

Public API:
    >>> from mlxyolos import YOLO
    >>> model = YOLO("yolov8-pose.yaml", scale="n",
    ...              weights="yolov8n-pose.safetensors")
    >>> results = model.predict("bus.jpg")

Currently supported tasks: ``pose`` (yolov8-pose family).
The package is structured so that adding new tasks (detect / segment /
obb) only requires (a) a YAML config, (b) a head class, and (c) a
post-processor entry — no per-model regex remappers in the converter.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = ["YOLO", "__version__"]

# Quiet by default; users can opt in via logging.getLogger("mlxyolos").setLevel(...).
_logger = _logging.getLogger(__name__)
if not _logger.handlers and not _logging.root.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(_logging.WARNING)


if TYPE_CHECKING:
    from mlxyolos.engine.model import YOLO  # noqa: F401


def __getattr__(name: str):
    if name == "YOLO":
        from mlxyolos.engine.model import YOLO

        return YOLO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
