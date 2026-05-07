# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""``YOLO`` — user-facing entry point.

Generic over the task: the YAML config tells us which head to instantiate,
and the matching post-processor in ``Predictor`` is selected from ``task``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mlxyolos.engine.predictor import Predictor
from mlxyolos.engine.results import Results
from mlxyolos.nn.modules.head import DetectV8, PoseV8
from mlxyolos.nn.tasks import BaseModel, build_model, load_model_config

logger = logging.getLogger(__name__)


# Map a head class to the predictor's task name. Adding a new head =
# one entry here and one entry in ``predictor.POSTPROCESSORS``.
_HEAD_TO_TASK: dict[type, str] = {
    DetectV8: "detect",
    PoseV8: "pose",
}


class YOLO:
    """Loads a YAML config + (optionally) MLX weights and runs inference.

    Examples
    --------
    >>> from mlxyolos import YOLO
    >>> model = YOLO("yolov8-pose.yaml", scale="n", weights="yolov8n-pose.safetensors")
    >>> results = model.predict("bus.jpg")
    """

    def __init__(
        self,
        cfg: str | Path | dict,
        *,
        weights: str | Path | None = None,
        scale: str | None = None,
        nc: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.cfg_dict = load_model_config(cfg)
        self.scale = scale or self.cfg_dict.get("scale", "n")
        self.model: BaseModel = build_model(
            self.cfg_dict, ch=3, nc=nc, scale=self.scale, verbose=verbose
        )
        self.task = self._infer_task()
        self.nc = self.cfg_dict.get("nc", 80) if nc is None else nc
        self.kpt_shape: tuple[int, int] | None = None
        if self.task == "pose":
            ks = self.cfg_dict.get("kpt_shape", [17, 3])
            self.kpt_shape = (int(ks[0]), int(ks[1]))
        self.names: dict[int, str] = {i: str(i) for i in range(self.nc)}
        if weights is not None:
            self.load(weights)

    # ----- model lifecycle ---------------------------------------------------

    def _infer_task(self) -> str:
        head = self.model.model[-1]
        for cls, task in _HEAD_TO_TASK.items():
            if isinstance(head, cls):
                return task
        raise ValueError(
            f"Could not infer task from head {type(head).__name__!r}; "
            f"register it in mlxyolos.engine.model._HEAD_TO_TASK"
        )

    def load(self, weights: str | Path) -> None:
        """Load MLX-format weights (``.safetensors``)."""
        self.model.load_weights(str(weights), strict=True)
        self.model.eval()

    # ----- inference ---------------------------------------------------------

    def predict(
        self,
        source: str | Path | np.ndarray | Image.Image | list,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[Results]:
        predictor = Predictor(
            model=self.model,
            task=self.task,
            nc=self.nc,
            names=self.names,
            kpt_shape=self.kpt_shape,
        )
        return predictor(source, imgsz=imgsz, conf=conf, iou=iou)

    # convenience: callable shorthand
    def __call__(self, source: Any, **kw: Any) -> list[Results]:
        return self.predict(source, **kw)
