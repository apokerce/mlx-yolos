# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Weight converters: Ultralytics ``.pt`` → mlx-yolos ``.safetensors``."""

from .ultralytics_pt import convert_ultralytics_checkpoint

__all__ = ["convert_ultralytics_checkpoint"]
