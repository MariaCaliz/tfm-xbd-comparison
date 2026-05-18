"""Utilidades transversales."""

from src.utils.config import load_config
from src.utils.seed import (
    count_parameters,
    get_device,
    set_seed,
    setup_logger,
)

__all__ = ["load_config", "set_seed", "get_device", "setup_logger", "count_parameters"]
