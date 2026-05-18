"""Utilidades transversales."""

# Importaciones ligeras (sin torch): siempre disponibles
from src.utils.config import load_config

__all__ = ["load_config", "set_seed", "get_device", "setup_logger", "count_parameters"]

# Importaciones pesadas (requieren torch): se cargan solo si torch está instalado
def __getattr__(name):
    _torch_exports = {"set_seed", "get_device", "setup_logger", "count_parameters"}
    if name in _torch_exports:
        from src.utils.seed import set_seed, get_device, setup_logger, count_parameters
        return locals()[name]
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")
