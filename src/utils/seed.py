"""Utilidades transversales: reproducibilidad, logging y helpers.

Estas funciones se usan desde scripts y notebooks para garantizar
reproducibilidad y facilitar el seguimiento de los experimentos.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fija la semilla aleatoria en todas las librerías relevantes.

    Garantiza reproducibilidad de los experimentos. Es importante
    llamarla *antes* de instanciar modelos o crear DataLoaders.

    Args:
        seed: Valor de la semilla. Por defecto 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Para determinismo completo (a coste de algo de velocidad):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Devuelve el dispositivo de cómputo a usar.

    Args:
        prefer: 'auto' elige cuda > mps > cpu; 'cuda', 'mps' o 'cpu'
                fuerza el dispositivo indicado.

    Returns:
        torch.device configurado.
    """
    if prefer in ("cuda", "mps", "cpu"):
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(name: str, log_file: Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Configura un logger con salida a consola y opcionalmente a fichero.

    Args:
        name: Nombre del logger (típicamente __name__ del módulo).
        log_file: Ruta al fichero de log. Si es None, solo consola.
        level: Nivel de logging.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Cuenta parámetros totales y entrenables de un modelo.

    Returns:
        (total, entrenables)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
