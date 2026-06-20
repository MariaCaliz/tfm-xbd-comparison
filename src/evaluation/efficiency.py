"""Medición de eficiencia computacional por modelo."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """Cuenta parámetros totales y entrenables del modelo.

    Args:
        model: Modelo nn.Module.

    Returns:
        Dict con claves 'total_params' y 'trainable_params'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def checkpoint_size_mb(model_or_path: nn.Module | str | Path) -> float:
    """Calcula el tamaño en MB del state_dict del modelo o de un checkpoint en disco.

    Args:
        model_or_path: Modelo nn.Module (se serializa temporalmente) o ruta
                       a un archivo .pt/.pth existente.

    Returns:
        Tamaño en megabytes (float).
    """
    if isinstance(model_or_path, (str, Path)):
        return Path(model_or_path).stat().st_size / (1024 ** 2)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        torch.save(model_or_path.state_dict(), tmp_path)
        return tmp_path.stat().st_size / (1024 ** 2)
    finally:
        tmp_path.unlink(missing_ok=True)


def benchmark_inference_time(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Mide el tiempo de inferencia por imagen con n_runs pasadas cronometradas.

    Hace n_warmup pasadas previas sin medir para estabilizar caché y JIT.
    Los tiempos se miden con time.perf_counter() (resolución de nanosegundos).

    Args:
        model:      Modelo nn.Module a evaluar.
        input_size: Forma del tensor de entrada (batch, C, H, W).
        device:     'cpu' o 'cuda'.
        n_warmup:   Pasadas de calentamiento (no se miden).
        n_runs:     Pasadas cronometradas.

    Returns:
        Dict con claves 'mean_ms', 'std_ms', 'median_ms'.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
    }


def compute_efficiency_metrics(
    model: nn.Module,
    model_name: str,
    checkpoint_path: str | Path | None = None,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Combina conteo de parámetros, tamaño y tiempo de inferencia en un único dict.

    Args:
        model:           Modelo a evaluar.
        model_name:      Nombre identificador del modelo.
        checkpoint_path: Ruta a un .pt existente para medir tamaño desde disco.
                         Si es None, se serializa el modelo temporalmente.
        n_warmup:        Pasadas de calentamiento para el benchmark.
        n_runs:          Pasadas cronometradas para el benchmark.

    Returns:
        Dict con claves: model_name, total_params, trainable_params,
        checkpoint_size_mb, mean_ms, std_ms, median_ms.
    """
    params = count_parameters(model)
    size_source = checkpoint_path if checkpoint_path is not None else model
    size_mb = checkpoint_size_mb(size_source)
    timing = benchmark_inference_time(model, n_warmup=n_warmup, n_runs=n_runs)
    return {
        "model_name": model_name,
        **params,
        "checkpoint_size_mb": round(size_mb, 3),
        **timing,
    }
