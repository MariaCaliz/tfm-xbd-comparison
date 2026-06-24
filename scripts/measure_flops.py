#!/usr/bin/env python3
"""Mide FLOPs (fvcore) y, si hay GPU, el pico de memoria de inferencia.

Completa las métricas de eficiencia de OE5 que no cubría el análisis en CPU.
Uso:  python scripts/measure_flops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fvcore.nn import FlopCountAnalysis

from src.models import get_model

MODELS = ["mobilenetv2", "efficientnetb0", "resnet50", "vit"]


def _cfg(name: str) -> dict:
    return {"model": {"name": name, "num_classes": 4,
                      "pretrained": False, "dropout": 0.2}}


def main() -> None:
    print(f"{'Modelo':16s} {'GFLOPs':>10s} {'Mem GPU (MB)':>14s}")
    print("-" * 42)
    for name in MODELS:
        model = get_model(_cfg(name)).eval()

        # ── FLOPs (en CPU, con un tensor de 1x3x224x224) ──
        x = torch.randn(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = flops.total() / 1e9

        # ── Pico de memoria GPU en inferencia (solo si hay CUDA) ──
        mem_mb = None
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            xg = x.cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model_gpu(xg)
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            del model_gpu, xg
            torch.cuda.empty_cache()

        mem_str = f"{mem_mb:14.1f}" if mem_mb is not None else f"{'n/a (CPU)':>14s}"
        print(f"{name:16s} {gflops:10.3f} {mem_str}")


if __name__ == "__main__":
    main()