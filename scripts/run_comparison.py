#!/usr/bin/env python3
"""Genera la tabla comparativa y las figuras transversales del Capítulo 5/6.

Mide eficiencia en CPU (representativo de despliegue con recursos limitados)
y la combina con las métricas de test ya calculadas por cada modelo.

Uso:
  python scripts/run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluation.compare import (
    load_all_metrics,
    build_comparison_table,
    plot_confusion_matrices_grid,
    plot_f1_comparison,
    plot_efficiency_tradeoff,
)
from src.evaluation.efficiency import compute_efficiency_metrics
from src.models import get_model

MODELS   = ["mobilenetv2", "efficientnetb0", "resnet50", "vit"]
METRICS  = Path("results/metrics_final")          # los 4 *_metrics.json
FIGURES  = Path("results/figures")
DEVICE   = "cpu"                                   # CPU = realista para campo

# Config mínima para instanciar cada modelo (sin pesos preentrenados: da igual,
# solo medimos nº de parámetros y tiempo de inferencia, no calidad).
def _cfg(name: str) -> dict:
    return {"model": {"name": name, "num_classes": 4,
                      "pretrained": False, "dropout": 0.2}}

def main() -> None:
    torch.set_num_threads(torch.get_num_threads())  # usa los hilos del equipo

    # 1) Eficiencia (params + latencia en CPU). Sin checkpoint_path: no usa los .pt
    efficiency_by_model = {}
    for name in MODELS:
        model = get_model(_cfg(name))
        eff = compute_efficiency_metrics(
            model, model_name=name,
            checkpoint_path=None,        # mide tamaño serializando en memoria
            n_warmup=10, n_runs=100,      # 50 pasadas; sube a 100 si quieres más estabilidad
        )
        efficiency_by_model[name] = eff
        print(f"{name:15s} | params={eff['total_params']:,} | "
              f"inferencia={eff['mean_ms']:.1f} ms | tamaño={eff['checkpoint_size_mb']:.0f} MB")

    # 2) Métricas de test (los 4 JSON copiados de Drive)
    metrics_by_model = load_all_metrics(METRICS, MODELS)

    # 3) Tabla comparativa
    df = build_comparison_table(metrics_by_model, efficiency_by_model)
    print("\n=== Tabla comparativa ===")
    print(df.to_string())
    df.to_csv(FIGURES.parent / "comparison_table.csv")

    # 4) Figuras transversales
    plot_confusion_matrices_grid(metrics_by_model, FIGURES / "confusion_matrices.png")
    plot_f1_comparison(metrics_by_model, FIGURES / "f1_comparison.png")
    plot_efficiency_tradeoff(metrics_by_model, efficiency_by_model,
                             FIGURES / "efficiency_tradeoff.png")
    print("\nListo. Figuras en", FIGURES)

if __name__ == "__main__":
    main()