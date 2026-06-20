"""Módulos de evaluación: métricas, eficiencia y comparativa."""

from src.evaluation.compare import (
    build_comparison_table,
    load_all_metrics,
    plot_confusion_matrices_grid,
    plot_efficiency_tradeoff,
    plot_f1_comparison,
)
from src.evaluation.efficiency import (
    benchmark_inference_time,
    checkpoint_size_mb,
    compute_efficiency_metrics,
    count_parameters,
)
from src.evaluation.metrics import compute_metrics, save_metrics

__all__ = [
    "compute_metrics",
    "save_metrics",
    "count_parameters",
    "checkpoint_size_mb",
    "benchmark_inference_time",
    "compute_efficiency_metrics",
    "load_all_metrics",
    "build_comparison_table",
    "plot_confusion_matrices_grid",
    "plot_f1_comparison",
    "plot_efficiency_tradeoff",
]
