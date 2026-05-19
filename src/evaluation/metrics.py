"""Cálculo y persistencia de métricas de clasificación."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

_LABEL_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
_LABELS = [0, 1, 2, 3]


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 4,
) -> dict:
    """Calcula todas las métricas de clasificación en una sola llamada.

    Args:
        y_true:      Etiquetas reales (enteros 0-3).
        y_pred:      Predicciones del modelo (enteros 0-3).
        num_classes: Número de clases (por defecto 4).

    Returns:
        Diccionario con las siguientes claves:
          accuracy, precision_macro, recall_macro, f1_macro,
          precision_per_class, recall_per_class, f1_per_class,
          support_per_class, confusion_matrix, confusion_matrix_normalized.
        Las métricas per_class son dicts con nombres de clase como claves.
        confusion_matrix_normalized normaliza por filas (recall por clase);
        filas con support=0 se dejan a cero.
    """
    labels = list(range(num_classes))
    label_names = _LABEL_NAMES[:num_classes]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        average=None,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    def _per_class(values: np.ndarray) -> dict:
        return {name: float(v) for name, v in zip(label_names, values)}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_per_class": _per_class(precision),
        "recall_per_class": _per_class(recall),
        "f1_per_class": _per_class(f1),
        "support_per_class": {name: int(v) for name, v in zip(label_names, support)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Guarda el diccionario de métricas como JSON con indent=2.

    Args:
        metrics: Diccionario devuelto por compute_metrics().
        path:    Ruta del archivo de salida.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
