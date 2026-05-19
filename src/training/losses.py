"""Función de pérdida para clasificación con desbalanceo de clases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

_NUM_CLASSES = 4
_LABEL_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def compute_class_weights(train_csv_path: str | Path) -> torch.Tensor:
    """Calcula pesos de clase balanceados a partir del CSV de train.

    Usa el método "balanced" de sklearn:
        weight_i = total / (num_classes * count_i)

    Las clases con menos muestras reciben mayor peso, compensando el
    fuerte desbalanceo (no-damage ~77 % del train set).

    Args:
        train_csv_path: Ruta a train.csv (data/splits/train.csv).

    Returns:
        Tensor float32 de shape (4,) con pesos en orden
        [no-damage, minor-damage, major-damage, destroyed].
    """
    df = pd.read_csv(train_csv_path)
    total = len(df)
    counts = df["label"].value_counts().sort_index()

    weights = torch.tensor(
        [total / (_NUM_CLASSES * counts.get(i, 1)) for i in range(_NUM_CLASSES)],
        dtype=torch.float32,
    )
    return weights


def get_loss(config: dict, train_csv_path: str | Path) -> nn.Module:
    """Factoría de función de pérdida según la configuración.

    Lee config['training']['loss']. Para 'weighted_cross_entropy',
    calcula los pesos de clase desde el CSV de train (o usa los pesos
    explícitos si config['training']['class_weights'] es una lista).

    Args:
        config:         Configuración fusionada de load_config().
        train_csv_path: Ruta a train.csv, usada solo si class_weights=='auto'.

    Returns:
        nn.Module listo para llamar con loss(logits, labels).

    Raises:
        ValueError: Si config['training']['loss'] no está soportado.
    """
    training_cfg = config.get("training", {})
    loss_name: str = training_cfg.get("loss", "weighted_cross_entropy")

    if loss_name == "weighted_cross_entropy":
        class_weights_cfg = training_cfg.get("class_weights", "auto")
        if class_weights_cfg == "auto":
            weights = compute_class_weights(train_csv_path)
        else:
            weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    raise ValueError(
        f"Función de pérdida '{loss_name}' no soportada. "
        f"Opciones: ['weighted_cross_entropy']"
    )
