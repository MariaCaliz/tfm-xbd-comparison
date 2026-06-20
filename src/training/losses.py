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


def compute_class_weights_effective_number(
    class_counts: list[int],
    beta: float = 0.999,
) -> torch.Tensor:
    """Calcula pesos usando "Effective Number of Samples" (Cui et al., 2019).

    En lugar de ponderar por la inversa del conteo bruto, pondera por la
    inversa del número efectivo de muestras, que satura logarítmicamente
    al crecer la clase mayoritaria:

        effective_num_i = (1 - beta^count_i) / (1 - beta)
        weight_i        = 1 / effective_num_i

    Los pesos se normalizan para que la media sea 1 (suma == num_classes),
    manteniendo magnitudes comparables con el esquema 'balanced'.

    A mayor beta (→ 1), más suave es la corrección y menor es el ratio
    max/min resultante. El valor 0.999 es el recomendado por los autores
    para datasets de escala de decenas de miles de muestras por clase.

    Args:
        class_counts: Lista de enteros donde class_counts[i] = nº muestras
                      de la clase i, en el mismo orden que las etiquetas.
        beta:         Hiperparámetro en [0, 1). Típicamente 0.9, 0.99,
                      0.999 o 0.9999.

    Returns:
        Tensor float32 de shape (len(class_counts),) con pesos normalizados.
    """
    num_classes = len(class_counts)
    effective_num = [(1.0 - beta ** n) / (1.0 - beta) for n in class_counts]
    raw_weights = [1.0 / en for en in effective_num]
    mean_w = sum(raw_weights) / num_classes
    weights = [w / mean_w for w in raw_weights]
    return torch.tensor(weights, dtype=torch.float32)


def get_loss(config: dict, train_csv_path: str | Path) -> nn.Module:
    """Factoría de función de pérdida según la configuración.

    Lee config['training']['loss']. Para 'weighted_cross_entropy', calcula
    los pesos de clase según config['training']['weight_scheme']:
      - 'balanced':          total / (num_classes * count_i)  [sklearn]
      - 'effective_number':  Cui et al. (2019), suaviza el ratio max/min

    Si config['training']['class_weights'] es una lista, se usa directamente
    ignorando weight_scheme.

    Args:
        config:         Configuración fusionada de load_config().
        train_csv_path: Ruta a train.csv, usada solo si class_weights=='auto'.

    Returns:
        nn.Module listo para llamar con loss(logits, labels).

    Raises:
        ValueError: Si config['training']['loss'] o weight_scheme no están
                    soportados.
    """
    training_cfg = config.get("training", {})
    loss_name: str = training_cfg.get("loss", "weighted_cross_entropy")

    if loss_name == "weighted_cross_entropy":
        class_weights_cfg = training_cfg.get("class_weights", "auto")
        if class_weights_cfg == "auto":
            weight_scheme = training_cfg.get("weight_scheme", "balanced")
            if weight_scheme == "effective_number":
                beta = float(training_cfg.get("beta", 0.999))
                df = pd.read_csv(train_csv_path)
                counts_series = df["label"].value_counts().sort_index()
                counts_list = [int(counts_series.get(i, 1)) for i in range(_NUM_CLASSES)]
                weights = compute_class_weights_effective_number(counts_list, beta=beta)
            elif weight_scheme == "balanced":
                weights = compute_class_weights(train_csv_path)
            else:
                raise ValueError(
                    f"weight_scheme '{weight_scheme}' no soportado. "
                    f"Opciones: ['balanced', 'effective_number']"
                )
        else:
            weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    raise ValueError(
        f"Función de pérdida '{loss_name}' no soportada. "
        f"Opciones: ['weighted_cross_entropy']"
    )
