"""Definiciones de los modelos comparados en el TFM."""

from __future__ import annotations

import torch.nn as nn

from src.models.mobilenetv2 import MobileNetV2DamageClassifier


def get_model(config: dict) -> nn.Module:
    """Factoría que devuelve el modelo según config['model']['name'].

    Args:
        config: Configuración fusionada de load_config().

    Returns:
        Modelo nn.Module listo para mover al device y entrenar.

    Raises:
        ValueError: Si el nombre de modelo no está soportado.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "")
    if name == "mobilenetv2":
        return MobileNetV2DamageClassifier(
            num_classes=int(model_cfg.get("num_classes", 4)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Modelo '{name}' no soportado. Opciones: ['mobilenetv2']")


__all__ = ["MobileNetV2DamageClassifier", "get_model"]
