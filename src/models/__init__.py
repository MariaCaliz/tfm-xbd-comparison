"""Definiciones de los modelos comparados en el TFM."""

from __future__ import annotations

import torch.nn as nn

from src.models.efficientnet import EfficientNetB0DamageClassifier
from src.models.mobilenetv2 import MobileNetV2DamageClassifier
from src.models.resnet50 import ResNet50DamageClassifier
from src.models.vit import ViTDamageClassifier


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
    kwargs = dict(
        num_classes=int(model_cfg.get("num_classes", 4)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    if name == "mobilenetv2":
        return MobileNetV2DamageClassifier(**kwargs)
    elif name == "resnet50":
        return ResNet50DamageClassifier(**kwargs)
    elif name == "efficientnetb0":
        return EfficientNetB0DamageClassifier(**kwargs)
    elif name == "vit":
        return ViTDamageClassifier(**kwargs)
    raise ValueError(
        f"Modelo '{name}' no soportado. Opciones: ['mobilenetv2', 'resnet50', 'efficientnetb0', 'vit']"
    )


__all__ = [
    "MobileNetV2DamageClassifier",
    "ResNet50DamageClassifier",
    "EfficientNetB0DamageClassifier",
    "ViTDamageClassifier",
    "get_model",
]
