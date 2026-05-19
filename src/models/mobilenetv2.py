"""Wrapper de MobileNetV2 para clasificación de daños en xBD.

Adapta el modelo preentrenado en ImageNet (disponible en torchvision)
a la tarea de clasificación en 4 niveles de daño, reemplazando la
cabeza de clasificación y permitiendo congelar/descongelar capas para
una estrategia de fine-tuning en dos etapas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2DamageClassifier(nn.Module):
    """MobileNetV2 con cabeza de clasificación adaptada a 4 clases de daño."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            num_classes: Número de clases de salida. Por defecto 4 (niveles de daño xBD).
            pretrained: Si True, carga pesos preentrenados en ImageNet.
            dropout: Tasa de dropout en la cabeza de clasificación.
        """
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Congela todas las capas excepto la cabeza de clasificación."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")

    def unfreeze_all(self) -> None:
        """Descongela todas las capas del modelo."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Descongela solo los últimos N bloques del backbone + la cabeza.

        MobileNetV2 organiza el backbone como features = Sequential(...18 bloques...).
        Esta función deja entrenables los últimos N bloques y la cabeza,
        y congela el resto.

        Args:
            n: Número de bloques finales a descongelar.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        features = self.backbone.features
        for block in features[-n:]:
            for param in block.parameters():
                param.requires_grad = True
