"""Wrapper de EfficientNet-B0 para clasificación de daños en xBD."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0DamageClassifier(nn.Module):
    """EfficientNet-B0 con cabeza de clasificación adaptada a 4 clases de daño."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Congela todas las capas excepto el classifier nuevo."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Descongela los últimos N elementos de features + el classifier.

        EfficientNet-B0: features tiene 9 elementos (stem + 7 bloques MBConv + head conv).
        Para n=3 (config por defecto): descongela features[-3:].

        Args:
            n: Número de elementos finales de features a descongelar.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        features = self.backbone.features
        for block in features[-n:]:
            for param in block.parameters():
                param.requires_grad = True
