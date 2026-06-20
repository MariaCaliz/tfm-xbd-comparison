"""Wrapper de ResNet50 para clasificación de daños en xBD."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50DamageClassifier(nn.Module):
    """ResNet50 con cabeza de clasificación adaptada a 4 clases de daño."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        if dropout > 0:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Congela todas las capas excepto la capa fc final."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Descongela los últimos N grupos de capas del backbone + fc.

        ResNet50 tiene 4 grupos: layer1, layer2, layer3, layer4.
        Para n=2 (config por defecto): descongela layer4 y layer3.

        Args:
            n: Número de grupos a descongelar (contando desde layer4 hacia atrás).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        layers = [
            self.backbone.layer4,
            self.backbone.layer3,
            self.backbone.layer2,
            self.backbone.layer1,
        ]
        for layer in layers[:n]:
            for param in layer.parameters():
                param.requires_grad = True
