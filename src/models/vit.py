"""Wrapper de ViT-Base/16 para clasificación de daños en xBD."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ViTDamageClassifier(nn.Module):
    """ViT-Base/16 (timm) con cabeza de clasificación adaptada a 4 clases de daño."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # timm reemplaza la cabeza directamente si se pasa num_classes
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Congela todas las capas excepto la cabeza de clasificación."""
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("head")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Descongela los últimos N bloques Transformer + norm + head.

        ViT-Base tiene 12 bloques en self.model.blocks.
        Para n=2 (config por defecto): descongela blocks[-2:] y norm.

        Args:
            n: Número de bloques Transformer finales a descongelar.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        for block in self.model.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.model, "norm"):
            for param in self.model.norm.parameters():
                param.requires_grad = True
