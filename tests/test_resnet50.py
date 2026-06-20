"""Smoke tests para ResNet50DamageClassifier."""

import torch
import pytest

from src.models.resnet50 import ResNet50DamageClassifier

NUM_CLASSES = 4
BATCH = 2
INPUT = torch.randn(BATCH, 3, 224, 224)


@pytest.fixture(scope="module")
def model():
    return ResNet50DamageClassifier(num_classes=NUM_CLASSES, pretrained=False, dropout=0.2)


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_forward(model):
    model.eval()
    with torch.no_grad():
        out = model(INPUT)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_freeze_backbone(model):
    model.freeze_backbone()
    trainable = count_trainable(model)
    # Solo la cabeza fc debe quedar entrenable
    assert trainable > 0
    # La cabeza es Linear(2048, 4) + Dropout → 2048*4 + 4 = 8196 parámetros entrenables
    assert trainable < 10_000


def test_unfreeze_last_n_blocks_increases_trainable(model):
    model.freeze_backbone()
    frozen_trainable = count_trainable(model)
    model.unfreeze_last_n_blocks(2)  # valor del config resnet50.yaml
    unfrozen_trainable = count_trainable(model)
    assert unfrozen_trainable > frozen_trainable
