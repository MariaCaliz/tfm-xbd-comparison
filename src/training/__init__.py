"""Módulos de entrenamiento: pérdidas y trainer."""

from src.training.losses import compute_class_weights, get_loss
from src.training.trainer import Trainer

__all__ = ["compute_class_weights", "get_loss", "Trainer"]
