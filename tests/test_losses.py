"""Smoke tests para losses.py — función compute_class_weights_effective_number."""

import torch
import pytest

from src.training.losses import compute_class_weights_effective_number

# Conteos reales del train set (no-damage es la clase mayoritaria)
COUNTS = [164_379, 17_016, 16_550, 15_172]

# Pesos balanced replicados inline para la comparación de ratio (sin leer CSV)
_TOTAL = sum(COUNTS)
_BALANCED = [_TOTAL / (4 * c) for c in COUNTS]
_BALANCED_RATIO = max(_BALANCED) / min(_BALANCED)


def test_returns_tensor_of_correct_shape():
    w = compute_class_weights_effective_number(COUNTS)
    assert isinstance(w, torch.Tensor)
    assert w.shape == (4,)


def test_all_weights_positive():
    w = compute_class_weights_effective_number(COUNTS)
    assert (w > 0).all()


def test_minority_heavier_than_majority():
    # destroyed (idx 3) es la clase minoritaria; no-damage (idx 0) es la mayoritaria
    w = compute_class_weights_effective_number(COUNTS, beta=0.99999)
    assert w[3] > w[0], f"Se esperaba w[3]>{w[0]:.4f}, got w[3]={w[3]:.4f}"


def test_ratio_lower_than_balanced():
    # Con beta=0.99999, el ratio debe estar entre 1x y el de balanced (10.83x)
    w = compute_class_weights_effective_number(COUNTS, beta=0.99999)
    eff_ratio = (w.max() / w.min()).item()
    assert 1.0 < eff_ratio < _BALANCED_RATIO, (
        f"Ratio eff_num ({eff_ratio:.2f}x) debe estar en (1, {_BALANCED_RATIO:.2f}x)"
    )


def test_mean_weight_is_one():
    w = compute_class_weights_effective_number(COUNTS, beta=0.999)
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_higher_beta_gives_lower_ratio():
    # Mayor beta → corrección más suave → ratio más cercano a 1
    w_low = compute_class_weights_effective_number(COUNTS, beta=0.9999)
    w_high = compute_class_weights_effective_number(COUNTS, beta=0.99999)
    ratio_low = (w_low.max() / w_low.min()).item()
    ratio_high = (w_high.max() / w_high.min()).item()
    assert ratio_low < ratio_high
