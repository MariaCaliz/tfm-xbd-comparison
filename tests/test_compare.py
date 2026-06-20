"""Smoke tests para compare.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.compare import (
    build_comparison_table,
    load_all_metrics,
    plot_confusion_matrices_grid,
    plot_efficiency_tradeoff,
    plot_f1_comparison,
)

MODEL_NAMES = ["mobilenetv2", "resnet50", "efficientnetb0", "vit"]


def _make_fake_metrics(seed: int = 0) -> dict:
    """Genera un dict con la misma estructura que compute_metrics()."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(10, 100, size=(4, 4)).astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    cm_norm = (counts / row_sums).tolist()
    f1_vals = rng.uniform(0.55, 0.90, size=4)
    return {
        "accuracy": round(float(rng.uniform(0.65, 0.90)), 4),
        "precision_macro": round(float(f1_vals.mean()), 4),
        "recall_macro": round(float(f1_vals.mean()), 4),
        "f1_macro": round(float(f1_vals.mean()), 4),
        "precision_per_class": dict(zip(
            ["no-damage", "minor-damage", "major-damage", "destroyed"],
            [round(float(v), 4) for v in f1_vals],
        )),
        "recall_per_class": dict(zip(
            ["no-damage", "minor-damage", "major-damage", "destroyed"],
            [round(float(v), 4) for v in f1_vals],
        )),
        "f1_per_class": dict(zip(
            ["no-damage", "minor-damage", "major-damage", "destroyed"],
            [round(float(v), 4) for v in f1_vals],
        )),
        "support_per_class": {
            "no-damage": 100, "minor-damage": 80,
            "major-damage": 60, "destroyed": 40,
        },
        "confusion_matrix": counts.astype(int).tolist(),
        "confusion_matrix_normalized": cm_norm,
    }


FAKE_EFFICIENCY = {
    "mobilenetv2":   {"model_name": "mobilenetv2",   "total_params": 3_500_000,  "trainable_params": 3_500_000,  "checkpoint_size_mb": 13.4,  "mean_ms": 5.2,  "std_ms": 0.3, "median_ms": 5.1},
    "resnet50":      {"model_name": "resnet50",      "total_params": 25_600_000, "trainable_params": 25_600_000, "checkpoint_size_mb": 97.8,  "mean_ms": 12.1, "std_ms": 0.5, "median_ms": 12.0},
    "efficientnetb0":{"model_name": "efficientnetb0","total_params": 5_300_000,  "trainable_params": 5_300_000,  "checkpoint_size_mb": 20.4,  "mean_ms": 7.3,  "std_ms": 0.4, "median_ms": 7.2},
    "vit":           {"model_name": "vit",           "total_params": 86_600_000, "trainable_params": 86_600_000, "checkpoint_size_mb": 330.2, "mean_ms": 45.6, "std_ms": 1.2, "median_ms": 45.4},
}


@pytest.fixture(scope="module")
def metrics_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("metrics")
    for i, name in enumerate(MODEL_NAMES):
        path = d / f"{name}_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(_make_fake_metrics(seed=i), f)
    return d


@pytest.fixture(scope="module")
def metrics_by_model(metrics_dir):
    return load_all_metrics(metrics_dir, MODEL_NAMES)


# ── load_all_metrics ──────────────────────────────────────────────────────────

def test_load_all_metrics_keys(metrics_by_model):
    assert set(metrics_by_model.keys()) == set(MODEL_NAMES)


def test_load_all_metrics_structure(metrics_by_model):
    for m in metrics_by_model.values():
        assert "accuracy" in m
        assert "f1_macro" in m
        assert "f1_per_class" in m
        assert "confusion_matrix_normalized" in m


# ── build_comparison_table ────────────────────────────────────────────────────

def test_build_comparison_table_shape(metrics_by_model):
    df = build_comparison_table(metrics_by_model, FAKE_EFFICIENCY)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(MODEL_NAMES)


def test_build_comparison_table_columns(metrics_by_model):
    df = build_comparison_table(metrics_by_model, FAKE_EFFICIENCY)
    for col in ("accuracy", "f1_macro", "total_params",
                "checkpoint_size_mb", "inference_mean_ms"):
        assert col in df.columns, f"Falta columna: {col}"


def test_build_comparison_table_index(metrics_by_model):
    df = build_comparison_table(metrics_by_model, FAKE_EFFICIENCY)
    assert set(df.index) == set(MODEL_NAMES)


# ── plot functions ────────────────────────────────────────────────────────────

def test_plot_confusion_matrices_grid_creates_file(metrics_by_model, tmp_path):
    out = tmp_path / "cm_grid.png"
    plot_confusion_matrices_grid(metrics_by_model, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_f1_comparison_creates_file(metrics_by_model, tmp_path):
    out = tmp_path / "f1_comparison.png"
    plot_f1_comparison(metrics_by_model, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_efficiency_tradeoff_creates_file(metrics_by_model, tmp_path):
    out = tmp_path / "tradeoff.png"
    plot_efficiency_tradeoff(metrics_by_model, FAKE_EFFICIENCY, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_efficiency_tradeoff_raises_if_model_missing(metrics_by_model, tmp_path):
    incomplete = {k: v for k, v in FAKE_EFFICIENCY.items() if k != "vit"}
    with pytest.raises(ValueError, match="vit"):
        plot_efficiency_tradeoff(metrics_by_model, incomplete,
                                 save_path=tmp_path / "tradeoff_err.png")


def test_plot_efficiency_tradeoff_raises_if_mean_ms_missing(metrics_by_model, tmp_path):
    bad_eff = {k: {kk: vv for kk, vv in v.items() if kk != "mean_ms"}
               for k, v in FAKE_EFFICIENCY.items()}
    with pytest.raises(ValueError, match="mean_ms"):
        plot_efficiency_tradeoff(metrics_by_model, bad_eff,
                                 save_path=tmp_path / "tradeoff_err2.png")
