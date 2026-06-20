"""Smoke tests para efficiency.py."""

import pytest
import torch

from src.evaluation.efficiency import (
    benchmark_inference_time,
    checkpoint_size_mb,
    compute_efficiency_metrics,
    count_parameters,
)
from src.models.mobilenetv2 import MobileNetV2DamageClassifier


@pytest.fixture(scope="module")
def model():
    return MobileNetV2DamageClassifier(num_classes=4, pretrained=False, dropout=0.2)


def test_count_parameters_keys(model):
    result = count_parameters(model)
    assert "total_params" in result
    assert "trainable_params" in result


def test_count_parameters_values(model):
    result = count_parameters(model)
    assert result["total_params"] > 0
    assert result["trainable_params"] > 0
    assert result["trainable_params"] <= result["total_params"]


def test_checkpoint_size_mb_positive(model):
    size = checkpoint_size_mb(model)
    assert isinstance(size, float)
    assert size > 0.0


def test_checkpoint_size_mb_from_path(model, tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    size_from_path = checkpoint_size_mb(path)
    size_from_model = checkpoint_size_mb(model)
    assert size_from_path > 0.0
    assert abs(size_from_path - size_from_model) < 0.01


def test_benchmark_inference_time_keys(model):
    result = benchmark_inference_time(model, n_warmup=1, n_runs=5)
    assert "mean_ms" in result
    assert "std_ms" in result
    assert "median_ms" in result


def test_benchmark_inference_time_values(model):
    result = benchmark_inference_time(model, n_warmup=1, n_runs=5)
    assert result["mean_ms"] > 0.0
    assert result["median_ms"] > 0.0
    assert result["std_ms"] >= 0.0


def test_compute_efficiency_metrics_shape(model):
    result = compute_efficiency_metrics(
        model, "mobilenetv2_test", n_warmup=1, n_runs=5
    )
    assert result["model_name"] == "mobilenetv2_test"
    assert result["total_params"] > 0
    assert result["checkpoint_size_mb"] > 0.0
    assert result["mean_ms"] > 0.0
