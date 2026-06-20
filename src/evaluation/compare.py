"""Comparativa entre modelos: carga de métricas, tablas y figuras."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LABEL_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
_LABEL_SHORT = ["No dmg", "Minor", "Major", "Destroyed"]
_MODEL_COLORS = ["#2c3e50", "#3498db", "#e67e22", "#27ae60"]

_FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def _build_color_map(model_names: list[str]) -> dict[str, str]:
    """Asigna un color fijo a cada modelo según su posición en model_names.

    El mapeo es estable: el mismo nombre siempre obtiene el mismo color
    independientemente del orden de iteración posterior (ej. tras ordenar por F1).
    """
    return {name: _MODEL_COLORS[i % len(_MODEL_COLORS)]
            for i, name in enumerate(model_names)}


def load_all_metrics(metrics_dir: str | Path, model_names: list[str]) -> dict:
    """Lee los JSON de métricas por modelo y los devuelve en un dict.

    Espera un archivo {model_name}_metrics.json por modelo, con la
    estructura generada por compute_metrics().

    Args:
        metrics_dir: Directorio que contiene los JSON de métricas.
        model_names: Lista de nombres de modelo a cargar.

    Returns:
        Dict {model_name: metrics_dict}.
    """
    metrics_dir = Path(metrics_dir)
    result = {}
    for name in model_names:
        path = metrics_dir / f"{name}_metrics.json"
        with path.open("r", encoding="utf-8") as f:
            result[name] = json.load(f)
    return result


def build_comparison_table(
    metrics_by_model: dict,
    efficiency_by_model: dict,
) -> pd.DataFrame:
    """Construye una tabla comparativa uniendo rendimiento y eficiencia.

    Args:
        metrics_by_model:    Dict {model_name: metrics_dict} de compute_metrics().
        efficiency_by_model: Dict {model_name: efficiency_dict} de
                             compute_efficiency_metrics().

    Returns:
        DataFrame indexado por nombre de modelo con columnas de rendimiento
        (accuracy, f1_macro, f1 por clase) y eficiencia (params, tamaño, ms).
    """
    rows = []
    for model_name, m in metrics_by_model.items():
        eff = efficiency_by_model.get(model_name, {})
        f1_cls = m.get("f1_per_class", {})
        row = {
            "model": model_name,
            "accuracy": round(m["accuracy"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "f1_no_damage": round(f1_cls.get("no-damage", float("nan")), 4),
            "f1_minor_damage": round(f1_cls.get("minor-damage", float("nan")), 4),
            "f1_major_damage": round(f1_cls.get("major-damage", float("nan")), 4),
            "f1_destroyed": round(f1_cls.get("destroyed", float("nan")), 4),
            "total_params": eff.get("total_params"),
            "checkpoint_size_mb": eff.get("checkpoint_size_mb"),
            "inference_mean_ms": eff.get("mean_ms"),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_confusion_matrices_grid(
    metrics_by_model: dict,
    save_path: str | Path | None = None,
) -> None:
    """Figura 2×2 con las matrices de confusión normalizadas de los 4 modelos.

    Args:
        metrics_by_model: Dict {model_name: metrics_dict} de compute_metrics().
        save_path:        Ruta de salida. Por defecto results/figures/confusion_matrices.png.
    """
    if save_path is None:
        save_path = _FIGURES_DIR / "confusion_matrices.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(metrics_by_model.keys())
    ncols = 2
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for idx, model_name in enumerate(models):
        ax = axes_flat[idx]
        cm = np.array(metrics_by_model[model_name]["confusion_matrix_normalized"])
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(_LABEL_SHORT)))
        ax.set_yticks(range(len(_LABEL_SHORT)))
        ax.set_xticklabels(_LABEL_SHORT, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(_LABEL_SHORT, fontsize=8)
        ax.set_xlabel("Predicción", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        for i in range(len(_LABEL_SHORT)):
            for j in range(len(_LABEL_SHORT)):
                val = cm[i, j]
                text_color = "white" if val > 0.6 else "#222222"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Matrices de confusión normalizadas", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en: {save_path}")


def plot_f1_comparison(
    metrics_by_model: dict,
    save_path: str | Path | None = None,
) -> None:
    """Gráfico de barras con el F1 macro de cada modelo, ordenado de mayor a menor.

    Args:
        metrics_by_model: Dict {model_name: metrics_dict} de compute_metrics().
        save_path:        Ruta de salida. Por defecto results/figures/f1_comparison.png.
    """
    if save_path is None:
        save_path = _FIGURES_DIR / "f1_comparison.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = _build_color_map(list(metrics_by_model.keys()))
    items = sorted(
        [(name, m["f1_macro"]) for name, m in metrics_by_model.items()],
        key=lambda t: t[1],
        reverse=True,
    )
    names = [t[0] for t in items]
    values = [t[1] for t in items]
    colors = [color_map[name] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, color="#333333",
        )

    ax.set_ylabel("F1 macro", fontsize=12)
    ax.set_title("Comparativa de F1 macro por modelo", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(values) * 1.15))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en: {save_path}")


def plot_efficiency_tradeoff(
    metrics_by_model: dict,
    efficiency_by_model: dict,
    save_path: str | Path | None = None,
) -> None:
    """Scatter trade-off rendimiento/coste: eje X = latencia (ms), eje Y = F1 macro.

    Figura central del Capítulo 5: cada punto representa un modelo con su nombre
    etiquetado. Permite comparar visualmente coste computacional vs calidad.

    Args:
        metrics_by_model:    Dict {model_name: metrics_dict} de compute_metrics().
        efficiency_by_model: Dict {model_name: efficiency_dict} de
                             compute_efficiency_metrics().
        save_path:           Ruta de salida. Por defecto results/figures/efficiency_tradeoff.png.
    """
    if save_path is None:
        save_path = _FIGURES_DIR / "efficiency_tradeoff.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [m for m in metrics_by_model if m not in efficiency_by_model]
    if missing:
        raise ValueError(
            f"Faltan datos de eficiencia para: {missing}. "
            f"Modelos disponibles en efficiency_by_model: {list(efficiency_by_model.keys())}"
        )
    no_mean_ms = [m for m in metrics_by_model if "mean_ms" not in efficiency_by_model[m]]
    if no_mean_ms:
        raise ValueError(
            f"La clave 'mean_ms' no está presente en los datos de eficiencia de: {no_mean_ms}"
        )

    color_map = _build_color_map(list(metrics_by_model.keys()))
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in metrics_by_model:
        eff = efficiency_by_model[model_name]
        x = eff["mean_ms"]
        y = metrics_by_model[model_name]["f1_macro"]
        color = color_map[model_name]
        ax.scatter(x, y, color=color, s=130, zorder=4,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            model_name,
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=10,
            color=color,
        )

    ax.set_xlabel("Tiempo de inferencia media (ms / imagen)", fontsize=12)
    ax.set_ylabel("F1 macro", fontsize=12)
    ax.set_title("Trade-off rendimiento / coste computacional", fontsize=13,
                 fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en: {save_path}")
