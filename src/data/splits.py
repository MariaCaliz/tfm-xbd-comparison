"""Particionado del dataset por evento de desastre.

Implementa una asignación multi-objetivo que equilibra simultáneamente:
  (1) el número total de crops por split (objetivo 70/15/15), y
  (2) la distribución de clases dentro de cada split (debe reflejar
      la distribución global del dataset).

Motivación:
  Los eventos de tipo tsunami (palu-tsunami, sunda-tsunami) tienen
  prácticamente cero muestras de 'minor-damage'. Un greedy que solo
  optimiza tamaño total puede agrupar estos eventos en el mismo split,
  dejándolo sin representación de clases minoritarias — lo que hace
  indefendible el F1-macro resultante (Benson y Ecker, 2020).

Algoritmo (greedy multi-objetivo + búsqueda local):
  Fase 1 — Greedy:
    Ordena los eventos de mayor a menor número de crops (tiebreaker
    alfabético para determinismo). Para cada evento, elige el split
    que minimiza:
      score = alpha * fill_ratio_after + (1 - alpha) * class_dist_after
    donde:
      fill_ratio_after = (size_actual + crops_evento) / size_objetivo
      class_dist_after = media de |prop_clase_split - prop_clase_global|
    Un fill_ratio bajo significa que el split está "hambriento" de
    crops; una class_dist baja significa que el split está bien
    representado en todas las clases.

  Fase 2 — Búsqueda local (hill climbing):
    Intenta intercambiar pares de eventos entre splits. Acepta el
    intercambio que más reduzca el score total (greedy best-swap).
    Repite hasta converger o alcanzar max_iterations.
    La función de score incluye una penalización dura si algún split
    queda con menos de MIN_SAMPLES_PER_CLASS muestras de alguna clase
    (umbral para que F1-macro sea estadísticamente defendible).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ── Parámetros ────────────────────────────────────────────────────────────────

_TARGET_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
_SEED = 42                    # El algoritmo es determinista; seed queda documentada.
_ALPHA_GREEDY = 0.9           # Greedy: fill_ratio domina → respeta 70/15/15.
_ALPHA_LOCAL = 0.1            # Búsqueda local: clase domina (90%) para refinar
                              # distribución sin romper el tamaño.
_MIN_SAMPLES_PER_CLASS = 500  # Umbral duro: mínimo de muestras por clase en c/split.
_VIOLATION_PENALTY = 10.0     # Penalización por violar el umbral (domina el score).
_SIZE_TOLERANCE = 0.07        # Búsqueda local: máx. desviación ±7pp del objetivo.
_LABEL_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
_MAX_CLASS_DEVIATION = 0.10   # Búsqueda local: umbral de desviación de clase (10pp).


# ── Funciones de score ────────────────────────────────────────────────────────

def _split_score(
    size: int,
    class_counts: dict[str, int],
    target_size: float,
    global_props: dict[str, float],
    alpha: float,
    include_violation: bool = False,
    deviation: bool = False,
) -> float:
    """Score de un único split (menor = mejor).

    Combina la desviación de tamaño y la desviación de distribución de
    clases respecto a la distribución global.

    deviation=False (greedy): fill = size/target → favorece splits
      "hambrientos" (fill < 1) durante la asignación.
    deviation=True (búsqueda local): fill = |size/target − 1| → mide
      desviación simétrica respecto al objetivo; necesario para evaluar
      correctamente si un intercambio mejora o empeora el balance.
    """
    raw_fill = size / target_size if target_size > 0 else 0.0
    fill = abs(raw_fill - 1.0) if deviation else raw_fill

    if size > 0:
        class_dev = sum(
            abs(class_counts.get(c, 0) / size - global_props[c])
            for c in _LABEL_NAMES
        ) / len(_LABEL_NAMES)
    else:
        class_dev = 0.0

    violation = 0.0
    if include_violation:
        violation = sum(
            (_MIN_SAMPLES_PER_CLASS - class_counts.get(c, 0)) / _MIN_SAMPLES_PER_CLASS
            for c in _LABEL_NAMES
            if class_counts.get(c, 0) < _MIN_SAMPLES_PER_CLASS
        )

    return alpha * fill + (1 - alpha) * class_dev + _VIOLATION_PENALTY * violation


# ── Fase 1: greedy multi-objetivo ─────────────────────────────────────────────

def _greedy_assign(
    event_counts: dict[str, int],
    event_class_counts: dict[str, dict[str, int]],
    global_props: dict[str, float],
    total: int,
    targets: dict[str, float],
    alpha: float,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, dict[str, int]]]:
    split_names = list(_TARGET_RATIOS.keys())
    events_sorted = sorted(event_counts, key=lambda e: (-event_counts[e], e))

    sizes: dict[str, int] = {s: 0 for s in split_names}
    cc: dict[str, dict[str, int]] = {s: {c: 0 for c in _LABEL_NAMES} for s in split_names}
    assignment: dict[str, list[str]] = {s: [] for s in split_names}

    for event in events_sorted:
        ev_n = event_counts[event]
        ev_cls = event_class_counts[event]

        best_split = min(
            split_names,
            key=lambda s, _ev_n=ev_n, _ev_cls=ev_cls: _split_score(
                sizes[s] + _ev_n,
                {c: cc[s][c] + _ev_cls.get(c, 0) for c in _LABEL_NAMES},
                targets[s],
                global_props,
                alpha,
                include_violation=False,  # violaciones solo en búsqueda local
            ),
        )
        assignment[best_split].append(event)
        sizes[best_split] += ev_n
        for c in _LABEL_NAMES:
            cc[best_split][c] += ev_cls.get(c, 0)

    return assignment, sizes, cc


# ── Fase 2: búsqueda local (hill climbing por intercambio de pares) ───────────

def _local_search(
    assignment: dict[str, list[str]],
    sizes: dict[str, int],
    cc: dict[str, dict[str, int]],
    event_counts: dict[str, int],
    event_class_counts: dict[str, dict[str, int]],
    global_props: dict[str, float],
    targets: dict[str, float],
    alpha: float,
    total: int = 0,
    max_iterations: int = 500,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, dict[str, int]]]:
    """Intercambia pares de eventos entre splits mientras mejore el score total."""
    split_names = list(_TARGET_RATIOS.keys())

    for _ in range(max_iterations):
        best_delta = 0.0
        best_swap: tuple | None = None

        for i, s1 in enumerate(split_names):
            for s2 in split_names[i + 1:]:
                for e1 in assignment[s1]:
                    for e2 in assignment[s2]:
                        n1, n2 = event_counts[e1], event_counts[e2]
                        cls1, cls2 = event_class_counts[e1], event_class_counts[e2]

                        new_size_s1 = sizes[s1] - n1 + n2
                        new_size_s2 = sizes[s2] - n2 + n1

                        if (abs(new_size_s1 / total - _TARGET_RATIOS[s1]) > _SIZE_TOLERANCE or
                                abs(new_size_s2 / total - _TARGET_RATIOS[s2]) > _SIZE_TOLERANCE):
                            continue

                        new_cc_s1 = {
                            c: cc[s1][c] - cls1.get(c, 0) + cls2.get(c, 0)
                            for c in _LABEL_NAMES
                        }
                        new_cc_s2 = {
                            c: cc[s2][c] - cls2.get(c, 0) + cls1.get(c, 0)
                            for c in _LABEL_NAMES
                        }

                        old_contrib = (
                            _split_score(sizes[s1], cc[s1], targets[s1], global_props, alpha,
                                         include_violation=True, deviation=True)
                            + _split_score(sizes[s2], cc[s2], targets[s2], global_props, alpha,
                                           include_violation=True, deviation=True)
                        )
                        new_contrib = (
                            _split_score(new_size_s1, new_cc_s1, targets[s1], global_props, alpha,
                                         include_violation=True, deviation=True)
                            + _split_score(new_size_s2, new_cc_s2, targets[s2], global_props, alpha,
                                           include_violation=True, deviation=True)
                        )
                        delta = new_contrib - old_contrib

                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (s1, s2, e1, e2, new_size_s1, new_size_s2, new_cc_s1, new_cc_s2)

        if best_swap is None:
            break

        s1, s2, e1, e2, ns1, ns2, nc1, nc2 = best_swap
        assignment[s1].remove(e1)
        assignment[s2].remove(e2)
        assignment[s1].append(e2)
        assignment[s2].append(e1)
        sizes[s1], sizes[s2] = ns1, ns2
        cc[s1], cc[s2] = nc1, nc2

    return assignment, sizes, cc


# ── Función pública ───────────────────────────────────────────────────────────

def make_splits(
    config: dict,
    processed_dir: str | Path,
    splits_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Particiona metadata.csv por evento y escribe los CSVs de split.

    Lee data/processed/metadata.csv, calcula la distribución global de
    clases, aplica el algoritmo greedy multi-objetivo y refina con
    búsqueda local. Escribe train.csv, val.csv, test.csv y
    split_assignments.json (con trazabilidad completa para la memoria).

    Args:
        config:        Configuración fusionada de load_config().
        processed_dir: Directorio con metadata.csv (data/processed/).
        splits_dir:    Directorio de salida de los CSVs (data/splits/).

    Returns:
        Dict {'train': df, 'val': df, 'test': df}.
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "metadata.csv")
    total = len(df)

    # Distribución global de clases
    global_props: dict[str, float] = {
        c: (df["label_name"] == c).sum() / total for c in _LABEL_NAMES
    }

    # Estadísticas por evento
    event_counts: dict[str, int] = df.groupby("event").size().to_dict()
    event_class_counts: dict[str, dict[str, int]] = {
        event: group.groupby("label_name").size().reindex(_LABEL_NAMES, fill_value=0).to_dict()
        for event, group in df.groupby("event")
    }

    targets: dict[str, float] = {s: total * r for s, r in _TARGET_RATIOS.items()}

    # ── Fase 1: greedy (tamaño dominante) ─────────────────────────────────────
    assignment, sizes, cc = _greedy_assign(
        event_counts, event_class_counts, global_props, total, targets, _ALPHA_GREEDY
    )

    # ── Fase 2: búsqueda local ─────────────────────────────────────────────────
    # Se activa si hay violaciones del umbral mínimo O si la distribución de
    # clases en algún split se desvía > MAX_CLASS_DEVIATION del global.
    # Solo acepta intercambios que mantengan ratios dentro de ±SIZE_TOLERANCE.
    has_violations = any(
        cc[split].get(c, 0) < _MIN_SAMPLES_PER_CLASS
        for split in _TARGET_RATIOS
        for c in _LABEL_NAMES
    )
    has_class_skew = any(
        abs(cc[split].get(c, 0) / sizes[split] - global_props[c]) > _MAX_CLASS_DEVIATION
        for split in _TARGET_RATIOS
        for c in _LABEL_NAMES
        if sizes[split] > 0
    )
    if has_violations or has_class_skew:
        assignment, sizes, cc = _local_search(
            assignment, sizes, cc,
            event_counts, event_class_counts, global_props, targets, _ALPHA_LOCAL,
            total=total,
        )

    # ── Construir DataFrames y escribir CSVs ───────────────────────────────────
    split_dfs: dict[str, pd.DataFrame] = {}
    for split, events in assignment.items():
        mask = df["event"].isin(events)
        split_df = df[mask].reset_index(drop=True)
        split_df.to_csv(splits_dir / f"{split}.csv", index=False)
        split_dfs[split] = split_df

    # ── split_assignments.json ─────────────────────────────────────────────────
    class_dist: dict[str, dict] = {}
    totals: dict[str, int] = {}
    for split, sdf in split_dfs.items():
        totals[split] = len(sdf)
        class_dist[split] = (
            sdf.groupby("label_name").size()
            .reindex(_LABEL_NAMES, fill_value=0)
            .to_dict()
        )

    global_props_rounded = {c: round(global_props[c], 4) for c in _LABEL_NAMES}
    class_dev_pp = {
        s: {
            c: round((class_dist[s][c] / totals[s] - global_props[c]) * 100, 1)
            if totals[s] > 0 else 0.0
            for c in _LABEL_NAMES
        }
        for s in _TARGET_RATIOS
    }
    max_dev_pp = round(
        max(abs(v) for s in class_dev_pp.values() for v in s.values()), 1
    )

    assignments_doc = {
        "seed": _SEED,
        "algorithm": "greedy_multiobjective_with_local_search",
        "alpha_greedy": _ALPHA_GREEDY,
        "alpha_local_search": _ALPHA_LOCAL,
        "min_samples_per_class": _MIN_SAMPLES_PER_CLASS,
        "target_ratios": _TARGET_RATIOS,
        "actual_ratios": {s: round(totals[s] / total, 4) for s in _TARGET_RATIOS},
        "train": sorted(assignment["train"]),
        "val": sorted(assignment["val"]),
        "test": sorted(assignment["test"]),
        "totals_per_split": totals,
        "class_distribution_per_split": class_dist,
        "global_class_proportions": global_props_rounded,
        "class_deviation_from_global_pp": class_dev_pp,
        "max_class_deviation_pp": max_dev_pp,
    }

    with open(splits_dir / "split_assignments.json", "w", encoding="utf-8") as f:
        json.dump(assignments_doc, f, indent=2, ensure_ascii=False)

    return split_dfs
