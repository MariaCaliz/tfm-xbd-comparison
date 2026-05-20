#!/usr/bin/env python3
# Figura 1: Distribucion de clases por split
# Lee los CSVs de data/splits/ y produce results/figures/class_distribution.png

import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_LABELS = ["No damage", "Minor damage", "Major damage", "Destroyed"]

counts = {}
for split in ["train", "val", "test"]:
    counter = Counter()
    with open(os.path.join(SPLITS_DIR, f"{split}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            counter[row["label_name"]] += 1
    counts[split] = counter

global_counter = Counter()
for c in counts.values():
    global_counter.update(c)
counts["global"] = global_counter

splits = ["global", "train", "val", "test"]
split_labels = ["Global", "Train", "Val", "Test"]

pcts = {}
for s in splits:
    total = sum(counts[s].values())
    pcts[s] = [counts[s][cls] / total * 100 for cls in CLASS_ORDER]

n_classes = len(CLASS_ORDER)
n_splits = len(splits)
x = np.arange(n_classes)
bar_width = 0.18
offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * bar_width

SPLIT_COLORS = ["#2c3e50", "#3498db", "#e67e22", "#27ae60"]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (split, slabel) in enumerate(zip(splits, split_labels)):
    bars = ax.bar(
        x + offsets[i],
        pcts[split],
        width=bar_width,
        label=slabel,
        color=SPLIT_COLORS[i],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for bar, pct in zip(bars, pcts[split]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS, fontsize=12)
ax.set_ylabel("Porcentaje de muestras (%)", fontsize=12)
ax.set_xlabel("Clase de dano", fontsize=12)
ax.set_title("Figura 1: Distribucion de clases por split", fontsize=14, fontweight="bold", pad=14)
ax.legend(title="Split", fontsize=10, title_fontsize=10, loc="upper right")
ax.set_ylim(0, max(max(v) for v in pcts.values()) * 1.18)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "class_distribution.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figura guardada en: {out_path}")
