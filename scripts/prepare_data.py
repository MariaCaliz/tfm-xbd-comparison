#!/usr/bin/env python3
"""Extrae crops del dataset xBD y genera las particiones de splits.

Uso:
  python scripts/prepare_data.py \\
      --config configs/mobilenetv2.yaml \\
      --raw-dir data/raw/downloads \\
      --processed-dir data/processed \\
      --splits-dir data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parse_xbd import parse_xbd
from src.data.splits import make_splits
from src.utils.config import load_config
from src.utils.seed import set_seed, setup_logger


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae crops de xBD y genera las particiones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Ruta al YAML de configuración.")
    parser.add_argument("--raw-dir", required=True,
                        help="Directorio raíz del dataset xBD (contiene train/, tier3/).")
    parser.add_argument("--processed-dir", required=True,
                        help="Salida de parse_xbd (crops + metadata.csv).")
    parser.add_argument("--splits-dir", required=True,
                        help="Salida de make_splits (train.csv, val.csv, test.csv).")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Saltar parse_xbd y usar metadata.csv existente.")
    parser.add_argument("--events-filter", nargs="+", metavar="EVENT",
                        help="Procesar solo estos eventos (útil para smoke test).")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("experiment", {}).get("seed", 42))
    logger = setup_logger("prepare_data")

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)

    # ── parse_xbd ─────────────────────────────────────────────────────────────
    if not args.skip_parse:
        logger.info("Iniciando extracción de crops (parse_xbd)...")
        df = parse_xbd(
            config=config,
            raw_dir=args.raw_dir,
            processed_dir=processed_dir,
            events_filter=args.events_filter,
        )
        logger.info("parse_xbd finalizado: %d crops extraídos", len(df))
    else:
        logger.info("--skip-parse activo: usando metadata.csv en %s", processed_dir)

    # ── make_splits ───────────────────────────────────────────────────────────
    logger.info("Generando particiones (make_splits)...")
    split_dfs = make_splits(config=config, processed_dir=processed_dir, splits_dir=splits_dir)

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = sum(len(v) for v in split_dfs.values())
    disk_gb = _dir_size_gb(processed_dir) if processed_dir.exists() else 0.0

    logger.info("=" * 56)
    logger.info("RESUMEN")
    logger.info("  Total crops:       %d", total)
    logger.info("  Disco processed/:  %.2f GB", disk_gb)
    for split, sdf in split_dfs.items():
        pct = len(sdf) / total * 100 if total else 0
        logger.info("  %s  %d muestras (%.1f%%)", split.ljust(5), len(sdf), pct)
    logger.info("Archivos generados:")
    for fname in ["metadata.csv", "stats.json"]:
        p = processed_dir / fname
        logger.info("  %s — %s", p, "OK" if p.exists() else "FALTA")
    for fname in ["train.csv", "val.csv", "test.csv", "split_assignments.json"]:
        p = splits_dir / fname
        logger.info("  %s — %s", p, "OK" if p.exists() else "FALTA")
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
