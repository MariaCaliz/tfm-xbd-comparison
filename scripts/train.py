#!/usr/bin/env python3
"""Entrena un modelo de clasificación de daños en xBD.

Uso:
  python scripts/train.py --config configs/mobilenetv2.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import XBDDataset
from src.data.transforms import get_transforms
from src.evaluation.metrics import compute_metrics, save_metrics
from src.models import get_model
from src.training.losses import get_loss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import count_parameters, get_device, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de clasificación de daños en xBD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Ruta al YAML del experimento.")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directorio de crops (data/processed/).")
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Directorio de splits (data/splits/).")
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de resultados (default: results/{experiment_name}).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Dispositivo de cómputo.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limitar muestras de train a N (smoke test).")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config.get("experiment", {}).get("name", "experiment")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / exp_name

    set_seed(config.get("experiment", {}).get("seed", 42))
    device = get_device(args.device)
    logger = setup_logger("train", log_file=output_dir / "train.log")

    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    logger.info("Experimento : %s", exp_name)
    logger.info("Config      : %s", args.config)
    logger.info("Device      : %s", device)

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = XBDDataset(train_csv, processed_dir, get_transforms("train", config))
    val_ds = XBDDataset(val_csv, processed_dir, get_transforms("val", config))
    test_ds = XBDDataset(test_csv, processed_dir, get_transforms("test", config))

    if args.max_samples is not None:
        if args.max_samples < len(train_ds):
            train_ds.df = train_ds.df.head(args.max_samples).reset_index(drop=True)
        # Limitar val/test proporcional a max_samples para que el smoke test sea rápido
        val_limit = max(64, args.max_samples // 3)
        if val_limit < len(val_ds):
            val_ds.df = val_ds.df.head(val_limit).reset_index(drop=True)
        if val_limit < len(test_ds):
            test_ds.df = test_ds.df.head(val_limit).reset_index(drop=True)
        logger.info(
            "Smoke test: train=%d | val=%d | test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )

    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    # pin_memory no funciona en MPS (ni siquiera en CPU cuando MPS está disponible)
    _mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    pin_memory = bool(data_cfg.get("pin_memory", True)) and not _mps_available

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.get_weighted_sampler(),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # ── Modelo ─────────────────────────────────────────────────────────────────
    model = get_model(config)
    total_params, trainable = count_parameters(model)
    logger.info(
        "Modelo: %s | Params: %s total, %s entrenables al inicio",
        config["model"]["name"], f"{total_params:,}", f"{trainable:,}",
    )

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    criterion = get_loss(config, train_csv)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    t0 = time.perf_counter()
    best = trainer.train_full()
    elapsed = time.perf_counter() - t0

    # ── Evaluación en test ─────────────────────────────────────────────────────
    logger.info("Evaluando en test set...")
    model.eval()
    model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(model(images).argmax(dim=1).cpu().tolist())
    test_metrics = compute_metrics(y_true, y_pred)
    save_metrics(test_metrics, output_dir / "test_metrics.json")

    # ── Resumen ────────────────────────────────────────────────────────────────
    val_f1 = best.get("val_metrics", {}).get("f1_macro", 0.0)
    logger.info("=" * 56)
    logger.info("RESUMEN FINAL")
    logger.info("  Mejor epoch:    Stage %s, epoch %s", best.get("stage"), best.get("epoch"))
    logger.info("  Val  F1-macro:  %.4f", val_f1)
    logger.info("  Test F1-macro:  %.4f", test_metrics["f1_macro"])
    logger.info("  Test accuracy:  %.4f", test_metrics["accuracy"])
    logger.info("  Tiempo total:   %.1f min", elapsed / 60)
    logger.info("  Outputs:        %s", output_dir)
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
