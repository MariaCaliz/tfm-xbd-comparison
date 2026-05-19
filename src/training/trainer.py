"""Entrenamiento en dos etapas (fine-tuning escalonado) para clasificación xBD."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


class _EarlyStopping:
    """Early stopping por val_f1_macro (modo max). Se reinicia entre stages."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best = -1.0
        self._wait = 0

    def step(self, val: float) -> bool:
        """Devuelve True si se debe detener el entrenamiento."""
        if val > self._best:
            self._best = val
            self._wait = 0
            return False
        self._wait += 1
        return self._wait >= self.patience

    def reset(self) -> None:
        self._best = -1.0
        self._wait = 0


class Trainer:
    """Orquesta el fine-tuning en dos etapas sobre un clasificador xBD.

    El modelo debe implementar freeze_backbone() y unfreeze_last_n_blocks(n).
    El mejor checkpoint global (a través de ambos stages) se guarda en
    output_dir/best_model.pt según val_f1_macro.

    Args:
        model:        Modelo nn.Module con interfaz de freeze/unfreeze.
        train_loader: DataLoader del split de entrenamiento.
        val_loader:   DataLoader del split de validación.
        criterion:    Función de pérdida (ej. CrossEntropyLoss con pesos).
        config:       Configuración fusionada de load_config().
        device:       Dispositivo de cómputo (cpu / cuda / mps).
        output_dir:   Directorio de salida para checkpoints e historiales.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: dict,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._tcfg = config.get("training", {})
        patience = self._tcfg.get("early_stopping", {}).get("patience", 5)
        self._es = _EarlyStopping(patience)

        self._best_val_f1: float = -1.0
        self._best_checkpoint: dict | None = None

    # ── Bucles internos ────────────────────────────────────────────────────────

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for images, labels in tqdm(self.train_loader, leave=False, desc="  train"):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
        return total_loss / n if n > 0 else 0.0

    def _val_epoch(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss, n = 0.0, 0
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                total_loss += self.criterion(logits, labels).item() * len(labels)
                n += len(labels)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        return total_loss / n if n > 0 else 0.0, compute_metrics(y_true, y_pred)

    def _run_stage(
        self,
        stage: int,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ) -> dict:
        stage_cfg = self._tcfg[f"stage{stage}"]
        num_epochs = int(stage_cfg["epochs"])
        self._es.reset()

        history: dict = {
            "stage": stage,
            "config": {f"stage{stage}": stage_cfg},
            "epochs": [],
            "best_epoch": None,
            "best_val_f1_macro": None,
            "stopped_by": "max_epochs",
        }
        stage_best = -1.0

        hdr = f"{'ep':>4} {'train_loss':>12} {'val_loss':>10} {'f1_macro':>10} {'acc':>8} {'s':>6}"
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(1, num_epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_epoch(optimizer)
            val_loss, val_metrics = self._val_epoch()
            if scheduler is not None:
                scheduler.step()
            elapsed = time.perf_counter() - t0

            val_f1 = val_metrics["f1_macro"]
            print(
                f"{epoch:>4}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
                f"{val_f1:>10.6f}  {val_metrics['accuracy']:>8.4f}  {elapsed:>6.1f}"
            )

            history["epochs"].append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_metrics": val_metrics,
            })

            # Mejor epoch de este stage (para el historial)
            if val_f1 > stage_best:
                stage_best = val_f1
                history["best_epoch"] = epoch
                history["best_val_f1_macro"] = round(val_f1, 6)

            # Mejor checkpoint global (a través de ambos stages)
            if val_f1 > self._best_val_f1:
                self._best_val_f1 = val_f1
                self._best_checkpoint = {
                    "stage": stage,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "stage": stage,
                        "metrics": val_metrics,
                    },
                    self.output_dir / "best_model.pt",
                )

            if self._es.step(val_f1):
                history["stopped_by"] = "early_stopping"
                print(f"  Early stopping en epoch {epoch} (patience={self._es.patience})")
                break

        return history

    # ── API pública ────────────────────────────────────────────────────────────

    def run_stage1(self) -> None:
        """Entrena solo la cabeza de clasificación con backbone congelado."""
        s1 = self._tcfg["stage1"]
        self.model.freeze_backbone()
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(s1["learning_rate"]),
            weight_decay=float(self._tcfg.get("weight_decay", 1e-4)),
        )
        print(f"\n=== Stage 1 — {s1['epochs']} epochs, backbone congelado ===")
        h = self._run_stage(1, opt)
        with open(self.output_dir / "history_stage1.json", "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2, ensure_ascii=False)

    def run_stage2(self) -> None:
        """Descongela últimas N capas y entrena con LR reducida."""
        s2 = self._tcfg["stage2"]
        n_blocks = int(s2.get("unfreeze_last_n_blocks", 2))
        self.model.unfreeze_last_n_blocks(n_blocks)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(s2["learning_rate"]),
            weight_decay=float(self._tcfg.get("weight_decay", 1e-4)),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(s2["epochs"]))
        print(f"\n=== Stage 2 — {s2['epochs']} epochs, últimos {n_blocks} bloques descongelados ===")
        h = self._run_stage(2, opt, sched)
        with open(self.output_dir / "history_stage2.json", "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2, ensure_ascii=False)

    def train_full(self) -> dict:
        """Ejecuta stage1 + stage2 y devuelve las métricas del mejor epoch."""
        self.run_stage1()
        self.run_stage2()
        best = self._best_checkpoint or {}
        with open(self.output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        return best
