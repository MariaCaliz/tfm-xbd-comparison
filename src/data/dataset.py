"""Dataset PyTorch para la clasificación de daños en edificaciones (xBD)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class XBDDataset(Dataset):
    """Dataset de crops de edificios xBD para clasificación de 4 clases.

    Espera que processed_dir contenga la subcarpeta crops/ con los JPEGs
    generados por parse_xbd.py, y que el CSV de split tenga al menos las
    columnas 'image_path' (relativa a processed_dir) y 'label' (int 0-3).

    Args:
        csv_path:      Ruta al CSV de split (train.csv, val.csv o test.csv).
        processed_dir: Directorio raíz de los crops (data/processed/).
        transform:     Pipeline de Albumentations (get_transforms(split, cfg))
                       o None para devolver arrays numpy sin transformar.
    """

    def __init__(
        self,
        csv_path: str | Path,
        processed_dir: str | Path,
        transform=None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.processed_dir = Path(processed_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Devuelve (tensor CHW float32, label int)."""
        row = self.df.iloc[idx]
        img_path = self.processed_dir / row["image_path"]

        # Albumentations espera uint8 HWC numpy array
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, int(row["label"])

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Sampler con peso inversamente proporcional a la frecuencia de clase.

        Úsalo solo en el DataLoader de entrenamiento para compensar el
        fuerte desbalanceo (no-damage ~77 % del dataset).

        Returns:
            WeightedRandomSampler con replacement=True y num_samples=len(dataset).
        """
        counts = Counter(self.df["label"].tolist())
        sample_weights = [1.0 / counts[int(lbl)] for lbl in self.df["label"]]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
