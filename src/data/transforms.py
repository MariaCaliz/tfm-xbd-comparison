"""Pipelines de transformación de imagen para entrenamiento y evaluación.

Usa Albumentations 2.x. Las probabilidades de augmentation se leen del
config (cfg['augmentation']['train']) para no hardcodear valores.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Valores de normalización ImageNet — comunes a los 4 modelos comparados
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: str, config: dict) -> A.Compose:
    """Devuelve el pipeline de transformación para el split indicado.

    Los crops ya son 224×224 desde parse_xbd.py, así que no se aplica
    resize. El pipeline de train añade augmentation geométrica y de color
    leída del config; val y test solo normalizan.

    Args:
        split:  'train', 'val' o 'test'.
        config: Configuración fusionada de load_config().

    Returns:
        A.Compose listo para llamar con transform(image=np_array)['image'].
    """
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        aug = config["augmentation"]["train"]
        return A.Compose([
            A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug.get("vertical_flip", 0.5)),
            # En Albumentations 2.x el default de RandomRotate90 es p=1,
            # por lo que hay que leerlo explícitamente del config.
            A.RandomRotate90(p=aug.get("rotate90", 0.5)),
            A.RandomBrightnessContrast(p=aug.get("brightness_contrast", 0.3)),
            # blur_limit=0 activa el modo sigma-driven de la API 2.x.
            A.GaussianBlur(
                blur_limit=0,
                sigma_limit=(0.5, 1.5),
                p=aug.get("gaussian_blur", 0.1),
            ),
            normalize,
            ToTensorV2(),
        ])

    return A.Compose([normalize, ToTensorV2()])
