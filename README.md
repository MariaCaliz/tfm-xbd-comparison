# Comparativa de arquitecturas CNN/ViT para clasificación de daños en edificaciones sobre xBD

Trabajo Fin de Máster — Máster Universitario en Inteligencia Artificial (UNIR)

**Autora:** María Cáliz González
**Director:** Wilmer Efrén Pereira González
**Tipo:** Comparativa de soluciones (Tipo 3)

---

## Objetivo

Comparativa experimental de cuatro arquitecturas de deep learning para la
clasificación automática del nivel de daño en edificaciones a partir de
imágenes satelitales post-catástrofe del dataset
[xBD](https://xview2.org/). Las cuatro clases corresponden al estándar JRC:
**sin daño · daño leve · daño mayor · destruido**.

| Arquitectura    | Familia             |
|-----------------|---------------------|
| MobileNetV2     | CNN ligera          |
| EfficientNet-B0 | CNN escalable       |
| ResNet50        | CNN clásica         |
| ViT-Base/16     | Vision Transformer  |

---

## Resultados finales

| Modelo          | Accuracy | F1-macro  | F1 sin daño | F1 leve | F1 mayor | F1 destruido |
|-----------------|----------|-----------|-------------|---------|----------|--------------|
| ResNet50        | 0.664    | **0.465** | 0.798       | 0.305   | 0.184    | 0.574        |
| ViT-Base/16     | 0.592    | 0.435     | 0.733       | 0.301   | 0.166    | 0.539        |
| EfficientNet-B0 | 0.526    | 0.408     | 0.660       | 0.297   | 0.169    | 0.505        |
| MobileNetV2     | 0.344    | 0.311     | 0.415       | 0.291   | 0.114    | 0.424        |

Eficiencia (inferencia en CPU, batch=1, imagen 224×224):

| Modelo          | Parámetros | GFLOPs | Latencia (ms) | Tamaño (MB) |
|-----------------|------------|--------|---------------|-------------|
| MobileNetV2     | 2.2 M      | 0.31   | ~19           | 9           |
| EfficientNet-B0 | 4.0 M      | 0.40   | ~86           | 16          |
| ResNet50        | 23.5 M     | 4.11   | ~21           | 90          |
| ViT-Base/16     | 85.8 M     | 16.87  | ~54           | 327         |

---

## Estructura del repositorio

```
tfm-xbd-comparison/
├── configs/                   # Configuraciones YAML de los experimentos
│   ├── base.yaml              # Parámetros comunes (heredados por todos)
│   ├── mobilenetv2.yaml
│   ├── efficientnetb0.yaml
│   ├── resnet50.yaml
│   └── vit.yaml
├── data/                      # Dataset xBD (NO versionado)
│   └── splits/                # CSVs de partición train/val/test
├── notebooks/                 # Notebooks para entrenamiento en Google Colab
├── results/
│   ├── figures/               # Figuras de la comparativa (versionadas)
│   ├── metrics_final/         # Métricas de test de cada modelo (JSON)
│   └── comparison_table.csv   # Tabla comparativa completa
├── scripts/
│   ├── prepare_data.py        # Preprocesamiento del dataset
│   ├── train.py               # Entrenamiento de un modelo
│   ├── run_comparison.py      # Genera tabla comparativa y figuras
│   └── measure_flops.py       # Mide FLOPs y latencia de inferencia
├── src/
│   ├── data/                  # Dataset, transforms, splits
│   ├── models/                # Definición de los 4 modelos
│   ├── training/              # Trainer, losses
│   ├── evaluation/            # Métricas, eficiencia, comparativa
│   └── utils/                 # Config, seed
├── tests/                     # Tests unitarios
└── requirements.txt
```

---

## Instalación

```bash
git clone <url-del-repo>
cd tfm-xbd-comparison
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

> Para instalar PyTorch con soporte CUDA o MPS (Apple Silicon) consulta
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
> antes de instalar el resto de dependencias.

---

## Preparación de los datos

1. Descarga el dataset xBD desde [xview2.org](https://xview2.org/) y
   descomprímelo en `data/raw/`.
2. Genera los crops de 224×224 y las particiones train/val/test:

```bash
python scripts/prepare_data.py
```

Esto produce ~302 933 crops (~2 GB) en `data/processed/` y los CSV de
partición en `data/splits/`.

---

## Entrenamiento

Cada modelo tiene un YAML en `configs/` que hereda los parámetros comunes
de `configs/base.yaml`.

```bash
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/mobilenetv2.yaml
python scripts/train.py --config configs/efficientnetb0.yaml
python scripts/train.py --config configs/vit.yaml
```

Los checkpoints se guardan en `results/checkpoints/` y las métricas de
validación en `results/metrics/`.

Los notebooks en `notebooks/` están preparados para ejecutar el
entrenamiento en Google Colab con GPU.

---

## Evaluación

Las métricas de test finales están en `results/metrics_final/`. Para
regenerar las métricas desde un checkpoint, carga el modelo con `get_model`
y ejecuta el bucle de evaluación del notebook de Colab (ver `notebooks/`).

---

## Análisis comparativo y de eficiencia

```bash
# Tabla comparativa + figuras → results/figures/ y results/comparison_table.csv
python scripts/run_comparison.py

# FLOPs y latencia de inferencia
python scripts/measure_flops.py
```

---

## Tests

```bash
pytest tests/
```
