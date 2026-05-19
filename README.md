# TFM Clasificación de daños en edificaciones con visión artificial

Trabajo Fin de Máster del Máster Universitario en Inteligencia Artificial (UNIR).

**Autora:** María Cáliz González
**Director:** Wilmer Efrén Pereira González
**Tipo de trabajo:** Tipo 3 - Comparativa de soluciones

## Descripción

Comparativa experimental de cuatro arquitecturas de deep learning
(MobileNetV2, ResNet50, EfficientNet-B0 y Vision Transformer) para la
clasificación automática del nivel de daño en edificaciones a partir de
imágenes satelitales del dataset [xBD](https://xview2.org/).

## Estructura del repositorio

```
tfm-xbd-comparison/
├── configs/        # Configuraciones YAML de los experimentos
├── data/           # Datos del dataset xBD (no versionados)
├── notebooks/      # Notebooks de análisis exploratorio
├── results/        # Métricas, figuras y checkpoints
├── scripts/        # Scripts ejecutables
├── src/            # Código fuente modular
└── requirements.txt
```

## Instalación y entornos

El proyecto usa tres archivos de dependencias según el entorno:

| Archivo | Propósito |
|---------|-----------|
| `requirements.txt` | Dependencias lógicas sin versionar (referencia) |
| `requirements-local.txt` | Lock completo para desarrollo local en Mac M4 (incluye torch 2.12 con soporte MPS) |
| `requirements-colab.txt` | Lock para Google Colab con CUDA (se genera al configurar Colab) |

**Entorno local (Mac M4):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

**Entorno Colab:** el notebook `notebooks/colab_mobilenetv2.ipynb` gestiona
la instalación automáticamente usando el torch preinstalado en Colab.

## Estado actual

- [x] Estructura inicial del proyecto
- [x] Configuraciones base de los experimentos
- [x] Preprocesamiento del dataset (302 933 crops, 2 GB)
- [ ] Entrenamiento de los modelos
- [ ] Evaluación y análisis comparativo