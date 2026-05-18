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

## Estado actual

🚧 Trabajo en curso.

- [x] Estructura inicial del proyecto
- [x] Configuraciones base de los experimentos
- [ ] Preprocesamiento del dataset
- [ ] Entrenamiento de los modelos
- [ ] Evaluación y análisis comparativo