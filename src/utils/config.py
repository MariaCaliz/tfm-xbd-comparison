"""Carga y fusión de configuraciones YAML con soporte de herencia."""

from __future__ import annotations

from pathlib import Path

import yaml

_REQUIRED_KEYS = {"experiment", "data", "augmentation", "training", "evaluation"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona *override* sobre *base* de forma recursiva.

    Los valores de *override* tienen precedencia. Los subdicts se fusionan
    en profundidad en lugar de reemplazarse enteros.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict:
    """Carga un archivo YAML resolviendo la herencia declarada en ``inherits``.

    Si el YAML contiene ``inherits: <nombre>.yaml``, se carga el archivo padre
    desde el mismo directorio y se fusiona con el hijo. El hijo sobreescribe
    al padre en cualquier clave que redefina.

    Args:
        config_path: Ruta al archivo YAML (absoluta o relativa al CWD).

    Returns:
        Diccionario de configuración fusionado y listo para usar.

    Raises:
        FileNotFoundError: Si el archivo config o el padre heredado no existen.
        ValueError: Si faltan claves de primer nivel requeridas.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)

    if "inherits" in config:
        parent_name: str = config.pop("inherits")
        parent_path = config_path.parent / parent_name
        if not parent_path.exists():
            raise FileNotFoundError(
                f"Config heredado '{parent_name}' no encontrado en {config_path.parent}"
            )
        with parent_path.open(encoding="utf-8") as f:
            base: dict = yaml.safe_load(f)
        config = _deep_merge(base, config)

    missing = _REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Faltan claves requeridas en la configuración: {sorted(missing)}"
        )

    return config
