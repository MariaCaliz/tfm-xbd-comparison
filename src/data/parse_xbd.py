"""Extracción de crops de edificios individuales desde el dataset xBD.

Recorre los JSON de anotación post-disaster de xBD (train + tier3),
extrae el bounding box de cada edificio etiquetado, aplica un margen
configurable, recorta la imagen original y guarda cada crop
redimensionado a 224×224 px.

Formato de salida JPEG:
  Se elige JPEG (quality=85) en lugar de PNG por dos razones:
  (1) Consistencia de dominio: MobileNetV2, ResNet50, EfficientNet-B0 y
      ViT-Base/16 están preentrenados en ImageNet, que distribuye sus
      imágenes en JPEG. Guardar los crops también en JPEG mantiene el
      dominio de entrada alineado con el preentrenamiento.
  (2) Eficiencia en disco: JPEG quality=85 genera ~6-10 KB por crop
      frente a ~40 KB de PNG comprimido. Para ~304 000 crops, la
      diferencia es ≈10 GB, crítica en Google Colab con almacenamiento
      limitado en Drive. El parámetro crop_quality es configurable en
      data.crop_quality del YAML para ajustarlo sin tocar el código.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from shapely import wkt as shapely_wkt
from tqdm import tqdm

# ── Constantes ────────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}
_VALID_SUBTYPES = frozenset(LABEL_MAP.keys())
_SPLITS = ("train", "tier3")


# ── Logger ────────────────────────────────────────────────────────────────────

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("parse_xbd")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Solo WARNING+ al fichero → trazabilidad de skips sin ruido de progreso
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Helpers de geometría ──────────────────────────────────────────────────────

def _apply_margin_and_clip(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    margin_ratio: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Expande el bbox en margin_ratio×tamaño a cada lado y recorta al tile."""
    bw = maxx - minx
    bh = maxy - miny
    mx = bw * margin_ratio
    my = bh * margin_ratio
    return (
        max(0.0, minx - mx),
        max(0.0, miny - my),
        min(float(img_w), maxx + mx),
        min(float(img_h), maxy + my),
    )


# ── Procesamiento por tile ────────────────────────────────────────────────────

def _process_tile(
    json_path: Path,
    img_path: Path,
    crops_dir: Path,
    source_split: str,
    margin_ratio: float,
    min_side_px: float,
    img_size: int,
    crop_quality: int,
    crop_ext: str,
    logger: logging.Logger,
) -> tuple[list[dict], Counter]:
    """Extrae todos los crops válidos de un tile.

    Returns:
        (records, skips): lista de dicts para el CSV y contador de skips.
    """
    records: list[dict] = []
    skips: Counter = Counter()

    # ── Carga JSON ────────────────────────────────────────────────────────────
    try:
        with open(json_path, encoding="utf-8") as f:
            label_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("TILE_SKIP json_error — %s: %s", json_path.name, exc)
        return records, Counter({"json_error": 1})

    # ── Carga imagen ──────────────────────────────────────────────────────────
    try:
        img = Image.open(img_path).convert("RGB")
        img.load()  # detecta corrupción antes de iterar edificios
    except OSError as exc:
        logger.warning("TILE_SKIP image_error — %s: %s", img_path.name, exc)
        return records, Counter({"image_error": 1})

    meta = label_data.get("metadata", {})
    img_w: int = meta.get("width", 1024)
    img_h: int = meta.get("height", 1024)
    event: str = meta.get("disaster", "unknown")
    tile_id: str = json_path.stem.replace("_post_disaster", "")

    # ── Iterar edificios ──────────────────────────────────────────────────────
    for feat in label_data["features"]["xy"]:
        props = feat["properties"]
        subtype: str = props.get("subtype", "")
        uid: str = props.get("uid", "")

        if subtype == "un-classified":
            skips["unclassified"] += 1
            continue

        if subtype not in _VALID_SUBTYPES:
            logger.warning("BLDG_SKIP unknown_subtype '%s' uid=%s", subtype, uid)
            skips["unknown_subtype"] += 1
            continue

        # Parsear WKT y obtener bbox en píxeles
        try:
            poly = shapely_wkt.loads(feat["wkt"])
            minx, miny, maxx, maxy = poly.bounds
        except Exception as exc:
            logger.warning("BLDG_SKIP wkt_error uid=%s: %s", uid, exc)
            skips["wkt_error"] += 1
            continue

        bw = maxx - minx
        bh = maxy - miny

        # Polígono genuinamente fuera del tile (tolerancia 0.1 px para fp noise)
        if maxx < -0.1 or maxy < -0.1 or minx > img_w + 0.1 or miny > img_h + 0.1:
            logger.warning(
                "BLDG_SKIP out_of_bounds uid=%s bounds=(%.1f,%.1f,%.1f,%.1f)",
                uid, minx, miny, maxx, maxy,
            )
            skips["out_of_bounds"] += 1
            continue

        # Edificio demasiado pequeño para ser útil
        if min(bw, bh) < min_side_px:
            logger.warning(
                "BLDG_SKIP too_small uid=%s size=(%.1f×%.1f px)", uid, bw, bh,
            )
            skips["too_small"] += 1
            continue

        x0, y0, x1, y1 = _apply_margin_and_clip(
            minx, miny, maxx, maxy, margin_ratio, img_w, img_h
        )

        crop = img.crop((x0, y0, x1, y1)).resize(
            (img_size, img_size), Image.LANCZOS
        )

        crop_filename = f"{uid}.{crop_ext}"
        crop_path = crops_dir / crop_filename
        if crop_ext in ("jpg", "jpeg"):
            crop.save(crop_path, format="JPEG", quality=crop_quality)
        else:
            crop.save(crop_path, format="PNG")

        records.append(
            {
                "uid": uid,
                "event": event,
                "source_split": source_split,
                "tile_id": tile_id,
                "label": LABEL_MAP[subtype],
                "label_name": subtype,
                # Relativo a processed_dir para portabilidad entre máquinas
                "image_path": f"crops/{crop_filename}",
                "source_tile_path": str(img_path),
                "bbox_xyxy": f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}",
            }
        )

    img.close()
    return records, skips


# ── Función principal ─────────────────────────────────────────────────────────

def parse_xbd(
    config: dict,
    raw_dir: str | Path,
    processed_dir: str | Path,
    events_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Parsea las anotaciones xBD y extrae crops individuales de edificios.

    Procesa los splits train/ y tier3/ (no test, que carece de labels de
    nivel de daño aprovechables). Solo imágenes post-disaster. Filtra los
    edificios 'un-classified'. Guarda crops en JPEG, el CSV maestro, el
    log de warnings y un resumen de estadísticas.

    Args:
        config:        Configuración fusionada devuelta por load_config().
        raw_dir:       Raíz de las descargas (contiene train/, tier3/).
        processed_dir: Directorio de salida. Se crean crops/, metadata.csv,
                       parse_xbd.log y stats.json dentro de él.
        events_filter: Si se especifica, solo se procesan los eventos de
                       esta lista (útil para smoke-tests rápidos).

    Returns:
        DataFrame con una fila por crop, esquema idéntico a metadata.csv.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    crops_dir = processed_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(processed_dir / "parse_xbd.log")

    data_cfg = config["data"]
    img_size: int = config.get("model", {}).get("input_size", 224)
    margin_ratio: float = float(data_cfg.get("patch_margin_ratio", 0.15))
    min_side_px: float = float(data_cfg.get("min_building_side_px", 5))
    crop_quality: int = int(data_cfg.get("crop_quality", 85))
    crop_fmt: str = str(data_cfg.get("crop_format", "jpg")).lower()
    crop_ext = "jpg" if crop_fmt in ("jpg", "jpeg") else crop_fmt

    logger.info("═" * 60)
    logger.info("xBD crop extraction — START")
    logger.info("  raw_dir         = %s", raw_dir)
    logger.info("  processed_dir   = %s", processed_dir)
    logger.info("  img_size        = %d", img_size)
    logger.info("  margin_ratio    = %.2f", margin_ratio)
    logger.info("  min_side_px     = %.1f", min_side_px)
    logger.info("  crop_quality    = %d", crop_quality)
    logger.info("  crop_format     = %s", crop_ext)
    if events_filter:
        logger.info("  events_filter   = %s", events_filter)
    logger.info("═" * 60)

    all_records: list[dict] = []
    total_skips: Counter = Counter()
    tiles_processed = 0
    t0 = time.perf_counter()

    for source_split in _SPLITS:
        labels_dir = raw_dir / source_split / "labels"
        images_dir = raw_dir / source_split / "images"

        if not labels_dir.exists():
            logger.warning("Split '%s' not found at %s — skipping", source_split, labels_dir)
            continue

        json_files = sorted(labels_dir.glob("*_post_disaster.json"))

        if events_filter:
            json_files = [
                jf for jf in json_files
                if any(jf.name.startswith(ev) for ev in events_filter)
            ]

        logger.info("Split %s: %d post-disaster tiles to process", source_split, len(json_files))

        for json_path in tqdm(json_files, desc=f"  {source_split}", unit="tile"):
            img_name = json_path.stem + ".png"
            img_path = images_dir / img_name

            if not img_path.exists():
                logger.warning("TILE_SKIP missing_image — %s", img_name)
                total_skips["missing_image"] += 1
                continue

            records, skips = _process_tile(
                json_path=json_path,
                img_path=img_path,
                crops_dir=crops_dir,
                source_split=source_split,
                margin_ratio=margin_ratio,
                min_side_px=min_side_px,
                img_size=img_size,
                crop_quality=crop_quality,
                crop_ext=crop_ext,
                logger=logger,
            )
            all_records.extend(records)
            total_skips.update(skips)
            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    # ── DataFrame y CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    if not df.empty:
        df["label"] = df["label"].astype("int8")
        metadata_path = processed_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        logger.info("metadata.csv → %d rows saved to %s", len(df), metadata_path)
    else:
        logger.warning("No crops were generated — check events_filter and raw_dir")

    # ── Stats JSON ────────────────────────────────────────────────────────────
    class_dist = (
        df.groupby("label_name").size().sort_index().to_dict()
        if not df.empty
        else {}
    )
    stats: dict = {
        "total_crops": len(df),
        "tiles_processed": tiles_processed,
        "skips": dict(total_skips),
        "class_distribution": class_dist,
        "processing_time_seconds": round(elapsed, 1),
        "events_processed": (
            sorted(df["event"].unique().tolist()) if not df.empty else []
        ),
    }
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("═" * 60)
    logger.info("DONE in %.1f s — %d crops, %d skips",
                elapsed, len(df), sum(total_skips.values()))
    logger.info("  class distribution: %s", class_dist)
    logger.info("  skips by reason:    %s", dict(total_skips))
    logger.info("═" * 60)

    return df
