"""Mask statistics and JSON/mask I/O helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import GroundTruth


class GroundTruthValidationError(ValueError):
    """Raised when annotation and mask bytes disagree."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    """Pixel-space statistics for a non-empty binary mask.

    Bounding boxes are half-open pixel boxes: [min_x, min_y, max_x + 1, max_y + 1].
    Centroids use pixel centres (index + 0.5).
    """
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {
            "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
            "centroid_xy": [0.0, 0.0],
            "area_px": 0,
        }
    return {
        "bbox_xyxy": [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ],
        "centroid_xy": [
            round(float(xs.mean() + 0.5), 6),
            round(float(ys.mean() + 0.5), 6),
        ],
        "area_px": int(m.sum()),
    }


def normalized_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    return [
        round(float(bbox[0] / width), 6),
        round(float(bbox[1] / height), 6),
        round(float(bbox[2] / width), 6),
        round(float(bbox[3] / height), 6),
    ]


def write_png_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("mask must be 2-D")
    Image.fromarray(arr, mode="L").save(path, format="PNG", optimize=False)


def write_png_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", optimize=False)


def save_annotation(path: str | Path, ground_truth: GroundTruth) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ground_truth.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def load_annotation(path: str | Path) -> GroundTruth:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GroundTruth.model_validate(data)


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.array(image.convert("L"), dtype=np.uint8)
    return arr
