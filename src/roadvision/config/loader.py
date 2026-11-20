"""YAML loading and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import RoadVisionConfig


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def load_config(path: str | Path) -> RoadVisionConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    try:
        return RoadVisionConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration {path}:\n{exc}") from exc


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for hashes and reproducibility comparisons."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def config_payload(config: RoadVisionConfig) -> dict[str, Any]:
    """The generation-relevant configuration, excluding orchestration fields.

    Dataset name and split never affect pixels, so they are excluded. This
    keeps an image ID stable when the same scene is used in different splits.
    """
    data = config.model_dump(mode="json")
    data.pop("dataset", None)
    return data


def image_config_payload(config: RoadVisionConfig) -> dict[str, Any]:
    """Fields that can change pixels or geometric ground truth."""
    data = config_payload(config)
    data.pop("questions", None)
    return data


def config_hash(config: RoadVisionConfig) -> str:
    return _sha256(config_payload(config))


def image_config_hash(config: RoadVisionConfig) -> str:
    return _sha256(image_config_payload(config))
