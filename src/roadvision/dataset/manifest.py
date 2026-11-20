"""Dataset manifests and split-leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..version import GENERATOR_VERSION, SCHEMA_VERSION


class ManifestImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    seed: int
    family_id: str
    image_path: str
    annotation_path: str
    image_sha256: str
    annotation_sha256: str
    defect_count: int
    occluder_count: int
    difficulty: str
    weather: str
    lighting: str
    shadow_present: bool
    scenario_tags: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION
    name: str
    split: str
    seed_start: int
    num_images: int
    family_block: int
    config_hash: str
    image_config_hash: str
    images: list[ManifestImage]
    task_counts: dict[str, int] = Field(default_factory=dict)
    task_counts_by_capability: dict[str, int] = Field(default_factory=dict)
    task_counts_by_difficulty: dict[str, int] = Field(default_factory=dict)
    internal_tasks: str
    public_tasks: str
    annotation_validated: int
    tasks_validated: int

    @model_validator(mode="after")
    def _seed_span(self) -> "DatasetManifest":
        if len(self.images) != self.num_images:
            raise ValueError("manifest image count does not match num_images")
        seeds = [image.seed for image in self.images]
        if seeds != sorted(seeds):
            raise ValueError("manifest images are not sorted by seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("manifest contains duplicate seeds")
        if seeds and (seeds[0] != self.seed_start or seeds[-1] != self.seed_start + self.num_images - 1):
            raise ValueError("manifest seeds do not match the configured range")
        return self


def manifest_path(root: str | Path, name: str) -> Path:
    return Path(root) / "manifests" / f"{name}.json"


def save_manifest(manifest: DatasetManifest, root: str | Path) -> Path:
    path = manifest_path(root, manifest.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(data)


def family_block_of(seed: int, block: int) -> int:
    return seed // block


def block_span(seed_start: int, num_images: int, block: int) -> tuple[int, int]:
    first = family_block_of(seed_start, block)
    last = family_block_of(seed_start + num_images - 1, block)
    return first, last


def check_split_disjoint(
    root: str | Path,
    *,
    name: str,
    split: str,
    seed_start: int,
    num_images: int,
    family_block: int,
) -> None:
    """Reject a dataset whose family blocks overlap another split.

    Blocks, not individual seeds, define a scenario family, so two splits that
    share a block would share scene structure even if the seeds differ.
    """
    manifests_dir = Path(root) / "manifests"
    if not manifests_dir.is_dir():
        return
    first, last = block_span(seed_start, num_images, family_block)
    for path in sorted(manifests_dir.glob("*.json")):
        if path.stem == name:
            continue
        other = load_manifest(path)
        if other.split == split:
            continue
        other_first, other_last = block_span(
            other.seed_start, other.num_images, other.family_block
        )
        if not (last < other_first or other_last < first):
            raise ValueError(
                f"split leakage: {name} ({split}) blocks {first}-{last} overlap "
                f"{other.name} ({other.split}) blocks {other_first}-{other_last}"
            )


def summarize(manifests: list[DatasetManifest]) -> dict[str, Any]:
    return {
        "datasets": [
            {
                "name": m.name,
                "split": m.split,
                "images": len(m.images),
                "tasks": sum(m.task_counts.values()),
                "config_hash": m.config_hash,
                "generator_version": m.generator_version,
            }
            for m in manifests
        ],
        "total_images": sum(len(m.images) for m in manifests),
        "total_tasks": sum(sum(m.task_counts.values()) for m in manifests),
    }
