"""Validation of a materialised dataset on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ground_truth.io import load_annotation, sha256_file
from ..ground_truth.validate import validate_ground_truth_files
from ..questions.models import Task
from .manifest import DatasetManifest, check_split_disjoint, load_manifest, manifest_path

FORBIDDEN_PUBLIC_KEYS = {
    "reference_answer",
    "verification",
    "related_object_ids",
    "visual_conditions",
    "defect_types",
    "details",
}


class DatasetValidationError(ValueError):
    """Raised when a dataset on disk violates the annotation/task contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{path}:{line_number}: invalid JSON ({exc})"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(f"{path}:{line_number}: not an object")
            records.append(record)
    return records


def validate_dataset(
    datasets_root: str | Path,
    name: str,
    *,
    check_image_hash: bool = True,
) -> dict[str, Any]:
    root = Path(datasets_root)
    manifest = load_manifest(manifest_path(root, name))
    errors: list[str] = []

    check_split_disjoint(
        root,
        name=manifest.name,
        split=manifest.split,
        seed_start=manifest.seed_start,
        num_images=manifest.num_images,
        family_block=manifest.family_block,
    )

    ground_truths = {}
    for image in manifest.images:
        annotation_path = root / image.annotation_path
        if not annotation_path.is_file():
            errors.append(f"{image.image_id}: missing annotation {image.annotation_path}")
            continue
        gt = load_annotation(annotation_path)
        ground_truths[image.image_id] = gt
        if gt.image_id != image.image_id:
            errors.append(f"{image.image_id}: annotation image_id mismatch")
        if gt.seed != image.seed:
            errors.append(f"{image.image_id}: annotation seed mismatch")
        if gt.family_id != image.family_id:
            errors.append(f"{image.image_id}: annotation family mismatch")
        if gt.image.sha256 != image.image_sha256:
            errors.append(f"{image.image_id}: manifest image hash mismatch")
        annotation_hash = sha256_file(annotation_path)
        if annotation_hash != image.annotation_sha256:
            errors.append(f"{image.image_id}: manifest annotation hash mismatch")
        try:
            validate_ground_truth_files(gt, root, check_image_hash=check_image_hash)
        except Exception as exc:  # noqa: BLE001 - surfaced as a validation error
            errors.append(f"{image.image_id}: {exc}")

        min_questions = int(
            gt.generation.config.get("questions", {}).get("min_questions", 3)
        )
        if len(gt.objects) < 2:
            errors.append(f"{image.image_id}: fewer than two defects")

    internal_records = _read_jsonl(root / manifest.internal_tasks)
    public_records = _read_jsonl(root / manifest.public_tasks)
    if len(internal_records) != len(public_records):
        errors.append(
            f"task files differ in length: {len(internal_records)} internal vs "
            f"{len(public_records)} public"
        )
    if len(internal_records) != manifest.tasks_validated:
        errors.append("manifest task count does not match tasks.jsonl")

    tasks_by_image: dict[str, list[Task]] = {}
    task_ids: set[str] = set()
    for record in internal_records:
        try:
            task = Task.model_validate(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid internal task: {exc}")
            continue
        if task.task_id in task_ids:
            errors.append(f"duplicate task id {task.task_id}")
        task_ids.add(task.task_id)
        if task.image_id not in ground_truths:
            errors.append(f"{task.task_id}: unknown image {task.image_id}")
            continue
        gt = ground_truths[task.image_id]
        known = {obj.id for obj in gt.objects}
        unknown = set(task.related_object_ids) - known
        if unknown:
            errors.append(f"{task.task_id}: unknown object IDs {sorted(unknown)}")
        original = task
        reconstructed = Task.model_validate(
            {**original.model_dump(mode="json"), "task_id": original.task_id}
        )
        try:
            reconstructed.assert_no_leakage()
        except ValueError as exc:
            errors.append(str(exc))
        tasks_by_image.setdefault(task.image_id, []).append(task)

    public_by_id = {record.get("task_id"): record for record in public_records}
    if len(public_by_id) != len(public_records):
        errors.append("duplicate public task IDs")
    for task in [t for tasks in tasks_by_image.values() for t in tasks]:
        public = public_by_id.get(task.task_id)
        if public is None:
            errors.append(f"{task.task_id}: missing public record")
            continue
        forbidden = FORBIDDEN_PUBLIC_KEYS & set(public)
        if forbidden:
            errors.append(f"{task.task_id}: public record leaks {sorted(forbidden)}")
        expected = task.public_payload()
        for key, value in expected.items():
            if public.get(key) != value:
                errors.append(f"{task.task_id}: public field {key!r} mismatch")

    for image_id, gt in ground_truths.items():
        image_tasks = tasks_by_image.get(image_id, [])
        min_questions = int(
            gt.generation.config.get("questions", {}).get("min_questions", 3)
        )
        if len(image_tasks) < min_questions:
            errors.append(
                f"{image_id}: {len(image_tasks)} tasks, minimum is {min_questions}"
            )
        if len({t.task_id for t in image_tasks}) != len(image_tasks):
            errors.append(f"{image_id}: duplicate task IDs")
        if len({t.capability for t in image_tasks}) < 3:
            errors.append(f"{image_id}: tasks cover fewer than three capabilities")

    if errors:
        raise DatasetValidationError("; ".join(errors))

    return {
        "dataset": name,
        "split": manifest.split,
        "images": len(manifest.images),
        "tasks": len(internal_records),
        "capabilities": manifest.task_counts_by_capability,
        "difficulty": manifest.task_counts_by_difficulty,
        "generator_version": manifest.generator_version,
        "config_hash": manifest.config_hash,
    }


def validate_all(datasets_root: str | Path) -> list[dict[str, Any]]:
    root = Path(datasets_root)
    reports = []
    for path in sorted((root / "manifests").glob("*.json")):
        reports.append(validate_dataset(root, path.stem))
    if not reports:
        raise DatasetValidationError("no dataset manifests found")
    return reports


__all__ = [
    "DatasetValidationError",
    "validate_all",
    "validate_dataset",
]
