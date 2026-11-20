"""Independent validation of saved annotations against saved masks.

This is intentionally a second implementation of the statistics: if the builder
and the masks ever disagree, validation must fail rather than trust the JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .io import GroundTruthValidationError, load_mask, sha256_file
from .models import GroundTruth


def _norm_box(box: list[float], width: int, height: int) -> list[float]:
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def _box_close(a: Iterable[float], b: Iterable[float], tol: float = 0.02) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise GroundTruthValidationError(
            f"path escapes dataset root: {relative}"
        ) from exc
    return candidate


def validate_ground_truth_files(
    ground_truth: GroundTruth,
    datasets_root: str | Path,
    *,
    check_image_hash: bool = True,
) -> None:
    root = Path(datasets_root)
    errors: list[str] = []
    w, h = ground_truth.image.width, ground_truth.image.height

    image_path = _resolve_inside(root, ground_truth.image.path)
    if not image_path.is_file():
        errors.append(f"missing image: {image_path}")
    else:
        if check_image_hash:
            digest = sha256_file(image_path)
            if digest != ground_truth.image.sha256:
                errors.append(
                    f"image sha256 mismatch: expected {ground_truth.image.sha256}, got {digest}"
                )

    objects = ground_truth.objects
    if not objects:
        raise GroundTruthValidationError("annotation has no objects")
    instance_path = _resolve_inside(root, objects[0].mask_path)
    visible_path = _resolve_inside(root, objects[0].visible_mask_path)
    if not instance_path.is_file():
        errors.append(f"missing instance mask: {instance_path}")
    if not visible_path.is_file():
        errors.append(f"missing visible mask: {visible_path}")
    if errors:
        raise GroundTruthValidationError("; ".join(errors))

    instance = load_mask(instance_path)
    visible_map = load_mask(visible_path)
    if instance.shape != (h, w):
        errors.append(f"instance mask shape {instance.shape} != {(h, w)}")
    if visible_map.shape != (h, w):
        errors.append(f"visible mask shape {visible_map.shape} != {(h, w)}")

    if not errors:
        expected_indices = {obj.mask_index for obj in objects}
        present = set(int(v) for v in np.unique(instance) if v != 0)
        if present != expected_indices:
            errors.append(
                f"instance mask values {sorted(present)} != expected {sorted(expected_indices)}"
            )
        visible_indices = set(int(v) for v in np.unique(visible_map) if v != 0)
        if not visible_indices.issubset(expected_indices):
            errors.append(
                f"visible mask contains unknown indices {sorted(visible_indices - expected_indices)}"
            )
        total_area = sum(obj.area_px for obj in objects)
        if total_area != int(np.count_nonzero(instance)):
            errors.append(
                "object areas overlap or do not cover the instance mask: "
                f"sum={total_area}, unique={int(np.count_nonzero(instance))}"
            )

        for obj in objects:
            mask = instance == obj.mask_index
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                errors.append(f"{obj.id}: missing pixels in instance mask")
                continue
            bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            area = int(mask.sum())
            centroid = [float(xs.mean() + 0.5), float(ys.mean() + 0.5)]
            if area != obj.area_px:
                errors.append(f"{obj.id}: area {area} != annotation {obj.area_px}")
            if not _box_close(bbox, obj.bbox_xyxy):
                errors.append(f"{obj.id}: bbox {bbox} != annotation {obj.bbox_xyxy}")
            if not _box_close(_norm_box(bbox, w, h), obj.bbox_normalized, 1e-5):
                errors.append(f"{obj.id}: normalized bbox mismatch")
            if not _box_close(centroid, obj.centroid_xy, 0.01):
                errors.append(f"{obj.id}: centroid mismatch")

            vmask = visible_map == obj.mask_index
            vys, vxs = np.nonzero(vmask)
            varea = int(vmask.sum())
            if varea != obj.visible_area_px:
                errors.append(
                    f"{obj.id}: visible area {varea} != annotation {obj.visible_area_px}"
                )
            if varea == 0:
                if any(abs(v) > 1e-9 for v in obj.bbox_visible_xyxy):
                    errors.append(f"{obj.id}: hidden object has a non-empty visible bbox")
            else:
                vbbox = [vxs.min(), vys.min(), vxs.max() + 1, vys.max() + 1]
                if not _box_close(vbbox, obj.bbox_visible_xyxy):
                    errors.append(
                        f"{obj.id}: visible bbox {vbbox} != annotation {obj.bbox_visible_xyxy}"
                    )
            if not np.all(vmask <= mask):
                errors.append(f"{obj.id}: visible mask is not a subset of full mask")
            vf = varea / max(1, area)
            if abs(vf - obj.visible_fraction) > 1e-5:
                errors.append(
                    f"{obj.id}: visible fraction {vf:.6f} != annotation {obj.visible_fraction}"
                )

    for occ in ground_truth.occluders:
        path = _resolve_inside(root, occ.mask_path)
        if not path.is_file():
            errors.append(f"missing occluder mask: {path}")
            continue
        occ_map = load_mask(path)
        if occ_map.shape != (h, w):
            errors.append(f"{occ.id}: occluder mask shape mismatch")
            continue
        mask = occ_map == occ.mask_index
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            errors.append(f"{occ.id}: empty occluder mask")
            continue
        area = int(mask.sum())
        bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        if area != occ.area_px:
            errors.append(f"{occ.id}: area {area} != annotation {occ.area_px}")
        if not _box_close(bbox, occ.bbox_xyxy):
            errors.append(f"{occ.id}: bbox mismatch")

    if errors:
        raise GroundTruthValidationError("; ".join(errors))
