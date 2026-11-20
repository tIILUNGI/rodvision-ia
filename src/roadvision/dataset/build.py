"""Dataset construction and single-image generation."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..config.loader import config_hash, image_config_hash
from ..config.models import RoadVisionConfig
from ..generator.scene import ScenePlan, make_scene_plan
from ..ground_truth.builder import build_ground_truth
from ..ground_truth.io import (
    save_annotation,
    sha256_bytes,
    sha256_file,
    write_png_mask,
    write_png_rgb,
)
from ..ground_truth.models import GroundTruth
from ..ground_truth.validate import validate_ground_truth_files
from ..questions.factory import generate_tasks
from ..questions.models import Task
from ..renderer.compose import RenderResult, render_scene
from ..version import GENERATOR_VERSION
from .manifest import (
    DatasetManifest,
    ManifestImage,
    check_split_disjoint,
    save_manifest,
)


@dataclass
class GenerationArtifacts:
    plan: ScenePlan
    render: RenderResult
    ground_truth: GroundTruth
    image_path: Path
    annotation_path: Path
    elapsed_sec: float


def _combined_mask(
    masks: dict[int, np.ndarray], height: int, width: int
) -> np.ndarray:
    combined = np.zeros((height, width), dtype=np.uint8)
    for index, mask in masks.items():
        combined[np.asarray(mask, dtype=bool)] = index
    return combined


def _paths(root: Path, dataset: str, image_id: str) -> dict[str, Path]:
    return {
        "image": root / f"generated/{dataset}/images/{image_id}.png",
        "annotation": root / f"annotations/{dataset}/{image_id}.json",
        "instances": root / f"annotations/{dataset}/masks/{image_id}_instances.png",
        "visible": root / f"annotations/{dataset}/masks/{image_id}_visible.png",
        "occluders": root / f"annotations/{dataset}/masks/{image_id}_occluders.png",
    }


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def generate_image(
    config: RoadVisionConfig,
    seed: int,
    *,
    datasets_root: str | Path = "datasets",
) -> GenerationArtifacts:
    started = time.perf_counter()
    root = Path(datasets_root)
    dataset = config.dataset.name
    plan = make_scene_plan(config, seed)
    render = render_scene(plan, config)
    paths = _paths(root, dataset, plan.image_id)

    write_png_rgb(paths["image"], render.image)
    image_hash = sha256_file(paths["image"])

    instance_mask = _combined_mask(
        {i: d.full_mask for i, d in enumerate(render.defects, start=1)},
        plan.height,
        plan.width,
    )
    visible_mask = _combined_mask(
        {
            i: render.visible_masks[d.object_id]
            for i, d in enumerate(render.defects, start=1)
        },
        plan.height,
        plan.width,
    )
    occluder_mask = _combined_mask(
        {1: render.occluder.mask} if render.occluder is not None else {},
        plan.height,
        plan.width,
    )
    write_png_mask(paths["instances"], instance_mask)
    write_png_mask(paths["visible"], visible_mask)
    write_png_mask(paths["occluders"], occluder_mask)

    ground_truth = build_ground_truth(
        plan,
        render,
        config,
        image_path=_relative(root, paths["image"]),
        mask_path=_relative(root, paths["instances"]),
        visible_mask_path=_relative(root, paths["visible"]),
        occluder_mask_path=_relative(root, paths["occluders"]),
        image_sha256=image_hash,
    )
    save_annotation(paths["annotation"], ground_truth)
    validate_ground_truth_files(ground_truth, root)

    return GenerationArtifacts(
        plan=plan,
        render=render,
        ground_truth=ground_truth,
        image_path=paths["image"],
        annotation_path=paths["annotation"],
        elapsed_sec=round(time.perf_counter() - started, 4),
    )


def write_tasks(
    tasks: Iterable[Task], root: str | Path, dataset: str
) -> tuple[Path, Path]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    ids = [task.task_id for task in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    task_dir = Path(root) / "tasks" / dataset
    task_dir.mkdir(parents=True, exist_ok=True)
    internal = task_dir / "tasks.jsonl"
    public = task_dir / "public_tasks.jsonl"
    with internal.open("w", encoding="utf-8", newline="\n") as fh:
        for task in ordered:
            task.assert_no_leakage()
            fh.write(json.dumps(task.internal_payload(), ensure_ascii=True) + "\n")
    with public.open("w", encoding="utf-8", newline="\n") as fh:
        for task in ordered:
            fh.write(json.dumps(task.public_payload(), ensure_ascii=True) + "\n")
    return internal, public


def _clear_dataset_dirs(root: Path, dataset: str) -> None:
    for path in (
        root / "generated" / dataset,
        root / "annotations" / dataset,
        root / "tasks" / dataset,
    ):
        if path.exists():
            shutil.rmtree(path)


def build_dataset(
    config: RoadVisionConfig,
    *,
    datasets_root: str | Path = "datasets",
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> DatasetManifest:
    root = Path(datasets_root)
    cfg_dataset = config.dataset
    check_split_disjoint(
        root,
        name=cfg_dataset.name,
        split=cfg_dataset.split,
        seed_start=cfg_dataset.seed_start,
        num_images=cfg_dataset.num_images,
        family_block=cfg_dataset.family_block,
    )

    existing = [
        p
        for p in (
            root / "generated" / cfg_dataset.name,
            root / "annotations" / cfg_dataset.name,
            root / "tasks" / cfg_dataset.name,
        )
        if p.exists() and any(p.rglob("*"))
    ]
    if existing and not force:
        raise FileExistsError(
            f"dataset {cfg_dataset.name!r} already exists at {existing[0]}; "
            "pass force=True to rebuild"
        )
    if force:
        _clear_dataset_dirs(root, cfg_dataset.name)

    images: list[ManifestImage] = []
    all_tasks: list[Task] = []
    for offset in range(cfg_dataset.num_images):
        seed = cfg_dataset.seed_start + offset
        artifacts = generate_image(config, seed, datasets_root=root)
        gt = artifacts.ground_truth
        tasks = generate_tasks(gt, config)
        for task in tasks:
            known = {obj.id for obj in gt.objects}
            if not set(task.related_object_ids) <= known:
                raise ValueError(
                    f"{task.task_id}: references unknown object IDs"
                )
        all_tasks.extend(tasks)
        images.append(
            ManifestImage(
                image_id=gt.image_id,
                seed=seed,
                family_id=gt.family_id,
                image_path=gt.image.path,
                annotation_path=_relative(root, artifacts.annotation_path),
                image_sha256=gt.image.sha256,
                annotation_sha256=sha256_file(artifacts.annotation_path),
                defect_count=len(gt.objects),
                occluder_count=len(gt.occluders),
                difficulty=gt.scene.difficulty,
                weather=gt.scene.weather,
                lighting=gt.scene.lighting,
                shadow_present=gt.scene.shadow_present,
                scenario_tags=gt.scene.scenario_tags,
            )
        )
        if progress is not None:
            progress(offset + 1, cfg_dataset.num_images, gt.image_id)

    internal, public = write_tasks(all_tasks, root, cfg_dataset.name)

    by_capability: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for task in all_tasks:
        by_capability[task.capability.value] = (
            by_capability.get(task.capability.value, 0) + 1
        )
        by_difficulty[task.difficulty] = by_difficulty.get(task.difficulty, 0) + 1

    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION,
        name=cfg_dataset.name,
        split=cfg_dataset.split,
        seed_start=cfg_dataset.seed_start,
        num_images=cfg_dataset.num_images,
        family_block=cfg_dataset.family_block,
        config_hash=config_hash(config),
        image_config_hash=image_config_hash(config),
        images=images,
        task_counts={
            "total": len(all_tasks),
            "images_with_tasks": len({t.image_id for t in all_tasks}),
        },
        task_counts_by_capability=dict(sorted(by_capability.items())),
        task_counts_by_difficulty=dict(sorted(by_difficulty.items())),
        internal_tasks=_relative(root, internal),
        public_tasks=_relative(root, public),
        annotation_validated=len(images),
        tasks_validated=len(all_tasks),
    )
    save_manifest(manifest, root)
    return manifest


__all__ = [
    "GenerationArtifacts",
    "build_dataset",
    "generate_image",
    "write_tasks",
]
