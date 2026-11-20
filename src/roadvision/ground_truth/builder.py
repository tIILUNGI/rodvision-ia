"""Build a GroundTruth record from the render result."""

from __future__ import annotations

from typing import Any

from ..config.models import DefectClass, RoadVisionConfig
from ..config.loader import config_payload
from ..generator.scene import ScenePlan
from ..renderer.compose import RenderResult
from ..version import GENERATOR_VERSION, SCHEMA_VERSION
from .io import mask_stats, normalized_bbox
from .models import (
    Counts,
    GenerationInfo,
    GroundTruth,
    ImageInfo,
    ObjectInfo,
    OccluderInfo,
    SceneInfo,
)


def _visibility_label(visible_fraction: float) -> str:
    if visible_fraction >= 0.95:
        return "fully_visible"
    if visible_fraction >= 0.15:
        return "partially_occluded"
    return "severely_occluded"


def build_ground_truth(
    plan: ScenePlan,
    render: RenderResult,
    config: RoadVisionConfig,
    *,
    image_path: str,
    mask_path: str,
    visible_mask_path: str,
    occluder_mask_path: str,
    image_sha256: str,
) -> GroundTruth:
    width, height = plan.width, plan.height
    planned_by_id = {d.id: d for d in plan.defects}

    objects: list[ObjectInfo] = []
    by_class = {c.value: 0 for c in DefectClass}
    visibility_counts = {
        "fully_visible": 0,
        "partially_occluded": 0,
        "severely_occluded": 0,
    }

    for index, defect in enumerate(render.defects, start=1):
        planned = planned_by_id[defect.object_id]
        full = mask_stats(defect.full_mask)
        visible_mask = render.visible_masks[defect.object_id]
        visible = mask_stats(visible_mask)
        vf = float(visible["area_px"]) / float(max(1, full["area_px"]))
        visibility = _visibility_label(vf)
        by_class[defect.defect_type.value] += 1
        visibility_counts[visibility] += 1
        attributes = {
            **defect.attributes,
            "center_uv": list(planned.center_uv),
            "width_uv": planned.width_uv,
            "height_uv": planned.height_uv,
            "rotation_deg": planned.rotation_deg,
        }
        objects.append(
            ObjectInfo(
                id=defect.object_id,
                type=defect.defect_type,
                bbox_xyxy=full["bbox_xyxy"],
                bbox_normalized=normalized_bbox(full["bbox_xyxy"], width, height),
                bbox_visible_xyxy=visible["bbox_xyxy"],
                bbox_visible_normalized=normalized_bbox(
                    visible["bbox_xyxy"], width, height
                ),
                centroid_xy=full["centroid_xy"],
                centroid_normalized=[
                    round(full["centroid_xy"][0] / width, 6),
                    round(full["centroid_xy"][1] / height, 6),
                ],
                area_px=full["area_px"],
                visible_area_px=visible["area_px"],
                visible_fraction=round(vf, 6),
                visibility=visibility,
                occluded=vf < 0.95,
                mask_index=index,
                mask_path=mask_path,
                visible_mask_path=visible_mask_path,
                attributes=attributes,
            )
        )

    occluders: list[OccluderInfo] = []
    if render.occluder is not None:
        stats = mask_stats(render.occluder.mask)
        occluders.append(
            OccluderInfo(
                id=render.occluder.object_id,
                type=render.occluder.occluder_type.value,
                bbox_xyxy=stats["bbox_xyxy"],
                bbox_normalized=normalized_bbox(stats["bbox_xyxy"], width, height),
                area_px=stats["area_px"],
                mask_index=1,
                mask_path=occluder_mask_path,
                attributes=render.occluder.attributes,
            )
        )

    scene = SceneInfo(
        weather=plan.weather.value,
        lighting=plan.lighting.value,
        shadow_present=plan.shadow_present,
        shadow_coverage=render.shadow_coverage,
        wet_coverage=render.wet_coverage,
        perspective_strength=config.render.perspective_strength,
        difficulty=plan.difficulty,
        defect_count=len(objects),
        occluder_count=len(occluders),
        scenario_tags=plan.scenario_tags,
        surface_attributes=render.surface.attributes,
        render_attributes=render.attributes,
    )

    counts = Counts(
        pothole=by_class["pothole"],
        crack=by_class["crack"],
        alligator_crack=by_class["alligator_crack"],
        patch=by_class["patch"],
        total=len(objects),
        fully_visible=visibility_counts["fully_visible"],
        partially_occluded=visibility_counts["partially_occluded"],
        severely_occluded=visibility_counts["severely_occluded"],
    )

    generation = GenerationInfo(
        seed=plan.seed,
        family_id=plan.family_id,
        config_hash=plan.config_hash,
        image_config_hash=plan.image_config_hash,
        generator_version=GENERATOR_VERSION,
        config=config_payload(config),
    )

    return GroundTruth(
        schema_version=SCHEMA_VERSION,
        generator_version=GENERATOR_VERSION,
        image_id=plan.image_id,
        seed=plan.seed,
        family_id=plan.family_id,
        config_hash=plan.config_hash,
        image_config_hash=plan.image_config_hash,
        image=ImageInfo(
            path=image_path,
            width=width,
            height=height,
            sha256=image_sha256,
        ),
        scene=scene,
        objects=objects,
        occluders=occluders,
        counts=counts,
        generation=generation,
    )
