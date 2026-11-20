"""Scene composition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.models import RoadVisionConfig
from ..generator.rng import object_rng, stream_rng
from ..generator.scene import ScenePlan
from .defects import DefectRender, render_defect
from .environment import apply_lighting, apply_shadows, apply_weather, finalize_image
from .occluders import OccluderRender, render_occluder
from .surface import SurfaceResult, render_surface


@dataclass
class RenderResult:
    image: np.ndarray
    defects: list[DefectRender]
    occluder: OccluderRender | None
    visible_masks: dict[str, np.ndarray]
    surface: SurfaceResult
    shadow_mask: np.ndarray
    shadow_coverage: float
    wet_coverage: float
    attributes: dict = field(default_factory=dict)


def render_scene(plan: ScenePlan, config: RoadVisionConfig) -> RenderResult:
    surface_rng = stream_rng(plan.seed, f"{plan.image_id}:surface")
    surface = render_surface(plan.geometry, config.render, surface_rng)
    image = surface.image.copy()

    defects: list[DefectRender] = []
    for planned in plan.defects:
        rng = object_rng(plan.seed, plan.image_id, planned.id)
        defects.append(render_defect(image, planned, plan.geometry, rng))

    light_rng = stream_rng(plan.seed, f"{plan.image_id}:lighting")
    image, light_attrs = apply_lighting(image, plan.lighting, light_rng)

    shadow_mask = np.zeros((plan.height, plan.width), dtype=np.float32)
    shadow_coverage = 0.0
    if plan.shadow_present:
        shadow_rng = stream_rng(plan.seed, f"{plan.image_id}:shadow")
        image, shadow_mask, shadow_coverage = apply_shadows(
            image,
            plan.geometry,
            plan.shadow_strength,
            plan.shadow_coverage,
            strong=plan.lighting.value == "harsh" or plan.shadow_strength >= 0.4,
            rng=shadow_rng,
        )

    weather_rng = stream_rng(plan.seed, f"{plan.image_id}:weather")
    image, wet_coverage = apply_weather(
        image,
        plan.geometry,
        plan.weather,
        config.environment.wet_puddle_strength,
        weather_rng,
    )

    occluder: OccluderRender | None = None
    if plan.occluder is not None:
        target = next(
            (d for d in defects if d.object_id == plan.occluder.target_object_id), None
        )
        if target is None:  # pragma: no cover - planner invariant
            raise RuntimeError("occluder target missing from rendered defects")
        occ_rng = object_rng(plan.seed, plan.image_id, plan.occluder.id)
        occluder = render_occluder(
            image, plan.occluder, target.full_mask, plan.geometry, occ_rng
        )

    visible_masks: dict[str, np.ndarray] = {}
    occluder_mask = occluder.mask if occluder else np.zeros((plan.height, plan.width), dtype=bool)
    for defect in defects:
        visible_masks[defect.object_id] = defect.full_mask & ~occluder_mask

    final_rng = stream_rng(plan.seed, f"{plan.image_id}:finalize")
    image = finalize_image(image, config.render, final_rng)

    return RenderResult(
        image=image,
        defects=defects,
        occluder=occluder,
        visible_masks=visible_masks,
        surface=surface,
        shadow_mask=shadow_mask,
        shadow_coverage=shadow_coverage,
        wet_coverage=wet_coverage,
        attributes={**surface.attributes, **light_attrs},
    )
