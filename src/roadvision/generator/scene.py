"""Deterministic scene planning.

Planning decides *what* exists and *where* it is. It does not paint pixels.
Every decision is made from a named random stream derived from the image seed,
so the same seed and configuration always produce the same plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.loader import config_hash, image_config_hash
from ..config.models import (
    DefectClass,
    Lighting,
    OccluderType,
    RoadVisionConfig,
    Weather,
)
from ..renderer.geometry import RoadGeometry
from .rng import stream_rng


@dataclass
class PlannedDefect:
    id: str
    type: DefectClass
    center_uv: tuple[float, float]
    radius_uv: float
    width_uv: float
    height_uv: float
    rotation_deg: float
    size_scale: float
    shape_seed: int


@dataclass
class PlannedOccluder:
    id: str
    type: OccluderType
    target_object_id: str
    target_visible_fraction: float
    direction_deg: float
    size_scale: float
    shape_seed: int


@dataclass
class ScenePlan:
    seed: int
    image_id: str
    family_id: str
    config_hash: str
    image_config_hash: str
    width: int
    height: int
    geometry: RoadGeometry
    defects: list[PlannedDefect]
    weather: Weather
    lighting: Lighting
    shadow_present: bool
    shadow_strength: float
    shadow_coverage: float
    occluder: PlannedOccluder | None
    difficulty: str
    scenario_tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


def _sample_size(
    defect_type: DefectClass,
    rng: np.random.Generator,
    scale: float,
) -> tuple[float, float, float]:
    """Return (half_width, half_height, radius) in road-plane units."""
    if defect_type == DefectClass.POTHOLE:
        rx = rng.uniform(0.048, 0.088) * scale
        ry = rng.uniform(0.038, 0.072) * scale
    elif defect_type == DefectClass.CRACK:
        length = rng.uniform(0.22, 0.46) * scale
        rx = length / 2.0
        ry = rng.uniform(0.003, 0.006) * scale
    elif defect_type == DefectClass.ALLIGATOR_CRACK:
        rx = rng.uniform(0.060, 0.115) * scale
        ry = rng.uniform(0.048, 0.095) * scale
    elif defect_type == DefectClass.PATCH:
        rx = rng.uniform(0.050, 0.105) * scale
        ry = rng.uniform(0.040, 0.085) * scale
    else:  # pragma: no cover - enum is closed
        raise ValueError(f"unsupported defect type {defect_type}")
    radius = float(np.hypot(rx, ry))
    return float(rx), float(ry), radius


def _oriented_corners(
    cx: float, cy: float, rx: float, ry: float, rotation_deg: float
) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    local = np.array(
        [[-rx, -ry], [rx, -ry], [rx, ry], [-rx, ry]], dtype=np.float64
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.array([cx, cy], dtype=np.float64)


def _uv_aabb(corners: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )


def _aabb_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    padding: float = 0.010,
) -> bool:
    return not (
        a[2] + padding <= b[0]
        or b[2] + padding <= a[0]
        or a[3] + padding <= b[1]
        or b[3] + padding <= a[1]
    )


def _projected_extent_ok(
    geometry: RoadGeometry,
    corners: np.ndarray,
    defect_type: DefectClass,
    margin: float = 5.0,
) -> bool:
    pts = geometry.project(corners)
    if pts[:, 0].min() < margin or pts[:, 1].min() < margin:
        return False
    if pts[:, 0].max() > geometry.width - margin:
        return False
    if pts[:, 1].max() > geometry.height - margin:
        return False
    span_x = float(pts[:, 0].max() - pts[:, 0].min())
    span_y = float(pts[:, 1].max() - pts[:, 1].min())
    if defect_type == DefectClass.CRACK:
        return max(span_x, span_y) >= 24.0
    return span_x >= 16.0 and span_y >= 12.0


def _place_defect(
    geometry: RoadGeometry,
    rng: np.random.Generator,
    defect_type: DefectClass,
    scale: float,
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float, float, float]:
    for attempt in range(900):
        relax = 1.0 - min(attempt / 1300.0, 0.24)
        rx, ry, radius = _sample_size(defect_type, rng, scale * relax)
        rotation = rng.uniform(0.0, 180.0)
        # The painted silhouette can exceed the nominal half-extents (spikes
        # on potholes, branches on cracks). Reserve space for it.
        factor = (
            1.42
            if defect_type == DefectClass.POTHOLE
            else 1.30
            if defect_type == DefectClass.CRACK
            else 1.18
            if defect_type == DefectClass.ALLIGATOR_CRACK
            else 1.06
        )
        cx = rng.uniform(0.06, 0.94)
        cy = rng.uniform(0.06, 0.94)
        corners = _oriented_corners(cx, cy, rx * factor, ry * factor, rotation)
        if corners[:, 0].min() < 0.025 or corners[:, 0].max() > 0.975:
            continue
        if corners[:, 1].min() < 0.025 or corners[:, 1].max() > 0.975:
            continue
        if not _projected_extent_ok(geometry, corners, defect_type):
            continue
        box = _uv_aabb(corners)
        if any(_aabb_overlap(box, other) for other in occupied):
            continue
        return cx, cy, rx, ry, radius, rotation
    raise RuntimeError("could not place defect without overlap")



def _difficulty(
    count: int,
    weather: Weather,
    lighting: Lighting,
    shadow_present: bool,
    has_occluder: bool,
    severe: bool,
) -> str:
    score = 0
    if count >= 3:
        score += 1
    if count >= 5:
        score += 1
    if weather in (Weather.WET, Weather.DAMP):
        score += 1
    if lighting == Lighting.HARSH or shadow_present:
        score += 1
    if has_occluder:
        score += 1
    if severe:
        score += 2
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def make_scene_plan(config: RoadVisionConfig, seed: int) -> ScenePlan:
    seed = int(seed)
    img_hash = image_config_hash(config)
    cfg_hash = config_hash(config)
    image_id = f"RV_{seed:06d}_{img_hash[:6]}"
    family_id = f"fam_{seed // config.dataset.family_block:05d}"
    geometry = RoadGeometry(
        width=config.render.width,
        height=config.render.height,
        perspective=config.render.perspective_strength,
    )

    env_rng = stream_rng(seed, f"{image_id}:environment")
    layout_rng = stream_rng(seed, f"{image_id}:layout")
    occ_rng = stream_rng(seed, f"{image_id}:occlusion")

    weather = Weather(env_rng.choice([w.value for w in config.environment.weather]))
    lighting = Lighting(
        env_rng.choice([li.value for li in config.environment.lighting])
    )
    shadow_present = bool(env_rng.random() < config.environment.shadow_probability)
    strong_shadow = bool(env_rng.random() < config.environment.strong_shadow_probability)
    shadow_strength = (
        float(env_rng.uniform(0.38, 0.55))
        if strong_shadow
        else float(env_rng.uniform(0.16, 0.30))
    )
    shadow_coverage = (
        float(env_rng.uniform(0.35, 0.62))
        if shadow_present and strong_shadow
        else float(env_rng.uniform(0.15, 0.40))
    )

    guaranteed = list(config.defects.guarantee_classes)
    count = int(
        layout_rng.integers(
            max(config.defects.count_min, len(guaranteed) or 1),
            config.defects.count_max + 1,
        )
    )
    classes = list(config.defects.class_weights.keys())
    weights = np.array(
        [config.defects.class_weights[c] for c in classes], dtype=np.float64
    )
    weights = weights / weights.sum()
    chosen: list[DefectClass] = list(guaranteed)
    while len(chosen) < count:
        chosen.append(DefectClass(layout_rng.choice([c.value for c in classes], p=weights)))
    # Shuffle class order so IDs do not encode a fixed class priority.
    order = layout_rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]

    defects: list[PlannedDefect] = []
    occupied: list[tuple[float, float, float, float]] = []
    for index, defect_type in enumerate(chosen, start=1):
        cx, cy, rx, ry, radius, rotation = _place_defect(
            geometry, layout_rng, defect_type, config.defects.size_scale, occupied
        )
        corners = _oriented_corners(cx, cy, rx, ry, rotation)
        occupied.append(_uv_aabb(corners))
        defects.append(
            PlannedDefect(
                id=f"damage_{index:03d}",
                type=defect_type,
                center_uv=(round(cx, 8), round(cy, 8)),
                radius_uv=round(radius, 8),
                width_uv=round(rx, 8),
                height_uv=round(ry, 8),
                rotation_deg=round(rotation, 6),
                size_scale=config.defects.size_scale,
                shape_seed=int(layout_rng.integers(0, 2**31 - 1)),
            )
        )

    occluder: PlannedOccluder | None = None
    if config.occlusion.enabled and occ_rng.random() < config.occlusion.probability:
        severe = bool(occ_rng.random() < config.occlusion.severe_share)
        target = defects[int(occ_rng.integers(0, len(defects)))]
        target_visible = (
            float(occ_rng.uniform(0.04, 0.09))
            if severe
            else float(occ_rng.uniform(0.30, 0.60))
        )
        occluder = PlannedOccluder(
            id="occluder_001",
            type=OccluderType(
                occ_rng.choice([t.value for t in config.occlusion.types])
            ),
            target_object_id=target.id,
            target_visible_fraction=round(target_visible, 6),
            direction_deg=round(float(occ_rng.uniform(0, 360)), 6),
            size_scale=round(float(occ_rng.uniform(1.20, 1.55)), 6),
            shape_seed=int(occ_rng.integers(0, 2**31 - 1)),
        )

    difficulty = _difficulty(
        count=len(defects),
        weather=weather,
        lighting=lighting,
        shadow_present=shadow_present,
        has_occluder=occluder is not None,
        severe=bool(occluder and occluder.target_visible_fraction <= 0.10),
    )

    tags = [weather.value, lighting.value, f"n{len(defects)}"]
    if shadow_present:
        tags.append("shadow")
    if occluder:
        tags.append("severe_occlusion" if occluder.target_visible_fraction <= 0.10 else "partial_occlusion")

    return ScenePlan(
        seed=seed,
        image_id=image_id,
        family_id=family_id,
        config_hash=cfg_hash,
        image_config_hash=img_hash,
        width=config.render.width,
        height=config.render.height,
        geometry=geometry,
        defects=defects,
        weather=weather,
        lighting=lighting,
        shadow_present=shadow_present,
        shadow_strength=shadow_strength,
        shadow_coverage=shadow_coverage,
        occluder=occluder,
        difficulty=difficulty,
        scenario_tags=tags,
    )
