"""Ground-truth Pydantic models.

These models are deliberately strict. They are the contract between the
renderer, question factory, dataset validator, evaluator and Harbor exporter.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config.models import DefectClass
from ..version import GENERATOR_VERSION, SCHEMA_VERSION

BBox = Annotated[list[float], Field(min_length=4, max_length=4)]
Visibility = Literal["fully_visible", "partially_occluded", "severely_occluded"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ImageInfo(_Strict):
    path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SceneInfo(_Strict):
    surface: Literal["asphalt"] = "asphalt"
    weather: Literal["dry", "damp", "wet"]
    lighting: Literal["overcast", "golden", "harsh"]
    shadow_present: bool
    shadow_coverage: float = Field(ge=0.0, le=1.0)
    wet_coverage: float = Field(ge=0.0, le=1.0)
    perspective_strength: float = Field(ge=0.0, le=1.0)
    difficulty: Literal["easy", "medium", "hard"]
    defect_count: int = Field(ge=1)
    occluder_count: int = Field(ge=0)
    scenario_tags: list[str] = Field(default_factory=list)
    surface_attributes: dict[str, Any] = Field(default_factory=dict)
    render_attributes: dict[str, Any] = Field(default_factory=dict)


class ObjectInfo(_Strict):
    id: str = Field(pattern=r"^damage_[0-9]{3}$")
    type: DefectClass
    bbox_xyxy: BBox
    bbox_normalized: BBox
    bbox_visible_xyxy: BBox
    bbox_visible_normalized: BBox
    centroid_xy: list[float] = Field(min_length=2, max_length=2)
    centroid_normalized: list[float] = Field(min_length=2, max_length=2)
    area_px: int = Field(gt=0)
    visible_area_px: int = Field(ge=0)
    visible_fraction: float = Field(ge=0.0, le=1.0)
    visibility: Visibility
    occluded: bool
    mask_index: int = Field(ge=1)
    mask_path: str = Field(min_length=1)
    visible_mask_path: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "bbox_xyxy",
        "bbox_normalized",
        "bbox_visible_xyxy",
        "bbox_visible_normalized",
    )
    @classmethod
    def _finite_bbox(cls, value: list[float]) -> list[float]:
        if any(not (-1e-9 <= v <= 1e9) for v in value):
            raise ValueError("bbox values must be finite and in a sane range")
        return [round(float(v), 6) for v in value]


class OccluderInfo(_Strict):
    id: str = Field(pattern=r"^occluder_[0-9]{3}$")
    type: Literal["debris_sheet", "cardboard", "leaf_litter"]
    bbox_xyxy: BBox
    bbox_normalized: BBox
    area_px: int = Field(gt=0)
    mask_index: int = Field(ge=1)
    mask_path: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox_xyxy", "bbox_normalized")
    @classmethod
    def _finite(cls, value: list[float]) -> list[float]:
        if any(not (-1e-9 <= v <= 1e9) for v in value):
            raise ValueError("bbox values must be finite")
        return [round(float(v), 6) for v in value]


class Counts(_Strict):
    pothole: int = Field(ge=0)
    crack: int = Field(ge=0)
    alligator_crack: int = Field(ge=0)
    patch: int = Field(ge=0)
    total: int = Field(ge=1)
    fully_visible: int = Field(ge=0)
    partially_occluded: int = Field(ge=0)
    severely_occluded: int = Field(ge=0)


class GenerationInfo(_Strict):
    seed: int = Field(ge=0)
    family_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    generator_version: str = GENERATOR_VERSION
    config: dict[str, Any] = Field(default_factory=dict)


class GroundTruth(_Strict):
    schema_version: str = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION
    image_id: str = Field(pattern=r"^RV_[0-9]{6,10}_[a-f0-9]{6}$")
    seed: int = Field(ge=0)
    family_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    image: ImageInfo
    scene: SceneInfo
    objects: list[ObjectInfo] = Field(min_length=1)
    occluders: list[OccluderInfo] = Field(default_factory=list)
    counts: Counts
    generation: GenerationInfo

    @model_validator(mode="after")
    def _integrity(self) -> "GroundTruth":
        if self.generation.seed != self.seed:
            raise ValueError("generation.seed does not match top-level seed")
        if self.generation.family_id != self.family_id:
            raise ValueError("generation.family_id does not match")
        if self.generation.config_hash != self.config_hash:
            raise ValueError("generation.config_hash does not match")
        if self.generation.image_config_hash != self.image_config_hash:
            raise ValueError("generation.image_config_hash does not match")

        w, h = self.image.width, self.image.height
        ids = [o.id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object IDs are not unique")
        mask_indices = [o.mask_index for o in self.objects]
        if len(mask_indices) != len(set(mask_indices)):
            raise ValueError("mask indices are not unique")

        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"{obj.id}: bbox outside image or degenerate")
            nx1, ny1, nx2, ny2 = obj.bbox_normalized
            if not (0 <= nx1 < nx2 <= 1 and 0 <= ny1 < ny2 <= 1):
                raise ValueError(f"{obj.id}: normalized bbox invalid")
            expected = [x1 / w, y1 / h, x2 / w, y2 / h]
            if max(abs(a - b) for a, b in zip([nx1, ny1, nx2, ny2], expected)) > 5e-5:
                raise ValueError(f"{obj.id}: normalized bbox does not match pixels")
            if not (obj.visible_area_px <= obj.area_px):
                raise ValueError(f"{obj.id}: visible area exceeds total area")
            if abs(obj.visible_fraction - obj.visible_area_px / obj.area_px) > 5e-6:
                raise ValueError(f"{obj.id}: visible_fraction is inconsistent")
            expected_vis = (
                "fully_visible"
                if obj.visible_fraction >= 0.95
                else "partially_occluded"
                if obj.visible_fraction >= 0.15
                else "severely_occluded"
            )
            if obj.visibility != expected_vis:
                raise ValueError(f"{obj.id}: visibility label is inconsistent")
            if obj.occluded != (obj.visible_fraction < 0.95):
                raise ValueError(f"{obj.id}: occluded flag is inconsistent")
            cx, cy = obj.centroid_xy
            if not (x1 - 1e-6 <= cx <= x2 + 1e-6 and y1 - 1e-6 <= cy <= y2 + 1e-6):
                raise ValueError(f"{obj.id}: centroid outside bbox")
            ncx, ncy = obj.centroid_normalized
            if abs(ncx - cx / w) > 5e-5 or abs(ncy - cy / h) > 5e-5:
                raise ValueError(f"{obj.id}: normalized centroid inconsistent")
            if obj.visible_area_px == 0:
                if any(abs(v) > 1e-9 for v in obj.bbox_visible_xyxy):
                    raise ValueError(f"{obj.id}: visible bbox must be zeros when hidden")
            else:
                vx1, vy1, vx2, vy2 = obj.bbox_visible_xyxy
                if not (0 <= vx1 < vx2 <= w and 0 <= vy1 < vy2 <= h):
                    raise ValueError(f"{obj.id}: visible bbox invalid")

        occ_ids = [o.id for o in self.occluders]
        if len(occ_ids) != len(set(occ_ids)):
            raise ValueError("occluder IDs are not unique")
        occ_indices = [o.mask_index for o in self.occluders]
        if len(occ_indices) != len(set(occ_indices)):
            raise ValueError("occluder mask indices are not unique")
        for occ in self.occluders:
            x1, y1, x2, y2 = occ.bbox_xyxy
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"{occ.id}: bbox outside image")
            if max(
                abs(a - b)
                for a, b in zip(
                    occ.bbox_normalized,
                    [x1 / w, y1 / h, x2 / w, y2 / h],
                )
            ) > 5e-5:
                raise ValueError(f"{occ.id}: normalized bbox inconsistent")

        counts = self.counts
        by_class = {c.value: 0 for c in DefectClass}
        for obj in self.objects:
            by_class[obj.type.value] += 1
        for class_name, value in by_class.items():
            if getattr(counts, class_name) != value:
                raise ValueError(f"counts.{class_name} is inconsistent")
        if counts.total != len(self.objects):
            raise ValueError("counts.total is inconsistent")
        vis = {label: 0 for label in ("fully_visible", "partially_occluded", "severely_occluded")}
        for obj in self.objects:
            vis[obj.visibility] += 1
        if counts.fully_visible != vis["fully_visible"]:
            raise ValueError("counts.fully_visible is inconsistent")
        if counts.partially_occluded != vis["partially_occluded"]:
            raise ValueError("counts.partially_occluded is inconsistent")
        if counts.severely_occluded != vis["severely_occluded"]:
            raise ValueError("counts.severely_occluded is inconsistent")

        if self.scene.defect_count != len(self.objects):
            raise ValueError("scene.defect_count is inconsistent")
        if self.scene.occluder_count != len(self.occluders):
            raise ValueError("scene.occluder_count is inconsistent")
        return self
