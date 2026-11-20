"""Pydantic configuration models.

The configuration is intentionally small and explicit. Anything that changes
pixels or ground truth must be represented here so that it participates in the
canonical configuration hash.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DefectClass(str, Enum):
    POTHOLE = "pothole"
    CRACK = "crack"
    ALLIGATOR_CRACK = "alligator_crack"
    PATCH = "patch"


class Weather(str, Enum):
    DRY = "dry"
    DAMP = "damp"
    WET = "wet"


class Lighting(str, Enum):
    OVERCAST = "overcast"
    GOLDEN = "golden"
    HARSH = "harsh"


class OccluderType(str, Enum):
    DEBRIS_SHEET = "debris_sheet"
    CARDBOARD = "cardboard"
    LEAF_LITTER = "leaf_litter"


class Capability(str, Enum):
    COUNTING = "counting"
    CLASSIFICATION = "classification"
    LOCALIZATION = "localization"
    SPATIAL_RELATION = "spatial_relation"
    AREA_COMPARISON = "area_comparison"
    OCCLUSION = "occlusion"
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DatasetConfig(_Model):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    split: Literal["smoke", "dev", "test"]
    seed_start: int = Field(ge=0, le=2**32 - 1)
    num_images: int = Field(ge=1, le=2000)
    # Images whose seeds fall in the same block share a scenario family. Splits
    # must use disjoint blocks.
    family_block: int = Field(default=5, ge=1, le=10000)

    @model_validator(mode="after")
    def _seed_range(self) -> "DatasetConfig":
        if self.seed_start + self.num_images - 1 > 2**32 - 1:
            raise ValueError("seed range exceeds uint32")
        return self


class RenderConfig(_Model):
    width: int = Field(default=768, ge=320, le=2048)
    height: int = Field(default=512, ge=240, le=2048)
    perspective_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    grain_sigma: float = Field(default=2.2, ge=0.0, le=10.0)
    blur_sigma: float = Field(default=0.35, ge=0.0, le=3.0)
    vignette: float = Field(default=0.12, ge=0.0, le=0.8)

    @model_validator(mode="after")
    def _aspect(self) -> "RenderConfig":
        ratio = self.width / self.height
        if not (1.0 <= ratio <= 2.4):
            raise ValueError("width/height ratio must be between 1.0 and 2.4")
        return self


class DefectConfig(_Model):
    count_min: int = Field(default=1, ge=1, le=12)
    count_max: int = Field(default=4, ge=1, le=12)
    class_weights: dict[DefectClass, float]
    guarantee_classes: list[DefectClass] = Field(default_factory=list)
    size_scale: float = Field(default=1.0, ge=0.4, le=2.5)

    @field_validator("class_weights")
    @classmethod
    def _weights(cls, value: dict[DefectClass, float]) -> dict[DefectClass, float]:
        if not value:
            raise ValueError("class_weights must not be empty")
        for key, weight in value.items():
            if weight <= 0:
                raise ValueError(f"class weight for {key} must be positive")
        return value

    @model_validator(mode="after")
    def _counts(self) -> "DefectConfig":
        if self.count_min > self.count_max:
            raise ValueError("count_min must be <= count_max")
        if len(self.guarantee_classes) > self.count_max:
            raise ValueError("guarantee_classes cannot exceed count_max")
        if len(set(self.guarantee_classes)) != len(self.guarantee_classes):
            raise ValueError("guarantee_classes contains duplicates")
        return self


class EnvironmentConfig(_Model):
    weather: list[Weather] = Field(default_factory=lambda: [Weather.DRY, Weather.WET])
    lighting: list[Lighting] = Field(
        default_factory=lambda: [Lighting.OVERCAST, Lighting.GOLDEN, Lighting.HARSH]
    )
    shadow_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    strong_shadow_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    wet_puddle_strength: float = Field(default=0.55, ge=0.0, le=1.0)

    @field_validator("weather", "lighting")
    @classmethod
    def _non_empty_unique(cls, value: list) -> list:
        if not value:
            raise ValueError("condition list must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("condition list contains duplicates")
        return value


class OcclusionConfig(_Model):
    enabled: bool = True
    probability: float = Field(default=0.45, ge=0.0, le=1.0)
    severe_share: float = Field(default=0.40, ge=0.0, le=1.0)
    types: list[OccluderType] = Field(
        default_factory=lambda: [
            OccluderType.DEBRIS_SHEET,
            OccluderType.CARDBOARD,
        ]
    )

    @field_validator("types")
    @classmethod
    def _types(cls, value: list[OccluderType]) -> list[OccluderType]:
        if not value:
            raise ValueError("occlusion.types must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("occlusion.types contains duplicates")
        return value


class QuestionConfig(_Model):
    capabilities: list[Capability] = Field(
        default_factory=lambda: list(Capability)
    )
    max_per_image: int = Field(default=7, ge=1, le=30)
    min_questions: int = Field(default=3, ge=1, le=30)
    include_abstention_controls: bool = True

    @field_validator("capabilities")
    @classmethod
    def _capabilities(cls, value: list[Capability]) -> list[Capability]:
        if not value:
            raise ValueError("questions.capabilities must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("questions.capabilities contains duplicates")
        return value

    @model_validator(mode="after")
    def _minimum(self) -> "QuestionConfig":
        if self.min_questions > self.max_per_image:
            raise ValueError("min_questions must be <= max_per_image")
        return self


class RoadVisionConfig(_Model):
    dataset: DatasetConfig
    render: RenderConfig = Field(default_factory=RenderConfig)
    defects: DefectConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    occlusion: OcclusionConfig = Field(default_factory=OcclusionConfig)
    questions: QuestionConfig = Field(default_factory=QuestionConfig)

    @field_validator("dataset")
    @classmethod
    def _dataset_name(cls, value: DatasetConfig) -> DatasetConfig:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", value.name):
            raise ValueError("dataset.name must be filesystem-safe")
        return value
