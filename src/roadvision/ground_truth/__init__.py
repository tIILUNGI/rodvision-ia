"""Ground-truth engine."""

from .builder import build_ground_truth
from .io import (
    GroundTruthValidationError,
    load_annotation,
    load_mask,
    mask_stats,
    normalized_bbox,
    save_annotation,
    sha256_file,
    write_png_mask,
    write_png_rgb,
)
from .models import Counts, GenerationInfo, GroundTruth, ImageInfo, ObjectInfo, OccluderInfo, SceneInfo
from .validate import validate_ground_truth_files

__all__ = [
    "Counts",
    "GenerationInfo",
    "GroundTruth",
    "GroundTruthValidationError",
    "ImageInfo",
    "ObjectInfo",
    "OccluderInfo",
    "SceneInfo",
    "build_ground_truth",
    "load_annotation",
    "load_mask",
    "mask_stats",
    "normalized_bbox",
    "save_annotation",
    "sha256_file",
    "validate_ground_truth_files",
    "write_png_mask",
    "write_png_rgb",
]
