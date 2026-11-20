from __future__ import annotations

from pathlib import Path

from roadvision.config import load_config
from roadvision.dataset.build import generate_image
from roadvision.ground_truth import load_annotation, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_same_seed_is_reproducible(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    a = generate_image(config, 10452, datasets_root=tmp_path / "a")
    b = generate_image(config, 10452, datasets_root=tmp_path / "b")

    assert a.plan.image_id == b.plan.image_id
    assert sha256_file(a.image_path) == sha256_file(b.image_path)
    assert sha256_file(a.annotation_path) == sha256_file(b.annotation_path)

    gt_a = load_annotation(a.annotation_path)
    gt_b = load_annotation(b.annotation_path)
    assert gt_a.model_dump(mode="json") == gt_b.model_dump(mode="json")
    assert gt_a.image.sha256 == gt_b.image.sha256


def test_milestone_seed_has_controlled_defects(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    art = generate_image(config, 10452, datasets_root=tmp_path)
    gt = art.ground_truth
    classes = {obj.type.value for obj in gt.objects}
    assert len(gt.objects) >= 2
    assert {"pothole", "crack"}.issubset(classes) or len(classes) >= 2
    assert gt.scene.defect_count == len(gt.objects)
    assert gt.counts.total == len(gt.objects)
    assert all(obj.area_px > 0 for obj in gt.objects)
    assert all(obj.visible_area_px <= obj.area_px for obj in gt.objects)
