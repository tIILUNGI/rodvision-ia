from __future__ import annotations

import json
from pathlib import Path

from roadvision.config import load_config
from roadvision.dataset.build import build_dataset, generate_image
from roadvision.dataset.validate import validate_dataset
from roadvision.ground_truth import load_annotation, load_mask, validate_ground_truth_files

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_annotation_matches_masks_and_counts(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    art = generate_image(config, 10452, datasets_root=tmp_path)
    gt = load_annotation(art.annotation_path)
    validate_ground_truth_files(gt, tmp_path)

    instance = load_mask(tmp_path / gt.objects[0].mask_path)
    visible = load_mask(tmp_path / gt.objects[0].visible_mask_path)
    assert instance.shape == (gt.image.height, gt.image.width)
    assert visible.shape == (gt.image.height, gt.image.width)

    present = {int(v) for v in instance.reshape(-1) if v != 0}
    assert present == {obj.mask_index for obj in gt.objects}

    by_class = {}
    for obj in gt.objects:
        by_class[obj.type.value] = by_class.get(obj.type.value, 0) + 1
        assert obj.bbox_normalized == [
            round(obj.bbox_xyxy[0] / gt.image.width, 6),
            round(obj.bbox_xyxy[1] / gt.image.height, 6),
            round(obj.bbox_xyxy[2] / gt.image.width, 6),
            round(obj.bbox_xyxy[3] / gt.image.height, 6),
        ]
        vf = obj.visible_area_px / obj.area_px
        assert abs(vf - obj.visible_fraction) <= 1e-6
    assert gt.counts.total == sum(by_class.values())
    assert gt.counts.pothole == by_class.get("pothole", 0)
    assert gt.counts.crack == by_class.get("crack", 0)


def test_dataset_manifest_and_public_tasks_do_not_leak(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    build_dataset(config, datasets_root=tmp_path, force=True)
    report = validate_dataset(tmp_path, "smoke")
    assert report["images"] == 4
    assert report["tasks"] >= 12

    public_path = tmp_path / "tasks" / "smoke" / "public_tasks.jsonl"
    for line in public_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "reference_answer" not in record
        assert "verification" not in record
        assert "related_object_ids" not in record
        assert "visual_conditions" not in record
        assert "defect_types" not in record
