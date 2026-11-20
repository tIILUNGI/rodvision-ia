# RoadVision AI — Data Schema Specification

## 1. Ground Truth Schema (`schema_version: "1.0"`)

Ground truth records are stored in `datasets/annotations/<dataset>/<image_id>.json`.

```json
{
  "schema_version": "1.0",
  "generator_version": "0.1.0",
  "image_id": "RV_010452_56ac55",
  "seed": 10452,
  "family_id": "fam_02613",
  "config_hash": "a4b2c1...",
  "image_config_hash": "56ac55...",
  "image": {
    "path": "generated/smoke/images/RV_010452_56ac55.png",
    "width": 768,
    "height": 512,
    "sha256": "8f2d...3b1"
  },
  "scene": {
    "surface": "asphalt",
    "weather": "wet",
    "lighting": "harsh",
    "shadow_present": true,
    "shadow_coverage": 0.28,
    "wet_coverage": 0.35,
    "perspective_strength": 0.55,
    "difficulty": "medium",
    "defect_count": 2,
    "occluder_count": 1,
    "scenario_tags": ["wet", "harsh", "shadow", "partial_occlusion"]
  },
  "objects": [
    {
      "id": "damage_001",
      "type": "pothole",
      "bbox_xyxy": [180.0, 210.0, 310.0, 330.0],
      "bbox_normalized": [0.234375, 0.410156, 0.403646, 0.644531],
      "bbox_visible_xyxy": [180.0, 210.0, 310.0, 330.0],
      "bbox_visible_normalized": [0.234375, 0.410156, 0.403646, 0.644531],
      "centroid_xy": [245.0, 270.0],
      "centroid_normalized": [0.31901, 0.527344],
      "area_px": 14250,
      "visible_area_px": 14250,
      "visible_fraction": 1.0,
      "visibility": "fully_visible",
      "occluded": false,
      "mask_index": 1,
      "mask_path": "annotations/smoke/masks/RV_010452_56ac55_instances.png",
      "visible_mask_path": "annotations/smoke/masks/RV_010452_56ac55_visible.png",
      "attributes": {}
    }
  ],
  "occluders": [],
  "counts": {
    "pothole": 1,
    "crack": 1,
    "alligator_crack": 0,
    "patch": 0,
    "total": 2,
    "fully_visible": 2,
    "partially_occluded": 0,
    "severely_occluded": 0
  }
}
```

## 2. Public vs Internal Task Schema

Tasks are stored in `datasets/tasks/<dataset>/`:
- `public_tasks.jsonl`: Strip reference answers, verification rules, and object IDs to prevent any data leakage to evaluated models.
- `tasks.jsonl`: Full internal records utilized by the deterministic verifier.
