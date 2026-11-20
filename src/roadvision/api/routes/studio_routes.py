"""Studio and visual generation routes."""

import base64
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from ...config.loader import load_config
from ...dataset.build import generate_image
from ...ground_truth.io import load_annotation
from ...questions.factory import generate_tasks
from ..schemas import GenerateRequest

router = APIRouter(prefix="/api/studio", tags=["studio"])


@router.post("/generate")
def generate_scene(req: GenerateRequest):
    cfg_path = Path("configs") / f"{req.config_name}.yaml"
    if not cfg_path.exists():
        cfg_path = Path("configs/smoke.yaml")

    config = load_config(cfg_path)
    artifacts = generate_image(config, req.seed, datasets_root="datasets")
    gt = artifacts.ground_truth
    tasks = generate_tasks(gt, config)

    # Encode main image and segmentation masks as base64 for immediate visual display
    img_bytes = artifacts.image_path.read_bytes()
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    root = Path("datasets")
    masks_b64 = {}
    inst_p = root / gt.mask_path
    if inst_p.exists():
        masks_b64["instances"] = "data:image/png;base64," + base64.b64encode(inst_p.read_bytes()).decode("utf-8")
    vis_p = root / gt.visible_mask_path
    if vis_p.exists():
        masks_b64["visible"] = "data:image/png;base64," + base64.b64encode(vis_p.read_bytes()).decode("utf-8")
    occ_p = root / gt.occluder_mask_path
    if occ_p.exists():
        masks_b64["occluders"] = "data:image/png;base64," + base64.b64encode(occ_p.read_bytes()).decode("utf-8")

    return {
        "image_id": gt.image_id,
        "seed": req.seed,
        "image_path": str(artifacts.image_path).replace("\\", "/"),
        "image_base64": f"data:image/png;base64,{b64_img}",
        "masks_base64": masks_b64,
        "width": gt.image.width,
        "height": gt.image.height,
        "defects_count": len(gt.objects),
        "occluders_count": len(gt.occluders),
        "scene": gt.scene.model_dump(mode="json"),
        "objects": [o.model_dump(mode="json") for o in gt.objects],
        "occluders": [occ.model_dump(mode="json") for occ in gt.occluders],
        "tasks": [t.public_payload() for t in tasks],
        "elapsed_seconds": artifacts.elapsed_sec,
    }


@router.get("/image/{dataset}/{image_id}")
def get_image(dataset: str, image_id: str):
    path = Path("datasets/generated") / dataset / "images" / f"{image_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/mask/{dataset}/{image_id}/{mask_type}")
def get_mask(dataset: str, image_id: str, mask_type: str):
    # mask_type: instances, visible, occluders
    path = Path("datasets/annotations") / dataset / "masks" / f"{image_id}_{mask_type}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Mask {mask_type} not found")
    return FileResponse(path, media_type="image/png")


@router.get("/annotation/{dataset}/{image_id}")
def get_annotation(dataset: str, image_id: str):
    path = Path("datasets/annotations") / dataset / f"{image_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotation not found")
    gt = load_annotation(path)
    return gt.model_dump(mode="json")
