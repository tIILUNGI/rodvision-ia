"""Harbor package export routes."""

import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...harbor.exporter import export_harbor_dataset
from ..schemas import HarborExportRequest

router = APIRouter(prefix="/api/harbor", tags=["harbor"])


@router.post("/export")
def export_dataset_for_harbor(req: HarborExportRequest):
    out_dir = Path("benchmarks/harbor_export") / req.dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    tasks = export_harbor_dataset(
        dataset_name=req.dataset_name,
        datasets_root="datasets",
        output_dir=out_dir,
        limit=req.limit,
    )

    # Create zip archive for immediate download
    zip_base = Path("reports") / f"harbor_{req.dataset_name}_v1_4"
    zip_path = shutil.make_archive(str(zip_base), "zip", out_dir)

    return {
        "status": "success",
        "tasks_exported": len(tasks),
        "harbor_schema": "1.4",
        "output_directory": str(out_dir).replace("\\", "/"),
        "archive_path": str(zip_path).replace("\\", "/"),
        "download_url": f"/api/harbor/download/{req.dataset_name}",
    }


@router.get("/download/{dataset_name}")
def download_harbor_archive(dataset_name: str):
    zip_file = Path("reports") / f"harbor_{dataset_name}_v1_4.zip"
    if not zip_file.exists():
        raise HTTPException(status_code=404, detail="Archive not generated yet")
    return FileResponse(zip_file, media_type="application/zip", filename=f"roadvision_harbor_{dataset_name}.zip")
