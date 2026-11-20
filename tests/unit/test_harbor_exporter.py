"""Unit tests for Harbor 1.4 package exporter."""

from pathlib import Path
from roadvision.harbor.exporter import export_harbor_dataset, export_harbor_task
from roadvision.questions.models import AnswerFormat, Capability, Task, Verification


def test_export_single_harbor_task(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    img_dir = datasets_root / "images"
    img_dir.mkdir(parents=True)
    img_file = img_dir / "sample.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfake_image")

    task = Task(
        task_id="RV_001001_123456_q01",
        image_id="RV_001001_123456",
        image_path="images/sample.png",
        split="smoke",
        capability=Capability.COUNTING,
        difficulty="easy",
        question_family="count_potholes",
        prompt="How many potholes are visible in this road?",
        answer_format=AnswerFormat.INTEGER,
        answer_hint="A single non-negative integer.",
        reference_answer=2,
        verification=Verification(mode="exact_int", value=2),
    )

    out_harbor = tmp_path / "harbor_tasks"
    pkg_dir = export_harbor_task(task, datasets_root, out_harbor)

    assert pkg_dir.exists()
    assert (pkg_dir / "task.toml").exists()
    assert (pkg_dir / "instruction.md").exists()
    assert (pkg_dir / "input" / "image.png").exists()
    assert (pkg_dir / "tests" / "test.sh").exists()
    assert (pkg_dir / "solution" / "solve.sh").exists()

    # Verify no leakage in instruction.md
    inst = (pkg_dir / "instruction.md").read_text(encoding="utf-8")
    assert "2" not in inst.split()  # The reference count 2 must NOT appear
    assert "task.toml" in [f.name for f in pkg_dir.iterdir()]
