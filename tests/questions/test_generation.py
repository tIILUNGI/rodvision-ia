from __future__ import annotations

from pathlib import Path

from roadvision.config import load_config
from roadvision.dataset.build import generate_image
from roadvision.questions import generate_tasks

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_milestone_seed_generates_required_questions(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    art = generate_image(config, 10452, datasets_root=tmp_path)
    tasks = generate_tasks(art.ground_truth, config)

    assert len(tasks) >= 3
    assert len({t.capability.value for t in tasks}) >= 3
    assert all(t.reference_answer is not None for t in tasks)
    assert len({t.task_id for t in tasks}) == len(tasks)

    for task in tasks:
        task.assert_no_leakage()
        public = task.public_payload()
        assert "reference_answer" not in public
        assert "verification" not in public
        assert public["task_id"] == task.task_id
        assert public["prompt"]


def test_question_generation_is_deterministic(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    art = generate_image(config, 10452, datasets_root=tmp_path)
    a = [t.internal_payload() for t in generate_tasks(art.ground_truth, config)]
    b = [t.internal_payload() for t in generate_tasks(art.ground_truth, config)]
    assert a == b
    assert any(t["capability"] == "counting" for t in a)
    assert any(t["capability"] == "classification" for t in a)
    assert any(t["capability"] == "localization" for t in a)
