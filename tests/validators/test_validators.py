from __future__ import annotations

from pathlib import Path

from roadvision.config import load_config
from roadvision.dataset.build import generate_image
from roadvision.questions import generate_tasks
from roadvision.validators import validate

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _tasks(tmp_path: Path):
    config = load_config(SMOKE)
    art = generate_image(config, 10452, datasets_root=tmp_path)
    return generate_tasks(art.ground_truth, config)


def test_validators_accept_correct_answers(tmp_path: Path) -> None:
    tasks = {t.capability.value: t for t in _tasks(tmp_path)}
    assert validate(tasks["counting"], tasks["counting"].reference_answer).correct
    assert validate(tasks["classification"], tasks["classification"].reference_answer).correct
    assert validate(tasks["localization"], tasks["localization"].reference_answer).correct
    assert validate(tasks["spatial_relation"], tasks["spatial_relation"].reference_answer).correct
    assert validate(tasks["area_comparison"], tasks["area_comparison"].reference_answer).correct
    assert validate(tasks["occlusion"], tasks["occlusion"].reference_answer).correct


def test_validators_accept_reasonable_variants(tmp_path: Path) -> None:
    tasks = {t.capability.value: t for t in _tasks(tmp_path)}
    assert validate(tasks["classification"], "A pothole").correct
    assert validate(tasks["classification"], "alligator cracking").correct is False
    assert validate(tasks["counting"], "Answer: 1").correct
    box = tasks["localization"].reference_answer
    shifted = [box[0] + 0.01, box[1] + 0.01, box[2] - 0.01, box[3] - 0.01]
    assert validate(tasks["localization"], shifted).correct
    relation = tasks["spatial_relation"].reference_answer
    if relation == "right of":
        assert validate(tasks["spatial_relation"], "to the right").correct


def test_validators_reject_incorrect_or_unparseable_answers(tmp_path: Path) -> None:
    tasks = {t.capability.value: t for t in _tasks(tmp_path)}
    assert not validate(tasks["counting"], 99).correct
    assert not validate(tasks["counting"], "many").correct
    assert not validate(tasks["classification"], "banana").correct
    assert not validate(tasks["localization"], [0.0, 0.0, 0.05, 0.05]).correct
    assert not validate(tasks["localization"], "not a box").correct
    wrong_relation = {
        "left of": "right of",
        "right of": "left of",
        "above": "below",
        "below": "above",
    }[tasks["spatial_relation"].reference_answer]
    assert not validate(tasks["spatial_relation"], wrong_relation).correct
    assert not validate(tasks["occlusion"], 999).correct


def test_abstention_validator(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    # Seed 10452 contains a severely/partially occluded defect in this config
    # only in some renders; search a few seeds for an abstention task.
    abstention = None
    for seed in range(10452, 10456):
        art = generate_image(config, seed, datasets_root=tmp_path / f"s{seed}")
        for task in generate_tasks(art.ground_truth, config):
            if task.capability.value == "evidence_sufficiency":
                abstention = task
                break
        if abstention:
            break
    if abstention is None:
        return
    if abstention.reference_answer == "insufficient":
        assert validate(abstention, "I cannot determine from the image").correct
        assert not validate(abstention, "sufficient").correct
    else:
        assert validate(abstention, "sufficient").correct
        assert not validate(abstention, "insufficient evidence").correct
