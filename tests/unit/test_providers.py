"""Unit tests for multimodal providers and interfaces."""

from pathlib import Path
from roadvision.providers import DeterministicMockProvider, ManualFileProvider, get_provider


def test_deterministic_mock_provider(tmp_path: Path):
    p1 = DeterministicMockProvider(seed=123)
    p2 = DeterministicMockProvider(seed=123)

    resp1 = p1.predict(
        task_id="RV_001_q01",
        image_path="dummy.png",
        prompt="How many potholes are visible?",
        answer_format="integer",
    )
    resp2 = p2.predict(
        task_id="RV_001_q01",
        image_path="dummy.png",
        prompt="How many potholes are visible?",
        answer_format="integer",
    )

    # Identical seed and task ID must yield identical answer
    assert resp1.raw_text == resp2.raw_text
    assert resp1.is_success
    assert resp1.usage.total_tokens > 0


def test_manual_file_provider(tmp_path: Path):
    ans_file = tmp_path / "answers.json"
    ans_file.write_text('{"RV_001_q01": "2", "RV_001_q02": "pothole"}', encoding="utf-8")

    provider = ManualFileProvider(answers_file=ans_file)
    r1 = provider.predict("RV_001_q01", "dummy.png", "Count?", "integer")
    assert r1.raw_text == "2"
    assert r1.is_success

    r_missing = provider.predict("RV_UNKNOWN", "dummy.png", "Count?", "integer")
    assert not r_missing.is_success
    assert "not found" in r_missing.error


def test_provider_factory():
    mock_p = get_provider("mock", model_name="test-mock")
    assert isinstance(mock_p, DeterministicMockProvider)
