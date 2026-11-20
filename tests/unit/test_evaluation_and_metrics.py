"""Unit tests for evaluation session, metrics calculation and reports."""

from roadvision.evaluation.models import EvaluationSession, SingleEvaluationRecord
from roadvision.reporting.metrics import MetricsEngine


def test_metrics_engine_calculation():
    records = [
        SingleEvaluationRecord(
            task_id="t1",
            image_id="img1",
            split="test",
            capability="counting",
            difficulty="easy",
            question_family="count_potholes",
            model_name="mock-model",
            prompt="How many?",
            answer_format="integer",
            raw_response="2",
            parsed_answer=2,
            reference_answer=2,
            correct=True,
            score=1.0,
            max_score=1.0,
            verification_mode="exact_int",
            is_abstention=False,
            elapsed_seconds=0.1,
            visual_conditions={"weather": "wet", "lighting": "harsh"},
        ),
        SingleEvaluationRecord(
            task_id="t2",
            image_id="img1",
            split="test",
            capability="localization",
            difficulty="medium",
            question_family="bbox_largest",
            model_name="mock-model",
            prompt="Return bbox",
            answer_format="bbox_normalized",
            raw_response="[0.1, 0.1, 0.5, 0.5]",
            parsed_answer=[0.1, 0.1, 0.5, 0.5],
            reference_answer=[0.1, 0.1, 0.5, 0.5],
            correct=True,
            score=1.0,
            max_score=1.0,
            verification_mode="bbox_iou",
            is_abstention=False,
            elapsed_seconds=0.2,
            details={"iou": 0.85},
            visual_conditions={"weather": "wet", "lighting": "harsh"},
        ),
        SingleEvaluationRecord(
            task_id="t3",
            image_id="img2",
            split="test",
            capability="counting",
            difficulty="easy",
            question_family="count_potholes",
            model_name="mock-model",
            prompt="How many?",
            answer_format="integer",
            raw_response="0",
            parsed_answer=0,
            reference_answer=1,
            correct=False,
            score=0.0,
            max_score=1.0,
            verification_mode="exact_int",
            is_abstention=False,
            elapsed_seconds=0.1,
            visual_conditions={"weather": "dry", "lighting": "overcast"},
        ),
    ]

    session = EvaluationSession(
        session_id="test_sess",
        dataset_name="smoke",
        split="test",
        model_name="mock-model",
        provider_type="DeterministicMockProvider",
        timestamp="2026-11-20T21:00:00Z",
        total_tasks=3,
        completed_tasks=3,
        correct_tasks=2,
        accuracy=2 / 3,
        average_latency_sec=0.133,
        records=records,
    )

    engine = MetricsEngine(session)
    overall = engine.overall()
    assert overall.total == 3
    assert overall.correct == 2
    assert abs(overall.accuracy - 0.6667) < 0.001

    by_cap = engine.by_capability()
    assert "counting" in by_cap
    assert by_cap["counting"].total == 2
    assert by_cap["counting"].correct == 1
    assert "localization" in by_cap
    assert by_cap["localization"].additional["mean_iou"] == 0.85

    by_cond = engine.by_visual_condition()
    assert by_cond["weather"]["wet"].correct == 2
    assert by_cond["weather"]["dry"].correct == 0
