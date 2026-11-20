"""Evaluation runner engine.

Executes a benchmark run against a dataset, preserving strict evaluation hygiene:
1. Public tasks and images only are sent to the provider.
2. The model response is parsed and checked by deterministic validators.
3. Every step is logged with exact timings and error tracking.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..ground_truth.models import GroundTruth
from ..providers.base import BaseMultimodalProvider
from ..questions.models import Task
from ..validators.core import validate
from .models import EvaluationSession, SingleEvaluationRecord


def run_evaluation(
    dataset_name: str,
    provider: BaseMultimodalProvider,
    *,
    datasets_root: str | Path = "datasets",
    max_tasks: int | None = None,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> EvaluationSession:
    """Run an evaluation benchmark on the specified dataset using the given provider."""
    root = Path(datasets_root)
    internal_tasks_path = root / f"tasks/{dataset_name}/tasks.jsonl"
    if not internal_tasks_path.exists():
        raise FileNotFoundError(f"Internal tasks file not found: {internal_tasks_path}")

    # Load tasks (with internal verification metadata)
    tasks: list[Task] = []
    with internal_tasks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                tasks.append(Task.model_validate_json(line))

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    records: list[SingleEvaluationRecord] = []
    total = len(tasks)
    correct_count = 0
    total_elapsed = 0.0

    for idx, task in enumerate(tasks, start=1):
        # Resolve image path relative to datasets_root
        img_full_path = root / task.image_path

        # STRICT ISOLATION: Provide only public inputs to the model
        t0 = time.perf_counter()
        resp = provider.predict(
            task_id=task.task_id,
            image_path=img_full_path,
            prompt=task.prompt,
            answer_format=task.answer_format.value,
            choices=task.choices,
            answer_hint=task.answer_hint,
        )
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        # Deterministic verification
        val_result = validate(task, resp.raw_text)
        if val_result.correct:
            correct_count += 1

        rec = SingleEvaluationRecord(
            task_id=task.task_id,
            image_id=task.image_id,
            split=task.split,
            capability=task.capability.value,
            difficulty=task.difficulty,
            question_family=task.question_family,
            model_name=provider.model_name,
            prompt=task.prompt,
            answer_format=task.answer_format.value,
            raw_response=resp.raw_text,
            parsed_answer=val_result.parsed,
            reference_answer=task.reference_answer,
            correct=val_result.correct,
            score=val_result.score,
            max_score=task.max_score,
            verification_mode=val_result.mode,
            is_abstention=val_result.is_abstention,
            elapsed_seconds=round(resp.elapsed_seconds, 3),
            details=val_result.details,
            visual_conditions=task.visual_conditions,
            defect_types=task.defect_types,
        )
        records.append(rec)

        if progress_callback:
            progress_callback(idx, total, task.task_id, val_result.correct)

    accuracy = correct_count / total if total > 0 else 0.0
    avg_latency = total_elapsed / total if total > 0 else 0.0

    session = EvaluationSession(
        session_id=session_id,
        dataset_name=dataset_name,
        split=tasks[0].split if tasks else "unknown",
        model_name=provider.model_name,
        provider_type=provider.__class__.__name__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_tasks=total,
        completed_tasks=len(records),
        correct_tasks=correct_count,
        accuracy=round(accuracy, 4),
        average_latency_sec=round(avg_latency, 3),
        records=records,
    )

    # Automatically persist session
    session.save(root.parent / "reports" / "evaluations")
    return session
