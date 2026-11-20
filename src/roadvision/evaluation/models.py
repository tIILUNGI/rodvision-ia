"""Evaluation data models and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..version import GENERATOR_VERSION, SCHEMA_VERSION


@dataclass
class SingleEvaluationRecord:
    """Evaluation result for one task."""
    task_id: str
    image_id: str
    split: str
    capability: str
    difficulty: str
    question_family: str
    model_name: str
    prompt: str
    answer_format: str
    raw_response: str
    parsed_answer: Any
    reference_answer: Any
    correct: bool
    score: float
    max_score: float
    verification_mode: str
    is_abstention: bool
    elapsed_seconds: float
    details: dict[str, Any] = field(default_factory=dict)
    visual_conditions: dict[str, str] = field(default_factory=dict)
    defect_types: list[str] = field(default_factory=list)


@dataclass
class EvaluationSession:
    """Full evaluation run metadata and records."""
    session_id: str
    dataset_name: str
    split: str
    model_name: str
    provider_type: str
    timestamp: str
    total_tasks: int
    completed_tasks: int
    correct_tasks: int
    accuracy: float
    average_latency_sec: float
    records: list[SingleEvaluationRecord]
    generator_version: str = GENERATOR_VERSION
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path | str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        file_path = out / f"eval_{self.model_name}_{self.dataset_name}_{self.session_id}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
        return file_path
