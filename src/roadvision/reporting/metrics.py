"""Scientific metrics engine for RoadVision benchmark evaluations.

Calculates fine-grained performance across capabilities, difficulty levels,
visual conditions (weather, lighting, shadows), and defect types.
Enforces honest metrics with explicit sample size denominators.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..evaluation.models import EvaluationSession, SingleEvaluationRecord


@dataclass
class MetricSummary:
    name: str
    total: int
    correct: int
    accuracy: float
    additional: dict[str, Any] = field(default_factory=dict)


class MetricsEngine:
    """Computes comprehensive scientific benchmark breakdowns from evaluation records."""

    def __init__(self, session: EvaluationSession) -> None:
        self.session = session
        self.records = session.records

    def overall(self) -> MetricSummary:
        total = len(self.records)
        correct = sum(1 for r in self.records if r.correct)
        acc = correct / total if total > 0 else 0.0
        return MetricSummary(
            name="Overall Benchmark",
            total=total,
            correct=correct,
            accuracy=round(acc, 4),
            additional={
                "average_latency_sec": self.session.average_latency_sec,
                "model_name": self.session.model_name,
            },
        )

    def by_capability(self) -> dict[str, MetricSummary]:
        groups: dict[str, list[SingleEvaluationRecord]] = defaultdict(list)
        for r in self.records:
            groups[r.capability].append(r)

        results: dict[str, MetricSummary] = {}
        for cap, items in sorted(groups.items()):
            n = len(items)
            c = sum(1 for x in items if x.correct)
            acc = c / n if n > 0 else 0.0
            extra: dict[str, Any] = {}

            # Capability-specific telemetry
            if cap == "localization":
                ious = [
                    x.details.get("iou", 0.0)
                    for x in items
                    if "iou" in x.details
                ]
                extra["mean_iou"] = round(sum(ious) / len(ious), 4) if ious else 0.0

            elif cap == "evidence_sufficiency":
                tp = sum(1 for x in items if x.correct and x.is_abstention)
                fp = sum(1 for x in items if not x.correct and x.is_abstention)
                fn = sum(1 for x in items if not x.correct and not x.is_abstention and x.reference_answer == "insufficient")
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                extra["abstention_precision"] = round(precision, 4)
                extra["abstention_recall"] = round(recall, 4)

            results[cap] = MetricSummary(
                name=cap,
                total=n,
                correct=c,
                accuracy=round(acc, 4),
                additional=extra,
            )
        return results

    def by_difficulty(self) -> dict[str, MetricSummary]:
        groups: dict[str, list[SingleEvaluationRecord]] = defaultdict(list)
        for r in self.records:
            groups[r.difficulty].append(r)

        results: dict[str, MetricSummary] = {}
        for diff in ("easy", "medium", "hard"):
            items = groups.get(diff, [])
            n = len(items)
            c = sum(1 for x in items if x.correct)
            acc = c / n if n > 0 else 0.0
            results[diff] = MetricSummary(
                name=diff,
                total=n,
                correct=c,
                accuracy=round(acc, 4),
            )
        return results

    def by_visual_condition(self) -> dict[str, dict[str, MetricSummary]]:
        """Accuracy breakdown by weather (dry/damp/wet) and lighting (overcast/golden/harsh)."""
        breakdown: dict[str, dict[str, MetricSummary]] = {"weather": {}, "lighting": {}}

        for cond_key in ("weather", "lighting"):
            grouped: dict[str, list[SingleEvaluationRecord]] = defaultdict(list)
            for r in self.records:
                val = r.visual_conditions.get(cond_key, "unknown")
                grouped[val].append(r)

            for val, items in sorted(grouped.items()):
                n = len(items)
                c = sum(1 for x in items if x.correct)
                acc = c / n if n > 0 else 0.0
                breakdown[cond_key][val] = MetricSummary(
                    name=f"{cond_key}:{val}",
                    total=n,
                    correct=c,
                    accuracy=round(acc, 4),
                )
        return breakdown

    def full_report_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session.session_id,
            "model_name": self.session.model_name,
            "dataset_name": self.session.dataset_name,
            "split": self.session.split,
            "timestamp": self.session.timestamp,
            "overall": self.overall().__dict__,
            "capabilities": {k: v.__dict__ for k, v in self.by_capability().items()},
            "difficulty": {k: v.__dict__ for k, v in self.by_difficulty().items()},
            "visual_conditions": {
                k: {sub_k: sub_v.__dict__ for sub_k, sub_v in sub_dict.items()}
                for k, sub_dict in self.by_visual_condition().items()
            },
        }
