"""Export benchmark reports to Markdown, CSV, and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..evaluation.models import EvaluationSession
from .metrics import MetricsEngine


def export_markdown_report(session: EvaluationSession, output_path: Path | str) -> Path:
    """Generate a clean, high-impact scientific Markdown summary report."""
    engine = MetricsEngine(session)
    report = engine.full_report_dict()
    overall = report["overall"]
    caps = report["capabilities"]
    diffs = report["difficulty"]
    conditions = report["visual_conditions"]

    md = []
    md.append(f"# RoadVision AI — Evaluation Report")
    md.append(f"**Model:** `{session.model_name}` | **Dataset:** `{session.dataset_name}` (`{session.split}`) | **Date:** {session.timestamp[:19]}")
    md.append(f"**Overall Accuracy:** **{overall['accuracy'] * 100:.1f}%** ({overall['correct']}/{overall['total']} tasks)")
    md.append(f"**Average Latency:** {overall['additional'].get('average_latency_sec', 0):.2f}s per task\n")

    md.append("## 1. Performance by Visual Reasoning Capability")
    md.append("| Capability | Accuracy | Correct | Total | Special Metrics |")
    md.append("|---|---|---|---|---|")
    for cap_name, data in caps.items():
        extras = []
        if "mean_iou" in data.get("additional", {}):
            extras.append(f"Mean IoU: {data['additional']['mean_iou']:.3f}")
        if "abstention_precision" in data.get("additional", {}):
            extras.append(f"Abst. Prec: {data['additional']['abstention_precision']:.2f}")
            extras.append(f"Abst. Rec: {data['additional']['abstention_recall']:.2f}")
        extra_str = ", ".join(extras) if extras else "-"
        md.append(f"| **{cap_name.replace('_', ' ').title()}** | **{data['accuracy']*100:.1f}%** | {data['correct']} | {data['total']} | {extra_str} |")
    md.append("")

    md.append("## 2. Performance by Difficulty Tier")
    md.append("| Tier | Accuracy | Correct / Total |")
    md.append("|---|---|---|")
    for d_name, data in diffs.items():
        md.append(f"| {d_name.capitalize()} | **{data['accuracy']*100:.1f}%** | {data['correct']} / {data['total']} |")
    md.append("")

    md.append("## 3. Environmental Robustness")
    md.append("### Weather Conditions")
    md.append("| Weather | Accuracy | Tasks |")
    md.append("|---|---|---|")
    for w_name, data in conditions.get("weather", {}).items():
        md.append(f"| {w_name.capitalize()} | {data['accuracy']*100:.1f}% | {data['total']} |")
    md.append("")

    md.append("### Lighting Variations")
    md.append("| Lighting | Accuracy | Tasks |")
    md.append("|---|---|---|")
    for l_name, data in conditions.get("lighting", {}).items():
        md.append(f"| {l_name.capitalize()} | {data['accuracy']*100:.1f}% | {data['total']} |")
    md.append("")

    md.append("---")
    md.append("*Generated automatically by RoadVision AI Benchmark Framework.*")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def export_csv_report(session: EvaluationSession, output_path: Path | str) -> Path:
    """Export item-level results to CSV for tabular analytics and pandas loading."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_id",
        "image_id",
        "split",
        "capability",
        "difficulty",
        "question_family",
        "correct",
        "score",
        "verification_mode",
        "raw_response",
        "parsed_answer",
        "reference_answer",
        "elapsed_seconds",
        "weather",
        "lighting",
    ]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in session.records:
            writer.writerow({
                "task_id": r.task_id,
                "image_id": r.image_id,
                "split": r.split,
                "capability": r.capability,
                "difficulty": r.difficulty,
                "question_family": r.question_family,
                "correct": r.correct,
                "score": r.score,
                "verification_mode": r.verification_mode,
                "raw_response": str(r.raw_response),
                "parsed_answer": str(r.parsed_answer),
                "reference_answer": str(r.reference_answer),
                "elapsed_seconds": r.elapsed_seconds,
                "weather": r.visual_conditions.get("weather", ""),
                "lighting": r.visual_conditions.get("lighting", ""),
            })
    return out
