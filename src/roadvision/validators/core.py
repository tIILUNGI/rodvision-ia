"""Deterministic answer validation.

Validation is deliberately strict about the answer *format* and tolerant of
presentation noise (case, whitespace, punctuation, code fences). Tolerances
are documented in ``docs/VALIDATION.md`` and mirrored here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..config.models import DefectClass
from ..questions.models import AnswerFormat, Task

CLASS_ALIASES: dict[str, str] = {
    "pothole": "pothole",
    "potholes": "pothole",
    "pot_hole": "pothole",
    "pot_holes": "pothole",
    "hole": "pothole",
    "crack": "crack",
    "cracks": "crack",
    "cracking": "crack",
    "linear_crack": "crack",
    "line_crack": "crack",
    "alligator_crack": "alligator_crack",
    "alligator_cracks": "alligator_crack",
    "alligator_cracking": "alligator_crack",
    "alligator": "alligator_crack",
    "crocodile_crack": "alligator_crack",
    "crocodile_cracking": "alligator_crack",
    "fatigue_crack": "alligator_crack",
    "fatigue_cracking": "alligator_crack",
    "patch": "patch",
    "patches": "patch",
    "patching": "patch",
    "repaired_patch": "patch",
    "repair": "patch",
}

CHOICE_ALIASES: dict[str, str] = {
    "left": "left of",
    "to_the_left": "left of",
    "left_of": "left of",
    "right": "right of",
    "to_the_right": "right of",
    "right_of": "right of",
    "above": "above",
    "top": "above",
    "upper": "above",
    "below": "below",
    "bottom": "below",
    "lower": "below",
}

ABSTAIN_PATTERNS = (
    "insufficient",
    "insufficient evidence",
    "cannot determine",
    "can not determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "not enough",
    "not sufficient",
    "unknown",
    "abstain",
    "n/a",
    "no answer",
)

YES = {"yes", "y", "true", "1", "correct", "present"}
NO = {"no", "n", "false", "0", "incorrect", "absent"}


@dataclass
class ValidationResult:
    task_id: str
    correct: bool
    score: float
    mode: str
    expected: Any
    parsed: Any
    is_abstention: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "correct": self.correct,
            "score": self.score,
            "mode": self.mode,
            "expected": self.expected,
            "parsed": self.parsed,
            "is_abstention": self.is_abstention,
            "details": self.details,
        }


def _strip_noise(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'` ")
    text = re.sub(r"[.;!]+$", "", text)
    return text.strip()


def _snake(value: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", value.strip().lower())
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    return cleaned


def _tokens(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip().split()


def _has_phrase(tokens: list[str], phrase: str) -> bool:
    target = phrase.replace("_", " ").split()
    if not target:
        return False
    width = len(target)
    return any(
        tokens[i : i + width] == target for i in range(len(tokens) - width + 1)
    )


def _unwrap(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("answer", "response", "output", "text"):
            if key in raw:
                return raw[key]
    return raw


def parse_class(raw: Any) -> str | None:
    value = _unwrap(raw)
    if not isinstance(value, str):
        return None
    tokens = _tokens(_strip_noise(value))
    for alias in sorted(CLASS_ALIASES, key=len, reverse=True):
        if _has_phrase(tokens, alias):
            return CLASS_ALIASES[alias]
    return None


def parse_int(raw: Any) -> int | None:
    value = _unwrap(raw)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else None
    if isinstance(value, str):
        text = _strip_noise(value)
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.0+", text):
            return int(float(text))
        numbers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text)
        if len(numbers) == 1:
            return int(numbers[0])
    return None


def parse_bool(raw: Any) -> bool | None:
    value = _unwrap(raw)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = _strip_noise(value).lower()
        if token in YES:
            return True
        if token in NO:
            return False
    return None


def is_abstention(raw: Any) -> bool:
    value = _unwrap(raw)
    if not isinstance(value, str):
        return False
    text = _strip_noise(value).lower()
    if text in ("insufficient", "unknown", "abstain"):
        return True
    return any(pattern in text for pattern in ABSTAIN_PATTERNS)


def parse_choice(raw: Any, choices: list[str]) -> str | None:
    value = _unwrap(raw)
    if not isinstance(value, str):
        return None
    text = _strip_noise(value).lower()
    for choice in choices:
        if text == choice.lower():
            return choice
    # Letter answers: A, B, C, D map to declaration order.
    if len(text) == 1 and text.isalpha():
        index = ord(text) - ord("a")
        if 0 <= index < len(choices):
            return choices[index]
    tokens = _tokens(text)
    for choice in choices:
        if _has_phrase(tokens, choice):
            return choice
    for alias, canonical in CHOICE_ALIASES.items():
        if _has_phrase(tokens, alias) and canonical in choices:
            return canonical
    # Class names can be answers to choice questions.
    class_value = parse_class(value)
    if class_value in choices:
        return class_value
    return None


def parse_bbox(raw: Any, width: int | None = None, height: int | None = None) -> list[float] | None:
    value = _unwrap(raw)
    if isinstance(value, str):
        text = _strip_noise(value)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(numbers) != 4:
                return None
            value = [float(n) for n in numbers]
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in value for k in keys):
            value = [value[k] for k in keys]
        elif "bbox" in value:
            value = value["bbox"]
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(map(lambda v: v == v and abs(v) != float("inf"), box)):
        return None
    if all(0.0 <= v <= 1.0 for v in box):
        return box
    # Pixel coordinates are accepted only when the image size is known.
    if width and height and any(v > 1.0 for v in box):
        if -1.0 <= box[0] <= width and -1.0 <= box[1] <= height:
            return [
                max(0.0, min(1.0, box[0] / width)),
                max(0.0, min(1.0, box[1] / height)),
                max(0.0, min(1.0, box[2] / width)),
                max(0.0, min(1.0, box[3] / height)),
            ]
    return None


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def validate(task: Task, raw_answer: Any) -> ValidationResult:
    verification = task.verification
    expected = verification.value
    parsed: Any = None
    correct = False
    abstained = is_abstention(raw_answer)
    details: dict[str, Any] = {}

    if verification.mode == "exact_int":
        parsed = parse_int(raw_answer)
        correct = parsed is not None and int(parsed) == int(expected)
    elif verification.mode == "exact_label":
        parsed = parse_class(raw_answer)
        correct = parsed is not None and parsed == expected
        if parsed is None:
            details["reason"] = "unrecognized class label"
    elif verification.mode == "bbox_iou":
        parsed = parse_bbox(
            raw_answer, verification.image_width, verification.image_height
        )
        if parsed is not None:
            iou = bbox_iou(parsed, [float(v) for v in expected])
            details["iou"] = round(iou, 6)
            details["threshold"] = verification.iou_threshold
            correct = iou >= verification.iou_threshold
        else:
            details["reason"] = "unparseable bounding box"
    elif verification.mode == "exact_choice":
        parsed = parse_choice(raw_answer, task.choices or [])
        correct = parsed is not None and parsed == expected
    elif verification.mode == "exact_bool":
        parsed = parse_bool(raw_answer)
        correct = parsed is not None and parsed == bool(expected)
    elif verification.mode == "abstention":
        if abstained:
            parsed = "abstain"
            correct = expected == "insufficient"
            details["abstained"] = True
        else:
            parsed = "sufficient"
            correct = expected == "sufficient"
            details["abstained"] = False
    else:  # pragma: no cover
        raise ValueError(f"unknown verification mode {verification.mode}")

    if task.answer_format == AnswerFormat.ABSTAIN_LABEL:
        details["abstained"] = abstained

    return ValidationResult(
        task_id=task.task_id,
        correct=bool(correct),
        score=float(task.max_score) if correct else 0.0,
        mode=verification.mode,
        expected=expected,
        parsed=parsed,
        is_abstention=abstained,
        details=details,
    )


__all__ = [
    "CLASS_ALIASES",
    "ValidationResult",
    "bbox_iou",
    "is_abstention",
    "parse_bbox",
    "parse_class",
    "parse_choice",
    "parse_int",
    "validate",
]
