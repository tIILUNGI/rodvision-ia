"""Task (question) models.

A task is the unit that a model sees and that the validator scores. The public
payload deliberately omits the reference answer, the verification rule and the
object IDs involved: anything that would leak the answer or the object layout.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config.models import Capability, DefectClass


class AnswerFormat(str, Enum):
    INTEGER = "integer"
    CLASS_LABEL = "class_label"
    BBOX_NORMALIZED = "bbox_normalized"
    CHOICE = "choice"
    YES_NO = "yes_no"
    ABSTAIN_LABEL = "abstain_label"


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "exact_int",
        "exact_label",
        "bbox_iou",
        "exact_choice",
        "exact_bool",
        "abstention",
    ]
    value: Any
    tolerance: float = Field(default=0.0, ge=0.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(pattern=r"^RV_[0-9]{6,10}_[a-f0-9]{6}_q[0-9]{2}$")
    image_id: str = Field(pattern=r"^RV_[0-9]{6,10}_[a-f0-9]{6}$")
    image_path: str = Field(min_length=1)
    split: Literal["smoke", "dev", "test"]
    capability: Capability
    difficulty: Literal["easy", "medium", "hard"]
    question_family: str = Field(min_length=1)
    prompt: str = Field(min_length=8)
    answer_format: AnswerFormat
    choices: list[str] | None = None
    answer_hint: str = Field(min_length=1)
    max_score: float = Field(default=1.0, gt=0.0)

    # Internal fields: never serialised to the public task file.
    reference_answer: Any
    verification: Verification
    related_object_ids: list[str] = Field(default_factory=list)
    visual_conditions: dict[str, str] = Field(default_factory=dict)
    defect_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "Task":
        if self.answer_format in (AnswerFormat.CHOICE, AnswerFormat.CLASS_LABEL, AnswerFormat.YES_NO):
            if not self.choices:
                raise ValueError("choice-like tasks must define choices")
        if self.answer_format == AnswerFormat.CLASS_LABEL:
            allowed = {c.value for c in DefectClass}
            if set(self.choices or []) - allowed:
                raise ValueError("class_label choices must be defect classes")
        if self.answer_format == AnswerFormat.CHOICE:
            if self.reference_answer not in (self.choices or []):
                raise ValueError("choice reference answer must be one of the choices")
        if self.answer_format == AnswerFormat.INTEGER:
            if not isinstance(self.reference_answer, int) or isinstance(self.reference_answer, bool):
                raise ValueError("integer tasks need an int reference")
        return self

    def public_payload(self) -> dict[str, Any]:
        """Fields that a model may see. No answers, no object IDs."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "split": self.split,
            "capability": self.capability.value,
            "difficulty": self.difficulty,
            "question_family": self.question_family,
            "prompt": self.prompt,
            "answer_format": self.answer_format.value,
            "answer_hint": self.answer_hint,
            "max_score": self.max_score,
        }
        if self.choices is not None:
            payload["choices"] = list(self.choices)
        return payload

    def internal_payload(self) -> dict[str, Any]:
        payload = self.public_payload()
        payload["reference_answer"] = self.reference_answer
        payload["verification"] = self.verification.model_dump(mode="json")
        payload["related_object_ids"] = list(self.related_object_ids)
        payload["visual_conditions"] = dict(self.visual_conditions)
        payload["defect_types"] = list(self.defect_types)
        return payload

    def assert_no_leakage(self) -> None:
        """Cheap structural guard against answers placed in the prompt.

        Numeric answers are never embedded in prompts; label answers may only
        appear in the options list, never in the prompt text.
        """
        prompt = self.prompt.lower()
        if self.answer_format == AnswerFormat.INTEGER:
            token = str(self.reference_answer)
            if token in prompt.split() or f" {token} " in f" {prompt} ":
                raise ValueError(f"{self.task_id}: prompt contains the count answer")
        elif self.answer_format == AnswerFormat.CLASS_LABEL:
            answer = str(self.reference_answer).replace("_", " ").lower()
            if answer and answer in prompt:
                raise ValueError(
                    f"{self.task_id}: prompt contains the label answer ({answer!r})"
                )
        # Multiple-choice prompts must list their options, so the reference
        # string necessarily appears there; that is not leakage.


__all__ = ["AnswerFormat", "Task", "Verification"]
