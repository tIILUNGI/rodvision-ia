"""Manual file-based provider.

Reads pre-computed or human-annotated answers from a JSON/JSONL file.
Enables offline, perfectly reproducible evaluation runs without API dependence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseMultimodalProvider, ProviderResponse, TokenUsage


class ManualFileProvider(BaseMultimodalProvider):
    """Answers tasks from a loaded answer mapping."""

    def __init__(
        self,
        answers_file: Path | str,
        model_name: str = "manual-reference",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.answers_file = Path(answers_file)
        self.answers: dict[str, str] = {}
        self._load_answers()

    def _load_answers(self) -> None:
        if not self.answers_file.exists():
            raise FileNotFoundError(f"Answers file not found: {self.answers_file}")

        text = self.answers_file.read_text(encoding="utf-8")
        if self.answers_file.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                tid = record.get("task_id")
                ans = record.get("answer") or record.get("response") or record.get("raw_text")
                if tid and ans is not None:
                    self.answers[str(tid)] = str(ans)
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                self.answers = {str(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                for item in data:
                    tid = item.get("task_id")
                    ans = item.get("answer") or item.get("response")
                    if tid and ans is not None:
                        self.answers[str(tid)] = str(ans)

    def predict(
        self,
        task_id: str,
        image_path: Path | str,
        prompt: str,
        answer_format: str,
        choices: list[str] | None = None,
        answer_hint: str | None = None,
    ) -> ProviderResponse:
        ans = self.answers.get(task_id)
        if ans is None:
            return ProviderResponse(
                task_id=task_id,
                model_name=self.model_name,
                raw_text="",
                elapsed_seconds=0.001,
                error=f"Task ID {task_id} not found in {self.answers_file.name}",
            )

        return ProviderResponse(
            task_id=task_id,
            model_name=self.model_name,
            raw_text=ans,
            elapsed_seconds=0.002,
            usage=TokenUsage(),
            metadata={"source": str(self.answers_file)},
        )
