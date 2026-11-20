"""Base interface for multimodal model providers.

This interface abstracts all model interaction. It enforces strict separation:
a provider only receives public task representations (prompt, image, format, choices),
never ground truth, annotations, or reference answers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption reported by the provider (if available)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Standardized response from any model provider."""
    task_id: str
    model_name: str
    raw_text: str
    elapsed_seconds: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None and bool(self.raw_text.strip())


class BaseMultimodalProvider(ABC):
    """Abstract base class for all vision-language model providers."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.options = kwargs

    @abstractmethod
    def predict(
        self,
        task_id: str,
        image_path: Path | str,
        prompt: str,
        answer_format: str,
        choices: list[str] | None = None,
        answer_hint: str | None = None,
    ) -> ProviderResponse:
        """Submit a task to the model and return the raw output.

        Strict benchmark rule:
        This method MUST NOT receive ground truth, bbox coordinates, or reference answers.
        """
        pass

    def close(self) -> None:
        """Release any underlying client resources."""
        pass
