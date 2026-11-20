"""Providers factory and exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseMultimodalProvider, ProviderResponse, TokenUsage
from .manual import ManualFileProvider
from .mock import DeterministicMockProvider
from .nvidia_provider import NvidiaNimProvider
from .openai_provider import OpenAICompatibleProvider


def get_provider(
    provider_type: str,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseMultimodalProvider:
    """Factory to instantiate any supported provider."""
    p_type = provider_type.lower()
    if p_type in ("mock", "mock_deterministic", "simulated"):
        return DeterministicMockProvider(
            model_name=model_name or "mock-roadvision-sota",
            **kwargs,
        )
    elif p_type in ("manual", "file", "json"):
        answers_file = kwargs.pop("answers_file", None)
        if not answers_file:
            raise ValueError("ManualFileProvider requires 'answers_file' parameter")
        return ManualFileProvider(
            answers_file=answers_file,
            model_name=model_name or Path(answers_file).stem,
            **kwargs,
        )
    elif p_type in ("nvidia", "nvidia_nim", "nim"):
        return NvidiaNimProvider(
            model_name=model_name or "meta/llama-3.2-90b-vision-instruct",
            **kwargs,
        )
    elif p_type in ("openai", "openai_compatible", "vllm", "openrouter"):
        return OpenAICompatibleProvider(
            model_name=model_name or "gpt-4o-mini",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported provider type '{provider_type}'. "
            "Supported: 'mock', 'manual', 'nvidia', 'openai'."
        )


__all__ = [
    "BaseMultimodalProvider",
    "DeterministicMockProvider",
    "ManualFileProvider",
    "NvidiaNimProvider",
    "OpenAICompatibleProvider",
    "ProviderResponse",
    "TokenUsage",
    "get_provider",
]
