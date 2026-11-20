"""OpenAI-compatible and multimodal HTTP API provider.

Supports GPT-4o, GPT-4o-mini, Claude via proxy, vLLM, Ollama, OpenRouter, etc.
Requires API key from environment variable or explicit option.
Gracefully handles timeouts, retries, and rate limits.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from .base import BaseMultimodalProvider, ProviderResponse, TokenUsage


class OpenAICompatibleProvider(BaseMultimodalProvider):
    """Adapter for OpenAI-compatible multimodal chat completions."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def _encode_image(self, path: Path | str) -> str:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def predict(
        self,
        task_id: str,
        image_path: Path | str,
        prompt: str,
        answer_format: str,
        choices: list[str] | None = None,
        answer_hint: str | None = None,
    ) -> ProviderResponse:
        if not self.api_key:
            return ProviderResponse(
                task_id=task_id,
                model_name=self.model_name,
                raw_text="",
                elapsed_seconds=0.0,
                error="OPENAI_API_KEY is not configured.",
            )

        # Lazy import of httpx
        try:
            import httpx
        except ImportError:
            return ProviderResponse(
                task_id=task_id,
                model_name=self.model_name,
                raw_text="",
                elapsed_seconds=0.0,
                error="httpx library is required for API provider.",
            )

        b64_img = self._encode_image(image_path)
        system_instruction = (
            "You are evaluating a road pavement inspection benchmark. "
            "Follow the exact format requested. "
            f"Format hint: {answer_hint or answer_format}. "
            "Give only the concise final answer without explanations."
        )

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}",
                    "detail": "high",
                },
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_text = data["choices"][0]["message"]["content"].strip()
                    usage_data = data.get("usage", {})
                    usage = TokenUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )
                    elapsed = time.perf_counter() - t0
                    return ProviderResponse(
                        task_id=task_id,
                        model_name=self.model_name,
                        raw_text=raw_text,
                        elapsed_seconds=round(elapsed, 3),
                        usage=usage,
                        metadata={"attempt": attempt + 1},
                    )
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5 * (attempt + 1))

        return ProviderResponse(
            task_id=task_id,
            model_name=self.model_name,
            raw_text="",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=f"API request failed after {self.max_retries + 1} attempts: {last_error}",
        )
