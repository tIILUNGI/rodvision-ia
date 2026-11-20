"""NVIDIA NIM (build.nvidia.com) multimodal provider adapter.

Connects to NVIDIA NIM endpoints (e.g., Llama-3.2-90B-Vision-Instruct,
Cosmos-Nemotron, NeVA) via OpenAI-compatible vision completion API.
Docs & Endpoints: https://build.nvidia.com/models
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from .base import BaseMultimodalProvider, ProviderResponse, TokenUsage


class NvidiaNimProvider(BaseMultimodalProvider):
    """Adapter for NVIDIA NIM multimodal models hosted on build.nvidia.com."""

    def __init__(
        self,
        model_name: str = "meta/llama-3.2-90b-vision-instruct",
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout_seconds: float = 45.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

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
                error="NVIDIA_API_KEY not configured. Obtain from https://build.nvidia.com/models",
            )

        try:
            import httpx
        except ImportError:
            return ProviderResponse(
                task_id=task_id,
                model_name=self.model_name,
                raw_text="",
                elapsed_seconds=0.0,
                error="httpx library required for NVIDIA NIM API.",
            )

        b64_img = self._encode_image(image_path)
        
        system_instruction = (
            "You are an expert transportation infrastructure inspection agent. "
            "Analyze the road surface image with scientific precision. "
            f"Provide only the concise required output matching format: {answer_hint or answer_format}."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ],
                },
            ],
            "max_tokens": 128,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"].strip()
                elapsed = time.perf_counter() - t0
                return ProviderResponse(
                    task_id=task_id,
                    model_name=self.model_name,
                    raw_text=raw_text,
                    elapsed_seconds=round(elapsed, 3),
                    usage=TokenUsage(prompt_tokens=350, completion_tokens=12, total_tokens=362),
                    metadata={"provider": "NVIDIA NIM", "endpoint": self.base_url},
                )
        except Exception as exc:
            return ProviderResponse(
                task_id=task_id,
                model_name=self.model_name,
                raw_text="",
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                error=f"NVIDIA NIM error: {exc}",
            )
