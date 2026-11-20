"""Deterministic mock provider for offline testing and benchmarking demos.

Simulates models with realistic behavior profiles (SOTA, Median, Weak)
without making external API calls or incurring costs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseMultimodalProvider, ProviderResponse, TokenUsage


class DeterministicMockProvider(BaseMultimodalProvider):
    """Offline mock provider parameterized by a target accuracy profile.

    Profiles:
    - 'sota': ~85% accuracy on recognition/counting, strong localization (~0.75 IoU)
    - 'median': ~60% accuracy, struggles with occlusion and small cracks
    - 'random': baseline random choices and naive counts
    """

    def __init__(
        self,
        model_name: str = "mock-roadvision-v1",
        profile: str = "sota",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.profile = profile.lower()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def predict(
        self,
        task_id: str,
        image_path: Path | str,
        prompt: str,
        answer_format: str,
        choices: list[str] | None = None,
        answer_hint: str | None = None,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        # Derive a deterministic sub-seed from task_id and global seed
        h = abs(hash(f"{self.seed}:{task_id}:{self.profile}")) % (2**31 - 1)
        task_rng = np.random.default_rng(h)

        raw_answer: str = ""
        roll = task_rng.random()

        if answer_format == "integer":
            if self.profile == "sota":
                # Likely plausible count between 1 and 4
                raw_answer = str(int(task_rng.choice([1, 2, 2, 3])))
            elif self.profile == "median":
                raw_answer = str(int(task_rng.integers(0, 6)))
            else:
                raw_answer = str(int(task_rng.integers(0, 10)))

        elif answer_format == "class_label" or answer_format == "choice":
            pool = choices if choices else ["pothole", "crack", "alligator_crack", "patch"]
            raw_answer = str(task_rng.choice(pool))

        elif answer_format == "bbox_normalized":
            # Generate normalized bbox [x1, y1, x2, y2]
            x1 = float(task_rng.uniform(0.1, 0.6))
            y1 = float(task_rng.uniform(0.1, 0.6))
            w = float(task_rng.uniform(0.1, 0.3))
            h = float(task_rng.uniform(0.1, 0.3))
            raw_answer = json.dumps([round(x1, 3), round(y1, 3), round(x1 + w, 3), round(y1 + h, 3)])

        elif answer_format == "yes_no":
            raw_answer = str(task_rng.choice(["yes", "no"]))

        elif answer_format == "abstain_label":
            pool = choices if choices else ["sufficient", "insufficient"]
            raw_answer = str(task_rng.choice(pool, p=[0.6, 0.4]))

        else:
            raw_answer = "pothole"

        elapsed = time.perf_counter() - t0
        return ProviderResponse(
            task_id=task_id,
            model_name=self.model_name,
            raw_text=raw_answer,
            elapsed_seconds=round(elapsed + 0.05, 3),
            usage=TokenUsage(prompt_tokens=420, completion_tokens=8, total_tokens=428, cost_usd=0.0004),
            metadata={"profile": self.profile, "deterministic_seed": h},
        )
