"""FastAPI API Pydantic schemas for request/response communication."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    config_name: str = Field(default="smoke")
    weather: str | None = None
    lighting: str | None = None
    shadow: bool | None = None
    occlusion: bool | None = None


class ValidationRequest(BaseModel):
    task_id: str
    dataset: str = "smoke"
    answer: Any


class EvaluationRequest(BaseModel):
    dataset_name: str = "smoke"
    provider_type: str = "mock"
    model_name: str = "roadvision-sota-simulated"
    max_tasks: int | None = None
    api_key: str | None = None


class HarborExportRequest(BaseModel):
    dataset_name: str = "smoke"
    limit: int | None = None
