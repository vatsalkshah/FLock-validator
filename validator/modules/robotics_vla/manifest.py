from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field


class EpisodeSpec(BaseModel):
    task: str
    instruction: str
    seed: int
    horizon: int | None = None
    camera_name: str | None = None
    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    domain_randomization: dict[str, Any] = Field(default_factory=dict)
    episode_id: str | None = None
    difficulty: str | None = None
    tags: list[str] = Field(default_factory=list)
    partial_metrics: list[str] = Field(default_factory=list)


class ValidationManifest(BaseModel):
    suite_version: str = "panda_tabletop_v1"
    episodes: list[EpisodeSpec]


def load_manifest(url_or_path: str) -> ValidationManifest:
    data = _read_json(url_or_path)
    if isinstance(data, list):
        data = {"suite_version": "panda_tabletop_v1", "episodes": data}
    return ValidationManifest.model_validate(data)


def _read_json(url_or_path: str) -> Any:
    path = Path(url_or_path).expanduser()
    if path.exists():
        raw = path.read_text()
    else:
        response = requests.get(url_or_path, timeout=30)
        response.raise_for_status()
        raw = response.text

    if url_or_path.endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)
