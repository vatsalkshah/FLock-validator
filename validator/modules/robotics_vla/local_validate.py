from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from validator.config import load_config_for_task
from validator.modules.robotics_vla import (
    RoboticsVLAConfig,
    RoboticsVLAInputData,
    RoboticsVLAValidationModule,
)


def run_local_validation(
    hg_repo_id: str,
    revision: str = "main",
    validation_manifest_url: str | None = None,
    validation_data_url: str | None = None,
    domain_randomization_url: str | None = None,
    adapter_filename: str = "flock_robotics_adapter.py",
    max_params: int = 4_500_000_000,
    max_episodes: int | None = None,
    max_episode_horizon: int | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
    render_video: bool | None = None,
    video_dir: str | None = None,
    output_json: str | None = None,
    hf_token: str | None = None,
    config_dir: str = "configs",
) -> dict[str, Any]:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    config = load_config_for_task(
        task_id="local_robotics_vla",
        task_type="robotics_vla",
        config_model=RoboticsVLAConfig,
        config_dir=config_dir,
    )
    config_updates: dict[str, Any] = {}
    if max_episodes is not None:
        config_updates["max_episodes"] = max_episodes
    if max_episode_horizon is not None:
        config_updates["max_episode_horizon"] = max_episode_horizon
    if device is not None:
        config_updates["device"] = device
    if torch_dtype is not None:
        config_updates["torch_dtype"] = torch_dtype
    if render_video is not None:
        config_updates["render_video"] = render_video
    if video_dir is not None:
        config_updates["video_dir"] = video_dir
    if config_updates:
        config = config.model_copy(update=config_updates)

    data = RoboticsVLAInputData(
        hg_repo_id=hg_repo_id,
        revision=revision,
        validation_manifest_url=validation_manifest_url,
        validation_data_url=validation_data_url,
        max_params=max_params,
        adapter_filename=adapter_filename,
        domain_randomization_url=domain_randomization_url,
    )
    module = RoboticsVLAValidationModule(config=config)
    metrics = module.validate(data)
    result = metrics.model_dump()
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
    return result
