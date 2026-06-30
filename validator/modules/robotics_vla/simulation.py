from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from validator.modules.robotics_vla.adapter import retry_model_query
from validator.modules.robotics_vla.errors import RoboticsSubmissionError
from validator.modules.robotics_vla.manifest import EpisodeSpec
from validator.modules.robotics_vla.task_registry import (
    validate_episode_task_kwargs,
    validate_task_registry_spec,
)


SUPPORTED_TASKS = {
    "lift_cube": ("Lift", {}),
    "pick_place_can": ("PickPlace", {"single_object_mode": 2, "object_type": "can"}),
    "pick_place_milk": ("PickPlace", {"single_object_mode": 2, "object_type": "milk"}),
    "pick_place_bread": ("PickPlace", {"single_object_mode": 2, "object_type": "bread"}),
    "pick_place_cereal": ("PickPlace", {"single_object_mode": 2, "object_type": "cereal"}),
    "pick_place_clutter": ("PickPlace", {"single_object_mode": 0}),
    "stack_blocks": ("Stack", {}),
    "open_door": ("Door", {}),
    "nut_assembly": ("NutAssembly", {}),
    "nut_assembly_square": ("NutAssemblySquare", {}),
    "nut_assembly_round": ("NutAssemblyRound", {}),
    "wipe_table": ("Wipe", {}),
    "tool_hang": ("ToolHang", {}),
}

PARTIAL_CREDIT_CAP = 0.40
DIFFICULTY_WEIGHTS = {
    "low": 0.75,
    "medium": 1.00,
    "hard": 1.25,
    "very_high": 1.50,
}


@dataclass(frozen=True)
class RolloutSettings:
    seed: int
    suite_version: str
    robot: str
    controller: str
    horizon: int
    max_episode_horizon: int
    control_freq: int
    camera_name: str
    camera_height: int
    camera_width: int
    action_dim: int
    render_video: bool
    video_dir: str
    max_videos: int
    expose_raw_obs: bool = False
    task_registry: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    loss: float
    mean_episode_score: float
    weighted_episode_score: float
    success_rate: float
    mean_progress_score: float
    mean_return: float
    mean_episode_length: float
    episodes_completed: int
    video_paths: list[str] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)


def rollout_manifest(
    policy: Any,
    episodes: list[EpisodeSpec],
    settings: RolloutSettings,
) -> RolloutResult:
    if not episodes:
        raise ValueError("Validation manifest must contain at least one episode")

    returns = []
    lengths = []
    successes = []
    episode_scores = []
    progress_scores = []
    difficulty_weights = []
    video_paths: list[str] = []

    for idx, episode in enumerate(episodes):
        record_video = settings.render_video and len(video_paths) < settings.max_videos
        result = run_episode(policy, episode, settings, idx, record_video)
        returns.append(result["return"])
        lengths.append(result["length"])
        successes.append(1.0 if result["success"] else 0.0)
        episode_scores.append(result["episode_score"])
        progress_scores.append(result["progress_score"])
        difficulty_weights.append(result["difficulty_weight"])
        if result.get("video_path"):
            video_paths.append(result["video_path"])

    weights = np.asarray(difficulty_weights, dtype=np.float64)
    scores = np.asarray(episode_scores, dtype=np.float64)
    weighted_episode_score = compute_weighted_episode_score(scores, weights)
    loss = compute_weighted_loss(scores, weights)

    return RolloutResult(
        loss=loss,
        mean_episode_score=float(np.mean(scores)),
        weighted_episode_score=weighted_episode_score,
        success_rate=float(np.mean(successes)),
        mean_progress_score=float(np.mean(progress_scores)),
        mean_return=float(np.mean(returns)),
        mean_episode_length=float(np.mean(lengths)),
        episodes_completed=len(episodes),
        video_paths=video_paths,
        diagnostics={
            "suite_version": settings.suite_version,
            "primary_metric": "loss",
            "primary_metric_direction": "lower_is_better",
            "score_direction": "higher_is_better",
            "score_definition": (
                "score = weighted_mean(episode_score); "
                "loss = weighted_mean(1 - episode_score); "
                "episode_score = 1 if success else 0.40 * normalized_progress"
            ),
            "partial_credit_cap": str(PARTIAL_CREDIT_CAP),
            "difficulty_weights": ",".join(
                f"{difficulty}:{weight}" for difficulty, weight in DIFFICULTY_WEIGHTS.items()
            ),
        },
    )


def run_episode(
    policy: Any,
    episode: EpisodeSpec,
    settings: RolloutSettings,
    episode_index: int,
    record_video: bool = False,
) -> dict[str, Any]:
    task_entry = task_entry_for_episode(episode, settings)
    env = make_env(episode, settings)
    frames = []
    total_reward = 0.0
    best_reward = 0.0
    success = False
    horizon = min(
        episode.horizon if episode.horizon is not None else settings.horizon,
        settings.max_episode_horizon,
    )

    try:
        raw_obs = env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]

        for step_idx in range(horizon):
            obs = prepare_policy_obs(
                raw_obs=raw_obs,
                episode=episode,
                settings=settings,
                step_idx=step_idx,
            )
            action = adapt_action_for_env(query_policy_action(policy, obs, settings.action_dim), env)
            raw_obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            best_reward = max(best_reward, float(reward))
            progress_value = compute_task_progress(env, raw_obs, episode, task_entry, float(reward))
            best_reward = max(best_reward, progress_value)
            success = success or _extract_success(env, info, episode, task_entry)

            if record_video:
                frames.append(extract_frame(raw_obs, episode_camera_name(episode, settings)))

            if done or success:
                progress_score = compute_progress_score(success, best_reward)
                return {
                    "return": total_reward,
                    "length": step_idx + 1,
                    "success": success,
                    "progress_score": progress_score,
                    "episode_score": compute_episode_score(success, progress_score),
                    "difficulty_weight": difficulty_weight(episode),
                    "video_path": save_video(frames, episode, settings, episode_index)
                    if record_video
                    else None,
                }

        progress_score = compute_progress_score(success, best_reward)
        return {
            "return": total_reward,
            "length": horizon,
            "success": success,
            "progress_score": progress_score,
            "episode_score": compute_episode_score(success, progress_score),
            "difficulty_weight": difficulty_weight(episode),
            "video_path": save_video(frames, episode, settings, episode_index)
            if record_video
            else None,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def make_env(episode: EpisodeSpec, settings: RolloutSettings):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import robosuite as suite

    env_name, default_kwargs, task_entry = resolve_task_definition(episode, settings)
    if task_entry:
        validate_episode_task_kwargs(episode.task, task_entry, episode.task_kwargs)
        env_kwargs = dict(default_kwargs)
    else:
        env_kwargs = {**default_kwargs, **episode.task_kwargs}
    seed = int(settings.seed + episode.seed)
    camera_name = episode_camera_name(episode, settings)
    np.random.seed(seed)
    controller_config = _load_controller_config(settings.controller, settings.robot)
    env = suite.make(
        env_name=env_name,
        robots=settings.robot,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=[camera_name],
        camera_heights=[settings.camera_height],
        camera_widths=[settings.camera_width],
        camera_depths=False,
        reward_shaping=True,
        horizon=episode.horizon or settings.horizon,
        control_freq=settings.control_freq,
        hard_reset=False,
        ignore_done=False,
        **env_kwargs,
    )
    seed_method = getattr(env, "seed", None)
    if callable(seed_method):
        seed_method(seed)
    return env


def resolve_task_definition(
    episode: EpisodeSpec,
    settings: RolloutSettings,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if episode.task in SUPPORTED_TASKS:
        env_name, default_kwargs = SUPPORTED_TASKS[episode.task]
        return env_name, dict(default_kwargs), None

    registry = validate_task_registry_spec(settings.task_registry)
    task_entry = registry.get("tasks", {}).get(episode.task)
    if not task_entry:
        supported = sorted([*SUPPORTED_TASKS, *registry.get("tasks", {})])
        raise ValueError(f"Unsupported robotics task {episode.task!r}; supported: {', '.join(supported)}")
    return str(task_entry["base_env"]), dict(task_entry.get("env_kwargs", {})), task_entry


def task_entry_for_episode(episode: EpisodeSpec, settings: RolloutSettings) -> dict[str, Any] | None:
    if episode.task in SUPPORTED_TASKS:
        return None
    registry = validate_task_registry_spec(settings.task_registry)
    task_entry = registry.get("tasks", {}).get(episode.task)
    if task_entry:
        validate_episode_task_kwargs(episode.task, task_entry, episode.task_kwargs)
    return task_entry


def episode_camera_name(episode: EpisodeSpec, settings: RolloutSettings) -> str:
    return episode.camera_name or settings.camera_name


def _load_controller_config(controller: str, robot: str) -> dict[str, Any]:
    try:
        from robosuite.controllers import load_composite_controller_config

        return load_composite_controller_config(controller=controller, robot=robot)
    except (ImportError, AssertionError):
        from robosuite.controllers import load_part_controller_config

        return load_part_controller_config(default_controller=controller)


def prepare_policy_obs(
    raw_obs: dict[str, Any],
    episode: EpisodeSpec,
    settings: RolloutSettings,
    step_idx: int,
) -> dict[str, Any]:
    image = extract_frame(raw_obs, episode_camera_name(episode, settings))
    proprio = _flatten_named_keys(
        raw_obs,
        [
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ],
    )
    obs = {
        "image": image,
        "instruction": episode.instruction,
        "proprio": proprio,
        "task": episode.task,
        "step": step_idx,
        "difficulty": episode.difficulty,
        "horizon": episode.horizon or settings.horizon,
    }
    if settings.expose_raw_obs:
        obs["low_dim"] = _flatten_numeric_obs(raw_obs)
        obs["raw_obs"] = raw_obs
    return obs


def query_policy_action(policy: Any, obs: dict[str, Any], action_dim: int) -> np.ndarray:
    """Ask the policy for an action, retrying transient failures.

    Both an exception from ``policy.act`` and a malformed action are treated as
    retryable; after the retries are exhausted the submission is declared invalid
    rather than crashing the validator.
    """
    return retry_model_query(
        lambda: validate_action(policy.act(obs), action_dim),
        description="policy.act",
        failure_mode="policy_execution_failed",
    )


def validate_action(action: Any, action_dim: int) -> np.ndarray:
    try:
        array = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RoboticsSubmissionError(
            f"Policy action is not a numeric array: {exc}",
            failure_mode="invalid_action",
        ) from exc
    if array.shape == (1, action_dim):
        array = array[0]
    if array.shape != (action_dim,):
        raise RoboticsSubmissionError(
            f"Policy action must have shape ({action_dim},), got {array.shape}",
            failure_mode="invalid_action",
        )
    if not np.isfinite(array).all():
        raise RoboticsSubmissionError(
            "Policy action contains NaN or Inf",
            failure_mode="invalid_action",
        )
    return np.clip(array, -1.0, 1.0).astype(np.float32)


def adapt_action_for_env(action: np.ndarray, env: Any) -> np.ndarray:
    env_action_dim = int(getattr(env, "action_dim", action.shape[0]))
    if env_action_dim == action.shape[0]:
        return action
    if env_action_dim < action.shape[0]:
        return action[:env_action_dim].astype(np.float32)
    padded = np.zeros((env_action_dim,), dtype=np.float32)
    padded[: action.shape[0]] = action
    return padded


def compute_progress_score(success: bool, best_reward: float) -> float:
    if success:
        return 1.0
    return float(np.clip(best_reward, 0.0, 1.0))


def compute_episode_score(success: bool, progress_score: float) -> float:
    if success:
        return 1.0
    return float(PARTIAL_CREDIT_CAP * np.clip(progress_score, 0.0, 1.0))


def compute_weighted_episode_score(episode_scores: Any, weights: Any) -> float:
    return float(np.average(np.asarray(episode_scores, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def compute_weighted_loss(episode_scores: Any, weights: Any) -> float:
    scores = np.asarray(episode_scores, dtype=np.float64)
    return float(np.average(1.0 - scores, weights=np.asarray(weights, dtype=np.float64)))


def difficulty_weight(episode: EpisodeSpec) -> float:
    return float(DIFFICULTY_WEIGHTS.get(episode.difficulty or "medium", 1.0))


def extract_frame(raw_obs: dict[str, Any], camera_name: str) -> np.ndarray:
    key = f"{camera_name}_image"
    if key not in raw_obs:
        raise KeyError(f"Observation missing camera image key {key!r}")
    image = orient_camera_image(raw_obs[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera image must have shape HxWx3, got {image.shape}")
    return image


def orient_camera_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera image must have shape HxWx3, got {image.shape}")
    return np.ascontiguousarray(np.rot90(image, 2))


def save_video(
    frames: list[np.ndarray],
    episode: EpisodeSpec,
    settings: RolloutSettings,
    episode_index: int,
) -> str | None:
    if not frames:
        return None

    import imageio.v2 as imageio

    output_dir = Path(settings.video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode_index:03d}_{episode.task}.mp4"
    imageio.mimsave(output_path, frames, fps=max(settings.control_freq // 2, 1))
    return str(output_path.resolve())


def compute_task_progress(
    env: Any,
    raw_obs: dict[str, Any],
    episode: EpisodeSpec,
    task_entry: dict[str, Any] | None,
    reward: float,
) -> float:
    if not task_entry:
        return float(reward)
    checker = task_entry.get("progress_checker")
    if checker == "target_pick_place_progress":
        return target_pick_place_progress(env, raw_obs, episode)
    if checker == "target_nut_assembly_progress":
        return target_nut_assembly_progress(env, raw_obs, episode)
    return float(reward)


def _extract_success(
    env: Any,
    info: dict[str, Any],
    episode: EpisodeSpec | None = None,
    task_entry: dict[str, Any] | None = None,
) -> bool:
    if task_entry and episode is not None:
        checker = task_entry.get("success_checker")
        if checker == "target_object_in_matching_bin":
            return target_object_in_matching_bin_success(env, episode)
        if checker == "target_nut_on_matching_peg":
            return target_nut_on_matching_peg_success(env, episode)
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return bool(info[key])
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def target_object_in_matching_bin_success(env: Any, episode: EpisodeSpec) -> bool:
    target_object = str(episode.task_kwargs.get("target_object", "")).lower()
    object_index = _pick_place_object_index(env, target_object)
    if object_index is None:
        return False
    if hasattr(env, "_check_success"):
        env._check_success()
    objects_in_bins = getattr(env, "objects_in_bins", None)
    if objects_in_bins is not None and len(objects_in_bins) > object_index:
        return bool(objects_in_bins[object_index])
    obj_pos = _pick_place_object_position(env, object_index)
    if obj_pos is None or not hasattr(env, "not_in_bin"):
        return False
    return not bool(env.not_in_bin(obj_pos, object_index))


def target_pick_place_progress(env: Any, raw_obs: dict[str, Any], episode: EpisodeSpec) -> float:
    target_object = str(episode.task_kwargs.get("target_object", "")).lower()
    object_index = _pick_place_object_index(env, target_object)
    if object_index is None:
        return 0.0
    object_pos = _pick_place_object_position(env, object_index)
    eef_pos = _eef_position(env, raw_obs)
    if object_pos is None or eef_pos is None:
        return 0.0
    reach = 1.0 - np.tanh(10.0 * np.linalg.norm(eef_pos - object_pos))
    bin_pos = _pick_place_bin_position(env, object_index)
    lift = float(np.clip((object_pos[2] - 0.80) / 0.18, 0.0, 1.0))
    placement = 0.0
    if bin_pos is not None:
        placement = 1.0 - np.tanh(6.0 * np.linalg.norm(object_pos[:2] - bin_pos[:2]))
    if target_object_in_matching_bin_success(env, episode):
        return 1.0
    return float(np.clip(max(0.25 * reach, 0.45 * lift, 0.75 * placement), 0.0, 1.0))


def target_nut_on_matching_peg_success(env: Any, episode: EpisodeSpec) -> bool:
    target_nut = str(episode.task_kwargs.get("target_nut", "")).lower()
    nut_index = _nut_index(env, target_nut)
    if nut_index is None:
        return False
    if hasattr(env, "_check_success"):
        env._check_success()
    objects_on_pegs = getattr(env, "objects_on_pegs", None)
    if objects_on_pegs is not None and len(objects_on_pegs) > nut_index:
        return bool(objects_on_pegs[nut_index])
    nut_pos = _nut_position(env, nut_index)
    if nut_pos is None or not hasattr(env, "on_peg"):
        return False
    return bool(env.on_peg(nut_pos, nut_index))


def target_nut_assembly_progress(env: Any, raw_obs: dict[str, Any], episode: EpisodeSpec) -> float:
    target_nut = str(episode.task_kwargs.get("target_nut", "")).lower()
    nut_index = _nut_index(env, target_nut)
    if nut_index is None:
        return 0.0
    nut_pos = _nut_position(env, nut_index)
    peg_pos = _peg_position(env, nut_index)
    eef_pos = _eef_position(env, raw_obs)
    if nut_pos is None or eef_pos is None:
        return 0.0
    reach = 1.0 - np.tanh(10.0 * np.linalg.norm(eef_pos - nut_pos))
    lift = float(np.clip((nut_pos[2] - 0.80) / 0.16, 0.0, 1.0))
    alignment = 0.0
    if peg_pos is not None:
        alignment = 1.0 - np.tanh(20.0 * np.linalg.norm(nut_pos[:2] - peg_pos[:2]))
    if target_nut_on_matching_peg_success(env, episode):
        return 1.0
    return float(np.clip(max(0.25 * reach, 0.45 * lift, 0.75 * alignment), 0.0, 1.0))


def _pick_place_object_index(env: Any, target_object: str) -> int | None:
    object_to_id = getattr(env, "object_to_id", None)
    if isinstance(object_to_id, dict) and target_object in object_to_id:
        return int(object_to_id[target_object])
    for idx, obj in enumerate(getattr(env, "objects", [])):
        if target_object and target_object in getattr(obj, "name", "").lower():
            return idx
    return None


def _pick_place_object_position(env: Any, object_index: int) -> np.ndarray | None:
    objects = getattr(env, "objects", [])
    if object_index >= len(objects):
        return None
    obj_name = getattr(objects[object_index], "name", "")
    body_ids = getattr(env, "obj_body_id", {})
    if obj_name not in body_ids:
        return None
    return np.asarray(env.sim.data.body_xpos[body_ids[obj_name]], dtype=np.float32)


def _pick_place_bin_position(env: Any, object_index: int) -> np.ndarray | None:
    placements = getattr(env, "target_bin_placements", None)
    if placements is None or len(placements) <= object_index:
        return None
    return np.asarray(placements[object_index], dtype=np.float32)


def _nut_index(env: Any, target_nut: str) -> int | None:
    nut_to_id = getattr(env, "nut_to_id", None)
    if isinstance(nut_to_id, dict) and target_nut in nut_to_id:
        return int(nut_to_id[target_nut])
    for idx, nut in enumerate(getattr(env, "nuts", [])):
        if target_nut and target_nut in getattr(nut, "name", "").lower():
            return idx
    return None


def _nut_position(env: Any, nut_index: int) -> np.ndarray | None:
    nuts = getattr(env, "nuts", [])
    if nut_index >= len(nuts):
        return None
    nut_name = getattr(nuts[nut_index], "name", "")
    body_ids = getattr(env, "obj_body_id", {})
    if nut_name not in body_ids:
        return None
    return np.asarray(env.sim.data.body_xpos[body_ids[nut_name]], dtype=np.float32)


def _peg_position(env: Any, nut_index: int) -> np.ndarray | None:
    peg_body_ids = [getattr(env, "peg1_body_id", None), getattr(env, "peg2_body_id", None)]
    if nut_index >= len(peg_body_ids) or peg_body_ids[nut_index] is None:
        return None
    return np.asarray(env.sim.data.body_xpos[peg_body_ids[nut_index]], dtype=np.float32)


def _eef_position(env: Any, raw_obs: dict[str, Any]) -> np.ndarray | None:
    if "robot0_eef_pos" in raw_obs:
        return np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
    robot = getattr(env, "robots", [None])[0]
    if robot is None:
        return None
    eef_site_id = getattr(robot, "eef_site_id", None)
    if isinstance(eef_site_id, dict):
        eef_site_id = next(iter(eef_site_id.values()), None)
    if eef_site_id is None:
        return None
    return np.asarray(env.sim.data.site_xpos[eef_site_id], dtype=np.float32)


def _flatten_named_keys(raw_obs: dict[str, Any], keys: list[str]) -> np.ndarray:
    values = []
    for key in keys:
        if key in raw_obs:
            values.append(np.asarray(raw_obs[key], dtype=np.float32).reshape(-1))
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def _flatten_numeric_obs(raw_obs: dict[str, Any]) -> np.ndarray:
    values = []
    for key in sorted(raw_obs):
        if key.endswith("_image") or key.endswith("_depth") or key.endswith("_segmentation"):
            continue
        value = raw_obs[key]
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if array.ndim == 0:
            array = array.reshape(1)
        values.append(array.reshape(-1))
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32)
