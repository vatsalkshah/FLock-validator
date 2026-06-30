from __future__ import annotations

from copy import deepcopy
from typing import Any


TASK_REGISTRY_VERSION = "robotics_vla_task_registry_v1"

ALLOWED_BASE_ENVS = {
    "PickPlace",
    "NutAssembly",
}

ALLOWED_SUCCESS_CHECKERS = {
    "target_object_in_matching_bin",
    "target_nut_on_matching_peg",
}

ALLOWED_PROGRESS_CHECKERS = {
    "target_pick_place_progress",
    "target_nut_assembly_progress",
}

ALLOWED_TASK_KWARGS = {
    "target_object": {"milk", "bread", "cereal", "can"},
    "target_nut": {"round", "square"},
}

ALLOWED_ENV_KWARGS = {
    "single_object_mode": {0, 1, 2},
    "object_type": {"milk", "bread", "cereal", "can"},
}


def validate_task_registry_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    if not spec:
        return {"version": TASK_REGISTRY_VERSION, "tasks": {}}
    if spec.get("version") != TASK_REGISTRY_VERSION:
        raise ValueError(
            f"Unsupported robotics task registry version {spec.get('version')!r}; "
            f"expected {TASK_REGISTRY_VERSION!r}"
        )
    tasks = spec.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Robotics task registry must contain a 'tasks' object")
    normalized_tasks = {}
    for task_name, entry in tasks.items():
        normalized_tasks[str(task_name)] = _validate_task_entry(str(task_name), entry)
    return {"version": TASK_REGISTRY_VERSION, "tasks": normalized_tasks}


def task_registry_task_names(spec: dict[str, Any] | None) -> set[str]:
    registry = validate_task_registry_spec(spec)
    return set(registry.get("tasks", {}))


def _validate_task_entry(task_name: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Task registry entry {task_name!r} must be an object")
    base_env = entry.get("base_env")
    if base_env not in ALLOWED_BASE_ENVS:
        raise ValueError(
            f"Task registry entry {task_name!r} has unsupported base_env {base_env!r}; "
            f"allowed: {sorted(ALLOWED_BASE_ENVS)}"
        )
    success_checker = entry.get("success_checker")
    if success_checker not in ALLOWED_SUCCESS_CHECKERS:
        raise ValueError(
            f"Task registry entry {task_name!r} has unsupported success_checker {success_checker!r}; "
            f"allowed: {sorted(ALLOWED_SUCCESS_CHECKERS)}"
        )
    progress_checker = entry.get("progress_checker")
    if progress_checker not in ALLOWED_PROGRESS_CHECKERS:
        raise ValueError(
            f"Task registry entry {task_name!r} has unsupported progress_checker {progress_checker!r}; "
            f"allowed: {sorted(ALLOWED_PROGRESS_CHECKERS)}"
        )
    env_kwargs = entry.get("env_kwargs", {})
    if not isinstance(env_kwargs, dict):
        raise ValueError(f"Task registry entry {task_name!r} env_kwargs must be an object")
    _validate_env_kwargs(task_name, env_kwargs)
    allowed_task_kwargs = entry.get("allowed_task_kwargs", [])
    if not isinstance(allowed_task_kwargs, list):
        raise ValueError(f"Task registry entry {task_name!r} allowed_task_kwargs must be a list")
    for key in allowed_task_kwargs:
        if key not in ALLOWED_TASK_KWARGS:
            raise ValueError(f"Task registry entry {task_name!r} allows unsupported task kwarg {key!r}")
    partial_metrics = entry.get("partial_metrics", [])
    if not isinstance(partial_metrics, list) or not all(isinstance(metric, str) for metric in partial_metrics):
        raise ValueError(f"Task registry entry {task_name!r} partial_metrics must be a list of strings")
    return {
        "base_env": str(base_env),
        "env_kwargs": deepcopy(env_kwargs),
        "success_checker": str(success_checker),
        "progress_checker": str(progress_checker),
        "allowed_task_kwargs": [str(key) for key in allowed_task_kwargs],
        "partial_metrics": [str(metric) for metric in partial_metrics],
    }


def _validate_env_kwargs(task_name: str, env_kwargs: dict[str, Any]) -> None:
    unsupported = set(env_kwargs) - set(ALLOWED_ENV_KWARGS)
    if unsupported:
        raise ValueError(
            f"Task registry entry {task_name!r} has unsupported env_kwargs {sorted(unsupported)}; "
            f"allowed: {sorted(ALLOWED_ENV_KWARGS)}"
        )
    for key, value in env_kwargs.items():
        allowed_values = ALLOWED_ENV_KWARGS[key]
        normalized_value = str(value).lower() if isinstance(value, str) else value
        if normalized_value not in allowed_values:
            raise ValueError(
                f"Task registry entry {task_name!r} has invalid env_kwarg {key}={value!r}; "
                f"allowed: {sorted(allowed_values)}"
            )


def validate_episode_task_kwargs(task_name: str, entry: dict[str, Any], task_kwargs: dict[str, Any]) -> None:
    allowed = set(entry.get("allowed_task_kwargs", []))
    unsupported = set(task_kwargs) - allowed
    if unsupported:
        raise ValueError(
            f"Task {task_name!r} has unsupported task_kwargs {sorted(unsupported)}; "
            f"allowed: {sorted(allowed)}"
        )
    for key, value in task_kwargs.items():
        allowed_values = ALLOWED_TASK_KWARGS.get(key)
        if allowed_values is not None and str(value).lower() not in allowed_values:
            raise ValueError(
                f"Task {task_name!r} has invalid {key}={value!r}; "
                f"allowed: {sorted(allowed_values)}"
            )
