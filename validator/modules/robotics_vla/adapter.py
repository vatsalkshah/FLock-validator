from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from huggingface_hub import snapshot_download
from loguru import logger

from validator.modules.robotics_vla.errors import RoboticsSubmissionError

# A model query (loading the policy, or asking it for an action) is retried this
# many extra times on failure before the submission is declared invalid. This
# absorbs transient faults (a flaky download, a one-off CUDA hiccup) while still
# converging quickly on a deterministically broken submission.
MODEL_QUERY_RETRIES = 3


def resolve_model_dir(repo_id_or_path: str, revision: str = "main") -> Path:
    candidate = Path(repo_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    token = os.getenv("HF_TOKEN")
    path = snapshot_download(repo_id=repo_id_or_path, revision=revision, token=token)
    return Path(path).resolve()


def _load_python_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy_from_adapter(
    model_dir: Path,
    adapter_filename: str,
    device: str,
    torch_dtype: str,
) -> Any:
    model_root = model_dir.resolve()
    adapter_path = model_root / adapter_filename
    # Lexical check: guard against adapter_filename escaping via '..' or an
    # absolute path. We normalise without resolving symlinks here because
    # HuggingFace snapshots store repo files as symlinks into a sibling blobs/
    # directory — resolving would wrongly flag every real HF download.
    normalized = os.path.normpath(str(adapter_path))
    root_str = str(model_root)
    if normalized != root_str and not normalized.startswith(root_str + os.sep):
        raise RoboticsSubmissionError(
            "adapter_filename must resolve inside the model directory",
            failure_mode="adapter_contract",
        )
    if not adapter_path.exists():
        raise RoboticsSubmissionError(
            f"Missing robotics VLA adapter: {adapter_path}",
            failure_mode="adapter_missing",
        )
    # Symlink check: if the adapter is a symlink (e.g. HF hub cache links blobs/
    # next to snapshots/), ensure the real target stays within the model cache
    # root — not an arbitrary filesystem path a miner could embed via git symlink.
    if adapter_path.is_symlink():
        resolved_target = adapter_path.resolve()
        # HF cache layout: model_root = .../models--org--repo/snapshots/<hash>/
        # Blobs live at   : .../models--org--repo/blobs/<sha256>
        # Allow the target to be anywhere within the repo-level cache directory.
        allowed_root = model_root.parent.parent
        if not (
            str(resolved_target) == str(model_root)
            or str(resolved_target).startswith(str(model_root) + os.sep)
            or str(resolved_target).startswith(str(allowed_root) + os.sep)
        ):
            raise RoboticsSubmissionError(
                f"Adapter {adapter_filename!r} is a symlink to a path outside "
                "the model directory",
                failure_mode="adapter_symlink_escape",
            )

    try:
        module = _load_python_module(adapter_path)
    except RoboticsSubmissionError:
        raise
    except Exception as exc:  # noqa: BLE001 - adapter is untrusted miner code
        raise RoboticsSubmissionError(
            f"Robotics VLA adapter failed to import: {exc}",
            failure_mode="adapter_import_failed",
        ) from exc
    if not hasattr(module, "load_policy"):
        raise RoboticsSubmissionError(
            f"{adapter_path} must define load_policy(model_dir, device, dtype)",
            failure_mode="adapter_contract",
        )

    policy = retry_model_query(
        lambda: module.load_policy(
            model_dir=str(model_root),
            device=device,
            dtype=torch_dtype,
        ),
        description="load_policy",
        failure_mode="model_load_failed",
    )
    if not hasattr(policy, "act"):
        raise RoboticsSubmissionError(
            "Robotics VLA policy must expose act(obs) -> action",
            failure_mode="adapter_contract",
        )
    return policy


def retry_model_query(fn: Any, description: str, failure_mode: str) -> Any:
    """Call an untrusted model query, retrying transient failures.

    Retries up to ``MODEL_QUERY_RETRIES`` extra times. A ``RoboticsSubmissionError``
    raised by ``fn`` (e.g. malformed action) is treated as a query failure and
    retried too, but its more specific ``failure_mode`` is preserved when we
    finally give up. Any other exception means the miner's code blew up.
    """
    last_exc: Exception | None = None
    last_mode = failure_mode
    for attempt in range(1 + MODEL_QUERY_RETRIES):
        try:
            return fn()
        except RoboticsSubmissionError as exc:
            last_exc, last_mode = exc, exc.failure_mode
            logger.warning(f"{description} attempt {attempt + 1} rejected: {exc}")
        except Exception as exc:  # noqa: BLE001 - untrusted miner code
            last_exc, last_mode = exc, failure_mode
            logger.warning(f"{description} attempt {attempt + 1} failed: {exc}")
    raise RoboticsSubmissionError(
        f"{description} failed after {1 + MODEL_QUERY_RETRIES} attempts: {last_exc}",
        failure_mode=last_mode,
    ) from last_exc


def count_model_parameters(model_dir: Path) -> int | None:
    safetensor_count = _count_safetensors(model_dir)
    if safetensor_count is not None:
        return safetensor_count

    torch_count = _count_torch_state_dicts(model_dir)
    if torch_count is not None:
        return torch_count

    return None


def count_policy_parameters(policy: Any) -> int | None:
    """Best-effort parameter count of an already-loaded policy.

    Catches submissions whose repo weights look small (or are absent) but which
    pull a large base model in at ``load_policy`` time. Returns ``None`` when no
    torch parameters can be discovered (e.g. torch unavailable, or a non-torch
    policy) so the caller can decide how to treat an unverifiable model. Only
    ever reports a positive count it is confident about; it never guesses.
    """
    try:
        import torch
    except ImportError:
        return None

    seen: set[int] = set()
    total = 0
    found = False

    def _add(module: Any) -> None:
        nonlocal total, found
        params = getattr(module, "parameters", None)
        if not callable(params):
            return
        try:
            iterator = params()
        except Exception:  # noqa: BLE001 - untrusted policy object
            return
        for param in iterator:
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += int(param.numel())
            found = True

    if isinstance(policy, torch.nn.Module):
        _add(policy)
    for value in (vars(policy).values() if hasattr(policy, "__dict__") else []):
        if isinstance(value, torch.nn.Module):
            _add(value)

    return total if found else None


def enforce_parameter_limit(
    repo_count: int | None,
    policy_count: int | None,
    max_params: int,
) -> int:
    """Reject a submission whose parameter count exceeds, or cannot prove it is
    within, ``max_params``. Returns the best-known parameter count on success.

    A submission whose size cannot be determined at all is rejected rather than
    silently admitted, closing the gap where an unrecognized weight format (or a
    base model fetched at runtime) bypassed the cap entirely.
    """
    counts = [count for count in (repo_count, policy_count) if count is not None]
    if not counts:
        raise RoboticsSubmissionError(
            "Could not determine the model parameter count from repo weights or the "
            f"loaded policy; cannot verify it is within the {max_params} parameter cap",
            failure_mode="parameter_count_unknown",
        )
    best = max(counts)
    if best > max_params:
        raise RoboticsSubmissionError(
            f"Model parameters {best} exceed limit {max_params}",
            failure_mode="parameter_limit_exceeded",
        )
    return best


def _count_safetensors(model_dir: Path) -> int | None:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None

    from safetensors import safe_open

    total = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_tensor(key).shape
                params = 1
                for dim in shape:
                    params *= int(dim)
                total += params
    return total


def _count_torch_state_dicts(model_dir: Path) -> int | None:
    files = sorted(
        path
        for pattern in ("*.pt", "*.pth", "*.bin")
        for path in model_dir.glob(pattern)
        if path.name not in {"optimizer.pt", "scheduler.pt"}
    )
    if not files:
        return None

    import torch

    total = 0
    for path in files:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # weights_only is not available in very old PyTorch. Refuse to load
            # without it — pickle execution is a remote-code-execution risk.
            raise RoboticsSubmissionError(
                "PyTorch >= 2.0 is required to safely load model weights. "
                "Upgrade PyTorch to enable weights_only=True.",
                failure_mode="unsupported_torch_version",
            )
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            continue
        for value in obj.values():
            if hasattr(value, "numel"):
                total += int(value.numel())
    return total
