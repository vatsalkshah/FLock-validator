from __future__ import annotations

import importlib.util
import os
import sys
import types
from hashlib import sha256
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from huggingface_hub import errors as hf_errors
from huggingface_hub import snapshot_download
from loguru import logger

from validator.modules.robotics_vla.errors import RoboticsSubmissionError

# A model query (loading the policy, or asking it for an action) is retried this
# many extra times on failure before the submission is declared invalid. This
# absorbs transient faults (a flaky download, a one-off CUDA hiccup) while still
# converging quickly on a deterministically broken submission.
MODEL_QUERY_RETRIES = 3

DEFAULT_POLICY_LOAD_TIMEOUT_SECONDS = 10 * 60
DEFAULT_POLICY_ACTION_TIMEOUT_SECONDS = 30
# Authoritative model-size ceiling for the policy sandbox (GPU VRAM + system RAM).
DEFAULT_POLICY_MEMORY_LIMIT_BYTES = 18 * 1024**3
DEFAULT_POLICY_CPU_TIME_SECONDS = 60 * 60


def resolve_model_dir(repo_id_or_path: str, revision: str = "main") -> Path:
    candidate = Path(repo_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    token = os.getenv("HF_TOKEN")
    try:
        path = snapshot_download(
            repo_id=repo_id_or_path, revision=revision, token=token
        )
    except Exception as exc:
        if not _is_deterministic_hub_error(exc):
            raise
        raise RoboticsSubmissionError(
            f"Invalid Hugging Face model reference {repo_id_or_path!r} "
            f"at revision {revision!r}: {exc}",
            failure_mode="model_reference_invalid",
        ) from exc
    return Path(path).resolve()


def _is_deterministic_hub_error(exc: Exception) -> bool:
    """Return whether a Hub failure is caused by the submitted reference.

    Connection failures, server failures, request timeouts, and rate limits stay
    recoverable. Malformed IDs and stable 4xx responses are invalid submissions.
    """
    local_miss_type = getattr(hf_errors, "LocalEntryNotFoundError", None)
    if isinstance(local_miss_type, type) and isinstance(exc, local_miss_type):
        return False

    deterministic_types = tuple(
        error_type
        for error_type in (
            getattr(hf_errors, "HFValidationError", None),
            getattr(hf_errors, "RepositoryNotFoundError", None),
            getattr(hf_errors, "RevisionNotFoundError", None),
            getattr(hf_errors, "EntryNotFoundError", None),
            getattr(hf_errors, "GatedRepoError", None),
            getattr(hf_errors, "DisabledRepoError", None),
            getattr(hf_errors, "BadRequestError", None),
        )
        if isinstance(error_type, type)
    )
    if isinstance(exc, deterministic_types):
        return True

    http_error_type = getattr(hf_errors, "HfHubHTTPError", None)
    if isinstance(http_error_type, type) and isinstance(exc, http_error_type):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return (
            isinstance(status_code, int)
            and 400 <= status_code < 500
            and status_code
            not in {
                408,
                429,
            }
        )
    return False


def _load_python_module(path: Path) -> ModuleType:
    module_digest = sha256(os.fsencode(path.absolute())).hexdigest()[:16]
    module_name = f"_flock_robotics_adapter_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    module_dir = str(path.parent.absolute())
    inserted_module_dir = module_dir not in sys.path
    if inserted_module_dir:
        sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    finally:
        if inserted_module_dir:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    return module


def load_policy_from_adapter(
    model_dir: Path,
    adapter_filename: str,
    device: str,
    torch_dtype: str,
    *,
    load_timeout_seconds: float = DEFAULT_POLICY_LOAD_TIMEOUT_SECONDS,
    action_timeout_seconds: float = DEFAULT_POLICY_ACTION_TIMEOUT_SECONDS,
    memory_limit_bytes: int = DEFAULT_POLICY_MEMORY_LIMIT_BYTES,
    cpu_time_seconds: int = DEFAULT_POLICY_CPU_TIME_SECONDS,
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

    # Importing the adapter and invoking policy.act both happen in a persistent,
    # credential-free sandbox process. The parent only accepts bounded JSON
    # messages, never pickle data supplied by miner code.
    from validator.modules.robotics_vla.isolation import IsolatedPolicy

    return IsolatedPolicy.start(
        model_dir=model_root,
        adapter_filename=adapter_filename,
        device=device,
        torch_dtype=torch_dtype,
        load_timeout_seconds=load_timeout_seconds,
        action_timeout_seconds=action_timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        cpu_time_seconds=cpu_time_seconds,
    )


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


def count_policy_parameters(
    policy: Any,
    *,
    module_roots: Iterable[Path] | None = None,
) -> int | None:
    """Recursively audit parameters reachable from a loaded policy.

    Containers, object attributes, function closures, and referenced globals are
    walked with cycle protection. Modules and classes whose source files live
    below ``module_roots`` are part of the submitted policy namespace and are
    walked too; dependency and standard-library namespaces remain out of scope.
    Returning ``0`` means the graph was audited and contains no tensor parameters;
    ``None`` means some object was opaque or could not be inspected, so the caller
    must reject the policy as unaccounted.
    """
    # Only trust the validator-owned proxy metadata. An arbitrary miner policy
    # may define an ``audited_parameter_count`` attribute and must not self-attest.
    from validator.modules.robotics_vla.isolation import IsolatedPolicy

    if isinstance(policy, IsolatedPolicy):
        isolated_count = policy.audited_parameter_count
        if isinstance(isolated_count, int) and isolated_count >= 0:
            return isolated_count
        return None

    try:
        import torch
    except ImportError:
        torch = None

    roots = tuple(
        Path(os.path.abspath(os.fspath(root))) for root in (module_roots or ())
    )

    def _path_is_in_module_roots(path: Any) -> bool:
        if not roots or not isinstance(path, (str, os.PathLike)):
            return False
        try:
            candidate = Path(os.path.abspath(os.fspath(path)))
        except (OSError, TypeError, ValueError):
            return False
        return any(candidate == root or root in candidate.parents for root in roots)

    def _module_is_auditable(module: ModuleType) -> bool:
        return _path_is_in_module_roots(getattr(module, "__file__", None))

    def _class_is_auditable(klass: type) -> bool:
        defining_module = sys.modules.get(getattr(klass, "__module__", ""))
        return isinstance(defining_module, ModuleType) and _module_is_auditable(
            defining_module
        )

    def _foreign_module_weight_values(module: ModuleType) -> list[Any]:
        """Weight-bearing objects hidden on a non-submitted module's namespace.

        A policy cannot be trusted to keep its weights in its own namespace: it can
        stash a tensor on a dependency (``numpy.HIDDEN = model``) or on a bare
        ``types.ModuleType`` held in a global, then read it back from ``act``. This
        unwraps transparent built-in containers to reach any arrays/tensors/modules
        parked on such a module. Functions, classes, sub-modules, and opaque objects
        are deliberately ignored: following them would crawl into dependency
        internals and could downgrade a legitimate submission's count to 'unknown'.
        The returned carriers are handed back to the main walk to be counted.
        """
        try:
            pending = [
                item
                for name, item in vars(module).items()
                if not (name.startswith("__") and name.endswith("__"))
            ]
        except Exception:  # noqa: BLE001 - untrusted/foreign module object
            return []
        found: list[Any] = []
        local_seen: set[int] = set()
        container_types = (list, tuple, set, frozenset, deque)
        while pending:
            obj = pending.pop()
            marker = id(obj)
            if marker in local_seen:
                continue
            local_seen.add(marker)
            if isinstance(obj, np.ndarray):
                found.append(obj)
            elif torch is not None and isinstance(obj, (torch.Tensor, torch.nn.Module)):
                found.append(obj)
            elif isinstance(obj, Mapping):
                try:
                    pending.extend(obj.values())
                except Exception:  # noqa: BLE001 - untrusted container
                    continue
            elif isinstance(obj, container_types):
                try:
                    pending.extend(obj)
                except Exception:  # noqa: BLE001 - untrusted container
                    continue
            elif _class_is_auditable(type(obj)):
                # A submitted-code instance parked on a foreign module (e.g. a
                # wrapper holding the weights). Hand it to the main walk, which
                # accounts its nested state. Dependency-owned objects keep their
                # non-submitted class and are ignored, so the scan still never
                # crawls into a dependency's internals.
                found.append(obj)
        return found

    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()
    total = 0
    complete = True
    stack = [policy]
    try:
        stack.append(getattr(policy, "act"))
    except Exception:  # noqa: BLE001 - untrusted policy object
        return None

    atomic_types = (str, bytes, bytearray, int, float, complex, bool, type(None), Path)

    def _add_array(value: Any) -> None:
        nonlocal total
        marker = id(value)
        if marker in seen_arrays:
            return
        seen_arrays.add(marker)
        total += int(
            value.numel()
            if torch is not None and isinstance(value, torch.Tensor)
            else value.size
        )

    while stack:
        value = stack.pop()
        if isinstance(value, atomic_types):
            continue
        marker = id(value)
        if marker in seen_objects:
            continue
        seen_objects.add(marker)

        if torch is not None and isinstance(value, torch.Tensor):
            _add_array(value)
            continue
        if isinstance(value, np.ndarray):
            _add_array(value)
            continue
        if isinstance(value, Mapping):
            try:
                for key, item in value.items():
                    stack.extend((key, item))
            except Exception:  # noqa: BLE001 - untrusted container
                complete = False
            continue
        if isinstance(value, (list, tuple, set, frozenset, deque)):
            try:
                stack.extend(value)
            except Exception:  # noqa: BLE001 - untrusted container
                complete = False
            continue
        if isinstance(value, types.MethodType):
            stack.extend((value.__self__, value.__func__))
            continue
        if isinstance(value, types.FunctionType):
            stack.extend(value.__defaults__ or ())
            stack.extend((value.__kwdefaults__ or {}).values())
            if value.__closure__:
                for cell in value.__closure__:
                    try:
                        stack.append(cell.cell_contents)
                    except ValueError:
                        continue
            for name in value.__code__.co_names:
                global_value = value.__globals__.get(name)
                if _may_contain_parameters(global_value, torch):
                    stack.append(global_value)
            continue
        if isinstance(value, ModuleType):
            if not _module_is_auditable(value):
                # A dependency or synthetic module is not submitted code, but a
                # policy can still hide weights on it and read them from act().
                # Harvest only the weight-bearing state; never crawl the module's
                # functions or internals.
                stack.extend(_foreign_module_weight_values(value))
                continue
            try:
                namespace = vars(value)
            except Exception:  # noqa: BLE001 - untrusted submitted module
                complete = False
                continue
            for name, item in namespace.items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                # An imported dependency function is not submitted state. Its
                # bound/default/closure state is still reached if the policy
                # itself retains the callable.
                if isinstance(
                    item, types.FunctionType
                ) and not _path_is_in_module_roots(item.__code__.co_filename):
                    continue
                if _may_contain_parameters(item, torch):
                    stack.append(item)
            continue
        if isinstance(value, type):
            if not _class_is_auditable(value):
                continue
            try:
                class_namespace = vars(value)
            except TypeError:
                continue
            except Exception:  # noqa: BLE001 - untrusted submitted class
                complete = False
                continue
            for name, attr in class_namespace.items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                if isinstance(attr, (staticmethod, classmethod)):
                    stack.append(attr.__func__)
                elif isinstance(attr, property):
                    if attr.fget is not None:
                        stack.append(attr.fget)
                elif _may_contain_parameters(attr, torch):
                    stack.append(attr)
            continue
        if isinstance(value, (types.CodeType, types.BuiltinFunctionType)):
            continue
        if torch is not None and isinstance(value, (torch.device, torch.dtype)):
            continue
        if isinstance(value, np.dtype):
            continue

        attributes_found = False
        try:
            attributes = vars(value)
        except TypeError:
            attributes = None
        except Exception:  # noqa: BLE001 - untrusted policy object
            attributes = None
            complete = False
        if attributes is not None:
            attributes_found = True
            stack.extend(attributes.values())

        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            try:
                stack.append(getattr(value, slot))
                attributes_found = True
            except AttributeError:
                continue
            except Exception:  # noqa: BLE001 - untrusted policy object
                complete = False

        # Instance __dict__/__slots__ alone miss a model stashed at class level:
        # ``class P: weights = big_model`` or a ``@property`` that returns it.
        # Walk class-level *data* attributes (skipping methods and other
        # descriptors, whose globals would explode the traversal) and property
        # getters so weights hidden on the class cannot zero out the count.
        try:
            mro = type(value).__mro__
        except Exception:  # noqa: BLE001 - untrusted policy type
            mro = ()
            complete = False
        for klass in mro:
            if klass in (object,):
                continue
            try:
                class_dict = vars(klass)
            except TypeError:
                continue
            for name, attr in class_dict.items():
                if name.startswith("__") and name.endswith("__"):
                    continue
                if isinstance(attr, property):
                    if attr.fget is not None:
                        stack.append(attr.fget)
                    continue
                # Functions, staticmethod/classmethod, and custom descriptors are
                # not where weights live; skipping them also avoids re-walking
                # every method's module globals.
                if hasattr(type(attr), "__get__"):
                    continue
                if _may_contain_parameters(attr, torch):
                    stack.append(attr)

        if not attributes_found:
            # Any non-atomic opaque value could retain model state that is not
            # visible to the auditor (including generators and C extensions).
            complete = False

    return total if complete else None


def _may_contain_parameters(value: Any, torch: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
        return False
    # Module and class provenance is checked by the caller before their
    # namespaces are traversed.
    if isinstance(value, (ModuleType, type)):
        return True
    if isinstance(value, (np.ndarray, Mapping, list, tuple, set, frozenset, deque)):
        return True
    if torch is not None and isinstance(value, (torch.Tensor, torch.nn.Module)):
        return True
    return True


def _count_safetensors(model_dir: Path) -> int | None:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None

    from safetensors import safe_open

    total = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # Read the shape from the header without materializing the tensor;
                # counting a 4.5B model must not load the whole model into the
                # validator's memory. Fall back to a full read only if the slice
                # API is unavailable in an older safetensors build.
                try:
                    shape = handle.get_slice(key).get_shape()
                except (AttributeError, TypeError):
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
        if (
            isinstance(obj, dict)
            and "state_dict" in obj
            and isinstance(obj["state_dict"], dict)
        ):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            continue
        for value in obj.values():
            if hasattr(value, "numel"):
                total += int(value.numel())
    return total
