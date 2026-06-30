from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from validator.modules.robotics_vla.domain_randomization import (
    apply_domain_randomization_to_manifest,
    domain_randomization_summary,
    load_domain_randomization,
    validate_domain_randomization_spec,
)
from validator.modules.robotics_vla.manifest import ValidationManifest, load_manifest
from validator.modules.robotics_vla.task_registry import validate_task_registry_spec


PACKAGE_METADATA_FILENAME = "package.json"
DEFAULT_PACKAGE_CACHE_DIR = ".cache/robotics_vla/package_cache"


@dataclass(frozen=True)
class ResolvedValidationData:
    manifest: ValidationManifest
    task_registry: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, str] = field(default_factory=dict)


def resolve_validation_data_package(data: Any, cache_dir: str | Path) -> ResolvedValidationData:
    package_url = _first_present(
        data,
        "validation_data_url",
        "validation_zip_url",
        "validation_set_url",
    )
    domain_randomization_url = _first_present(data, "domain_randomization_url")
    manifest_url = _first_present(data, "validation_manifest_url")

    if package_url:
        return _resolve_zip_package(
            package_url=package_url,
            cache_dir=Path(cache_dir),
            domain_randomization_url=domain_randomization_url,
        )
    if not manifest_url:
        raise ValueError(
            "Robotics VLA validation requires either validation_data_url/validation_zip_url/"
            "validation_set_url or validation_manifest_url"
        )

    manifest = load_manifest(manifest_url)
    task_registry = validate_task_registry_spec(None)
    diagnostics = {
        "validation_data_source": "manifest",
        "validation_manifest_url": str(manifest_url),
    }
    if domain_randomization_url:
        spec_path = _materialize_to_cache(domain_randomization_url, Path(cache_dir), "domain_randomization.json")
        spec = load_domain_randomization(spec_path)
        manifest = apply_domain_randomization_to_manifest(manifest, spec)
        diagnostics.update(_stringify_diagnostics(domain_randomization_summary(manifest)))
        diagnostics["domain_randomization_source"] = str(domain_randomization_url)
    return ResolvedValidationData(manifest=manifest, task_registry=task_registry, diagnostics=diagnostics)


def _resolve_zip_package(
    package_url: str,
    cache_dir: Path,
    domain_randomization_url: str | None,
) -> ResolvedValidationData:
    package_path = _materialize_to_cache(package_url, cache_dir, "validation_package.zip")
    sha256 = _sha256(package_path)
    extract_dir = cache_dir / f"zip_{sha256[:16]}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_zip(package_path, extract_dir)

    metadata = _read_package_metadata(extract_dir)
    manifest_path = _find_packaged_file(
        extract_dir=extract_dir,
        explicit_path=metadata.get("manifest_path"),
        candidates=("manifest.json", "validation_manifest.json", "public_eval_manifest.json", "private_eval_manifest.json"),
    )
    manifest = load_manifest(str(manifest_path))
    task_registry_path = _find_packaged_file(
        extract_dir=extract_dir,
        explicit_path=metadata.get("task_registry_path"),
        candidates=("task_registry.json",),
        required=False,
    )
    if task_registry_path is not None:
        task_registry = validate_task_registry_spec(json.loads(task_registry_path.read_text()))
    else:
        task_registry = validate_task_registry_spec(None)

    randomization_path: Path | None = None
    if domain_randomization_url:
        randomization_path = _materialize_to_cache(
            domain_randomization_url,
            cache_dir,
            "domain_randomization.json",
        )
    else:
        explicit_randomization = metadata.get("domain_randomization_path")
        randomization_path = _find_packaged_file(
            extract_dir=extract_dir,
            explicit_path=explicit_randomization,
            candidates=("domain_randomization.json", "randomization.json"),
            required=False,
        )

    diagnostics = {
        "validation_data_source": "zip",
        "validation_package_url": str(package_url),
        "validation_package_sha256": sha256,
        "validation_manifest_path": str(manifest_path),
        "task_registry_tasks": ",".join(sorted(task_registry.get("tasks", {}))),
    }
    if task_registry_path is not None:
        diagnostics["task_registry_path"] = str(task_registry_path)
    if randomization_path is not None:
        spec = load_domain_randomization(randomization_path)
        manifest = apply_domain_randomization_to_manifest(manifest, spec)
        diagnostics["domain_randomization_path"] = str(randomization_path)
        diagnostics.update(_stringify_diagnostics(domain_randomization_summary(manifest)))

    return ResolvedValidationData(manifest=manifest, task_registry=task_registry, diagnostics=diagnostics)


def _read_package_metadata(extract_dir: Path) -> dict[str, Any]:
    metadata_path = extract_dir / PACKAGE_METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{PACKAGE_METADATA_FILENAME} must contain a JSON object")
    return metadata


def _find_packaged_file(
    extract_dir: Path,
    explicit_path: str | None,
    candidates: tuple[str, ...],
    required: bool = True,
) -> Path | None:
    if explicit_path:
        path = (extract_dir / explicit_path).resolve()
        try:
            path.relative_to(extract_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Packaged file path escapes zip root: {explicit_path}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Packaged file does not exist: {explicit_path}")
        return path

    for candidate in candidates:
        path = extract_dir / candidate
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(
            f"Validation package must contain one of: {', '.join(candidates)}"
        )
    return None


def _materialize_to_cache(url_or_path: str, cache_dir: Path, filename_hint: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(str(url_or_path))
    key = hashlib.sha256(str(url_or_path).encode("utf-8")).hexdigest()[:16]
    suffix = Path(parsed.path).suffix or Path(filename_hint).suffix
    output_path = cache_dir / f"{Path(filename_hint).stem}_{key}{suffix}"

    local_path = Path(str(url_or_path)).expanduser()
    if local_path.exists():
        shutil.copyfile(local_path, output_path)
        return output_path

    response = requests.get(str(url_or_path), timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            try:
                target.relative_to(output_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Zip entry escapes target directory: {member.filename}") from exc
        archive.extractall(output_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_present(data: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(data, name, None)
        if value:
            return str(value)
    return None


def _stringify_diagnostics(values: dict[str, Any]) -> dict[str, str]:
    return {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for key, value in values.items()}


def write_validation_package(
    manifest_path: Path,
    output_zip: Path,
    domain_randomization_path: Path | None = None,
    task_registry_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if domain_randomization_path is not None:
        spec = json.loads(domain_randomization_path.read_text())
        validate_domain_randomization_spec(spec)
    if task_registry_path is not None:
        spec = json.loads(task_registry_path.read_text())
        validate_task_registry_spec(spec)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    package_metadata = {
        "package_version": "robotics_vla_validation_package_v1",
        "manifest_path": "manifest.json",
    }
    if domain_randomization_path is not None:
        package_metadata["domain_randomization_path"] = "domain_randomization.json"
    if task_registry_path is not None:
        package_metadata["task_registry_path"] = "task_registry.json"
    if metadata:
        package_metadata.update(metadata)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, "manifest.json")
        archive.writestr(PACKAGE_METADATA_FILENAME, json.dumps(package_metadata, indent=2))
        if domain_randomization_path is not None:
            archive.write(domain_randomization_path, "domain_randomization.json")
        if task_registry_path is not None:
            archive.write(task_registry_path, "task_registry.json")
    return output_zip
