from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from validator.modules.robotics_vla.manifest import EpisodeSpec, ValidationManifest


DOMAIN_RANDOMIZATION_VERSION = "robotics_vla_domain_randomization_v1"
SUPPORTED_CAMERAS = {"agentview", "frontview"}


def load_domain_randomization(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    randomization_path = Path(path)
    if not randomization_path.exists():
        raise FileNotFoundError(f"Domain randomization file does not exist: {randomization_path}")
    spec = json.loads(randomization_path.read_text())
    validate_domain_randomization_spec(spec)
    return spec


def validate_domain_randomization_spec(spec: dict[str, Any]) -> None:
    version = spec.get("version")
    if version != DOMAIN_RANDOMIZATION_VERSION:
        raise ValueError(
            f"Unsupported robotics domain randomization version {version!r}; "
            f"expected {DOMAIN_RANDOMIZATION_VERSION!r}"
        )
    rules = spec.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("Domain randomization spec must contain a list field named 'rules'")
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Domain randomization rule {idx} must be an object")
        match = rule.get("match", {})
        if match is not None and not isinstance(match, dict):
            raise ValueError(f"Domain randomization rule {idx} match must be an object")
        if "camera_weights" in rule:
            camera_weights = rule["camera_weights"]
            if not isinstance(camera_weights, dict) or not camera_weights:
                raise ValueError(f"Domain randomization rule {idx} camera_weights must be a non-empty object")
            unsupported = set(camera_weights) - SUPPORTED_CAMERAS
            if unsupported:
                raise ValueError(
                    f"Domain randomization rule {idx} has unsupported camera(s): {sorted(unsupported)}"
                )
        if "horizon_jitter" in rule:
            jitter = rule["horizon_jitter"]
            if not isinstance(jitter, dict) or "min" not in jitter or "max" not in jitter:
                raise ValueError(f"Domain randomization rule {idx} horizon_jitter must have min and max")
            if int(jitter["min"]) > int(jitter["max"]):
                raise ValueError(f"Domain randomization rule {idx} horizon_jitter min exceeds max")
        if "seed_offset_range" in rule:
            offset = rule["seed_offset_range"]
            if not isinstance(offset, dict) or "min" not in offset or "max" not in offset:
                raise ValueError(f"Domain randomization rule {idx} seed_offset_range must have min and max")
            if int(offset["min"]) > int(offset["max"]):
                raise ValueError(f"Domain randomization rule {idx} seed_offset_range min exceeds max")


def apply_domain_randomization_to_manifest(
    manifest: ValidationManifest,
    spec: dict[str, Any] | None,
) -> ValidationManifest:
    if not spec:
        return manifest
    validate_domain_randomization_spec(spec)
    seed_salt = str(spec.get("seed_salt", "robotics_vla_domain_randomization"))
    rules = spec.get("rules", [])
    episodes = []
    for episode_index, episode in enumerate(manifest.episodes):
        payload = episode.model_dump()
        randomization_meta = deepcopy(payload.get("domain_randomization") or {})
        matched_rule_names = []
        for rule_index, rule in enumerate(rules):
            if not _rule_matches(episode, rule.get("match", {})):
                continue
            rule_name = str(rule.get("name", f"rule_{rule_index}"))
            matched_rule_names.append(rule_name)
            label_base = _label(seed_salt, episode, episode_index, rule_index, rule_name)

            if "camera_weights" in rule:
                payload["camera_name"] = _weighted_choice(f"{label_base}:camera", rule["camera_weights"])

            if "horizon_jitter" in rule:
                jitter = rule["horizon_jitter"]
                delta = _int_inclusive(f"{label_base}:horizon", int(jitter["min"]), int(jitter["max"]))
                base_horizon = int(payload.get("horizon") or 0)
                payload["horizon"] = int(max(80, min(640, base_horizon + delta)))
                randomization_meta.setdefault("horizon_jitter", []).append(delta)

            if "seed_offset_range" in rule:
                offset = rule["seed_offset_range"]
                delta = _int_inclusive(f"{label_base}:seed", int(offset["min"]), int(offset["max"]))
                payload["seed"] = int(payload["seed"]) + delta
                randomization_meta.setdefault("seed_offsets", []).append(delta)

            prefixes = rule.get("instruction_prefixes") or []
            if prefixes:
                prefix = prefixes[_hash_uint(f"{label_base}:instruction") % len(prefixes)]
                payload["instruction"] = _format_instruction(prefix, str(payload["instruction"]))

            tags = rule.get("tags") or rule.get("extra_tags") or []
            if tags:
                payload["tags"] = sorted(set([*payload.get("tags", []), *[str(tag) for tag in tags]]))

            modifiers = rule.get("modifiers") or {}
            if modifiers:
                randomization_meta.setdefault("modifiers", {}).update(deepcopy(modifiers))

        if matched_rule_names:
            payload["tags"] = sorted(set([*payload.get("tags", []), "domain_randomized"]))
            randomization_meta["matched_rules"] = matched_rule_names
        payload["domain_randomization"] = randomization_meta
        episodes.append(EpisodeSpec.model_validate(payload))
    return ValidationManifest(suite_version=manifest.suite_version, episodes=episodes)


def domain_randomization_summary(manifest: ValidationManifest) -> dict[str, Any]:
    randomized = [episode for episode in manifest.episodes if episode.domain_randomization]
    rule_counts: dict[str, int] = {}
    modifier_counts: dict[str, int] = {}
    for episode in randomized:
        for rule_name in episode.domain_randomization.get("matched_rules", []):
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        for modifier_name in episode.domain_randomization.get("modifiers", {}):
            modifier_counts[modifier_name] = modifier_counts.get(modifier_name, 0) + 1
    return {
        "domain_randomized_episodes": len(randomized),
        "rule_counts": dict(sorted(rule_counts.items())),
        "modifier_counts": dict(sorted(modifier_counts.items())),
    }


def _rule_matches(episode: EpisodeSpec, match: dict[str, Any] | None) -> bool:
    if not match:
        return True
    for key, expected in match.items():
        if key == "tag":
            values = set(_as_list(expected))
            if not values.intersection(set(episode.tags)):
                return False
            continue
        actual = getattr(episode, key, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _label(seed_salt: str, episode: EpisodeSpec, episode_index: int, rule_index: int, rule_name: str) -> str:
    episode_key = episode.episode_id or f"{episode.task}:{episode.seed}:{episode_index}"
    return f"{seed_salt}:{episode_key}:{rule_index}:{rule_name}"


def _weighted_choice(label: str, weights: dict[str, float]) -> str:
    total = float(sum(float(value) for value in weights.values()))
    if total <= 0:
        raise ValueError("Domain randomization weights must sum to a positive value")
    cursor = _unit_float(label) * total
    acc = 0.0
    for key, weight in weights.items():
        acc += float(weight)
        if cursor <= acc:
            return key
    return next(reversed(weights))


def _int_inclusive(label: str, lower: int, upper: int) -> int:
    span = upper - lower + 1
    return lower + (_hash_uint(label) % span)


def _unit_float(label: str) -> float:
    return _hash_uint(label) / float(2**64 - 1)


def _hash_uint(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _format_instruction(template: str, instruction: str) -> str:
    if "{instruction}" in template:
        return template.format(instruction=instruction, instruction_lower=_lower_first(instruction))
    return f"{template} {_lower_first(instruction)}"


def _lower_first(text: str) -> str:
    if not text:
        return text
    return f"{text[0].lower()}{text[1:]}"
