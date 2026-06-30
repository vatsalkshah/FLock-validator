import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from validator.modules.robotics_vla.adapter import (
    count_policy_parameters,
    enforce_parameter_limit,
    load_policy_from_adapter,
)
from validator.modules.robotics_vla import RoboticsVLAValidationModule
from validator.modules.robotics_vla.errors import RoboticsSubmissionError
from validator.modules.robotics_vla.data_package import (
    resolve_validation_data_package,
    write_validation_package,
)
from validator.modules.robotics_vla.domain_randomization import (
    DOMAIN_RANDOMIZATION_VERSION,
    apply_domain_randomization_to_manifest,
)
from validator.modules.robotics_vla.manifest import EpisodeSpec, ValidationManifest
from validator.modules.robotics_vla.manifest import load_manifest
from validator.modules.robotics_vla.simulation import (
    SUPPORTED_TASKS,
    RolloutSettings,
    adapt_action_for_env,
    compute_episode_score,
    compute_progress_score,
    compute_weighted_episode_score,
    compute_weighted_loss,
    difficulty_weight,
    extract_frame,
    prepare_policy_obs,
    query_policy_action,
    resolve_task_definition,
    target_nut_on_matching_peg_success,
    target_object_in_matching_bin_success,
    validate_action,
)
from validator.modules.robotics_vla.task_registry import (
    TASK_REGISTRY_VERSION,
    validate_task_registry_spec,
)


def test_manifest_loads_json_file(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "task": "lift_cube",
                        "instruction": "lift the cube",
                        "seed": 1,
                    }
                ],
            }
        )
    )

    manifest = load_manifest(str(manifest_path))

    assert manifest.suite_version == "panda_tabletop_v1"
    assert manifest.episodes[0].task == "lift_cube"


def test_validate_action_clips_and_rejects_bad_shapes():
    action = validate_action(np.array([2, -2, 0, 0, 0, 0, 0]), 7)

    assert action.tolist() == [1, -1, 0, 0, 0, 0, 0]

    with pytest.raises(RoboticsSubmissionError, match="shape") as excinfo:
        validate_action(np.zeros((6,), dtype=np.float32), 7)
    assert excinfo.value.failure_mode == "invalid_action"


def test_validate_action_rejects_nan():
    with pytest.raises(RoboticsSubmissionError, match="NaN") as excinfo:
        validate_action(np.array([0, 0, np.nan, 0, 0, 0, 0]), 7)
    assert excinfo.value.failure_mode == "invalid_action"


def test_adapt_action_for_env_matches_declared_env_dimension():
    class Env:
        action_dim = 6

    action = np.arange(7, dtype=np.float32)

    adapted = adapt_action_for_env(action, Env())

    np.testing.assert_array_equal(adapted, np.arange(6, dtype=np.float32))


def test_hybrid_strict_scoring_caps_failed_partial_credit():
    assert compute_episode_score(success=True, progress_score=0.0) == 1.0
    assert compute_progress_score(success=True, best_reward=0.0) == 1.0
    assert compute_episode_score(success=False, progress_score=1.0) == 0.40
    assert compute_episode_score(success=False, progress_score=0.5) == 0.20
    assert compute_progress_score(success=False, best_reward=2.0) == 1.0
    assert compute_progress_score(success=False, best_reward=-1.0) == 0.0


def test_difficulty_weights_are_simple_and_monotonic():
    weights = [
        difficulty_weight(EpisodeSpec(task="lift_cube", instruction="x", seed=1, difficulty=difficulty))
        for difficulty in ["low", "medium", "hard", "very_high"]
    ]

    assert weights == sorted(weights)
    assert weights == [0.75, 1.0, 1.25, 1.5]


def test_lower_is_better_weighted_loss_matches_score_complement():
    scores = np.array([1.0, 0.4, 0.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)

    weighted_score = compute_weighted_episode_score(scores, weights)
    loss = compute_weighted_loss(scores, weights)

    assert weighted_score == pytest.approx(0.45)
    assert loss == pytest.approx(0.55)


def test_robotics_metric_payload_separates_score_from_loss():
    invalid = RoboticsVLAValidationModule._invalid_metrics(
        RoboticsVLAValidationModule.__new__(RoboticsVLAValidationModule),
        "too many params",
    )

    assert invalid.score == 0.0
    assert invalid.loss == 1.0


def test_extract_frame_rotates_camera_180_degrees():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    frame = extract_frame({"agentview_image": image}, "agentview")

    np.testing.assert_array_equal(frame, np.rot90(image, 2))
    assert frame.flags.c_contiguous


def test_prepare_policy_obs_hides_raw_state_by_default():
    raw_obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
        "Can_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "object-state": np.ones(14, dtype=np.float32),
    }
    episode = EpisodeSpec(task="pick_place_can", instruction="pick can", seed=1)
    settings = RolloutSettings(
        seed=0,
        suite_version="panda_tabletop_v1",
        robot="Panda",
        controller="BASIC",
        horizon=320,
        max_episode_horizon=300,
        control_freq=20,
        camera_name="agentview",
        camera_height=2,
        camera_width=2,
        action_dim=7,
        render_video=False,
        video_dir="",
        max_videos=0,
    )

    obs = prepare_policy_obs(raw_obs, episode, settings, step_idx=3)

    assert set(obs) == {"image", "instruction", "proprio", "task", "step", "difficulty", "horizon"}
    assert obs["proprio"].shape == (25,)
    assert obs["difficulty"] is None
    assert obs["horizon"] == 320
    assert "raw_obs" not in obs
    assert "low_dim" not in obs


def test_prepare_policy_obs_uses_episode_camera_override():
    raw_obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "frontview_image": np.ones((2, 2, 3), dtype=np.uint8) * 255,
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
    }
    episode = EpisodeSpec(
        task="pick_place_can",
        instruction="pick can",
        seed=1,
        camera_name="frontview",
    )
    settings = RolloutSettings(
        seed=0,
        suite_version="panda_tabletop_v1",
        robot="Panda",
        controller="BASIC",
        horizon=320,
        max_episode_horizon=300,
        control_freq=20,
        camera_name="agentview",
        camera_height=2,
        camera_width=2,
        action_dim=7,
        render_video=False,
        video_dir="",
        max_videos=0,
    )

    obs = prepare_policy_obs(raw_obs, episode, settings, step_idx=0)

    assert obs["image"].mean() == 255


def test_domain_randomization_is_deterministic_and_manifest_level():
    manifest = ValidationManifest(
        episodes=[
            EpisodeSpec(
                episode_id="episode-1",
                task="pick_place_can",
                instruction="Pick up the can.",
                seed=100,
                horizon=120,
                camera_name="agentview",
                difficulty="low",
                tags=["public_eval"],
            ),
            EpisodeSpec(
                episode_id="episode-2",
                task="pick_place_milk",
                instruction="Pick up the milk.",
                seed=101,
                horizon=140,
                camera_name="agentview",
                difficulty="medium",
                tags=["public_eval"],
            ),
            EpisodeSpec(
                episode_id="episode-3",
                task="lift_cube",
                instruction="Lift the cube.",
                seed=102,
                horizon=90,
                camera_name="agentview",
                difficulty="low",
                tags=["public_eval"],
            ),
        ]
    )
    spec = {
        "version": DOMAIN_RANDOMIZATION_VERSION,
        "seed_salt": "unit-test",
        "rules": [
            {
                "name": "heldout_frontview_pick_place",
                "match": {"task": ["pick_place_can", "pick_place_milk"]},
                "camera_weights": {"frontview": 1.0},
                "horizon_jitter": {"min": 7, "max": 7},
                "seed_offset_range": {"min": 1000, "max": 1000},
                "instruction_prefixes": ["Held-out visual variant: {instruction_lower}"],
                "tags": ["heldout_visual_variant"],
                "modifiers": {"bin_texture": "matte_private_eval"},
            }
        ],
    }

    first = apply_domain_randomization_to_manifest(manifest, spec)
    second = apply_domain_randomization_to_manifest(manifest, spec)

    assert first.model_dump() == second.model_dump()
    randomized = [episode for episode in first.episodes if episode.task in {"pick_place_can", "pick_place_milk"}]
    assert randomized
    for original, episode in zip(manifest.episodes, first.episodes):
        if episode.task in {"pick_place_can", "pick_place_milk"}:
            assert episode.camera_name == "frontview"
            assert episode.horizon == (original.horizon or 0) + 7
            assert episode.seed == original.seed + 1000
            assert episode.instruction.startswith("Held-out visual variant:")
            assert "heldout_visual_variant" in episode.tags
            assert episode.domain_randomization["modifiers"]["bin_texture"] == "matte_private_eval"


def test_validation_zip_package_applies_domain_randomization(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    randomization_path = tmp_path / "domain_randomization.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "episode_id": "episode-1",
                        "task": "pick_place_can",
                        "instruction": "Pick up the can.",
                        "seed": 7,
                        "horizon": 120,
                        "camera_name": "agentview",
                        "difficulty": "low",
                    }
                ],
            }
        )
    )
    randomization_path.write_text(
        json.dumps(
            {
                "version": DOMAIN_RANDOMIZATION_VERSION,
                "seed_salt": "zip-test",
                "rules": [
                    {
                        "name": "force_frontview",
                        "match": {"task": "pick_place_can"},
                        "camera_weights": {"frontview": 1.0},
                        "horizon_jitter": {"min": 5, "max": 5},
                        "tags": ["zip_randomized"],
                    }
                ],
            }
        )
    )
    write_validation_package(manifest_path, output_zip, randomization_path, metadata={"split": "public_eval"})

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")
    episode = resolved.manifest.episodes[0]

    assert episode.camera_name == "frontview"
    assert episode.horizon == 125
    assert "zip_randomized" in episode.tags
    assert resolved.diagnostics["validation_data_source"] == "zip"
    assert resolved.diagnostics["domain_randomized_episodes"] == "1"


def test_validation_zip_package_loads_allowlisted_task_registry(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "task_registry.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "episode_id": "private-target-1",
                        "task": "targeted_clutter_pick_place",
                        "instruction": "Pick only the can into the matching bin.",
                        "seed": 9,
                        "horizon": 260,
                        "camera_name": "frontview",
                        "difficulty": "hard",
                        "task_kwargs": {"target_object": "can"},
                    }
                ],
            }
        )
    )
    registry_path.write_text(
        json.dumps(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "targeted_clutter_pick_place": {
                        "base_env": "PickPlace",
                        "env_kwargs": {"single_object_mode": 0},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": ["target_object"],
                        "partial_metrics": ["target_reach_score", "target_lift_score"],
                    }
                },
            }
        )
    )
    write_validation_package(manifest_path, output_zip, task_registry_path=registry_path)

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")
    episode = resolved.manifest.episodes[0]
    env_name, env_kwargs, task_entry = resolve_task_definition(
        episode,
        RolloutSettings(
            seed=0,
            suite_version="panda_tabletop_v1",
            robot="Panda",
            controller="BASIC",
            horizon=320,
            max_episode_horizon=300,
            control_freq=20,
            camera_name="agentview",
            camera_height=2,
            camera_width=2,
            action_dim=7,
            render_video=False,
            video_dir="",
            max_videos=0,
            task_registry=resolved.task_registry,
        ),
    )

    assert env_name == "PickPlace"
    assert env_kwargs == {"single_object_mode": 0}
    assert task_entry["success_checker"] == "target_object_in_matching_bin"
    assert resolved.diagnostics["task_registry_tasks"] == "targeted_clutter_pick_place"


def test_task_registry_rejects_non_allowlisted_code_or_env():
    with pytest.raises(ValueError, match="unsupported base_env"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "ArbitraryEnv",
                        "env_kwargs": {},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="unsupported success_checker"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "PickPlace",
                        "env_kwargs": {},
                        "success_checker": "__import__('os').system",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="unsupported env_kwargs"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "PickPlace",
                        "env_kwargs": {"robots": "Sawyer"},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )


def test_targeted_clutter_success_uses_target_object_only():
    class Obj:
        def __init__(self, name):
            self.name = name

    class Env:
        objects = [Obj("Milk"), Obj("Bread"), Obj("Cereal"), Obj("Can")]
        object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        objects_in_bins = np.array([0, 0, 0, 1])

        def _check_success(self):
            return False

    assert target_object_in_matching_bin_success(
        Env(),
        EpisodeSpec(
            task="targeted_clutter_pick_place",
            instruction="pick can",
            seed=1,
            task_kwargs={"target_object": "can"},
        ),
    )
    assert not target_object_in_matching_bin_success(
        Env(),
        EpisodeSpec(
            task="targeted_clutter_pick_place",
            instruction="pick milk",
            seed=1,
            task_kwargs={"target_object": "milk"},
        ),
    )


def test_targeted_nut_success_uses_target_nut_only():
    class Env:
        nut_to_id = {"square": 0, "round": 1}
        objects_on_pegs = np.array([0, 1])

        def _check_success(self):
            return False

    assert target_nut_on_matching_peg_success(
        Env(),
        EpisodeSpec(
            task="distractor_nut_assembly",
            instruction="place round nut",
            seed=1,
            task_kwargs={"target_nut": "round"},
        ),
    )
    assert not target_nut_on_matching_peg_success(
        Env(),
        EpisodeSpec(
            task="distractor_nut_assembly",
            instruction="place square nut",
            seed=1,
            task_kwargs={"target_nut": "square"},
        ),
    )


def test_validation_set_url_alias_loads_robotics_zip(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [{"task": "lift_cube", "instruction": "Lift.", "seed": 1}],
            }
        )
    )
    write_validation_package(manifest_path, output_zip)

    class Input:
        validation_data_url = None
        validation_zip_url = None
        validation_set_url = str(output_zip)
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")

    assert resolved.manifest.episodes[0].task == "lift_cube"
    assert resolved.diagnostics["validation_data_source"] == "zip"


def test_validation_zip_rejects_path_traversal(tmp_path: Path):
    output_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr("../manifest.json", "{}")

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    with pytest.raises(ValueError, match="escapes"):
        resolve_validation_data_package(Input(), tmp_path / "cache")


def test_adapter_contract_loads_policy(tmp_path: Path):
    adapter_path = tmp_path / "flock_robotics_adapter.py"
    adapter_path.write_text(
        """
import numpy as np

class Policy:
    def act(self, obs):
        return np.zeros(7, dtype=np.float32)

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(tmp_path, "flock_robotics_adapter.py", "cpu", "float32")

    assert policy.act({}).shape == (7,)


# --- Issue 1: bad submissions are scored, never crash the validator ----------


def test_adapter_loads_through_hf_style_symlink(tmp_path: Path):
    """HF snapshots symlink repo files into a sibling blobs/ dir; the adapter
    loader must follow that without flagging it as a path escape."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "deadbeef"
    blob.write_text(
        "import numpy as np\n"
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return np.zeros(7, dtype=np.float32)\n"
        "def load_policy(model_dir, device, dtype):\n"
        "    return Policy()\n"
    )
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "flock_robotics_adapter.py").symlink_to(blob)

    policy = load_policy_from_adapter(snapshot, "flock_robotics_adapter.py", "cpu", "float32")

    assert policy.act({}).shape == (7,)


def test_adapter_filename_escape_is_rejected(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text(
        "def load_policy(model_dir, device, dtype):\n    return object()\n"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(model_dir, "../flock_robotics_adapter.py", "cpu", "float32")
    assert excinfo.value.failure_mode == "adapter_contract"


def test_missing_adapter_is_submission_error(tmp_path: Path):
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(tmp_path, "flock_robotics_adapter.py", "cpu", "float32")
    assert excinfo.value.failure_mode == "adapter_missing"


def test_adapter_without_load_policy_is_contract_error(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text("X = 1\n")
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(tmp_path, "flock_robotics_adapter.py", "cpu", "float32")
    assert excinfo.value.failure_mode == "adapter_contract"


def test_policy_without_act_is_contract_error(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text(
        "def load_policy(model_dir, device, dtype):\n    return object()\n"
    )
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(tmp_path, "flock_robotics_adapter.py", "cpu", "float32")
    assert excinfo.value.failure_mode == "adapter_contract"


def test_failing_load_policy_is_retried_then_caught(tmp_path: Path):
    from validator.modules.robotics_vla.adapter import MODEL_QUERY_RETRIES

    counter = tmp_path / "load_attempts.txt"
    counter.write_text("0")
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
from pathlib import Path

def load_policy(model_dir, device, dtype):
    counter = Path(model_dir) / "load_attempts.txt"
    counter.write_text(str(int(counter.read_text()) + 1))
    raise RuntimeError("model would not load")
"""
    )
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(tmp_path, "flock_robotics_adapter.py", "cpu", "float32")
    assert excinfo.value.failure_mode == "model_load_failed"
    assert int(counter.read_text()) == 1 + MODEL_QUERY_RETRIES


def test_query_policy_action_retries_then_reports_invalid_action():
    from validator.modules.robotics_vla.adapter import MODEL_QUERY_RETRIES

    class BadShapePolicy:
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return np.zeros(6, dtype=np.float32)

    policy = BadShapePolicy()
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        query_policy_action(policy, {}, 7)
    assert excinfo.value.failure_mode == "invalid_action"
    assert policy.calls == 1 + MODEL_QUERY_RETRIES


def test_query_policy_action_recovers_from_transient_failure():
    class FlakyPolicy:
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient cuda error")
            return np.zeros(7, dtype=np.float32)

    policy = FlakyPolicy()
    action = query_policy_action(policy, {}, 7)
    assert action.shape == (7,)
    assert policy.calls == 3


def test_invalid_metrics_carry_failure_mode():
    metrics = RoboticsVLAValidationModule._invalid_metrics(
        RoboticsVLAValidationModule.__new__(RoboticsVLAValidationModule),
        "broken adapter",
        failure_mode="adapter_missing",
    )
    assert metrics.invalid_submission is True
    assert metrics.score == 0.0
    assert metrics.diagnostics["failure_mode"] == "adapter_missing"


def test_runner_does_not_crash_on_bad_submission():
    """A submission that raises must fail just that assignment, not exit(1)."""
    import sys as _sys

    from validator.validation_runner import ValidationRunner

    class BoomModule:
        def __init__(self):
            self.calls = 0

        def validate(self, data):
            self.calls += 1
            raise ValueError("policy action must have shape (7,)")

    class FakeApi:
        def __init__(self):
            self.failed = []

        def mark_assignment_as_failed(self, assignment_id):
            self.failed.append(assignment_id)

    runner = ValidationRunner.__new__(ValidationRunner)
    module = BoomModule()
    runner.task_id_to_module = {"task-1": module}
    runner.api = FakeApi()

    real_exit = _sys.exit
    _sys.exit = lambda *a: (_ for _ in ()).throw(AssertionError("runner called sys.exit"))
    try:
        result = runner.perform_validation("assignment-1", "task-1", object())
    finally:
        _sys.exit = real_exit

    assert result is None
    assert module.calls == 3
    assert runner.api.failed == ["assignment-1"]


# --- Issue 2: the parameter cap is actually enforced -------------------------


def test_enforce_parameter_limit_rejects_unknown_count():
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        enforce_parameter_limit(None, None, 4_500_000_000)
    assert excinfo.value.failure_mode == "parameter_count_unknown"


def test_enforce_parameter_limit_rejects_oversized_policy():
    # Repo weights look tiny, but the loaded policy is over the cap.
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        enforce_parameter_limit(1_000, 5_000_000_000, 4_500_000_000)
    assert excinfo.value.failure_mode == "parameter_limit_exceeded"


def test_enforce_parameter_limit_returns_best_known_count():
    assert enforce_parameter_limit(1_000, 350_000, 4_500_000_000) == 350_000
    assert enforce_parameter_limit(420_000, None, 4_500_000_000) == 420_000


def test_count_policy_parameters_handles_non_torch_policy():
    class Policy:
        def act(self, obs):
            return None

    # A policy with no torch modules cannot be counted -> None (not a crash).
    assert count_policy_parameters(Policy()) is None
