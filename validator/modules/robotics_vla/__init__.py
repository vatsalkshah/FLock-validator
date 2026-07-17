from __future__ import annotations

from loguru import logger
from pydantic import Field

from validator.modules.base import (
    BaseConfig,
    BaseInputData,
    BaseMetrics,
    BaseValidationModule,
)
from validator.modules.robotics_vla.adapter import (
    count_model_parameters,
    count_policy_parameters,
    load_policy_from_adapter,
    resolve_model_dir,
)
from validator.modules.robotics_vla.errors import RoboticsSubmissionError
from validator.modules.robotics_vla.data_package import (
    DEFAULT_PACKAGE_CACHE_DIR,
    resolve_validation_data_package,
)
from validator.modules.robotics_vla.simulation import RolloutSettings, rollout_manifest


WORST_POSSIBLE_LOSS = 1.0
LOWEST_POSSIBLE_RETURN = -999.0


class RoboticsVLAConfig(BaseConfig):
    seed: int = 42
    suite_version: str = "panda_tabletop_v1"
    robot: str = "Panda"
    controller: str = "BASIC"
    horizon: int = 200
    control_freq: int = 20
    camera_name: str = "agentview"
    camera_height: int = 224
    camera_width: int = 224
    action_dim: int = 7
    render_video: bool = False
    video_dir: str = "validation_outputs/robotics_vla"
    max_videos: int = 2
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_episodes: int | None = None
    max_episode_horizon: int = 300
    expose_raw_obs: bool = False
    package_cache_dir: str = DEFAULT_PACKAGE_CACHE_DIR
    policy_load_timeout_seconds: float = Field(default=600.0, gt=0)
    policy_action_timeout_seconds: float = Field(default=30.0, gt=0)
    # Authoritative model-size ceiling. The policy sandbox is held to this much
    # memory (GPU VRAM via the CUDA allocator, and system RAM via a runtime
    # monitor); a submission that fits may hold however many parameters it likes.
    # 18 GiB ~= a 4.5B-param model in fp32, or a larger quantized one.
    policy_memory_limit_gb: int = Field(default=18, ge=1)
    policy_cpu_time_seconds: int = Field(default=3600, ge=1)


class RoboticsVLAMetrics(BaseMetrics):
    score: float
    loss: float
    mean_episode_score: float
    weighted_episode_score: float
    success_rate: float
    mean_progress_score: float
    mean_return: float
    mean_episode_length: float
    episodes_completed: int
    invalid_submission: bool = False
    parameter_count: int | None = None
    video_paths: list[str] = Field(default_factory=list)
    diagnostics: dict[str, str] = Field(default_factory=dict)


class RoboticsVLAInputData(BaseInputData):
    hg_repo_id: str
    revision: str = "main"
    validation_manifest_url: str | None = None
    validation_data_url: str | None = None
    validation_zip_url: str | None = None
    validation_set_url: str | None = None
    domain_randomization_url: str | None = None
    max_params: int = 4_500_000_000
    adapter_filename: str = "flock_robotics_adapter.py"


class RoboticsVLAValidationModule(BaseValidationModule):
    config_schema = RoboticsVLAConfig
    metrics_schema = RoboticsVLAMetrics
    input_data_schema = RoboticsVLAInputData
    task_type = "robotics_vla"

    def __init__(self, config: RoboticsVLAConfig, **kwargs):
        self.config = config

    def validate(self, data: RoboticsVLAInputData, **kwargs) -> RoboticsVLAMetrics:
        parameter_count: int | None = None
        try:
            model_dir = resolve_model_dir(data.hg_repo_id, data.revision)
            # Parameter count is telemetry only. Model size is enforced at runtime
            # by the sandbox memory limit (see load_policy_from_adapter), which
            # bounds the policy however its weights are represented; a static count
            # can be defeated by reconstructing weights at inference time.
            parameter_count = count_model_parameters(model_dir)

            resolved_data = resolve_validation_data_package(
                data, self.config.package_cache_dir
            )
            manifest = resolved_data.manifest
            if manifest.suite_version != self.config.suite_version:
                # A package/config mismatch is an operator-side (infra) problem, not
                # the submitter's fault. Raise RecoverableException so the runner
                # lets the assignment time out and be re-queued rather than zeroing
                # the miner's score.
                from validator.exceptions import RecoverableException

                raise RecoverableException(
                    f"Manifest suite_version {manifest.suite_version!r} does not match "
                    f"config suite_version {self.config.suite_version!r}"
                )

            policy = load_policy_from_adapter(
                model_dir=model_dir,
                adapter_filename=data.adapter_filename,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                load_timeout_seconds=self.config.policy_load_timeout_seconds,
                action_timeout_seconds=self.config.policy_action_timeout_seconds,
                memory_limit_bytes=self.config.policy_memory_limit_gb * 1024**3,
                cpu_time_seconds=self.config.policy_cpu_time_seconds,
            )
            try:
                # Prefer the sandbox's live graph audit for the reported count when
                # it is available; it is telemetry, not a gate (memory is the gate).
                live_count = count_policy_parameters(policy)
                if isinstance(live_count, int):
                    parameter_count = (
                        live_count
                        if parameter_count is None
                        else max(parameter_count, live_count)
                    )

                episodes = manifest.episodes
                if self.config.max_episodes is not None:
                    episodes = episodes[: self.config.max_episodes]

                settings = RolloutSettings(
                    seed=self.config.seed,
                    suite_version=self.config.suite_version,
                    robot=self.config.robot,
                    controller=self.config.controller,
                    horizon=self.config.horizon,
                    max_episode_horizon=self.config.max_episode_horizon,
                    control_freq=self.config.control_freq,
                    camera_name=self.config.camera_name,
                    camera_height=self.config.camera_height,
                    camera_width=self.config.camera_width,
                    action_dim=self.config.action_dim,
                    render_video=self.config.render_video,
                    video_dir=self.config.video_dir,
                    max_videos=self.config.max_videos,
                    expose_raw_obs=self.config.expose_raw_obs,
                    task_registry=resolved_data.task_registry,
                )
                result = rollout_manifest(
                    policy=policy, episodes=episodes, settings=settings
                )
            finally:
                policy.close()
            return RoboticsVLAMetrics(
                score=result.weighted_episode_score,
                loss=result.loss,
                mean_episode_score=result.mean_episode_score,
                weighted_episode_score=result.weighted_episode_score,
                success_rate=result.success_rate,
                mean_progress_score=result.mean_progress_score,
                mean_return=result.mean_return,
                mean_episode_length=result.mean_episode_length,
                episodes_completed=result.episodes_completed,
                invalid_submission=False,
                parameter_count=parameter_count,
                video_paths=result.video_paths,
                diagnostics={**resolved_data.diagnostics, **result.diagnostics},
            )
        except RoboticsSubmissionError as exc:
            # The submission itself is invalid/unrunnable: score it 0 and keep the
            # validator alive. Genuine infra errors are intentionally NOT caught
            # here, so the runner can retry them or re-queue the assignment.
            logger.error(f"Invalid robotics VLA submission [{exc.failure_mode}]: {exc}")
            return self._invalid_metrics(
                str(exc), parameter_count=parameter_count, failure_mode=exc.failure_mode
            )

    def _invalid_metrics(
        self,
        reason: str,
        parameter_count: int | None = None,
        failure_mode: str = "submission_error",
    ) -> RoboticsVLAMetrics:
        return RoboticsVLAMetrics(
            score=0.0,
            loss=WORST_POSSIBLE_LOSS,
            mean_episode_score=0.0,
            weighted_episode_score=0.0,
            success_rate=0.0,
            mean_progress_score=0.0,
            mean_return=LOWEST_POSSIBLE_RETURN,
            mean_episode_length=0.0,
            episodes_completed=0,
            invalid_submission=True,
            parameter_count=parameter_count,
            video_paths=[],
            diagnostics={"reason": reason, "failure_mode": failure_mode},
        )

    def cleanup(self):
        pass


MODULE = RoboticsVLAValidationModule
