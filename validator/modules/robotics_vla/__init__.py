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
    enforce_parameter_limit,
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
            parameter_count = count_model_parameters(model_dir)
            # Reject an obviously oversized model up front, before paying to load it.
            if parameter_count is not None and parameter_count > data.max_params:
                raise RoboticsSubmissionError(
                    f"Model parameters {parameter_count} exceed limit {data.max_params}",
                    failure_mode="parameter_limit_exceeded",
                )

            resolved_data = resolve_validation_data_package(data, self.config.package_cache_dir)
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
            )

            # Re-check the cap against the loaded policy to catch submissions that
            # ship tiny repo weights but pull a large base model in at load time.
            parameter_count = enforce_parameter_limit(
                parameter_count, count_policy_parameters(policy), data.max_params
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
            result = rollout_manifest(policy=policy, episodes=episodes, settings=settings)
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
