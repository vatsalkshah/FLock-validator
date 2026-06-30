import sys
import json
import click
from validator.validation_runner import ValidationRunner

@click.command()
@click.argument(
    "module",
    type=str,
    required=True,
)
@click.option(
    "--task_ids",
    type=str,
    required=False,
    help="The ids of the task, separated by comma",
)
@click.option('--flock-api-key', envvar='FLOCK_API_KEY', required=False, help='Flock API key')
@click.option('--hf-token', envvar='HF_TOKEN', required=False, help='HuggingFace token')
@click.option('--time-sleep', envvar='TIME_SLEEP', default=60 * 3, type=int, show_default=True, help='Time to sleep between retries (seconds)')
@click.option('--assignment-lookup-interval', envvar='ASSIGNMENT_LOOKUP_INTERVAL', default=60 * 3, type=int, show_default=True, help='Assignment lookup interval (seconds)')
@click.option("--debug", is_flag=True)
@click.option("--local-validation", is_flag=True, help="Run one local validation and exit. Supported for robotics_vla.")
@click.option("--hg-repo-id", "--hf-model-repo", "local_hg_repo_id", type=str, help="HuggingFace repo id or local model directory for local robotics validation.")
@click.option("--revision", default="main", show_default=True, help="HuggingFace revision for local validation.")
@click.option("--validation-manifest-url", type=str, help="Manifest path/URL for local robotics validation.")
@click.option("--validation-data-url", "--validation-zip-url", "--validation-set-url", "validation_data_url", type=str, help="Zip package path/URL for local robotics validation.")
@click.option("--domain-randomization-url", type=str, help="Optional domain randomization JSON path/URL for local robotics validation.")
@click.option("--adapter-filename", default="flock_robotics_adapter.py", show_default=True, help="Adapter file inside the model repo.")
@click.option("--max-params", default=4_500_000_000, show_default=True, type=int, help="Maximum allowed model parameter count.")
@click.option("--max-episodes", type=int, help="Limit local robotics validation to the first N episodes.")
@click.option("--max-episode-horizon", type=int, help="Hard cap on steps per episode (overrides manifest horizon).")
@click.option("--device", type=str, help="Device for local robotics validation, e.g. cuda or cpu.")
@click.option("--torch-dtype", type=str, help="Torch dtype for local robotics validation.")
@click.option("--render-video", is_flag=True, help="Render local robotics validation videos.")
@click.option("--video-dir", type=str, help="Directory for local robotics validation videos.")
@click.option("--output-json", type=str, help="Write local validation metrics JSON to this path.")
def main(
    module: str,
    task_ids: str | None,
    flock_api_key: str | None,
    hf_token: str | None,
    time_sleep: int,
    assignment_lookup_interval: int,
    debug: bool,
    local_validation: bool,
    local_hg_repo_id: str | None,
    revision: str,
    validation_manifest_url: str | None,
    validation_data_url: str | None,
    domain_randomization_url: str | None,
    adapter_filename: str,
    max_params: int,
    max_episodes: int | None,
    max_episode_horizon: int | None,
    device: str | None,
    torch_dtype: str | None,
    render_video: bool,
    video_dir: str | None,
    output_json: str | None,
):
    """
    CLI entrypoint for running the validation process.
    Delegates core logic to ValidationRunner.
    """
    if local_validation:
        if module != "robotics_vla":
            raise click.ClickException("--local-validation is currently supported only for robotics_vla")
        if not local_hg_repo_id:
            raise click.ClickException("--local-validation requires --hg-repo-id/--hf-model-repo")
        if not validation_manifest_url and not validation_data_url:
            raise click.ClickException("--local-validation requires --validation-data-url or --validation-manifest-url")

        from validator.modules.robotics_vla.local_validate import run_local_validation

        result = run_local_validation(
            hg_repo_id=local_hg_repo_id,
            revision=revision,
            validation_manifest_url=validation_manifest_url,
            validation_data_url=validation_data_url,
            domain_randomization_url=domain_randomization_url,
            adapter_filename=adapter_filename,
            max_params=max_params,
            max_episodes=max_episodes,
            max_episode_horizon=max_episode_horizon,
            device=device,
            torch_dtype=torch_dtype,
            render_video=render_video,
            video_dir=video_dir,
            output_json=output_json,
            hf_token=hf_token,
        )
        click.echo(json.dumps(result, indent=2))
        return

    if not task_ids:
        raise click.ClickException("--task_ids is required unless --local-validation is used")
    if not flock_api_key:
        raise click.ClickException("--flock-api-key or FLOCK_API_KEY is required unless --local-validation is used")
    if not hf_token:
        raise click.ClickException("--hf-token or HF_TOKEN is required unless --local-validation is used")

    import logging
    _restart_sleep = 60
    while True:
        try:
            runner = ValidationRunner(
                module=module,
                task_ids=task_ids.split(","),
                flock_api_key=flock_api_key,
                hf_token=hf_token,
                time_sleep=time_sleep,
                assignment_lookup_interval=assignment_lookup_interval,
                debug=debug,
            )
            runner.run()
        except KeyboardInterrupt:
            click.echo("\nValidation interrupted by user.")
            sys.exit(0)
        except Exception as e:
            logging.error(
                f"Validator crashed unexpectedly: {e}. "
                f"Restarting in {_restart_sleep}s."
            )
            import time as _time
            _time.sleep(_restart_sleep)

if __name__ == "__main__":
    main()
