import importlib
import time
import sys
from loguru import logger
from .exceptions import RecoverableException
from .api import FedLedger
from .config import load_config_for_task
from .modules.base import BaseValidationModule, BaseConfig, BaseInputData, BaseMetrics

_RETRY_SLEEP = 60   # seconds to back off after unexpected errors in the main loop


class ValidationRunner:
    """
    Runs the validation process for a given module and set of task IDs.
    Handles assignment fetching, validation, error handling, and result submission.
    """
    def __init__(
        self,
        module: str,
        task_ids: list[str],
        flock_api_key: str,
        hf_token: str,
        time_sleep: int = 180,
        assignment_lookup_interval: int = 180,
        debug: bool = False,
    ):
        self.module = module
        self.task_ids = task_ids
        self.flock_api_key = flock_api_key
        self.hf_token = hf_token
        self.time_sleep = time_sleep
        self.assignment_lookup_interval = assignment_lookup_interval
        self.debug = debug
        self.api = FedLedger(flock_api_key)
        self._setup_modules()

    def _setup_modules(self):
        """Dynamically import and initialize validation modules for each task."""
        all_tasks = self.api.list_tasks()
        tasks = [task for task in all_tasks if task["id"] in self.task_ids]
        task_types = {task["id"]: task["task_type"] for task in tasks}
        if not all(task["task_type"] == self.module for task in tasks):
            raise ValueError(f"Module {self.module} is not valid for the given task ids. Check task types: {task_types}")
        module_mod = importlib.import_module(f"validator.modules.{self.module}")
        module_cls: type[BaseValidationModule] = module_mod.MODULE
        self.module_config_to_module: dict[BaseConfig, BaseValidationModule] = {}
        self.task_id_to_module: dict[str, BaseValidationModule] = {}
        for task_id in self.task_ids:
            config = load_config_for_task(task_id, self.module, module_cls.config_schema)
            self.module_config_to_module.setdefault(config, module_cls(config=config))
            self.task_id_to_module[task_id] = self.module_config_to_module[config]

    def perform_validation(self, assignment_id: str, task_id: str, input_data: BaseInputData) -> BaseMetrics | None:
        """
        Perform validation for a given assignment and input data.
        Retries up to 3 times on transient failures.  Returns None and marks the
        assignment failed if all retries are exhausted.
        """
        module_obj = self.task_id_to_module[task_id]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return module_obj.validate(input_data)
            except KeyboardInterrupt:
                raise
            except RecoverableException as e:
                # Infra-side issue (bad config, env problem) — not the miner's fault.
                # Log and return None without marking the assignment as failed; the
                # assignment will time out and be re-queued by FedLedger.
                logger.error(f"Recoverable infra exception (assignment {assignment_id}): {e}")
                return None
            except Exception as e:
                last_error = e
                logger.error(f"Validation attempt {attempt + 1}/3 failed for {assignment_id}: {e}")
        logger.error(f"Marking assignment {assignment_id} as failed after 3 attempts: {last_error}")
        try:
            self.api.mark_assignment_as_failed(assignment_id)
        except Exception as e:
            logger.error(f"Could not mark assignment {assignment_id} as failed: {e}")
        return None

    def _request_assignment(self, task_id: str):
        """Poll for an assignment, blocking until one is available or interrupted."""
        last_successful_request_time = time.time()
        while True:
            try:
                resp = self.api.request_validation_assignment(task_id)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Network error fetching assignment for task {task_id}: {e}")
                logger.info(f"Retrying in {self.time_sleep}s")
                time.sleep(self.time_sleep)
                continue

            if resp.status_code == 200:
                return resp

            try:
                resp_json = resp.json()
            except Exception:
                resp_json = None

            if resp_json == {"detail": "No task submissions available to validate"}:
                logger.info("No task submissions available to validate")
                time.sleep(self.assignment_lookup_interval)
            elif resp_json == {"detail": "Rate limit reached for validation assignment lookup: 1 per 3 minutes"}:
                time_since_last_success = time.time() - last_successful_request_time
                wait = max(0, self.assignment_lookup_interval - time_since_last_success)
                if wait > 0:
                    logger.info(f"Rate limited — sleeping {int(wait)}s")
                    time.sleep(wait)
            else:
                logger.error(f"Unexpected response fetching assignment: {resp.status_code} {resp.content}")
                time.sleep(self.time_sleep)

    def _submit_result(self, assignment_id: str, metrics: BaseMetrics) -> None:
        """Submit validation result; logs on failure but never raises."""
        try:
            resp = self.api.submit_validation_result(
                assignment_id=assignment_id,
                data=metrics.model_dump(),
            )
            if resp.status_code == 200:
                logger.info(f"Validation result submitted successfully for assignment {assignment_id}")
            else:
                logger.error(
                    f"Failed to submit result for {assignment_id}: "
                    f"HTTP {resp.status_code} {resp.content}"
                )
        except Exception as e:
            logger.error(f"Network error submitting result for {assignment_id}: {e}")

    def run(self):
        """
        Run the validation loop for all configured task IDs.
        This method blocks and runs indefinitely.  Unexpected exceptions within a
        single iteration are caught and logged so the loop never exits unintentionally.
        """
        while True:
            for task_id in self.task_ids:
                try:
                    resp = self._request_assignment(task_id)

                    resp_json = resp.json()
                    task_submission_data = resp_json["task_submission"]["data"]
                    validation_assignment_data = resp_json["data"]
                    merged_data = {**task_submission_data, **validation_assignment_data}
                    assignment_id = resp_json["id"]

                    module_obj = self.task_id_to_module[task_id]
                    input_data = module_obj.input_data_schema.model_validate(merged_data)

                    metrics = self.perform_validation(assignment_id, task_id, input_data)
                    if metrics is not None:
                        self._submit_result(assignment_id, metrics)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(
                        f"Unexpected error in validation loop for task {task_id}: {e}. "
                        f"Sleeping {_RETRY_SLEEP}s before retrying."
                    )
                    time.sleep(_RETRY_SLEEP)
