from __future__ import annotations


class RoboticsSubmissionError(Exception):
    """Raised when a miner's submission is invalid or unrunnable.

    These are the *submitter's* fault (a broken adapter, an unloadable model, a
    policy that emits malformed actions, or a model that violates the parameter
    cap). They must be scored as an invalid submission (score 0) and must NOT
    crash the long-running validator. Genuine infrastructure problems (a broken
    validation package, a simulator crash, network/disk failures) deliberately do
    NOT use this type, so they can propagate to the runner and be retried or have
    the assignment re-queued instead of unfairly zeroing the miner.

    ``failure_mode`` is a short, stable, machine-readable tag surfaced in the
    submitted metrics' diagnostics so failures can be triaged.
    """

    def __init__(self, message: str, failure_mode: str = "submission_error"):
        super().__init__(message)
        self.failure_mode = failure_mode
