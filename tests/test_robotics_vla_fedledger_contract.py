"""Logic check for the FedLedger production validation path.

Simulates the assignment payload the validator receives from FedLedger and
exercises the exact data flow `ValidationRunner.run()` performs:

    task_submission["data"]  +  assignment["data"]
        -> merged dict
        -> RoboticsVLAInputData.model_validate(...)
        -> module.validate(input_data) -> RoboticsVLAMetrics
        -> api.submit_validation_result(id, metrics.model_dump())

It does not hit the network; it validates that a representative payload parses,
that the metrics serialize for submission, and that malformed payloads are
rejected at parse time.
"""

import json

import pytest
from pydantic import ValidationError

from validator.modules.robotics_vla import RoboticsVLAInputData, RoboticsVLAMetrics
from validator.modules.robotics_vla.data_package import _first_present


def simulate_fedledger_assignment():
    """Shape mirrors resp.json() in ValidationRunner.run()."""
    return {
        "id": "assignment-abc123",
        "task_submission": {
            "data": {
                # The miner's submission fields.
                "hg_repo_id": "random-sequence/flock-robotics-qwen25vl-vla-4b",
                "revision": "main",
                # Real payloads carry extra fields that must be ignored.
                "submitter": "0xminer",
                "submission_round": 7,
            }
        },
        "data": {
            # The validation-assignment fields (see README contract).
            "validation_data_url": "https://fed-ledger.example/private_eval_validation_package.zip",
            "max_params": 4_500_000_000,
        },
    }


def merge_like_runner(resp_json):
    """Exactly what ValidationRunner.run() does to build input_data."""
    task_submission_data = resp_json["task_submission"]["data"]
    validation_assignment_data = resp_json["data"]
    return {**task_submission_data, **validation_assignment_data}


def test_fedledger_payload_parses_into_input_schema():
    resp = simulate_fedledger_assignment()
    merged = merge_like_runner(resp)

    data = RoboticsVLAInputData.model_validate(merged)

    assert data.hg_repo_id == "random-sequence/flock-robotics-qwen25vl-vla-4b"
    assert data.revision == "main"
    assert data.max_params == 4_500_000_000
    assert data.validation_data_url.endswith(".zip")
    assert data.adapter_filename == "flock_robotics_adapter.py"
    # The package resolver picks the right url from the alias group.
    assert _first_present(
        data, "validation_data_url", "validation_zip_url", "validation_set_url"
    ) == data.validation_data_url


@pytest.mark.parametrize("url_field", ["validation_data_url", "validation_zip_url", "validation_set_url"])
def test_fedledger_accepts_each_validation_url_alias(url_field):
    merged = {"hg_repo_id": "org/model", "max_params": 4_500_000_000, url_field: "https://x/p.zip"}

    data = RoboticsVLAInputData.model_validate(merged)

    assert _first_present(
        data, "validation_data_url", "validation_zip_url", "validation_set_url"
    ) == "https://x/p.zip"


def test_fedledger_rejects_payload_without_required_fields():
    # Missing hg_repo_id (and max_params) must fail fast at parse time.
    with pytest.raises(ValidationError):
        RoboticsVLAInputData.model_validate({"validation_data_url": "https://x/p.zip"})


def test_metrics_serialize_for_submission():
    # What perform_validation returns and what api.submit_validation_result sends.
    metrics = RoboticsVLAMetrics(
        score=0.0025,
        loss=0.9975,
        mean_episode_score=0.0025,
        weighted_episode_score=0.0025,
        success_rate=0.0,
        mean_progress_score=0.006,
        mean_return=0.02,
        mean_episode_length=150.0,
        episodes_completed=3,
        invalid_submission=False,
        parameter_count=4_056_760_327,
    )

    dump = metrics.model_dump()
    submission = {"status": "completed", "data": dump}

    # Must be JSON-serializable for the POST body.
    json.dumps(submission)
    for key in ("score", "loss", "success_rate", "invalid_submission", "parameter_count"):
        assert key in dump
    assert submission["data"]["parameter_count"] == 4_056_760_327


def test_invalid_submission_still_serializes_for_submission():
    # A rejected submission (e.g. over the param cap) is also submitted, scored 0.
    from validator.modules.robotics_vla import RoboticsVLAValidationModule

    metrics = RoboticsVLAValidationModule._invalid_metrics(
        RoboticsVLAValidationModule.__new__(RoboticsVLAValidationModule),
        "Model parameters 5000000000 exceed limit 4500000000",
        parameter_count=5_000_000_000,
        failure_mode="parameter_limit_exceeded",
    )
    dump = metrics.model_dump()
    json.dumps({"status": "completed", "data": dump})
    assert dump["invalid_submission"] is True
    assert dump["score"] == 0.0
    assert dump["diagnostics"]["failure_mode"] == "parameter_limit_exceeded"
