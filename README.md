# LLM Validator System

## Overview

The Flock LLM Validator script is a modular system for doing validation for Flock AI Arena validation assignments. It is designed to manage an isolated conda environment for each task type, fetch validation assignments from the Flock API, execute the validation, and submit results back to the API. The system is extensible, allowing new validation modules or task types to be added easily.

The project is split into two layers:
1. The outer layer that runs outside conda and is responsible for managing and running conda environments for each module.
2. The inner layer that runs inside conda and is responsible for all validation and assignment orchestration logic.

## Setup

1. Install miniconda (recommended, e.g. [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)). If `conda` is not available, the runner falls back to a local Python 3.10/3.11 venv.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Validation Jobs

Use the CLI to start the validation process. The CLI settings are defined in `environment_entrypoint.py`.

```bash
python run.py \
  <module> \
  --task_ids <task_id1,task_id2,...> \
  --flock-api-key <your_flock_api_key> \
  --hf-token <your_hf_token>
```

#### Arguments and Options
- `<module>`: Name of the validation module to use (e.g., `lora`).
- `--task_ids`: Comma-separated list of task IDs to validate.
- `--flock-api-key`: Flock API key (can also be set via the `FLOCK_API_KEY` environment variable).
- `--hf-token`: HuggingFace token (can also be set via the `HF_TOKEN` environment variable).
- `--time-sleep`: (Optional) Time to sleep between retries in seconds (default: 180).
- `--assignment-lookup-interval`: (Optional) Assignment lookup interval in seconds (default: 180).
- `--debug`: (Optional) Enable debug mode.

#### Example

```bash
python run.py lora --task_ids 123,456 --flock-api-key $FLOCK_API_KEY --hf-token $HF_TOKEN
```

#### Robotics VLA

```bash
python run.py robotics_vla --task_ids "$ROBOTICS_TASK_ID" --flock-api-key "$FLOCK_API_KEY" --hf-token "$HF_TOKEN"
```

Dependencies (MuJoCo, Robosuite, Torch, Hugging Face) are installed automatically on first run. For full setup instructions, scoring details, submission contract, and a troubleshooting FAQ see [`validator/modules/robotics_vla/README.md`](validator/modules/robotics_vla/README.md).

### Environment Variables
You can set the following environment variables instead of passing them as CLI options:
- `FLOCK_API_KEY`
- `HF_TOKEN`
- `TIME_SLEEP`
- `ASSIGNMENT_LOOKUP_INTERVAL`

### Adding New Validation Modules
1. Create a new subdirectory or file in `validator/modules/`.
2. Implement a class that inherits from `BaseValidationModule` and define the required schemas.
3. Expose your module class as `MODULE` in the module's `__init__.py`.
4. The system will dynamically load your module when specified via the CLI.

### Configuration
- Module and task-specific configuration files should be placed in the `configs/` directory.
- Per-task-type config: `configs/<module>.json`
- Per-task-id config (which overrides the per-task-type config): `configs/tasks/<task_id>.json`

## Error Handling
- Recoverable errors (subclass `RecoverableException`) will not mark the assignment as failed and will log the error.
- Other exceptions will mark the assignment as failed.
