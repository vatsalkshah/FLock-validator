import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_ROOT = REPO_ROOT / ".venv"


def run_command(cmd, **kwargs):
    # Ensure output is unbuffered
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs
    )
    for line in process.stdout:
        print(line, end='')  # Already includes newline
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def conda_available() -> bool:
    """Return True when the outer runner can use conda."""
    return shutil.which("conda") is not None

def env_exists(env_name: str) -> bool:
    """Check if a conda environment exists."""
    logger.info(f"Checking if environment {env_name} exists")
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True, text=True, check=True
    )
    return any(line.split()[0] == env_name for line in result.stdout.splitlines() if line and not line.startswith("#"))

def create_env(env_name: str, env_yml: Path, requirements_txt: Path):
    """Create a conda environment from environment.yml or requirements.txt."""
    logger.info(f"Creating environment {env_name} from {env_yml} and {requirements_txt}")
    run_command(["conda", "env", "create", "-n", env_name, "-f", str(env_yml)])
    # Install base requirements
    run_command([
        "conda", "run", "-n", env_name, "pip", "install", "-r", str(requirements_txt)
    ])
    
def update_env(env_name: str, env_yml: Path, requirements_txt: Path):
    """Update a conda environment from environment.yml or requirements.txt."""
    logger.info(f"Updating environment {env_name} from {env_yml} and {requirements_txt}")
    run_command(["conda", "env", "update", "-n", env_name, "-f", str(env_yml)])
    run_command(["conda", "run", "-n", env_name, "pip", "install", "-r", str(requirements_txt)])
    

def install_in_env(env_name: str, packages: List[str]):
    """Install additional packages into an existing conda environment."""
    run_command(["conda", "install", "-y", "-n", env_name] + packages)

def run_in_env(env_name: str, command: List[str], env_vars: Optional[dict] = None):
    """Run a command inside a conda environment."""
    logger.info(f"Running command {command} in environment {env_name}")
    cmd = ["conda", "run", "-n", env_name] + command
    run_command(cmd, env=env_vars)

def ensure_env_and_run(
    env_name: str,
    env_yml: Path,
    requirements_txt: Path,
    command: List[str],
    env_vars: Optional[dict] = None
):
    """Ensure the environment exists, create if needed, then run the command."""
    if conda_available():
        if not env_exists(env_name):
            create_env(env_name, env_yml, requirements_txt)
        else:
            update_env(env_name, env_yml, requirements_txt)
        run_in_env(env_name, command, env_vars)
        return

    logger.warning(f"conda was not found; falling back to a local venv for {env_name}")
    ensure_venv_and_run(env_name, env_yml, requirements_txt, command, env_vars)


def ensure_venv_and_run(
    env_name: str,
    env_yml: Path,
    requirements_txt: Path,
    command: List[str],
    env_vars: Optional[dict] = None,
):
    """Create or update a local venv from the module environment.yml pip dependencies."""
    if not (3, 10) <= sys.version_info[:2] < (3, 12):
        raise RuntimeError(
            "conda is not installed and the current Python is not 3.10 or 3.11. "
            "Install Miniconda or run the validator with Python 3.10/3.11."
        )

    venv_dir = VENV_ROOT / env_name
    python = _venv_python(venv_dir)
    if not python.exists():
        logger.info("Creating local venv %s", venv_dir)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
    else:
        logger.info("Updating local venv %s", venv_dir)

    run_command([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python), "-m", "pip", "install", "-r", str(requirements_txt)])
    env_packages = pip_packages_from_environment_yml(env_yml)
    if env_packages:
        run_command([str(python), "-m", "pip", "install", *env_packages])

    normalized_command = _strip_conda_run_options(command)
    if normalized_command and normalized_command[0] == "python":
        normalized_command = [str(python), *normalized_command[1:]]
    run_command(normalized_command, env=env_vars)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _strip_conda_run_options(command: List[str]) -> List[str]:
    return [part for part in command if part != "--no-capture-output"]


def pip_packages_from_environment_yml(env_yml: Path) -> List[str]:
    """Extract a small conda environment.yml subset that can be installed with pip."""
    packages: List[str] = []
    in_dependencies = False
    in_pip = False
    for raw_line in env_yml.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if stripped == "- pip:":
            in_pip = True
            continue
        if in_pip:
            if raw_line.startswith("    - "):
                packages.append(stripped[2:].strip())
                continue
            in_pip = False
        if stripped.startswith("- "):
            package = stripped[2:].strip()
            if package == "pip" or package.startswith("python"):
                continue
            packages.append(package)
    return packages
