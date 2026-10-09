"""Run one Terminal-Bench evaluation for an exported TMax checkpoint.

vLLM serves the model. Harbor runs Terminal-Bench with TMax's Vanillux agent
and task verifiers. TMax's ``compute_stats.py`` aggregates the rewards and
computes pass@k.

This module starts and stops vLLM, runs Harbor, copies its results, and records
the checkpoint and evaluation settings. Its defaults match the main settings
from TMax's manual Qwen3.5 command. It runs one job, not the complete multi-run
evaluation reported in the paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from priml.baselines.tmax.scripts import upstream
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable


class _Flags(Protocol):
    """Parsed command-line flags."""

    base_dir: Path
    working_dir: Path
    checkpoint: Path
    results_dir: Path
    job_name: str
    dataset: str
    gpus: int
    n_concurrent: int
    n_attempts: int
    n_tasks: int | None
    environment: str
    tool_call_parser: str
    max_format_errors: int
    port: int
    tensor_parallel_size: int | None
    data_parallel_size: int
    max_model_len: int | None
    upstream_root: Path


VLLM_VERSION = "0.19.1"
"""vLLM version used by TMax's evaluation command."""
DAYTONA_VERSION = "0.165.0"
"""Minimum Daytona SDK release required by TMax's locked Harbor 0.6.6."""
TOOL_CALL_PARSER = "qwen3_xml"
"""Tool-call parser used for Qwen3.5."""
DATASET = "terminal-bench@2.0"
"""Dataset used by TMax's manual evaluation command."""
AGENT_IMPORT_PATH = "Vanillux2Agent:Vanillux2Agent"
ENVIRONMENT = "daytona"
N_CONCURRENT = 16
N_ATTEMPTS = 5
MAX_FORMAT_ERRORS = 64


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Settings for one Terminal-Bench job."""

    checkpoint: Path
    results_dir: Path
    job_name: str
    served_model_name: str | None = None
    dataset: str = DATASET
    agent_import_path: str = AGENT_IMPORT_PATH
    environment: str = ENVIRONMENT
    vllm_version: str = VLLM_VERSION
    tool_call_parser: str = TOOL_CALL_PARSER
    port: int = 8008
    gpus: int = 8
    tensor_parallel_size: int | None = None
    data_parallel_size: int = 1
    n_concurrent: int = N_CONCURRENT
    n_attempts: int = N_ATTEMPTS
    max_format_errors: int = MAX_FORMAT_ERRORS
    n_tasks: int | None = None
    max_model_len: int | None = None
    tmax_root: Path = upstream.DEFAULT_CHECKOUT
    revision: str = "main"

    def __post_init__(self) -> None:
        """Reject command settings that vLLM or Harbor cannot use."""
        object.__setattr__(self, "checkpoint", self.checkpoint.resolve())
        object.__setattr__(self, "results_dir", self.results_dir.resolve())
        object.__setattr__(self, "tmax_root", self.tmax_root.resolve())
        if self.port <= 0 or self.gpus <= 0:
            raise ValueError("port and gpus must be positive.")
        if self.data_parallel_size <= 0 or self.n_concurrent <= 0:
            raise ValueError("data_parallel_size and n_concurrent must be positive.")
        if self.n_attempts <= 0:
            raise ValueError("n_attempts must be positive.")
        if self.max_format_errors <= 0:
            raise ValueError("max_format_errors must be positive.")
        if self.n_tasks is not None and self.n_tasks <= 0:
            raise ValueError("n_tasks must be positive when set.")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive when set.")
        if not self.tmax_root.is_dir():
            raise FileNotFoundError(f"TMax checkout is missing: {self.tmax_root}")
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"Exported checkpoint is missing: {self.checkpoint}",
            )

    @property
    def served_name(self) -> str:
        """Return the name passed to vLLM and Harbor."""
        return self.served_model_name or self.checkpoint.name

    @property
    def harbor_model_name(self) -> str:
        """Return Harbor's hosted-vLLM model identifier."""
        return f"hosted_vllm/{self.served_name}"

    @property
    def tensor_parallel(self) -> int:
        """Use all GPUs for tensor parallelism unless overridden."""
        return self.tensor_parallel_size or self.gpus

    @property
    def job_dir(self) -> Path:
        """Return the directory containing Harbor's trial results."""
        return self.results_dir / "harbor" / self.job_name

    @property
    def metrics_path(self) -> Path:
        """Return the path for the metrics file."""
        return self.results_dir / self.job_name / "metrics.json"

    @property
    def provenance_path(self) -> Path:
        """Return the path for the evaluation manifest."""
        return self.results_dir / self.job_name / "eval_manifest.json"


def build_vllm_command(config: EvalConfig) -> list[str]:
    """Build the vLLM command."""
    command = [
        "uvx",
        "--with",
        "fastapi<0.137",
        f"vllm=={config.vllm_version}",
        "serve",
        str(config.checkpoint),
        "--revision",
        config.revision,
        "--tokenizer-revision",
        config.revision,
        "--served-model-name",
        config.served_name,
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--tool-call-parser",
        config.tool_call_parser,
        "--port",
        str(config.port),
        "--gpu-memory-utilization",
        "0.85",
        "--tensor-parallel-size",
        str(config.tensor_parallel),
        "--data-parallel-size",
        str(config.data_parallel_size),
    ]
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    return command


def build_harbor_command(config: EvalConfig) -> list[str]:
    """Build the Harbor command."""
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
    ]
    if config.environment.casefold() == "daytona":
        command.extend(["--with", f"daytona=={DAYTONA_VERSION}"])
    command.extend(
        [
            "harbor",
            "run",
            "--dataset",
            config.dataset,
            "--model",
            config.harbor_model_name,
            "--env",
            config.environment,
            "--n-concurrent",
            str(config.n_concurrent),
            "--agent-kwarg",
            f"api_base=http://localhost:{config.port}/v1",
            "--agent-kwarg",
            f"model_info={_model_info(config)}",
            "--agent-kwarg",
            f"max_format_errors={config.max_format_errors}",
            "--job-name",
            config.job_name,
            "--jobs-dir",
            str(config.results_dir / "harbor"),
            "-k",
            str(config.n_attempts),
            "--agent-import-path",
            config.agent_import_path,
        ],
    )
    if config.n_tasks is not None:
        command.extend(["--n-tasks", str(config.n_tasks)])
    return command


def build_stats_command(config: EvalConfig) -> list[str]:
    """Build the TMax metrics command."""
    return [
        "uv",
        "run",
        "python",
        "scripts/compute_stats.py",
        str(config.job_dir),
        "--json-output",
        str(config.metrics_path),
    ]


def _harbor_environment(config: EvalConfig) -> dict[str, str]:
    """Set the local vLLM environment expected by Harbor and its agents.

    Vanillux also receives the local API address directly. These variables
    support Harbor adapters that read the address from the environment.
    """
    environment = dict(os.environ)
    environment.setdefault("OPENAI_API_KEY", "dummy")
    environment["OPENAI_API_BASE"] = f"http://localhost:{config.port}/v1"
    environment["OPENAI_BASE_URL"] = f"http://localhost:{config.port}/v1"
    return environment


def run(
    config: EvalConfig,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Serve the model, run the evaluation, compute metrics, and stop vLLM."""
    checkpoint_files = _checkpoint_files(config.checkpoint)
    checkpoint_identity = _checkpoint_identity(checkpoint_files)
    config = replace(
        config,
        served_model_name=(f"{config.served_name}-{checkpoint_identity}"),
    )
    _require_existing_job_identity(config, checkpoint_files)
    _preflight_environment(config, run_command=run_command)
    output_dir = config.results_dir / config.job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_provenance(config, checkpoint_files=checkpoint_files, status="started")
    log_path = output_dir / "vllm.log"
    with log_path.open("ab") as server_log:
        server = popen(
            build_vllm_command(config),
            cwd=config.tmax_root,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_server(
                config.port,
                config.served_name,
                server=server,
                sleep=sleep,
            )
            harbor = run_command(
                build_harbor_command(config),
                cwd=config.tmax_root,
                check=False,
                env=_harbor_environment(config),
            )
            _require_success(harbor)
            # Copy Harbor's trials before computing metrics from them.
            shutil.copytree(
                config.job_dir,
                output_dir,
                dirs_exist_ok=True,
            )
            config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            stats = run_command(
                build_stats_command(config),
                cwd=config.tmax_root,
                check=False,
            )
            _require_success(stats)
            _require_metrics(config.metrics_path)
            _write_provenance(
                config,
                checkpoint_files=checkpoint_files,
                status="completed",
            )
            return config.metrics_path
        except BaseException as error:
            _write_provenance(
                config,
                checkpoint_files=checkpoint_files,
                status="failed",
                error=type(error).__name__,
            )
            raise
        finally:
            _stop_server(server)


def _require_success(completed: subprocess.CompletedProcess[bytes]) -> None:
    """Raise the standard subprocess error for a failed upstream command."""
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cast(Sequence[str], completed.args),
        )


def _require_metrics(path: Path) -> None:
    """Require compute_stats.py to write the metrics file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"TMax compute_stats completed without writing {path} via --json-output.",
        )


def _checkpoint_identity(checkpoint_files: Mapping[str, str]) -> str:
    """Return a digest of the checkpoint file hashes."""
    canonical = json.dumps(
        checkpoint_files,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_files(checkpoint: Path) -> dict[str, str]:
    """Hash every exported checkpoint file used by the evaluation."""
    files = [file for file in sorted(checkpoint.rglob("*")) if file.is_file()]
    if not files:
        raise ValueError(f"Exported checkpoint contains no files: {checkpoint}")
    result: dict[str, str] = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[file.relative_to(checkpoint).as_posix()] = digest.hexdigest()
    return result


def _evaluation_identity(
    config: EvalConfig,
    checkpoint_files: Mapping[str, str],
) -> dict[str, object]:
    """Return the inputs used to prevent mismatched job reuse."""
    return {
        "schema": 1,
        "tmax_commit": upstream.UPSTREAM_COMMIT,
        "dataset": config.dataset,
        "agent_import_path": config.agent_import_path,
        "environment": config.environment,
        "n_attempts": config.n_attempts,
        "n_concurrent": config.n_concurrent,
        "n_tasks": config.n_tasks,
        "max_format_errors": config.max_format_errors,
        "vllm_version": config.vllm_version,
        "tool_call_parser": config.tool_call_parser,
        "port": config.port,
        "gpus": config.gpus,
        "tensor_parallel_size": config.tensor_parallel,
        "data_parallel_size": config.data_parallel_size,
        "max_model_len": config.max_model_len,
        "checkpoint": str(config.checkpoint),
        "checkpoint_files": dict(checkpoint_files),
        "checkpoint_identity": _checkpoint_identity(checkpoint_files),
        "served_model_name": config.served_name,
        "revision": config.revision,
    }


def _require_existing_job_identity(
    config: EvalConfig,
    checkpoint_files: Mapping[str, str],
) -> None:
    """Resume only when the existing manifest matches."""
    output_dir = config.results_dir / config.job_name
    existing_outputs = (config.job_dir.is_dir() and any(config.job_dir.iterdir())) or (
        output_dir.is_dir() and any(output_dir.iterdir())
    )
    if not existing_outputs:
        return
    if not config.provenance_path.is_file():
        raise FileExistsError(
            f"Evaluation job {config.job_name!r} already has outputs but no "
            "PriML manifest; choose a fresh --job-name.",
        )
    try:
        payload = cast(
            object,
            json.loads(config.provenance_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read evaluation manifest {config.provenance_path}.",
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Evaluation manifest {config.provenance_path} must be an object.",
        )
    expected = _evaluation_identity(config, checkpoint_files)
    changed = sorted(
        key for key, value in expected.items() if payload.get(key) != value
    )
    if changed:
        raise ValueError(
            f"Evaluation job {config.job_name!r} belongs to different inputs "
            f"({changed}); choose a fresh --job-name.",
        )


def _preflight_environment(
    config: EvalConfig,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    """Check Daytona requirements before loading the model."""
    if config.environment.casefold() != "daytona":
        return
    if not os.environ.get("DAYTONA_API_KEY", "").strip():
        raise RuntimeError(
            "Daytona evaluation requires DAYTONA_API_KEY before vLLM starts.",
        )
    completed = run_command(
        [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--with",
            f"daytona=={DAYTONA_VERSION}",
            "python",
            "-c",
            "import daytona",
        ],
        cwd=config.tmax_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "Daytona evaluation requires the pinned TMax environment and "
            f"Daytona SDK {DAYTONA_VERSION}; run `uv sync --frozen` from "
            f"{config.tmax_root}, then check that `uv run --frozen --no-sync "
            f'--with daytona=={DAYTONA_VERSION} python -c "import daytona"` '
            "succeeds there.",
        )


def _write_provenance(
    config: EvalConfig,
    *,
    checkpoint_files: dict[str, str],
    status: str,
    error: str | None = None,
) -> None:
    """Write the inputs and status used to guard job reuse."""
    payload = _evaluation_identity(config, checkpoint_files)
    payload.update(
        {
            "status": status,
        },
    )
    if error is not None:
        payload["error"] = error
    target = config.provenance_path
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def wait_for_server(
    port: int,
    served_name: str,
    *,
    server: subprocess.Popen[bytes] | None = None,
    timeout: float = 1_800.0,
    interval: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> None:
    """Wait until vLLM lists the requested model."""
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{port}/v1/models"
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before serving {served_name!r} on port {port}.",
            )
        response: object | None = None
        ready = False
        try:
            response = urlopen(url, timeout=min(interval, 10.0))
            read = getattr(response, "read", None)
            if callable(read):
                payload = _decode_json_response(read())
                if isinstance(payload, Mapping):
                    models = payload.get("data")
                    if isinstance(models, list):
                        ready = any(
                            isinstance(model, Mapping)
                            and model.get("id") == served_name
                            for model in cast("list[object]", models)
                        )
        except (
            OSError,
            TypeError,
            ValueError,
            urllib.error.URLError,
        ):
            pass
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if ready:
            return
        sleep(interval)
    raise TimeoutError(
        f"vLLM did not serve {served_name!r} at {url} within {timeout:.0f}s.",
    )


def _decode_json_response(encoded: object) -> object:
    """Decode a JSON response."""
    if not isinstance(encoded, (str, bytes, bytearray)):
        raise TypeError("The vLLM models response must be JSON bytes.")
    return cast(object, json.loads(encoded))


def _model_info(config: EvalConfig) -> str:
    """Build the model details passed to Harbor."""
    return json.dumps(
        {
            "max_input_tokens": config.max_model_len or 40_960,
            "max_output_tokens": 8_192,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        separators=(",", ":"),
    )


def _stop_server(server: subprocess.Popen[bytes]) -> None:
    """Stop vLLM, killing it if graceful shutdown times out."""
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=30)


def main() -> int:
    """Parse evaluation settings and run one job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/scratch"),
        help="Root for input assets and generated run files.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/runs/tmax/exp000"),
        help="Logical run directory below base-dir.",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--n-concurrent", type=int, default=N_CONCURRENT)
    parser.add_argument("--n-attempts", type=int, default=N_ATTEMPTS)
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--environment", default=ENVIRONMENT)
    parser.add_argument("--tool-call-parser", default=TOOL_CALL_PARSER)
    parser.add_argument("--max-format-errors", type=int, default=MAX_FORMAT_ERRORS)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/datasets/tmax/upstream/tmax"),
    )
    flags = cast(_Flags, parser.parse_args())
    working_dir = resolve_working_dir(flags.base_dir, flags.working_dir)
    checkpoint = resolve_working_dir(working_dir, flags.checkpoint)
    results_dir = resolve_working_dir(working_dir, flags.results_dir)
    upstream_root = resolve_working_dir(flags.base_dir, flags.upstream_root)
    # Verify at launch so config construction stays side-effect-free.
    tmax_root = upstream.verify_checkout(upstream_root)
    path = run(
        EvalConfig(
            checkpoint=checkpoint,
            results_dir=results_dir,
            job_name=flags.job_name,
            dataset=flags.dataset,
            gpus=flags.gpus,
            n_concurrent=flags.n_concurrent,
            n_attempts=flags.n_attempts,
            n_tasks=flags.n_tasks,
            environment=flags.environment,
            tool_call_parser=flags.tool_call_parser,
            max_format_errors=flags.max_format_errors,
            port=flags.port,
            tensor_parallel_size=flags.tensor_parallel_size,
            data_parallel_size=flags.data_parallel_size,
            max_model_len=flags.max_model_len,
            tmax_root=tmax_root,
        ),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
