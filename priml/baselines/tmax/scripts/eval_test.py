"""Tests for TMax evaluation commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import json
import subprocess
import time

from treekle import json_loads

import pytest

from priml.baselines.tmax.scripts import (
    eval as eval_module,
    upstream,
)
from priml.baselines.tmax.scripts.eval import (
    AGENT_IMPORT_PATH,
    DATASET,
    DAYTONA_VERSION,
    ENVIRONMENT,
    MAX_FORMAT_ERRORS,
    N_ATTEMPTS,
    N_CONCURRENT,
    TOOL_CALL_PARSER,
    VLLM_VERSION,
    EvalConfig,
    build_harbor_command,
    build_stats_command,
    build_vllm_command,
    wait_for_server,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


def _ignore(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _same_path(root: Path) -> Path:
    return root


def _config(tmp_path: Path) -> EvalConfig:
    checkpoint = tmp_path / "export"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    return EvalConfig(
        checkpoint=checkpoint,
        results_dir=tmp_path / "results",
        job_name="exp000-tb2",
        tmax_root=tmp_path,
        gpus=2,
    )


def test_vllm_command_pins_parser_and_version(tmp_path: Path) -> None:
    """The vLLM command sets the version and qwen3_xml parser."""
    command = build_vllm_command(_config(tmp_path))

    assert f"vllm=={VLLM_VERSION}" in command
    assert "--tool-call-parser" in command
    assert command[command.index("--tool-call-parser") + 1] == TOOL_CALL_PARSER
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-auto-tool-choice" in command


def test_harbor_command_uses_terminal_bench_and_vanillux(tmp_path: Path) -> None:
    """The Harbor command uses Terminal-Bench and the included Vanillux agent."""
    command = build_harbor_command(_config(tmp_path))

    assert command[command.index("--dataset") + 1] == DATASET
    assert command[command.index("--agent-import-path") + 1] == AGENT_IMPORT_PATH
    assert command[command.index("--env") + 1] == ENVIRONMENT
    assert command[:7] == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "--with",
        f"daytona=={DAYTONA_VERSION}",
        "harbor",
    ]
    assert command[command.index("-k") + 1] == str(N_ATTEMPTS)
    assert command[command.index("--n-concurrent") + 1] == str(N_CONCURRENT)
    assert command[command.index("--jobs-dir") + 1] == str(
        _config(tmp_path).results_dir / "harbor",
    )
    assert f"max_format_errors={MAX_FORMAT_ERRORS}" in command
    assert command[command.index("--model") + 1] == "hosted_vllm/export"
    model_info = command[command.index("--agent-kwarg") + 3]
    assert (
        json.loads(model_info.removeprefix("model_info="))["max_output_tokens"] == 8_192
    )


def test_harbor_command_skips_daytona_for_other_environments(tmp_path: Path) -> None:
    """Do not load the Daytona SDK when Harbor uses another environment."""
    command = build_harbor_command(replace(_config(tmp_path), environment="docker"))

    assert command[:5] == ["uv", "run", "--frozen", "--no-sync", "harbor"]
    assert "--with" not in command


def test_stats_command_delegates_to_tmax_script(tmp_path: Path) -> None:
    """The stats command runs TMax's compute_stats.py."""
    command = build_stats_command(_config(tmp_path))

    assert command[:4] == ["uv", "run", "python", "scripts/compute_stats.py"]
    assert command[-1].endswith("metrics.json")


def test_eval_identity_is_pinned_to_terminal_bench_and_tmax(tmp_path: Path) -> None:
    """The defaults match TMax's Qwen3.5 evaluation."""
    config = _config(tmp_path)

    assert config.dataset == "terminal-bench@2.0"
    assert config.agent_import_path == "Vanillux2Agent:Vanillux2Agent"
    assert config.environment == "daytona"
    assert config.n_attempts == 5
    assert config.n_concurrent == 16
    assert config.max_format_errors == 64
    assert config.provenance_path.name == "eval_manifest.json"


def test_run_writes_provenance_and_copies_harbor_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write evaluation metadata and copy the Harbor results."""
    config = _config(tmp_path)
    trial = config.job_dir / "trial-1" / "result.json"

    class _ExitedServer:
        def poll(self) -> int:
            return 0

    def fake_popen(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
        return cast(subprocess.Popen[bytes], _ExitedServer())

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        assert cwd == config.tmax_root
        assert check is False
        assert capture_output is ("-c" in command)
        if "-c" in command:
            assert command[command.index("--with") + 1] == (
                f"daytona=={DAYTONA_VERSION}"
            )
        if "harbor" in command:
            trial.parent.mkdir(parents=True)
            trial.write_text("{}")
            # Harbor copies this URL into each trial. Without it, LiteLLM sends
            # requests to OpenAI instead of the local vLLM server.
            assert env is not None
            assert env["OPENAI_API_KEY"] == "dummy"
            assert env["OPENAI_API_BASE"] == f"http://localhost:{config.port}/v1"
            assert env["OPENAI_BASE_URL"] == f"http://localhost:{config.port}/v1"
        if "scripts/compute_stats.py" in command:
            assert config.metrics_path.parent.is_dir()
            config.metrics_path.write_text('{"pass@1": 1.0}\n')
        return subprocess.CompletedProcess(args=list(command), returncode=0)

    def no_wait(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(eval_module, "wait_for_server", no_wait)
    # Remove any real API key so the test checks the dummy value.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")
    result = eval_module.run(
        config,
        popen=fake_popen,
        run_command=fake_run,
        sleep=lambda _seconds: None,
    )

    assert result == config.metrics_path
    assert config.metrics_path.is_file()
    payload = cast(
        dict[str, object],
        json.loads(config.provenance_path.read_text(encoding="utf-8")),
    )
    assert payload["status"] == "completed"
    assert payload["tmax_commit"] == upstream.UPSTREAM_COMMIT
    assert payload["dataset"] == "terminal-bench@2.0"
    assert payload["agent_import_path"] == "Vanillux2Agent:Vanillux2Agent"
    assert payload["vllm_version"] == VLLM_VERSION
    checkpoint_files = eval_module._checkpoint_files(config.checkpoint)
    assert payload["served_model_name"] == (
        f"export-{eval_module._checkpoint_identity(checkpoint_files)}"
    )
    assert payload["checkpoint_identity"] == eval_module._checkpoint_identity(
        checkpoint_files,
    )
    assert payload["environment"] == "daytona"
    assert payload["n_attempts"] == 5
    assert payload["n_concurrent"] == 16
    assert payload["max_format_errors"] == 64
    assert payload["tensor_parallel_size"] == 2
    assert payload["checkpoint_files"] == {
        "config.json": eval_module._checkpoint_files(config.checkpoint)["config.json"],
    }
    assert (config.results_dir / config.job_name / "trial-1" / "result.json").is_file()
    assert (config.results_dir / config.job_name / "vllm.log").is_file()


def test_the_harbor_run_carries_upstreams_agent_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harbor trials use the local vLLM server.

    Harbor copies ``OPENAI_BASE_URL`` into each trial. The dummy API key
    prevents requests from using a public OpenAI endpoint.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    environment = eval_module._harbor_environment(_config(tmp_path))
    assert environment["OPENAI_API_KEY"] == "dummy"
    assert environment["OPENAI_API_BASE"] == "http://localhost:8008/v1"
    assert environment["OPENAI_BASE_URL"] == "http://localhost:8008/v1"


def test_daytona_preflight_runs_before_vllm_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check Daytona's API key before starting vLLM."""
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DAYTONA_API_KEY"):
        eval_module.run(
            _config(tmp_path),
            popen=lambda *_args, **_kwargs: pytest.fail("must not start vLLM"),
        )


def test_daytona_preflight_requires_the_optional_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the Daytona SDK overlay before starting vLLM."""
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    def missing_extra(
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=1)

    with pytest.raises(RuntimeError, match=f"daytona=={DAYTONA_VERSION}"):
        eval_module.run(
            _config(tmp_path),
            popen=lambda *_args, **_kwargs: pytest.fail("must not start vLLM"),
            run_command=missing_extra,
        )


def test_existing_job_must_match_the_checkpoint_identity(
    tmp_path: Path,
) -> None:
    """A reused job name must use the checkpoint recorded in its manifest."""
    config = _config(tmp_path)
    checkpoint_files = eval_module._checkpoint_files(config.checkpoint)
    identity = eval_module._checkpoint_identity(checkpoint_files)
    bound = replace(
        config,
        served_model_name=f"{config.served_name}-{identity}",
    )
    bound.provenance_path.parent.mkdir(parents=True)
    eval_module._write_provenance(
        bound,
        checkpoint_files=checkpoint_files,
        status="failed",
    )
    config.job_dir.mkdir(parents=True)
    (config.job_dir / "old-trial").mkdir()
    (config.checkpoint / "config.json").write_text('{"changed": true}')

    with pytest.raises(ValueError, match="different inputs"):
        eval_module.run(
            config,
            popen=lambda *_args, **_kwargs: pytest.fail("must not start vLLM"),
        )


def test_eval_config_rejects_invalid_parallelism(tmp_path: Path) -> None:
    """A zero-GPU evaluation cannot serve the model."""
    with pytest.raises(ValueError, match="gpus"):
        _config(tmp_path).__class__(
            checkpoint=tmp_path / "export",
            results_dir=tmp_path / "results",
            job_name="bad",
            tmax_root=tmp_path,
            gpus=0,
        )


def test_eval_config_resolves_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve relative CLI paths against the caller's directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "export").mkdir()
    (tmp_path / "upstream").mkdir()
    config = EvalConfig(
        checkpoint=Path("export"),
        results_dir=Path("results"),
        job_name="job",
        tmax_root=Path("upstream"),
    )
    assert config.checkpoint == (tmp_path / "export").resolve()
    assert config.results_dir == (tmp_path / "results").resolve()
    assert config.tmax_root == (tmp_path / "upstream").resolve()


def test_eval_cli_resolves_paths_under_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve CLI paths beneath the experiment run directory."""
    run_dir = tmp_path / "runs/tmax/exp000"
    checkpoint = run_dir / "export"
    upstream_root = tmp_path / "datasets/tmax/upstream/tmax"
    checkpoint.mkdir(parents=True)
    upstream_root.mkdir(parents=True)
    captured: list[EvalConfig] = []

    def run(config: EvalConfig) -> Path:
        captured.append(config)
        return config.metrics_path

    monkeypatch.setattr(eval_module, "run", run)
    monkeypatch.setattr(upstream, "verify_checkout", Path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval",
            "--base-dir",
            str(tmp_path),
            "/export",
            "--results-dir",
            "/eval",
            "--job-name",
            "exp000-tb2",
        ],
    )

    assert eval_module.main() == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.checkpoint == checkpoint
    assert config.results_dir == run_dir / "eval"
    assert config.tmax_root == upstream_root
    assert config.job_dir == run_dir / "eval/harbor/exp000-tb2"


class _ModelsResponse:
    def __init__(self, model: str) -> None:
        self.model = model
        self.closed = False

    def read(self) -> bytes:
        return json.dumps({"data": [{"id": self.model}]}).encode()

    def close(self) -> None:
        self.closed = True


def test_server_readiness_requires_the_requested_model() -> None:
    """Readiness requires the server to list the requested model."""
    times = iter((0.0, 0.0, 2.0))
    response = _ModelsResponse("other-model")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(time, "monotonic", lambda: next(times))
        with pytest.raises(TimeoutError, match="requested-model"):
            wait_for_server(
                8008,
                "requested-model",
                timeout=1.0,
                sleep=lambda _: None,
                urlopen=lambda *_args, **_kwargs: response,
            )
    assert response.closed


def test_server_readiness_returns_once_the_model_is_listed() -> None:
    """Return as soon as the server lists the requested model."""
    response = _ModelsResponse("requested-model")
    wait_for_server(
        8008,
        "requested-model",
        timeout=1.0,
        sleep=lambda _: pytest.fail("must not wait once ready"),
        urlopen=lambda *_args, **_kwargs: response,
    )
    assert response.closed


def test_server_readiness_retries_after_a_refused_connection() -> None:
    """Keep polling while the port refuses connections."""
    attempts = iter((ConnectionRefusedError(), _ModelsResponse("requested-model")))

    def urlopen(*_args: object, **_kwargs: object) -> _ModelsResponse:
        attempt = next(attempts)
        if isinstance(attempt, OSError):
            raise attempt
        return attempt

    sleeps: list[float] = []
    wait_for_server(
        8008,
        "requested-model",
        timeout=60.0,
        interval=0.5,
        sleep=sleeps.append,
        urlopen=urlopen,
    )
    assert sleeps == [0.5]


def test_server_readiness_rejects_an_exited_process() -> None:
    """Treat an exited server as an error, not a slow start."""

    class _Exited:
        def poll(self) -> int:
            return 1

    server = cast(
        subprocess.Popen[bytes],
        cast(object, _Exited()),
    )
    with pytest.raises(RuntimeError, match="exited"):
        wait_for_server(
            8008,
            "requested-model",
            server=server,
            urlopen=lambda *_args, **_kwargs: pytest.fail("must not probe"),
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"data_parallel_size": 0}, "data_parallel_size and n_concurrent"),
        ({"n_attempts": 0}, "n_attempts must be positive"),
        ({"max_format_errors": 0}, "max_format_errors must be positive"),
        ({"n_tasks": 0}, "n_tasks must be positive"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size must be positive"),
    ],
)
def test_eval_config_rejects_invalid_counts(
    tmp_path: Path,
    overrides: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_config(tmp_path), **overrides)


def test_eval_config_requires_the_checkout_and_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="TMax checkout is missing"):
        replace(config, tmax_root=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="Exported checkpoint is missing"):
        replace(config, checkpoint=tmp_path / "absent")


def test_optional_limits_reach_the_commands(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_model_len=4_096, n_tasks=3)
    vllm = build_vllm_command(config)
    assert vllm[vllm.index("--max-model-len") + 1] == "4096"
    harbor = build_harbor_command(config)
    assert harbor[harbor.index("--n-tasks") + 1] == "3"


class _RunningServer:
    """A vLLM process that is still up, then exits on terminate or kill."""

    def __init__(self, *, ignores_terminate: bool = False) -> None:
        self.events: list[str] = []
        self.ignores_terminate = ignores_terminate

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")

    def wait(self, timeout: float) -> int:
        self.events.append(f"wait{timeout:.0f}")
        if self.ignores_terminate and self.events.count("kill") == 0:
            raise subprocess.TimeoutExpired("vllm", timeout)
        return 0


@pytest.mark.parametrize(
    ("ignores_terminate", "expected"),
    [
        (False, ["terminate", "wait30"]),
        (True, ["terminate", "wait30", "kill", "wait30"]),
    ],
)
def test_stopping_vllm_escalates_to_kill(
    *,
    ignores_terminate: bool,
    expected: list[str],
) -> None:
    server = _RunningServer(ignores_terminate=ignores_terminate)
    eval_module._stop_server(cast(subprocess.Popen[bytes], cast(object, server)))
    assert server.events == expected


def test_a_failed_harbor_run_records_the_failure_and_stops_vllm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(tmp_path), environment="docker")
    server = _RunningServer()

    def fail(
        command: Sequence[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=list(command), returncode=2)

    monkeypatch.setattr(eval_module, "wait_for_server", _ignore)
    with pytest.raises(subprocess.CalledProcessError):
        eval_module.run(
            config,
            popen=lambda *_args, **_kwargs: cast(
                subprocess.Popen[bytes],
                cast(object, server),
            ),
            run_command=fail,
        )
    payload = json_loads(config.provenance_path.read_bytes(), dict[str, object])
    assert payload["status"] == "failed"
    assert payload["error"] == "CalledProcessError"
    assert server.events[0] == "terminate"


def test_stats_must_write_the_metrics_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="without writing"):
        eval_module._require_metrics(tmp_path / "metrics.json")


def test_an_empty_checkpoint_cannot_be_evaluated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no files"):
        eval_module._checkpoint_files(tmp_path)


def test_existing_outputs_without_a_manifest_are_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.job_dir.mkdir(parents=True)
    (config.job_dir / "trial").mkdir()
    with pytest.raises(FileExistsError, match="no PriML manifest"):
        eval_module._require_existing_job_identity(config, {})


def test_an_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.provenance_path.parent.mkdir(parents=True)
    config.provenance_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read evaluation manifest"):
        eval_module._require_existing_job_identity(config, {})


def test_a_non_daytona_environment_needs_no_preflight(tmp_path: Path) -> None:
    eval_module._preflight_environment(
        replace(_config(tmp_path), environment="docker"),
        run_command=lambda *_args, **_kwargs: pytest.fail("must not probe"),
    )


class _GarbageResponse:
    def __init__(self) -> None:
        self.closed = False

    def read(self) -> bytes:
        return b"not json"

    def close(self) -> None:
        self.closed = True


def test_server_readiness_keeps_polling_past_an_unparseable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((0.0, 0.0, 2.0))
    response = _GarbageResponse()
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        wait_for_server(
            8008,
            "m",
            timeout=1.0,
            sleep=lambda _: None,
            urlopen=lambda *_args, **_kwargs: response,
        )
    assert response.closed


def test_main_runs_one_job_from_the_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs/tmax/exp000"
    (run_dir / "export").mkdir(parents=True)
    (run_dir / "export" / "config.json").write_text("{}", encoding="utf-8")
    checkout = tmp_path / "datasets/tmax/upstream/tmax"
    checkout.mkdir(parents=True)
    captured: list[EvalConfig] = []

    def run(config: EvalConfig) -> Path:
        captured.append(config)
        return config.metrics_path

    monkeypatch.setattr(eval_module, "run", run)
    monkeypatch.setattr(upstream, "verify_checkout", _same_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval",
            "/export",
            "--base-dir",
            str(tmp_path),
            "--results-dir",
            "/eval",
            "--job-name",
            "job",
            "--gpus",
            "1",
        ],
    )
    assert eval_module.main() == 0
    assert captured[0].checkpoint == run_dir / "export"
    assert captured[0].results_dir == run_dir / "eval"
    assert capsys.readouterr().out.strip().endswith("metrics.json")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
