"""Tests for the grading script's statistics and job driving."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import json
import subprocess

import pytest

from priml.baselines.openaimath.scripts import eval as eval_module
from priml.baselines.openaimath.scripts.eval import (
    ORACLE_AGENT_ARGS,
    TaskResult,
    _config,
    _report_payload,
    _task_names,
    _write_report,
    evaluate,
    job_directory,
    main,
    pass_at_k,
    summarize,
)
from priml.baselines.openaimath.scripts.runner import TrialOutcome


if TYPE_CHECKING:
    from priml.baselines.openaimath.scripts.eval import _Flags
    from priml.baselines.openaimath.scripts.runner import OpenMathRunnerConfig


TASK_NAMES = ("alpha", "beta", "gamma")
TASKS_LOGICAL_DIR = Path("/tasks")


def _scratch(root: Path, names: tuple[str, ...] = TASK_NAMES) -> Path:
    """Write a staged tasks tree the way prepare_data leaves one.

    The paths are logical, as they are in an experiment config: everything is
    resolved beneath ``base_dir``, so a test's scratch directory plays the part
    of ``/opt/scratch``.
    """
    scratch = root / "scratch"
    for name in names:
        directory = scratch / "tasks" / name
        directory.mkdir(parents=True)
        (directory / "instruction.md").write_text(f"# {name}\n", encoding="utf-8")
    return scratch


def _flags(scratch: Path, **overrides: object) -> _Flags:
    """Build the parsed-command-line shape the module reads."""
    values: dict[str, object] = {
        "base_dir": scratch,
        "tasks_dir": TASKS_LOGICAL_DIR,
        "snapshot": "openai-math-lean-test",
        "working_dir": Path("/runs/openaimath/eval"),
        "output_dir": Path(),
        "report": None,
        "base_url": None,
        "tokenizer_path": Path("/models/Qwen3.5-4B"),
        "served_model": None,
        "temperature": None,
        "top_p": None,
        "attempts": 1,
        "limit": 0,
        "task_names": None,
        "task_list": None,
        "seed": 42,
        "n_concurrent": None,
        "oracle": False,
    }
    values.update(overrides)
    return cast("_Flags", type("_Parsed", (), values)())


def _config_for(scratch: Path, **overrides: object) -> OpenMathRunnerConfig:
    """Finalize a config over a scratch directory."""
    return _config(_flags(scratch, **overrides))


@pytest.mark.parametrize(
    ("passes", "attempts", "k", "expected"),
    [
        (0, 8, 1, 0.0),
        (0, 8, 8, 0.0),
        (8, 8, 1, 1.0),
        (1, 2, 2, 1.0),
        (1, 4, 2, 0.5),
        (2, 4, 2, 1 - 1 / 6),
        (3, 4, 4, 1.0),
    ],
)
def test_pass_at_k_matches_the_unbiased_estimator(
    passes: int,
    attempts: int,
    k: int,
    expected: float,
) -> None:
    """pass@k is the chance that k fresh attempts hold no accepted one."""
    assert pass_at_k(passes, attempts, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("passes", "attempts", "k"),
    [(9, 8, 1), (0, 8, 0), (0, 8, 9)],
)
def test_pass_at_k_rejects_impossible_counts(
    passes: int,
    attempts: int,
    k: int,
) -> None:
    """Inconsistent counts are refused rather than estimated."""
    with pytest.raises(ValueError, match="not between"):
        pass_at_k(passes, attempts, k)


def test_task_result_separates_failures_from_missing_verdicts() -> None:
    """A trial that was never graded is not the same as a rejected proof."""
    result = TaskResult("alpha", (1.0, 0.0, None))
    assert result.attempts == 3
    assert result.passes == 1
    assert result.missing == 1
    assert result.to_payload() == {
        "task": "alpha",
        "rewards": [1.0, 0.0, None],
        "passes": 1,
        "missing": 1,
    }


def test_summarize_reports_both_estimates_and_the_missing_trials() -> None:
    """The summary carries pass@1, pass@k, and what never reached the grader."""
    summary = summarize(
        [
            TaskResult("alpha", (1.0, 0.0)),
            TaskResult("beta", (0.0, None)),
        ],
        k=2,
    )
    assert summary["tasks"] == 2
    assert summary["attempts_per_task"] == 2
    assert summary["pass@1"] == pytest.approx(0.25)
    assert summary["pass@k"] == pytest.approx(0.5)
    assert summary["mean_reward"] == pytest.approx(1 / 3)
    assert summary["tasks_with_a_pass"] == 1
    assert summary["missing_trials"] == 1


@pytest.mark.parametrize(
    ("results", "k", "match"),
    [
        ([], 1, "graded no task"),
        ([TaskResult("a", (1.0,)), TaskResult("b", (1.0, 0.0))], 1, "different"),
        ([TaskResult("a", (1.0,))], 2, "exceeds"),
    ],
)
def test_summarize_rejects_an_unusable_run(
    results: list[TaskResult],
    k: int,
    match: str,
) -> None:
    """A run that graded nothing, or graded unevenly, cannot be summarized."""
    with pytest.raises(ValueError, match=match):
        summarize(results, k=k)


def test_config_resolves_the_overrides_against_the_base_dir(tmp_path: Path) -> None:
    """Evaluation jobs read the same paths the training runner does."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch, base_url="http://127.0.0.1:9")
    assert config.tasks_dir == scratch / "tasks"
    assert config.output_dir == scratch / "runs/openaimath/eval"
    assert config.resolved_tokenizer_path == scratch / "models/Qwen3.5-4B"
    assert config.resolved_base_url == "http://127.0.0.1:9"


def test_config_requires_a_snapshot(tmp_path: Path) -> None:
    """Without the dataset's snapshot there is nothing to grade in."""
    with pytest.raises(ValueError, match="snapshot"):
        _config_for(tmp_path, snapshot="")


def test_task_names_draws_every_staged_task_by_default(tmp_path: Path) -> None:
    """The default run grades the whole staged set, once each."""
    scratch = _scratch(tmp_path)
    names = _task_names(_config_for(scratch), _flags(scratch))
    assert sorted(names) == sorted(TASK_NAMES)
    assert len(set(names)) == len(TASK_NAMES)


def test_task_names_limit_is_a_seeded_subset(tmp_path: Path) -> None:
    """A limited run takes the runner's own draw order, not the sorted one."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch)
    first = _task_names(config, _flags(scratch, limit=2))
    assert len(first) == 2
    assert first == _task_names(config, _flags(scratch, limit=2))
    full = _task_names(config, _flags(scratch))
    assert first == full[:2]


def test_task_names_keeps_an_explicit_order(tmp_path: Path) -> None:
    """Naming tasks grades exactly those, in the order given."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch)
    flags = _flags(scratch, task_names=["gamma", "alpha"])
    assert _task_names(config, flags) == ["gamma", "alpha"]


def test_task_names_rejects_an_unknown_task(tmp_path: Path) -> None:
    """A task that is not staged is refused before Harbor is invoked."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch)
    flags = _flags(scratch, task_names=["gamma", "delta"])
    with pytest.raises(FileNotFoundError, match="missing"):
        _task_names(config, flags)


def test_task_names_reads_a_task_list_under_the_base_dir(tmp_path: Path) -> None:
    """A list file is a logical path, and its blank lines are not tasks."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch)
    (scratch / "lists").mkdir()
    (scratch / "lists" / "official.txt").write_text(
        "beta\n\ngamma\n",
        encoding="utf-8",
    )
    flags = _flags(scratch, task_list=Path("/lists/official.txt"))
    assert _task_names(config, flags) == ["beta", "gamma"]


def test_task_names_refuses_both_a_list_and_names(tmp_path: Path) -> None:
    """Two sources for the task set would leave one silently ignored."""
    scratch = _scratch(tmp_path)
    config = _config_for(scratch)
    flags = _flags(scratch, task_names=["alpha"], task_list=Path("/lists/x.txt"))
    with pytest.raises(ValueError, match="not both"):
        _task_names(config, flags)


def test_evaluate_reports_one_result_per_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every graded task contributes its own verdicts, in order."""
    config = _config_for(_scratch(tmp_path))

    def run_job(
        _config: object,
        _name: str,
        _job_dir: Path,
        **_kwargs: object,
    ) -> list[TrialOutcome]:
        return [
            TrialOutcome("alpha__aaaa__agent", 1.0, None),
            TrialOutcome("alpha__bbbb__agent", 0.0, None),
        ]

    monkeypatch.setattr(eval_module, "run_job", run_job)
    results = evaluate(config, ["alpha", "beta"], attempts=2, oracle=False)
    assert [result.task for result in results] == ["alpha", "beta"]
    assert results[0].rewards == (1.0, 0.0)


def test_evaluate_refuses_a_run_that_never_reached_the_grader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No verdict at all is a harness failure, not a policy failure."""
    config = _config_for(_scratch(tmp_path))

    def run_job(
        _config: object,
        _name: str,
        _job_dir: Path,
        **_kwargs: object,
    ) -> list[TrialOutcome]:
        return [TrialOutcome("alpha__aaaa__agent", None, None)]

    monkeypatch.setattr(eval_module, "run_job", run_job)
    with pytest.raises(RuntimeError, match="never reached the grader"):
        evaluate(config, ["alpha"], attempts=1, oracle=False)


def test_evaluate_resets_and_runs_through_the_real_job_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eval job directory passes the runner's own reset guard.

    The reset refuses any directory not named for its kind, so evaluation must
    tell it the kind; a stale trial left from an earlier run must not be read.
    """
    config = _config_for(_scratch(tmp_path))
    stale = job_directory(config, "alpha") / "alpha__stale" / "verifier"
    stale.mkdir(parents=True)
    (stale / "reward.txt").write_text("1\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_harbor(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        job_dir = Path(command[command.index("-o") + 1])
        verdict = job_dir / "alpha__fresh" / "verifier"
        verdict.mkdir(parents=True)
        (verdict / "reward.txt").write_text("0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_harbor)
    results = evaluate(config, ["alpha"], attempts=1, oracle=True)
    assert results == [TaskResult("alpha", (0.0,))]
    assert len(commands) == 1
    assert commands[0][commands[0].index("-i") + 1] == "alpha"


def test_job_directory_is_scoped_to_the_run(tmp_path: Path) -> None:
    """Job directories are named for the run, so a reset stays scoped."""
    config = _config_for(_scratch(tmp_path))
    assert job_directory(config, "alpha").name == "openaimath_eval_alpha"


def test_report_round_trip_records_the_pins(tmp_path: Path) -> None:
    """The report carries what a reader needs to reproduce the run."""
    config = _config_for(_scratch(tmp_path), base_url="http://host:1")
    results = [TaskResult("alpha", (1.0, 0.0))]
    payload = _report_payload(config, results, summarize(results, k=2), oracle=False)
    target = tmp_path / "reports" / "eval_report.json"
    _write_report(target, payload)
    written = cast(dict[str, object], json.loads(target.read_text(encoding="utf-8")))
    summary = cast(dict[str, object], written["summary"])
    assert written["mode"] == "policy"
    assert written["base_url"] == "http://host:1"
    assert written["snapshot"] == "openai-math-lean-test"
    assert summary["pass@1"] == pytest.approx(0.5)
    assert written["results"] == [result.to_payload() for result in results]
    assert not target.with_suffix(".json.partial").exists()


def test_report_omits_the_policy_server_in_oracle_mode(tmp_path: Path) -> None:
    """The oracle needs no policy server, so the report names none."""
    config = _config_for(_scratch(tmp_path))
    results = [TaskResult("alpha", (1.0,))]
    payload = _report_payload(config, results, summarize(results, k=1), oracle=True)
    assert payload["mode"] == "oracle"
    assert payload["base_url"] is None
    assert payload["tokenizer_path"] is None


def test_main_grades_through_the_runner_and_writes_a_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy run drives the shared job runner and reports pass@1."""
    scratch = _scratch(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run_job(
        _config: object,
        name: str,
        job_dir: Path,
        **kwargs: object,
    ) -> list[TrialOutcome]:
        calls.append({"name": name, "job_dir": job_dir, **kwargs})
        return [
            TrialOutcome(f"{name}__aaaa__agent", 1.0, None),
            TrialOutcome(f"{name}__bbbb__agent", 0.0, None),
        ]

    monkeypatch.setattr(eval_module, "run_job", fake_run_job)
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.py",
            "--base-dir",
            str(scratch),
            "--tasks-dir",
            str(TASKS_LOGICAL_DIR),
            "--snapshot",
            "openai-math-lean-test",
            "--attempts",
            "2",
            "--task-names",
            "alpha",
            "--report",
            str(report),
        ],
    )
    assert main() == 0
    assert len(calls) == 1
    assert calls[0]["name"] == "alpha"
    assert calls[0]["agent_args"] is None
    assert calls[0]["attempts"] == 2
    written = cast(dict[str, object], json.loads(report.read_text(encoding="utf-8")))
    summary = cast(dict[str, object], written["summary"])
    assert summary["pass@1"] == pytest.approx(0.5)
    assert summary["pass@k"] == pytest.approx(1.0)


def test_main_runs_the_oracle_as_a_positive_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle run must be accepted everywhere, or the harness is broken."""
    scratch = _scratch(tmp_path)
    calls: list[dict[str, object]] = []
    rewards: list[float | None] = [1.0]

    def fake_run_job(
        _config: object,
        name: str,
        job_dir: Path,
        **kwargs: object,
    ) -> list[TrialOutcome]:
        calls.append({"name": name, "job_dir": job_dir, **kwargs})
        return [TrialOutcome(f"{name}__oracle__agent", rewards[0], None)]

    monkeypatch.setattr(eval_module, "run_job", fake_run_job)
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.py",
            "--base-dir",
            str(scratch),
            "--tasks-dir",
            str(TASKS_LOGICAL_DIR),
            "--snapshot",
            "openai-math-lean-test",
            "--oracle",
            "--task-names",
            "alpha",
            "--report",
            str(report),
        ],
    )
    assert main() == 0
    assert calls[0]["agent_args"] == ORACLE_AGENT_ARGS
    assert calls[0]["options"] == ()
    assert calls[0]["attempts"] == 1

    rewards[0] = 0.0
    assert main() == 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
