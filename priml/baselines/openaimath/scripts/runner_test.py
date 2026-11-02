"""Tests for the openai/math rollout runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import json
import threading

import pytest

from priml.baselines.openaimath.episode import TRANSCRIPT_NAME
from priml.baselines.openaimath.scripts.runner import (
    AGENT_IMPORT_PATH,
    VLLM_VERSION,
    OpenMathRolloutRunner,
    OpenMathRunnerConfig,
    TaskStore,
    TrialOutcome,
    attempt_line,
    check_rollout_gpus,
    collect_group,
    read_job,
    records_for_update,
    reset_job_dir,
    rollout_server_command,
    rollout_server_environment,
    run_updates,
)
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


TASK_NAMES = ("alpha", "beta", "gamma", "delta")


def _tasks(tmp_path: Path, names: Sequence[str] = TASK_NAMES) -> Path:
    """Write a staged tasks directory."""
    directory = tmp_path / "tasks"
    for name in names:
        (directory / name).mkdir(parents=True, exist_ok=True)
        (directory / name / "instruction.md").write_text(
            f"prove {name}\n",
            encoding="utf-8",
        )
    return directory


def _runner(tmp_path: Path, **overrides: object) -> OpenMathRolloutRunner:
    """Build a validated runner over a staged tasks directory."""
    _tasks(tmp_path)
    config = OpenMathRunnerConfig(
        base_dir=tmp_path,
        working_dir="run",
        output_dir=Path("rollouts"),
        tasks_dir=Path("tasks"),
        snapshot="snapshot-name",
        checkpoint="/models/policy",
        samples_per_prompt=2,
        num_prompts=2,
        num_updates=1,
        seed=0,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return OpenMathRolloutRunner(config.finalize())


def _transcript(*, response: int = 2) -> dict[str, object]:
    """Build one episode transcript."""
    return {
        "challenge": "PlaneColoring",
        "prompt_tokens": [1, 2, 3],
        "response_tokens": list(range(4, 4 + response)),
        "logprobs": [-0.5] * response,
        "tool_mask": [1] * response,
        "finish_reason": "stop",
        "turns": 1,
        "submitted": True,
    }


def _trial(
    job_dir: Path,
    name: str,
    *,
    reward: float | None = 0.0,
    transcript: Mapping[str, object] | None = None,
) -> None:
    """Write one trial directory the way Harbor lays it out."""
    trial = job_dir / name
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir(parents=True)
    if reward is not None:
        (trial / "verifier" / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    if transcript is not None:
        (trial / "agent" / TRANSCRIPT_NAME).write_text(
            json.dumps(dict(transcript)) + "\n",
            encoding="utf-8",
        )


def _outcomes(count: int, *, reward: float = 0.0) -> list[TrialOutcome]:
    """Build a group's worth of outcomes."""
    return [
        TrialOutcome(
            trial=f"task__{index:04d}__agent",
            reward=reward if index else 1.0,
            record=_transcript(),
        )
        for index in range(count)
    ]


def test_task_store_draws_deterministically_and_resumes(tmp_path: Path) -> None:
    """The draw order is fixed by the seed, and a cursor resumes it exactly."""
    tasks_dir = _tasks(tmp_path)
    first = TaskStore(tasks_dir, seed=7)
    drawn = [first.draw().name for _ in range(3)]
    state = first.state_dict()

    resumed = TaskStore(tasks_dir, seed=7)
    resumed.load_state_dict(state)
    assert resumed.draw().name == first.draw().name

    same_seed = TaskStore(tasks_dir, seed=7)
    assert [same_seed.draw().name for _ in range(3)] == drawn


def test_task_store_cycles_through_a_new_epoch(tmp_path: Path) -> None:
    """Every task is drawn before any repeats."""
    store = TaskStore(_tasks(tmp_path), seed=3)
    first_epoch = [store.draw().name for _ in TASK_NAMES]
    second_epoch = [store.draw().name for _ in TASK_NAMES]
    assert sorted(first_epoch) == sorted(TASK_NAMES)
    assert sorted(second_epoch) == sorted(TASK_NAMES)
    # Four draws consumed the first epoch; four more consumed the second.
    assert store.state_dict() == {"seed": 3, "epoch": 1, "position": 4}


def test_task_store_resumes_an_order_seeded_from_entropy(tmp_path: Path) -> None:
    """An unseeded order is saved with its cursor, so a resume continues it."""
    tasks_dir = _tasks(tmp_path)
    first = TaskStore(tasks_dir, seed=None)
    _ = first.draw()
    state = first.state_dict()
    assert state["seed"] == first.seed

    resumed = TaskStore(tasks_dir, seed=None)
    resumed.load_state_dict(state)
    assert [resumed.draw().name for _ in range(7)] == [
        first.draw().name for _ in range(7)
    ]


def test_task_store_reads_the_drawn_instruction(tmp_path: Path) -> None:
    """A drawn task carries the instruction that prompts the episode."""
    store = TaskStore(_tasks(tmp_path), seed=0)
    task = store.draw()
    assert task.instruction == f"prove {task.name}\n"


def test_task_store_rejects_a_missing_task(tmp_path: Path) -> None:
    """An explicit subset must be staged."""
    tasks_dir = _tasks(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        TaskStore(tasks_dir, seed=0, task_names=("alpha", "missing"))


def test_task_store_rejects_an_empty_directory(tmp_path: Path) -> None:
    """A run with no staged tasks cannot draw anything."""
    with pytest.raises(FileNotFoundError, match="No staged"):
        TaskStore(tmp_path / "nothing", seed=0)


def test_runner_rejects_an_unusable_configuration(tmp_path: Path) -> None:
    """Composition rules that would silently change the objective are refused."""
    with pytest.raises(ValueError, match="samples_per_prompt"):
        _runner(tmp_path, samples_per_prompt=1)
    with pytest.raises(ValueError, match="snapshot"):
        _runner(tmp_path, snapshot="")
    with pytest.raises(ValueError, match="active_sampling"):
        _runner(tmp_path, filter_zero_std_samples=True, active_sampling=False)
    with pytest.raises(ValueError, match="distinct"):
        _runner(tmp_path, rollout_gpus=(4, 5))
    with pytest.raises(ValueError, match="distinct"):
        _runner(tmp_path, rollout_gpus=(4, 4), data_parallel_size=2)


@pytest.mark.parametrize("name", ["working_dir", "output_dir", "num_updates"])
def test_runner_refuses_an_unset_per_run_setting(tmp_path: Path, name: str) -> None:
    """A per-run setting nobody filled is refused, not defaulted."""
    with pytest.raises(ValueError, match=name):
        _runner(tmp_path, **{name: None})


def test_runner_keeps_an_explicit_seed(tmp_path: Path) -> None:
    """The seed the caller set is the one the task order uses."""
    runner = _runner(tmp_path, seed=7)
    assert runner.tasks.seed == 7


def test_runner_refuses_a_draw_cap_below_one_update(tmp_path: Path) -> None:
    """An update must at least be able to draw one group per slot."""
    with pytest.raises(ValueError, match="max_draws_per_update"):
        _runner(tmp_path, max_draws_per_update=1)


def test_harbor_command_pins_the_agent_and_the_budgets(tmp_path: Path) -> None:
    """The job invokes Harbor with one task, one group, and the agent's settings."""
    runner = _runner(tmp_path, samples_per_prompt=4, n_concurrent=8)
    job_dir = tmp_path / "job"
    command = runner.harbor_command("alpha", job_dir)
    assert command[0] == "harbor"
    assert command[1] == "run"
    assert command[command.index("-p") + 1] == str(tmp_path / "tasks")
    assert command[command.index("-i") + 1] == "alpha"
    assert command[command.index("-k") + 1] == "4"
    assert command[command.index("-n") + 1] == "8"
    assert command[command.index("-o") + 1] == str(job_dir)
    assert command[command.index("--agent-import-path") + 1] == AGENT_IMPORT_PATH
    assert command[command.index("--ek") + 1] == "snapshot_template_name=snapshot-name"
    agent_options = {
        command[index + 1] for index, value in enumerate(command) if value == "--ak"
    }
    assert f"base_url={runner.config.resolved_base_url}" in agent_options
    assert f"tokenizer_path={runner.config.resolved_tokenizer_path}" in agent_options
    assert "max_turns=64" in agent_options
    assert "response_length=65536" in agent_options
    # One seed shared by every attempt would make the group identical.
    assert not any(option.startswith("seed=") for option in agent_options)


def test_harbor_command_passes_a_sampling_seed_when_configured(tmp_path: Path) -> None:
    """An explicit sampling seed reaches the agent."""
    runner = _runner(tmp_path, sample_seed=11)
    command = runner.harbor_command("alpha", tmp_path / "job")
    assert "seed=11" in command


def test_rollout_server_command_covers_the_episode_budget(tmp_path: Path) -> None:
    """The server's context bound holds a whole prompt plus a whole response."""
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "export"
    command = rollout_server_command(runner.config, checkpoint)
    assert command[:4] == ["uvx", f"vllm=={VLLM_VERSION}", "serve", str(checkpoint)]
    assert command[command.index("--served-model-name") + 1] == "policy"
    assert command[command.index("--max-model-len") + 1] == str(16_384 + 65_536)
    assert command[command.index("--data-parallel-size") + 1] == "1"
    assert "--enable-prefix-caching" in command


def test_rollout_server_command_can_name_an_installed_vllm(tmp_path: Path) -> None:
    """A node with vLLM installed can skip the uvx launcher."""
    runner = _runner(tmp_path, vllm_command=("/opt/vllm/bin/vllm",))
    command = rollout_server_command(runner.config, tmp_path / "export")
    assert command[:2] == ["/opt/vllm/bin/vllm", "serve"]


def test_rollout_server_environment_pins_the_rollout_gpus(tmp_path: Path) -> None:
    """The server sees only its own devices, numbered as the learner sees them."""
    runner = _runner(tmp_path, rollout_gpus=(2, 3), data_parallel_size=2)
    plain = rollout_server_environment(runner.config, {"PATH": "/bin"})
    assert plain == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "2,3"}
    empty = rollout_server_environment(
        runner.config,
        {"CUDA_VISIBLE_DEVICES": ""},
    )
    assert empty["CUDA_VISIBLE_DEVICES"] == "2,3"
    remapped = rollout_server_environment(
        runner.config,
        {"CUDA_VISIBLE_DEVICES": "1, 3,5,7"},
    )
    assert remapped["CUDA_VISIBLE_DEVICES"] == "5,7"
    with pytest.raises(ValueError, match="visible"):
        rollout_server_environment(runner.config, {"CUDA_VISIBLE_DEVICES": "0,1"})


def test_rollout_server_environment_without_rollout_gpus_is_unchanged(
    tmp_path: Path,
) -> None:
    """A standalone tool's server keeps whatever devices it was given."""
    runner = _runner(tmp_path)
    environ = {"CUDA_VISIBLE_DEVICES": "0"}
    assert rollout_server_environment(runner.config, environ) == environ


def test_check_rollout_gpus_refuses_the_learners_devices() -> None:
    """Local rank r holds cuda:r, so those devices are not the server's."""
    check_rollout_gpus((4, 5, 6, 7), learner_devices=4, available_devices=8)
    with pytest.raises(ValueError, match="overlap"):
        check_rollout_gpus((3, 4), learner_devices=4, available_devices=8)
    with pytest.raises(ValueError, match="not among"):
        check_rollout_gpus((4, 8), learner_devices=4, available_devices=8)


def test_reset_job_dir_is_scoped_to_its_kind(tmp_path: Path) -> None:
    """A reset removes only a directory named for its run and kind."""
    step_dir = tmp_path / "jobs" / "openaimath_step_00000000_slot_00_try_00"
    (step_dir / "trial").mkdir(parents=True)
    with pytest.raises(ValueError, match="foreign"):
        reset_job_dir(step_dir, "openaimath", kind="eval")
    reset_job_dir(step_dir, "openaimath")
    assert not step_dir.exists()
    with pytest.raises(ValueError, match="foreign"):
        reset_job_dir(tmp_path / "jobs" / "other_step_0", "openaimath")


def test_read_job_reads_verdicts_and_transcripts(tmp_path: Path) -> None:
    """A trial's verdict and transcript are read from Harbor's layout."""
    job_dir = tmp_path / "job"
    _trial(job_dir, "alpha__aaaa__agent", reward=1.0, transcript=_transcript())
    _trial(job_dir, "alpha__bbbb__agent", reward=0.0, transcript=_transcript())
    outcomes = read_job(job_dir)
    assert [outcome.trial for outcome in outcomes] == [
        "alpha__aaaa__agent",
        "alpha__bbbb__agent",
    ]
    assert [outcome.reward for outcome in outcomes] == [1.0, 0.0]
    assert outcomes[0].record is not None
    assert outcomes[0].record["finish_reason"] == "stop"  # pyright: ignore[reportIndexIssue]


def test_read_job_reports_a_trial_that_never_graded(tmp_path: Path) -> None:
    """A trial with no verdict still appears, with a missing reward."""
    job_dir = tmp_path / "job"
    _trial(job_dir, "alpha__aaaa__agent", reward=None, transcript=_transcript())
    outcomes = read_job(job_dir)
    assert outcomes[0].reward is None
    assert outcomes[0].record is not None


def test_read_job_reports_a_trial_that_wrote_no_transcript(tmp_path: Path) -> None:
    """A trial whose agent crashed is reported rather than dropped."""
    job_dir = tmp_path / "job"
    _trial(job_dir, "alpha__aaaa__agent", reward=0.0, transcript=None)
    outcomes = read_job(job_dir)
    assert outcomes[0].record is None


def test_read_job_rejects_a_directory_without_trials(tmp_path: Path) -> None:
    """A job that produced nothing is an error, not an empty group."""
    with pytest.raises(FileNotFoundError, match="No trials"):
        read_job(tmp_path / "empty")


def test_read_job_keeps_the_last_transcript_line(tmp_path: Path) -> None:
    """A resumed trial's newest episode is the one that is trained."""
    job_dir = tmp_path / "job"
    _trial(job_dir, "alpha__aaaa__agent", transcript=_transcript(response=2))
    path = job_dir / "alpha__aaaa__agent" / "agent" / TRANSCRIPT_NAME
    path.write_text(
        json.dumps(_transcript(response=5)) + "\n",
        encoding="utf-8",
    )
    outcomes = read_job(job_dir)
    assert outcomes[0].record is not None
    assert len(cast(list[object], outcomes[0].record["response_tokens"])) == 5


def test_read_job_reads_the_graders_reason_and_duration(tmp_path: Path) -> None:
    """The grader's own report is kept beside the reward it trained on."""
    job_dir = tmp_path / "job"
    _trial(job_dir, "alpha__aaaa__agent", transcript=_transcript())
    _trial(job_dir, "alpha__bbbb__agent", transcript=_transcript())
    (job_dir / "alpha__aaaa__agent" / "verifier" / "grading.json").write_text(
        json.dumps(
            {"reason": "comparator rejected", "started_at": 10.0, "finished_at": 52.5},
        ),
        encoding="utf-8",
    )
    (job_dir / "alpha__bbbb__agent" / "verifier" / "grading.json").write_text(
        "not json",
        encoding="utf-8",
    )
    graded, unreadable = read_job(job_dir)
    assert graded.verdict == "comparator rejected"
    assert graded.grade_seconds == pytest.approx(42.5)
    assert (graded.grade_started_at, graded.grade_finished_at) == (10.0, 52.5)
    # A grader report this baseline cannot read never costs the group its reward.
    assert unreadable.verdict is None
    assert unreadable.grade_seconds is None
    assert unreadable.grade_started_at is None
    assert unreadable.reward == 0.0


def test_collect_group_assembles_one_prompt_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group's records are ordered by trial name and labelled with their slot."""
    runner = _runner(tmp_path)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    task = runner.tasks.draw()
    records = collect_group(runner, 5, 1, task)
    assert records is not None
    # Slot 1 of groups of two: its samples follow slot 0's within the update.
    assert [record["sample_idx"] for record in records] == [2, 3]
    assert {record["prompt_idx"] for record in records} == {1}
    assert {record["step"] for record in records} == {5}
    rewards = [cast(float, record["reward"]) for record in records]
    assert sorted(rewards) == [0.0, 1.0]
    assert records[0]["tool_mask"] == [1, 1]
    assert records[0]["finish_reason"] == "stop"


def test_collect_group_redraws_a_group_missing_a_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ungraded group is redrawn rather than scored zero."""
    runner = _runner(tmp_path)
    attempts = [
        [TrialOutcome("alpha__aaaa__agent", None, _transcript())],
        _outcomes(2),
    ]

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return attempts.pop(0)

    monkeypatch.setattr(runner, "run_job", run_job)
    records = collect_group(runner, 0, 0, runner.tasks.draw())
    assert records is not None
    assert len(records) == 2


def test_collect_group_gives_up_after_the_attempt_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt that never produces a usable group fails the update loudly."""
    runner = _runner(tmp_path, max_group_attempts=2)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return [TrialOutcome("alpha__aaaa__agent", None, None)]

    monkeypatch.setattr(runner, "run_job", run_job)
    with pytest.raises(RuntimeError, match="no usable group"):
        collect_group(runner, 0, 0, runner.tasks.draw())


def test_collect_group_drops_a_zero_spread_group_when_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With filtering on, an all-equal group is dropped for a fresh draw."""
    runner = _runner(tmp_path, filter_zero_std_samples=True, active_sampling=True)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return [
            TrialOutcome("alpha__aaaa__agent", 0.0, _transcript()),
            TrialOutcome("alpha__bbbb__agent", 0.0, _transcript()),
        ]

    monkeypatch.setattr(runner, "run_job", run_job)
    assert collect_group(runner, 0, 0, runner.tasks.draw()) is None


def test_records_for_update_holds_one_group_per_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update holds exactly the promised records, grouped by slot."""
    runner = _runner(tmp_path)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    records = records_for_update(runner, 3)
    assert len(records) == 4
    assert [record["prompt_idx"] for record in records] == [0, 0, 1, 1]
    # The rollout writer keys records by (step, sample_idx), as TMax does.
    assert [record["sample_idx"] for record in records] == [0, 1, 2, 3]
    assert {record["step"] for record in records} == {3}


def test_records_for_update_commits_through_the_rollout_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update's records are exactly what the shared writer commits and replays."""
    runner = _runner(tmp_path)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    records = records_for_update(runner, 0)
    writer = RolloutWriter(tmp_path / "rollouts", "openaimath")
    writer.write_live(0, records, prompt_loader_state_after=runner.tasks.state_dict())
    resumed = RolloutWriter(tmp_path / "rollouts", "openaimath")
    assert resumed.committed_live_steps(4) == {0}
    assert resumed.live_cursor_after(0, 4) == runner.tasks.state_dict()
    replayed = resumed.records_for_step(0, 4)
    assert [record["task"] for record in replayed] == [
        record["task"] for record in records
    ]


def test_records_for_update_refills_a_dropped_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped group's slot is refilled from the next draw, in slot order."""
    runner = _runner(tmp_path, filter_zero_std_samples=True, active_sampling=True)
    order = TaskStore(tmp_path / "tasks", seed=0)
    first, second, third = (order.draw().name for _ in range(3))
    ran: dict[str, int] = {}

    def run_job(task: str, _job_dir: Path) -> list[TrialOutcome]:
        ran[task] = ran.get(task, 0) + 1
        if task == first:
            return [
                TrialOutcome(f"{task}__aaaa__agent", 0.0, _transcript()),
                TrialOutcome(f"{task}__bbbb__agent", 0.0, _transcript()),
            ]
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    records = records_for_update(runner, 0)
    assert len(records) == 4
    assert [record["prompt_idx"] for record in records] == [0, 0, 1, 1]
    assert ran == {first: 1, second: 1, third: 1}
    assert runner.tasks.state_dict() == {"seed": 0, "epoch": 0, "position": 3}


def test_records_for_update_stops_at_the_draw_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool with no mixed groups fails the update instead of drawing forever."""
    runner = _runner(tmp_path, max_draws_per_update=5)
    ran: list[str] = []

    def run_job(task: str, _job_dir: Path) -> list[TrialOutcome]:
        ran.append(task)
        return [
            TrialOutcome(f"{task}__aaaa__agent", 0.0, _transcript()),
            TrialOutcome(f"{task}__bbbb__agent", 0.0, _transcript()),
        ]

    monkeypatch.setattr(runner, "run_job", run_job)
    with pytest.raises(RuntimeError, match="cap is 5"):
        records_for_update(runner, 0)
    # Two rounds of two slots fit under the cap; a third round would not.
    assert len(ran) == 4


def test_records_for_update_keeps_the_lean_side_of_each_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each record names its task and carries the grader's reason and costs."""
    runner = _runner(tmp_path)

    def run_job(task: str, _job_dir: Path) -> list[TrialOutcome]:
        return [
            TrialOutcome(
                f"{task}__{index}__agent",
                float(index),
                {**_transcript(), "sandbox_seconds": 3.0 + index},
                verdict="accepted" if index else "rejected",
                grade_seconds=40.0,
            )
            for index in range(2)
        ]

    monkeypatch.setattr(runner, "run_job", run_job)
    records = records_for_update(runner, 0)
    assert {record["task"] for record in records} <= set(TASK_NAMES)
    assert [record["verdict"] for record in records[:2]] == ["rejected", "accepted"]
    assert [record["sandbox_seconds"] for record in records[:2]] == [3.0, 4.0]
    assert {record["grade_seconds"] for record in records} == {40.0}


def test_records_for_update_runs_its_groups_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every slot's job is in flight at once, so an update is not their sum."""
    runner = _runner(tmp_path)
    # Each job waits for the other; run one after another, they would time out.
    both_running = threading.Barrier(2, timeout=5.0)

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        both_running.wait()
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    assert len(records_for_update(runner, 0)) == 4


def test_job_directory_names_are_scoped_to_the_run(tmp_path: Path) -> None:
    """A job directory is named for its step, slot, draw, and try."""
    runner = _runner(tmp_path)
    assert (
        runner.job_dir(7, 2, 3, 1).name
        == "openaimath_step_00000007_slot_02_draw_03_try_01"
    )


def test_recollected_group_preserves_logged_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted update's retried job cannot erase its earlier trials."""
    runner = _runner(tmp_path)
    task = runner.tasks.draw()

    def run_job(_task: str, job_dir: Path) -> list[TrialOutcome]:
        reset_job_dir(job_dir, runner.config.run_name)
        _trial(job_dir, "alpha__aaaa__agent", reward=1.0, transcript=_transcript())
        _trial(job_dir, "alpha__bbbb__agent", reward=0.0, transcript=_transcript())
        return read_job(job_dir)

    monkeypatch.setattr(runner, "run_job", run_job)
    assert collect_group(runner, 0, 0, task) is not None
    assert collect_group(runner, 0, 0, task) is not None
    lines = _attempt_log(runner)
    jobs = [str(line["job"]) for line in lines]
    assert len(set(jobs)) == 2
    assert jobs[:2] == [jobs[0]] * 2
    assert jobs[2:] == [jobs[2]] * 2
    for line in lines:
        trial = runner.output_dir / "jobs" / str(line["job"]) / str(line["trial"])
        assert (trial / "agent" / TRANSCRIPT_NAME).is_file()
        assert (trial / "verifier" / "reward.txt").is_file()


def test_fixed_corpus_resume_restores_seed_and_draw_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed corpus resumes after the draws consumed by committed steps."""
    collected: list[tuple[int, list[str]]] = []

    def collect(runner: OpenMathRolloutRunner, step: int) -> list[dict[str, object]]:
        tasks = [runner.tasks.draw().name for _ in range(runner.config.num_prompts)]
        collected.append((step, tasks))
        return [
            {"step": step, "sample_idx": slot * 2 + index, "task": task}
            for slot, task in enumerate(tasks)
            for index in range(2)
        ]

    monkeypatch.setattr(
        "priml.baselines.openaimath.scripts.runner.records_for_update",
        collect,
    )
    first = _runner(tmp_path, seed=None)
    assert len(run_updates(first)) == 1
    expected = [first.tasks.draw().name for _ in range(first.config.num_prompts)]
    resumed = _runner(tmp_path, seed=None, num_updates=2)
    assert len(run_updates(resumed)) == 1
    assert collected == [(0, collected[0][1]), (1, expected)]
    assert resumed.tasks.seed == first.tasks.seed
    assert resumed.tasks.state_dict() == first.tasks.state_dict()


def test_fixed_corpus_refuses_legacy_shards_without_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never guess a cursor or discard data from an older fixed corpus."""
    runner = _runner(tmp_path)
    writer = RolloutWriter(runner.output_dir, runner.config.run_name)
    writer.write(
        0,
        [{"step": 0, "sample_idx": index} for index in range(4)],
    )

    def unexpected(_runner: OpenMathRolloutRunner, _step: int) -> None:
        raise AssertionError("must not recollect over an unmarked corpus")

    monkeypatch.setattr(
        "priml.baselines.openaimath.scripts.runner.records_for_update",
        unexpected,
    )
    with pytest.raises(ValueError, match="no task-cursor markers"):
        run_updates(runner)
    assert writer.completed_steps(4) == {0}


def _attempt_log(runner: OpenMathRolloutRunner) -> list[dict[str, object]]:
    """Read the runner's attempt log back."""
    if not runner.attempt_log_path.exists():
        return []
    return [
        cast(dict[str, object], json.loads(line))
        for line in runner.attempt_log_path.read_text(encoding="utf-8").splitlines()
    ]


def test_records_for_update_keeps_a_dropped_draw_on_disk_and_in_the_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refilled slot's dropped draw keeps its own job directory and log lines."""
    runner = _runner(tmp_path)
    first = TaskStore(tmp_path / "tasks", seed=0).draw().name
    job_dirs: dict[str, list[str]] = {}

    def run_job(task: str, job_dir: Path) -> list[TrialOutcome]:
        job_dirs.setdefault(task, []).append(job_dir.name)
        if task == first:
            return [
                TrialOutcome(f"{task}__aaaa__agent", 0.0, _transcript()),
                TrialOutcome(f"{task}__bbbb__agent", 0.0, _transcript()),
            ]
        return _outcomes(2)

    monkeypatch.setattr(runner, "run_job", run_job)
    records = records_for_update(runner, 4)
    assert len(records) == 4
    names = [name for task_dirs in job_dirs.values() for name in task_dirs]
    # Distinct names: resetting the refill's directory cannot remove the drop's.
    assert len(set(names)) == len(names) == 3
    assert job_dirs[first] == ["openaimath_step_00000004_slot_00_draw_00_try_00"]
    assert "openaimath_step_00000004_slot_00_draw_01_try_00" in names

    lines = _attempt_log(runner)
    assert len(lines) == 6
    dropped = [line for line in lines if line["task"] == first]
    assert {line["group"] for line in dropped} == {"zero_spread"}
    assert {line["job"] for line in dropped} == set(job_dirs[first])
    assert [line["group"] for line in lines].count("kept") == 4
    refill = [line for line in lines if line["slot"] == 0 and line["draw"] == 1]
    assert len(refill) == 2


def test_collect_group_logs_an_unusable_try_and_its_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun's first try is in the log, marked unusable, under its own try."""
    runner = _runner(tmp_path)
    attempts = [
        [
            TrialOutcome("alpha__aaaa__agent", None, _transcript()),
            TrialOutcome("alpha__bbbb__agent", 1.0, None),
        ],
        _outcomes(2),
    ]

    def run_job(_task: str, _job_dir: Path) -> list[TrialOutcome]:
        return attempts.pop(0)

    monkeypatch.setattr(runner, "run_job", run_job)
    assert collect_group(runner, 0, 1, runner.tasks.draw(), draw=2) is not None
    lines = _attempt_log(runner)
    assert [(line["try"], line["group"]) for line in lines] == [
        (0, "unusable"),
        (0, "unusable"),
        (1, "kept"),
        (1, "kept"),
    ]
    assert {line["draw"] for line in lines} == {2}
    assert lines[0]["reward"] is None
    assert lines[0]["has_transcript"] is True
    assert lines[1]["has_transcript"] is False
    assert lines[1]["finish_reason"] is None


def test_attempt_line_carries_cost_times_and_the_submission_digest() -> None:
    """A line holds what joins, counts, and labels one graded attempt."""
    outcome = TrialOutcome(
        "alpha__aaaa__agent",
        1.0,
        {
            **_transcript(response=3),
            "tool_mask": [1, 0, 1],
            "sandbox_seconds": 12.5,
            "started_at": 100.0,
            "finished_at": 160.0,
            "submission_sha256": "ab" * 32,
        },
        verdict="accepted",
        grade_seconds=30.0,
        grade_started_at=170.0,
        grade_finished_at=200.0,
    )
    line = attempt_line(
        outcome,
        task="alpha",
        step=3,
        slot=1,
        draw=0,
        attempt=0,
        job="openaimath_step_00000003_slot_01_draw_00_try_00",
        group="kept",
    )
    assert line["prompt_tokens"] == 3
    assert line["policy_tokens"] == 2
    assert line["sandbox_seconds"] == 12.5
    assert line["grade_seconds"] == 30.0
    assert (line["episode_started_at"], line["episode_finished_at"]) == (100.0, 160.0)
    assert (line["grade_started_at"], line["grade_finished_at"]) == (170.0, 200.0)
    assert line["submission_sha256"] == "ab" * 32
    assert line["verdict"] == "accepted"
    assert line["submitted"] is True
    json.dumps(line)


def test_log_attempts_serializes_concurrent_appends(tmp_path: Path) -> None:
    """Lines from the update's worker threads never interleave."""
    runner = _runner(tmp_path)
    line = {"payload": "x" * 50_000}

    def append() -> None:
        for _ in range(20):
            runner.log_attempts([line, line])

    threads = [threading.Thread(target=append) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = _attempt_log(runner)
    assert len(lines) == 160
    assert all(entry == line for entry in lines)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
