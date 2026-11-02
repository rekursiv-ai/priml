"""Tests for the live openai/math rollout seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import os
import signal
import subprocess

import pytest
import torch

from priml.baselines.openaimath import live as live_module
from priml.baselines.openaimath.data import OpenMathRolloutData
from priml.baselines.openaimath.live import (
    LiveOpenMathRolloutData,
    server_answers,
    wait_for_server,
)
from priml.baselines.openaimath.scripts.runner import (
    OpenMathRolloutRunner,
    OpenMathRunnerConfig,
)
from priml.baselines.tmax.train_step import TMaxDPPOTrainStep


if TYPE_CHECKING:
    from collections.abc import Iterator

    from priml.train.custom_types import TrainStepProtocol


@dataclass
class _FakeStep:
    """The learner fields the live seam reads before it starts a runtime."""

    global_step: int = 0
    pad_token_id: int = 0


def _ready(_url: str) -> bool:
    """Probe that always reports the server up."""
    return True


def _unready(_url: str) -> bool:
    """Probe that never reports the server up."""
    return False


def _no_sleep(_seconds: float) -> None:
    """Return without sleeping."""


def _config(tmp_path: Path, **overrides: object) -> LiveOpenMathRolloutData.Config:
    """Build a live configuration over a staged tasks directory."""
    tasks = tmp_path / "tasks" / "alpha"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / "instruction.md").write_text("prove alpha\n", encoding="utf-8")
    runner = OpenMathRunnerConfig(
        base_dir=tmp_path,
        working_dir="run",
        output_dir=Path("rollouts"),
        tasks_dir=Path("tasks"),
        snapshot="snapshot-name",
        samples_per_prompt=2,
        num_prompts=2,
        num_updates=1,
        seed=0,
        rollout_gpus=(1,),
    )
    config = LiveOpenMathRolloutData.Config(
        base_dir=tmp_path,
        working_dir="run/rollouts",
        mask_tool_use=False,
        runner=runner,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.finalize()


def _tmax_step(temperature: float) -> TMaxDPPOTrainStep:
    """Return a TMax step that carries only its config, for the binding check."""
    step = TMaxDPPOTrainStep.__new__(TMaxDPPOTrainStep)
    step.config = TMaxDPPOTrainStep.Config(temperature=temperature)
    return step


def test_wait_for_server_returns_when_the_server_answers(tmp_path: Path) -> None:
    """Readiness is a single successful probe."""
    slept: list[float] = []
    wait_for_server(
        "http://127.0.0.1:8000",
        timeout_s=10.0,
        alive=lambda: None,
        log_path=tmp_path / "server.log",
        answers=_ready,
        sleep=slept.append,
    )
    assert slept == []


def test_wait_for_server_reports_a_dead_server_with_its_log(tmp_path: Path) -> None:
    """A server that exits is reported with the tail of its log."""
    log = tmp_path / "server.log"
    log.write_text("line one\nout of memory\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="out of memory"):
        wait_for_server(
            "http://127.0.0.1:8000",
            timeout_s=10.0,
            alive=lambda: 9,
            log_path=log,
            answers=_ready,
            sleep=_no_sleep,
        )


def test_wait_for_server_times_out(tmp_path: Path) -> None:
    """A server that never answers is reported with its log's path."""
    with pytest.raises(RuntimeError, match="did not answer"):
        wait_for_server(
            "http://127.0.0.1:8000",
            timeout_s=0.0,
            alive=lambda: None,
            log_path=tmp_path / "server.log",
            answers=_unready,
            sleep=_no_sleep,
        )


def test_wait_for_server_polls_until_the_server_answers(tmp_path: Path) -> None:
    """A server that comes up on a later probe is waited for."""
    probes: list[str] = []

    def answers(url: str) -> bool:
        probes.append(url)
        return len(probes) == 3

    slept: list[float] = []
    wait_for_server(
        "http://127.0.0.1:8000/",
        timeout_s=60.0,
        alive=lambda: None,
        log_path=tmp_path / "server.log",
        answers=answers,
        sleep=slept.append,
    )
    assert probes == ["http://127.0.0.1:8000/"] * 3
    assert slept == [5.0, 5.0]


def test_server_answers_reports_a_closed_port() -> None:
    """A port nobody serves is not ready."""
    assert server_answers("http://127.0.0.1:1", timeout_s=1.0) is False


def test_live_config_takes_the_batch_shape_from_the_runner(
    tmp_path: Path,
) -> None:
    """The runner decides what an update is; the dataset cannot disagree."""
    config = _config(tmp_path, records_per_update=2, num_samples_per_prompt=1)
    assert config.records_per_update == 4
    assert config.num_samples_per_prompt == 2


def test_live_data_refuses_an_unfinalized_batch_shape_mismatch(
    tmp_path: Path,
) -> None:
    """A dataset and a runner that disagree would mislabel every group."""
    config = _config(tmp_path)
    config.records_per_update = 2
    with pytest.raises(ValueError, match="records_per_update"):
        LiveOpenMathRolloutData(config)


def test_live_data_requires_the_runner_to_agree_on_filtering(
    tmp_path: Path,
) -> None:
    """The zero-spread filter is the runner's decision, mirrored by the dataset."""
    with pytest.raises(ValueError, match="filter_zero_std_samples"):
        LiveOpenMathRolloutData(
            _config(tmp_path, filter_zero_std_samples=False, active_sampling=True),
        )
    with pytest.raises(ValueError, match="active_sampling"):
        LiveOpenMathRolloutData(
            _config(tmp_path, filter_zero_std_samples=True, active_sampling=False),
        )


def test_live_data_requires_rollout_gpus(tmp_path: Path) -> None:
    """A server left on the learner's devices would fight it for memory."""
    config = _config(tmp_path)
    config.runner.rollout_gpus = ()
    with pytest.raises(ValueError, match="rollout_gpus"):
        LiveOpenMathRolloutData(config)


def test_live_data_requires_rows_that_hold_one_episode(tmp_path: Path) -> None:
    """A packed row shorter than an episode would truncate it."""
    with pytest.raises(ValueError, match="pack_length"):
        LiveOpenMathRolloutData(_config(tmp_path, pack_length=1_024))


def test_live_data_refuses_a_learner_at_another_temperature(tmp_path: Path) -> None:
    """DPPO's ratio is meaningful only when both sides use one distribution."""
    data = LiveOpenMathRolloutData(_config(tmp_path))
    with pytest.raises(ValueError, match="temperature"):
        data.bind_step(cast("TrainStepProtocol", _tmax_step(0.6)))


def test_live_data_refuses_a_learner_with_another_pad_id(tmp_path: Path) -> None:
    """Packing and scoring must remove the same token."""
    data = LiveOpenMathRolloutData(_config(tmp_path, pad_token_id=1))
    with pytest.raises(ValueError, match="pad_token_id"):
        data.bind_step(cast("TrainStepProtocol", _FakeStep(pad_token_id=2)))


def test_live_data_requires_a_bound_step(tmp_path: Path) -> None:
    """Collecting without a learner would have no policy to serve."""
    data = LiveOpenMathRolloutData(_config(tmp_path))
    with pytest.raises(RuntimeError, match="bound to a train step"):
        next(iter(data.train_dataloader()))


def test_live_config_keeps_the_recorded_defaults(tmp_path: Path) -> None:
    """The live configuration inherits the substrate's grouping defaults."""
    config = _config(tmp_path)
    assert isinstance(config, OpenMathRolloutData.Config)
    assert config.pack_length == 81_920
    assert config.mask_tool_use is False


@dataclass
class _ExportingStep:
    """A learner that records each policy export the live seam asks for."""

    pad_token_id: int = 0
    global_step: int = 0
    exports: list[int] = field(default_factory=list)

    def export_checkpoint(self, export_dir: Path) -> Path:
        """Record the export and name a per-step checkpoint directory."""
        self.exports.append(self.global_step)
        return export_dir / f"step_{self.global_step}"


class _Server:
    """A rollout server process that is up until it is signalled."""

    def __init__(self, pid: int, *, stubborn: bool = False) -> None:
        self.pid = pid
        self.stubborn = stubborn

    def poll(self) -> int | None:
        """Report the server as running."""
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Exit, unless this server ignores the first stop request."""
        if self.stubborn:
            self.stubborn = False
            raise subprocess.TimeoutExpired("vllm", timeout or 0.0)
        return 0


class _Harness:
    """Fake the process and Harbor boundaries of a live run; keep the rest real.

    The rollout writer, the task cursor, the resume bookkeeping, and the packing
    are the real ones, so these tests check what a cluster run would commit and
    replay. Only the vLLM process, the readiness probe, and Harbor's episodes
    are stand-ins.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.servers: list[tuple[list[str], dict[str, str]]] = []
        self.signals: list[signal.Signals] = []
        self.collected: list[int] = []
        self.stubborn = False
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", self._popen)
        monkeypatch.setattr(live_module, "wait_for_server", self._wait)
        monkeypatch.setattr(os, "killpg", self._killpg)
        monkeypatch.setattr(live_module, "records_for_update", self._collect)

    def _popen(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        **_kwargs: object,
    ) -> _Server:
        self.servers.append((command, env))
        return _Server(pid=1_000 + len(self.servers), stubborn=self.stubborn)

    def _wait(self, *_args: object, **_kwargs: object) -> None:
        return None

    def _killpg(self, _pid: int, signum: signal.Signals) -> None:
        self.signals.append(signum)

    def _collect(
        self,
        runner: OpenMathRolloutRunner,
        step: int,
    ) -> list[dict[str, object]]:
        """Draw the update's tasks and return one mixed group per slot."""
        self.collected.append(step)
        size = runner.config.samples_per_prompt
        records: list[dict[str, object]] = []
        for slot in range(runner.config.num_prompts):
            task = runner.tasks.draw().name
            records.extend(
                {
                    "step": step,
                    "sample_idx": slot * size + index,
                    "prompt_idx": slot,
                    "prompt_tokens": [1, 2, 3],
                    "response_tokens": [4 + step, 5],
                    "logprobs": [-0.5, -0.25],
                    "reward": float(index % 2),
                    "finish_reason": "stop",
                    "tool_mask": [1, 1],
                    "task": task,
                }
                for index in range(size)
            )
        return records


def _live(
    tmp_path: Path,
    step: _ExportingStep,
    *,
    num_updates: int = 2,
) -> LiveOpenMathRolloutData:
    """Build a bound live dataset over the real writer in ``tmp_path``."""
    config = _config(tmp_path)
    config.runner.num_updates = num_updates
    data = LiveOpenMathRolloutData(config)
    data.bind_step(cast("TrainStepProtocol", step))
    return data


def _train(
    batches: Iterator[dict[str, object]],
    step: _ExportingStep,
) -> dict[str, object]:
    """Take one update, then advance the learner the way a train step does."""
    batch = next(batches)
    step.global_step += 1
    return batch


def test_live_run_serves_each_update_from_the_policy_that_trains_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each update is collected right after the previous one's weights are served."""
    harness = _Harness(monkeypatch)
    step = _ExportingStep()
    data = _live(tmp_path, step)
    batches = iter(data.train_dataloader())

    first = _train(batches, step)
    # The initial policy is served before the first episode samples it.
    assert step.exports == [0]
    assert harness.collected == [0]
    assert len(harness.servers) == 1

    _ = _train(batches, step)
    assert step.exports == [0, 1]
    assert harness.collected == [0, 1]
    command, env = harness.servers[1]
    assert str(tmp_path / "export" / "step_1") in command
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    # The old server is stopped, and its process group reaped, before a new one.
    assert harness.signals == [signal.SIGTERM, signal.SIGKILL]
    with pytest.raises(StopIteration):
        next(batches)

    assert first["response_token_count"] == 8
    state = data.state_dict()
    assert state["next_step"] == 2
    cursor = cast(dict[str, object], state["task_cursor"])
    assert cursor == data._runner.tasks.state_dict()
    assert "seed" in cursor
    data.close()
    assert harness.signals[-2:] == [signal.SIGTERM, signal.SIGKILL]


def test_live_resume_replays_a_committed_update_without_collecting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that crashed after committing an update trains it from disk."""
    harness = _Harness(monkeypatch)
    step = _ExportingStep()
    data = _live(tmp_path, step)
    batches = iter(data.train_dataloader())
    _ = _train(batches, step)
    # The learner checkpoints here, then commits update 1 and crashes.
    checkpoint = data.state_dict()
    original = _train(batches, step)
    cursor_after = data._runner.tasks.state_dict()

    resumed_step = _ExportingStep(global_step=1)
    resumed = LiveOpenMathRolloutData(data.live_config)
    resumed.load_state_dict(checkpoint)
    resumed.bind_step(cast("TrainStepProtocol", resumed_step))
    replayed = _train(iter(resumed.train_dataloader()), resumed_step)

    assert harness.collected == [0, 1]
    assert replayed["response_token_count"] == original["response_token_count"]
    for row, expected in zip(
        cast(tuple[dict[str, object], ...], replayed["rows"]),
        cast(tuple[dict[str, object], ...], original["rows"]),
        strict=True,
    ):
        assert row.keys() == expected.keys()
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                # Prompt positions carry NaN logprobs, which never compare equal.
                assert torch.allclose(
                    value,
                    cast(torch.Tensor, expected[key]),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                ), key
    # The task order continues where the committed update left it.
    assert resumed._runner.tasks.state_dict() == cursor_after
    assert resumed_step.exports == [1]


def test_live_resume_refuses_a_cursor_ahead_of_the_learner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset cursor from another checkpoint would train the wrong updates."""
    _ = _Harness(monkeypatch)
    step = _ExportingStep(global_step=1)
    data = _live(tmp_path, step)
    data.load_state_dict({"next_step": 3})
    with pytest.raises(ValueError, match="does not match the learner"):
        next(iter(data.train_dataloader()))


def test_live_resume_refuses_a_learner_before_the_start_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run configured to start later cannot begin from an earlier learner."""
    _ = _Harness(monkeypatch)
    step = _ExportingStep()
    data = _live(tmp_path, step)
    data._runner.config.start_step = 2
    with pytest.raises(ValueError, match="start_step"):
        next(iter(data.train_dataloader()))


def test_live_stop_kills_a_server_that_ignores_the_stop_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that will not exit is killed, so its GPUs are free for the next."""
    harness = _Harness(monkeypatch)
    harness.stubborn = True
    step = _ExportingStep()
    data = _live(tmp_path, step, num_updates=1)
    _ = _train(iter(data.train_dataloader()), step)
    data.close()
    assert harness.signals == [signal.SIGTERM, signal.SIGKILL, signal.SIGKILL]
    data.close()
    assert len(harness.signals) == 3


def test_live_stop_tolerates_a_server_that_already_exited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signalling a process group that is gone is not an error."""
    _ = _Harness(monkeypatch)

    def gone(_pid: int, _signum: signal.Signals) -> None:
        raise ProcessLookupError

    step = _ExportingStep()
    data = _live(tmp_path, step, num_updates=1)
    _ = _train(iter(data.train_dataloader()), step)
    monkeypatch.setattr(os, "killpg", gone)
    data.close()


def test_live_other_ranks_train_what_rank_zero_collected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero rank exports collectively but starts no server and collects nothing."""
    harness = _Harness(monkeypatch)
    records = harness._collect(
        LiveOpenMathRolloutData(_config(tmp_path / "zero"))._runner,
        0,
    )
    harness.collected.clear()
    broadcasts: list[object] = [0, records]

    def broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        payload[0] = broadcasts.pop(0)

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    step = _ExportingStep()
    data = _live(tmp_path / "one", step, num_updates=1)
    batch = _train(iter(data.train_dataloader()), step)
    assert batch["response_token_count"] == 8
    assert step.exports == [0]
    assert harness.servers == []
    assert harness.collected == []
    assert broadcasts == []


def test_live_config_narrows_only_a_live_config(tmp_path: Path) -> None:
    """The live-only settings are not readable off a recorded-data config."""
    data = LiveOpenMathRolloutData(_config(tmp_path))
    data.config = cast(
        "LiveOpenMathRolloutData.Config",
        OpenMathRolloutData.Config(working_dir="x.jsonl"),
    )
    with pytest.raises(TypeError, match="non-live"):
        _ = data.live_config


def test_live_data_refuses_a_group_size_mismatch(tmp_path: Path) -> None:
    """The dataset's group size is the runner's samples per prompt."""
    config = _config(tmp_path)
    config.num_samples_per_prompt = 4
    with pytest.raises(ValueError, match="num_samples_per_prompt"):
        LiveOpenMathRolloutData(config)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
