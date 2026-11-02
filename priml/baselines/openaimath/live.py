"""Interleave openai/math rollout collection with the PriML learner.

The live dataset owns only the seam between the two systems. Harbor still builds
the sandbox, runs the episode, and grades the proof; the learner still packs and
trains. What this module adds is the ordering: an update is collected immediately
before it is trained, from the policy the learner has just finished updating, and
the trained weights reach the rollout server before the next update's first token
is sampled.

Weight sync works differently here than in the TMax baseline, and the reason is
the server. There the vLLM engines live inside the trainer's own Ray cluster and
take tensors over NCCL. Here vLLM is its own process serving a checkpoint, so
each update stops it, exports the trained policy, and starts it again on the
export. The cost is a server start per update; the benefit is that the served
policy is exactly the artifact the evaluation serves and the checkpoint holds,
with no in-process transfer to keep in step.
"""

from __future__ import annotations

# ruff: noqa: S310 -- Every URL here is this baseline's own configured endpoint.
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# cast(Any, ...) reaches the train step's optional export surface, and the
# rollout server's HTTP response is untyped, as in the TMax baseline's live.py.
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, cast, override

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

from priml.baselines.openaimath.data import OpenMathRolloutData, record_from_payload
from priml.baselines.openaimath.scripts.runner import (
    OpenMathRolloutRunner,
    OpenMathRunnerConfig,
    check_rollout_gpus,
    records_for_update,
    rollout_server_command,
    rollout_server_environment,
)
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.baselines.tmax.train_step import TMaxDPPOTrainStep
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    import torch

    from priml.baselines.tmax.rollouts import RolloutRecord
    from priml.train.custom_types import TrainStepProtocol
else:
    from wrapt import lazy_import

    # ~1050 ms; only the rank-zero runtime reaches it.
    torch = lazy_import("torch")


MODELS_SUFFIX: Final = "/v1/models"
"""The rollout server's cheapest readiness route."""

SERVER_POLL_S: Final = 5.0
"""Seconds between readiness probes while a server starts."""

SERVER_STOP_TIMEOUT_S: Final = 60.0
"""Seconds to wait for a rollout server to stop before killing it."""

SERVER_LOG_LINES: Final = 20
"""Log lines quoted when a rollout server fails to come up."""


class LiveOpenMathRolloutData(OpenMathRolloutData):
    """Collect one openai/math update immediately before each learner update."""

    class Config(OpenMathRolloutData.Config):
        """Recorded-data settings plus the pinned live rollout runtime.

        The batch shape is the runner's: ``records_per_update`` and
        ``num_samples_per_prompt`` are derived from ``runner.num_prompts`` and
        ``runner.samples_per_prompt`` here, so set the shape on the runner.
        """

        runner: OpenMathRunnerConfig = field(default_factory=OpenMathRunnerConfig)
        """Harbor, rollout-server, and episode settings for this experiment."""

        export_dir: Path | str = "/export"
        """Where the trained policy is exported before each server restart.

        Relative to :attr:`base_dir`, the run directory.
        """

        @override
        def finalize(self) -> Self:
            """Derive the batch shape from the runner and resolve the export path."""
            self.records_per_update = self.runner.records_per_update
            self.num_samples_per_prompt = self.runner.samples_per_prompt
            self.export_dir = resolve_working_dir(self.base_dir, self.export_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Validate the live runtime without starting it.

        Startup waits for :meth:`train_dataloader`, after the learner has
        restored its checkpoint, so building the dataset never launches a server.

        Raises:
          ValueError: The dataset and its runner disagree on what an update is,
            or the rollout server has no devices of its own.

        """
        runner = config.runner
        if config.records_per_update != runner.records_per_update:
            raise ValueError(
                "Live openai/math records_per_update must equal the runner's "
                "num_prompts * samples_per_prompt.",
            )
        if config.num_samples_per_prompt != runner.samples_per_prompt:
            raise ValueError(
                "Live openai/math num_samples_per_prompt must equal the runner's "
                "samples_per_prompt.",
            )
        if config.filter_zero_std_samples != runner.filter_zero_std_samples:
            raise ValueError(
                "Live openai/math dataset.filter_zero_std_samples must match "
                "dataset.runner.filter_zero_std_samples.",
            )
        if config.active_sampling != runner.active_sampling:
            raise ValueError(
                "Live openai/math dataset.active_sampling must match "
                "dataset.runner.active_sampling.",
            )
        episode_tokens = runner.max_prompt_tokens + runner.response_length
        if config.pack_length < episode_tokens:
            raise ValueError(
                f"pack_length {config.pack_length} cannot hold one episode of "
                f"{episode_tokens} tokens (max_prompt_tokens + response_length).",
            )
        if not runner.rollout_gpus:
            raise ValueError(
                "Live openai/math needs runner.rollout_gpus: the rollout server "
                "runs beside the learner and must not share its devices.",
            )
        super().__init__(config)
        self.config = config
        self._runner: OpenMathRolloutRunner = runner.make()
        self._step: TrainStepProtocol | None = None
        self._writer: RolloutWriter | None = None
        self._server: subprocess.Popen[bytes] | None = None
        self._next_step = 0
        self._first_missing_step = 0
        self._task_cursor: Mapping[str, object] | None = None
        self._runtime_started = False
        self._synced_model_step: int | None = None

    @property
    def live_config(self) -> Config:
        """Narrow the inherited config to the live-only settings."""
        if not isinstance(self.config, LiveOpenMathRolloutData.Config):
            raise TypeError("Live openai/math data received a non-live config.")
        return self.config

    @override
    def bind_step(self, step: TrainStepProtocol) -> None:
        """Bind the pad id and the learner used for post-update export.

        Raises:
          ValueError: The learner scores at a different temperature than the
            rollout server samples at, so DPPO's ratio would compare two
            different distributions.

        """
        if (
            isinstance(step, TMaxDPPOTrainStep)
            and step.config.temperature != self._runner.config.temperature
        ):
            raise ValueError(
                f"runner.temperature {self._runner.config.temperature} must equal "
                f"the train step's temperature {step.config.temperature}.",
            )
        super().bind_step(step)
        self._step = step

    @override
    def train_dataloader(self) -> Iterator[dict[str, object]]:
        """Yield exactly one packed live update at a time.

        The generator resumes after the learner has completed the previous
        update. That resume point is the only place the next weight sync runs, so
        no episode can be collected from the policy the learner has already moved
        past.
        """
        if self._step is None:
            raise RuntimeError("Live openai/math data must be bound to a train step.")
        self._ensure_runtime()
        end = self._runner.config.start_step + self._runner.num_updates
        while self._next_step < end:
            if (
                self._next_step >= self._first_missing_step
                and self._synced_model_step != self._step.global_step
            ):
                self._sync_latest_policy()
            records = self._records_for_next_step()
            self._next_step += 1
            yield self.pack_update(records)
        return

    def _ensure_runtime(self) -> None:
        """Start the rollout runtime after TrainLoop has restored its checkpoint."""
        if self._runtime_started:
            return
        if self._step is None:
            raise RuntimeError("Live openai/math data must be bound to a train step.")
        restored_cursor = self._next_step
        self._next_step = self._step.global_step
        if restored_cursor not in (0, self._next_step):
            raise ValueError(
                "Live rollout cursor does not match the learner checkpoint: "
                f"dataset next_step={restored_cursor}, learner global_step="
                f"{self._next_step}.",
            )
        start_step = self._runner.config.start_step
        if self._next_step < start_step:
            raise ValueError(
                f"Learner resumed at step {self._next_step}, before the "
                f"configured rollout start_step {start_step}.",
            )
        if is_rank_zero():
            check_rollout_gpus(
                self._runner.config.rollout_gpus,
                learner_devices=_learner_processes_on_this_node(),
                available_devices=(
                    torch.cuda.device_count()
                    if torch.cuda.is_available()
                    # No CUDA here: vLLM reports that itself, more precisely.
                    else max(self._runner.config.rollout_gpus) + 1
                ),
            )
            self._start_rank_zero_runtime()
        frontier: list[object] = [
            self._first_missing_step if is_rank_zero() else None,
        ]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(frontier, src=0)
        self._first_missing_step = cast(int, frontier[0])
        # The policy must be served before the first episode samples it, on a
        # fresh run as well as a resumed one, so the initial sync is the first
        # thing that happens after the learner's checkpoint is restored.
        self._sync_latest_policy()
        self._runtime_started = True

    def _start_rank_zero_runtime(self) -> None:
        """Index the rollout shards already written and restore the task cursor."""
        runner = self._runner
        self._writer = RolloutWriter(runner.output_dir, runner.config.run_name)
        committed = self._writer.committed_live_steps(runner.records_per_update)
        cursor = self._next_step
        while cursor in committed:
            self._task_cursor = self._writer.live_cursor_after(
                cursor,
                runner.records_per_update,
            )
            cursor += 1
        later = sorted(step for step in committed if step > cursor)
        if later:
            raise ValueError(
                "Live rollout commit markers are not consecutive from learner "
                f"step {self._next_step}; first gap is {cursor}, but later "
                f"committed steps exist: {later}.",
            )
        self._first_missing_step = cursor
        self._writer.recover_uncommitted_live_tail(
            self._first_missing_step,
            runner.records_per_update,
        )
        if self._task_cursor is not None:
            self._runner.tasks.load_state_dict(self._task_cursor)

    def _records_for_next_step(self) -> list[RolloutRecord]:
        """Replay a complete staged step or collect and persist a fresh one."""
        if not is_rank_zero():
            payload: list[object] = [None]
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(payload, src=0)
            return _records_from_payload(cast(list[Mapping[str, object]], payload[0]))

        if self._writer is None:
            raise RuntimeError("Live openai/math runtime is not initialized.")
        runner = self._runner
        committed = self._writer.committed_live_steps(runner.records_per_update)
        if self._next_step in committed:
            raw = self._writer.records_for_step(
                self._next_step,
                runner.records_per_update,
            )
        else:
            if self._next_step != self._first_missing_step:
                raise ValueError(
                    f"Expected first missing live rollout step "
                    f"{self._first_missing_step}, got {self._next_step}.",
                )
            raw = records_for_update(self._runner, self._next_step)
            cursor = self._runner.tasks.state_dict()
            self._writer.write_live(
                self._next_step,
                raw,
                prompt_loader_state_after=cursor,
            )
            self._task_cursor = cursor
            self._first_missing_step += 1
        payload = [cast(object, [dict(record) for record in raw])]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        return _records_from_payload(raw)

    def _sync_latest_policy(self) -> None:
        """Stop the rollout server, export the trained policy, and serve it again."""
        if self._step is None:
            raise RuntimeError("Live openai/math data must be bound to a train step.")
        step = self._step.global_step
        if is_rank_zero():
            self._stop_rollout_server()
        # Collective under FSDP: every rank gathers its shards, rank zero writes.
        export_dir = Path(self.live_config.export_dir)
        written = cast(Any, self._step).export_checkpoint(export_dir)
        if is_rank_zero():
            checkpoint = Path(written) if written is not None else export_dir
            self._start_rollout_server(checkpoint, step=step)
        self._synced_model_step = step

    def _start_rollout_server(self, checkpoint: Path, *, step: int) -> None:
        """Serve an exported policy and wait until it answers.

        Args:
          checkpoint: The exported checkpoint to serve.
          step: The learner step it was exported at, for the log's name.

        Raises:
          RuntimeError: The server exited or never answered.

        """
        runner = self._runner.config
        working_dir = self._runner.working_dir
        log_path = working_dir / f"vllm_step_{step:08d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            self._server = subprocess.Popen(  # noqa: S603 -- The command is built from configuration, not from input.
                rollout_server_command(runner, checkpoint),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(working_dir),
                env=rollout_server_environment(runner, os.environ),
                # Its own process group, so stopping it also stops the engine
                # workers vLLM forks under the ``uvx`` launcher, and their GPU
                # memory is free before the next server starts.
                start_new_session=True,
            )
        server = self._server
        wait_for_server(
            runner.resolved_base_url,
            timeout_s=runner.server_startup_timeout_s,
            alive=server.poll,
            log_path=log_path,
        )

    def _stop_rollout_server(self) -> None:
        """Stop the rollout server this rank started, if any."""
        server = self._server
        self._server = None
        if server is None:
            return
        _signal_group(server, signal.SIGTERM)
        try:
            server.wait(timeout=SERVER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _signal_group(server, signal.SIGKILL)
            server.wait(timeout=SERVER_STOP_TIMEOUT_S)
        # The launcher can exit before its workers do; reap the rest of the group.
        _signal_group(server, signal.SIGKILL)

    @override
    def state_dict(self) -> dict[str, object]:
        """Return captured cursors without advancing or querying anything."""
        state = super().state_dict()
        state["next_step"] = self._next_step
        if self._task_cursor is not None:
            state["task_cursor"] = self._task_cursor
        return state

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the cursor after the learner checkpoint is loaded."""
        super().load_state_dict(state_dict)
        self._next_step = cast(int, state_dict.get("next_step", 0))
        cursor = state_dict.get("task_cursor")
        self._task_cursor = (
            cast(Mapping[str, object], cursor) if isinstance(cursor, Mapping) else None
        )

    def close(self) -> None:
        """Stop the rollout server if this rank started one."""
        self._stop_rollout_server()
        self._writer = None


def _signal_group(server: subprocess.Popen[bytes], signum: signal.Signals) -> None:
    """Signal a server's whole process group, if any of it is still running."""
    try:
        os.killpg(server.pid, signum)
    except ProcessLookupError:
        return


def _learner_processes_on_this_node() -> int:
    """Return how many learner processes share this node's GPUs.

    ``torch.distributed.run`` exports it as ``LOCAL_WORLD_SIZE``; a process
    started without it is the node's only learner.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def _records_from_payload(
    payload: Sequence[Mapping[str, object]],
) -> list[RolloutRecord]:
    """Validate a broadcast payload into the learner's record type."""
    return [record_from_payload(record) for record in payload]


def server_answers(base_url: str, *, timeout_s: float = 30.0) -> bool:
    """Report whether the rollout server's model list answers.

    Args:
      base_url: The server's root.
      timeout_s: Per-request timeout.

    Returns:
      ready: Whether the server answered.

    """
    url = base_url.rstrip("/") + MODELS_SUFFIX
    try:
        with urllib.request.urlopen(
            url,
            timeout=timeout_s,
        ) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def wait_for_server(
    base_url: str,
    *,
    timeout_s: float,
    alive: Callable[[], int | None],
    log_path: Path,
    poll_s: float = SERVER_POLL_S,
    answers: Callable[..., bool] = server_answers,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until the rollout server answers, or fail with its log.

    Args:
      base_url: The server's root.
      timeout_s: How long to wait before giving up.
      alive: Returns the server process's exit code, or ``None`` while it runs.
      log_path: The server's log, quoted when it dies or times out.
      poll_s: Seconds between probes.
      answers: The readiness probe; injectable so the wait is testable.
      sleep: The sleep used between probes; injectable for the same reason.

    Raises:
      RuntimeError: The server exited, or never answered in time.

    """
    deadline = time.monotonic() + timeout_s
    while True:
        exit_code = alive()
        if exit_code is not None:
            raise RuntimeError(
                f"Rollout server exited with code {exit_code}: {_log_tail(log_path)}",
            )
        if answers(base_url):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Rollout server did not answer at {base_url} within "
                f"{timeout_s:.0f}s: {_log_tail(log_path)}",
            )
        sleep(poll_s)


def _log_tail(path: Path, *, lines: int = SERVER_LOG_LINES) -> str:
    """Return the last lines of a log, or a note that it is missing."""
    if not path.exists():
        return f"{path} was not written"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])
