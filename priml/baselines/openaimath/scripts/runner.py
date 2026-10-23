"""Drive the openai/math Lean trials from PriML.

Harbor owns the sandbox, the shared snapshot, the collect hook, and the separate
verifier. The dataset owns the tasks and the grader. This module owns the part
neither does: which prompts an update draws, how many attempts each gets, where
the trials' output goes, and how a trial's verdict plus its token transcript
become one DPPO rollout record.

The boundary between collection and training is files, exactly as in the TMax
baseline. A trial writes its transcript into its own agent directory and its
verdict into its own verifier directory; this module reads those back and stages
them through the same writer the learner replays from. Nothing here holds a live
reference to Harbor or to the learner, so a collected update can be inspected,
replayed, and re-trained on.

One job per prompt, not one job per update: Harbor names trials by task and
attempt, so a job that holds a single prompt maps each of its trials to exactly
one sample index without guessing at Harbor's trial directory naming. An
update's prompt jobs run concurrently, because each one waits on sandboxes, not
on this process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Self, cast, override

import argparse
import json
import logging
import random
import secrets
import shutil
import subprocess
import threading

from configgle import Fig

from priml.baselines.openaimath.episode import (
    MAX_OBSERVATION_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TURNS,
    PER_TURN_MAX_TOKENS,
    RESPONSE_LENGTH,
    TRANSCRIPT_NAME,
)
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.paths import resolve_working_dir


logger = logging.getLogger(__name__)


AGENT_IMPORT_PATH: Final = "priml.baselines.openaimath.scripts.agent:OpenMathLeanAgent"
"""The Harbor agent this runner launches.

Harbor resolves this path inside the process that runs the learner, so the agent
can be a class in this repository rather than something installed into the
sandbox image.
"""

AGENT_ARGS: Final[tuple[str, ...]] = ("--agent-import-path", AGENT_IMPORT_PATH)
"""How Harbor is told to launch this baseline's Lean agent."""

INSTRUCTION_NAME: Final = "instruction.md"
"""The instruction file each staged task directory holds."""

REWARD_NAME: Final = "verifier/reward.txt"
"""Where Harbor records a trial's verdict, relative to the trial directory."""

GRADING_NAME: Final = "verifier/grading.json"
"""The dataset grader's own record of a verdict: its reason and timestamps."""

AGENT_LOG_DIR: Final = "agent"
"""The trial subdirectory Harbor mounts as the agent's own log directory."""

ATTEMPT_LOG_SUFFIX: Final = "attempts.jsonl"
"""The attempt log's name after the run name, in the output directory.

The rollout shards hold only the groups that train, so a group dropped for zero
reward spread, or rerun as unusable, has its verdicts and its Lean cost only
here: one line per graded attempt, kept or dropped, with what the line needs to
be joined back to its trial directory. It is the label log for a model that predicts
Lean's verdicts, and the sum of its seconds is the run's Lean cost.

A resumed run redraws the update it was interrupted in under the same job names,
so the interrupted jobs' lines stay (that Lean time was spent) while their trial
directories are replaced; the episode timestamps tell the two apart.
"""

SNAPSHOT_NAME: Final = "openai-math-lean-62f5d72497ea"
"""Daytona snapshot the pinned dataset revision's generator builds from ``image/``.

The name is a digest of the generator's image recipe, so it changes only when the
pinned revision does; ``prepare_data.py`` re-derives it and refuses a mismatch.
"""

VLLM_VERSION: Final = "0.19.1"
"""The vLLM release the rollout server runs.

Pinned because the sampler depends on vLLM-specific request fields
(``return_tokens_as_token_ids``, ``stop_token_ids``) and on the server keeping
the stop token in the sampled ids. The server runs under ``uvx`` in its own
environment, so its torch never has to agree with the learner's.
"""


@dataclass(frozen=True, slots=True)
class OpenMathTask:
    """One staged openai/math task: a name and the instruction that prompts it."""

    name: str
    """The task's directory name, which is also its Harbor task name."""

    instruction: str
    """The task's own instruction text, used as the episode's prompt."""

    @classmethod
    def load(cls, directory: Path) -> OpenMathTask:
        """Read one staged task directory.

        Args:
          directory: A task directory under the dataset's ``tasks/``.

        Returns:
          task: The task's name and instruction.

        Raises:
          FileNotFoundError: The directory holds no instruction.

        """
        return cls(
            name=directory.name,
            instruction=(directory / INSTRUCTION_NAME).read_text(encoding="utf-8"),
        )


class TaskStore:
    """Draw tasks for an update, deterministically and resumably.

    The order is a seeded shuffle of the staged task names, reshuffled into a new
    epoch when it runs out. An update's prompts are therefore reproducible from
    the seed and a cursor, without keeping the list of what was drawn, and a long
    campaign cycles through every task before repeating one. The seed is part of
    the cursor, so a run whose seed was drawn at random resumes the same order.
    """

    def __init__(
        self,
        tasks_dir: Path,
        *,
        seed: int | None,
        task_names: Sequence[str] = (),
    ) -> None:
        """Index the staged tasks and shuffle them into the first epoch.

        Args:
          tasks_dir: The directory of staged task directories.
          seed: Fixes the draw order; ``None`` draws one from OS entropy.
          task_names: An explicit subset to draw from, in any order. Empty draws
            from every staged task.

        Raises:
          FileNotFoundError: No staged tasks were found.

        """
        self.tasks_dir = Path(tasks_dir)
        if task_names:
            names = list(task_names)
        elif self.tasks_dir.is_dir():
            names = sorted(
                path.name for path in self.tasks_dir.iterdir() if path.is_dir()
            )
        else:
            names: list[str] = []
        if not names:
            raise FileNotFoundError(
                f"No staged openai/math tasks under {self.tasks_dir}.",
            )
        missing = [name for name in names if not (self.tasks_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Staged tasks are missing: {missing[:8]}.")
        self.seed = secrets.randbits(63) if seed is None else seed
        self.names = tuple(names)
        self._epoch = 0
        self._position = 0
        self._order = self._shuffled(0)

    def _shuffled(self, epoch: int) -> list[str]:
        """Return one epoch's draw order."""
        order = list(self.names)
        random.Random(f"{self.seed}:{epoch}").shuffle(order)  # noqa: S311 -- A seeded draw order, not a secret.
        return order

    def draw(self) -> OpenMathTask:
        """Draw the next task, starting a new epoch when the order runs out.

        Returns:
          task: The drawn task.

        """
        if self._position >= len(self._order):
            self._epoch += 1
            self._order = self._shuffled(self._epoch)
            self._position = 0
        name = self._order[self._position]
        self._position += 1
        return OpenMathTask.load(self.tasks_dir / name)

    def state_dict(self) -> dict[str, object]:
        """Return the seed and cursor, for a resumed run to restore."""
        return {"seed": self.seed, "epoch": self._epoch, "position": self._position}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a cursor written by :meth:`state_dict`."""
        if "seed" in state_dict:
            self.seed = cast(int, state_dict["seed"])
        self._epoch = cast(int, state_dict.get("epoch", 0))
        self._position = cast(int, state_dict.get("position", 0))
        self._order = self._shuffled(self._epoch)
        if self._position > len(self._order):
            raise ValueError(
                f"Restored task cursor {self._position} is past the end of "
                f"epoch {self._epoch}.",
            )


class OpenMathRunnerConfig(Fig["OpenMathRolloutRunner"]):
    """Rollout settings for the openai/math baseline.

    The sampling defaults are the TMax baseline's 4B recipe's. The token budgets
    are this substrate's (see :mod:`priml.baselines.openaimath.episode`): a Lean
    prompt carries a whole theorem statement rather than a shell task, and a
    proof attempt is a single long trajectory rather than a sequence of short
    tool calls.

    Fields defaulting to ``None`` are per-run settings. Inside an experiment the
    training loop fills each one it owns from its own value, and an explicit
    value set here wins; a standalone tool sets them itself. The runner refuses
    to start while any is still unset.
    """

    base_dir: Path | str | None = None
    """Root supplied by the experiment."""

    working_dir: Path | str | None = None
    """Run directory: rollout-server logs, and the root of :attr:`output_dir`.

    ``None`` inherits the experiment's working directory.
    """

    output_dir: Path | str | None = None
    """Where rollout shards and Harbor job directories are written.

    Relative to :attr:`working_dir`. ``None`` inherits the live dataset's
    ``working_dir``.
    """

    tasks_dir: Path | str = Path("/datasets/openaimath/openai-math/lemmas/train")
    """The directory of task directories to draw from.

    The default is the training lemma tasks ``prepare_data.py`` cuts from the
    training families' modules (see :mod:`lemma_tasks`). The published
    challenges are in the dataset's ``tasks/``, which evaluation reads.
    """

    snapshot: str = SNAPSHOT_NAME
    """Daytona snapshot template holding Lean, Mathlib, and Comparator.

    Built once from the dataset's ``generator/`` image, which takes about an hour
    and 9 GB. Harbor's own auto-snapshot gives up after ten minutes, which is why
    the name is pinned here rather than something this runner builds.
    """

    environment: str = "daytona"
    """Harbor's environment provider."""

    harbor_executable: str = "harbor"
    """The Harbor CLI to invoke.

    It must run in an environment where ``priml`` is importable, because Harbor
    imports :data:`AGENT_IMPORT_PATH` in its own process.
    """

    run_name: str = "openaimath"
    """Prefix for shard, marker, and job directory names."""

    checkpoint: Path | str | None = None
    """The policy checkpoint training starts from; ``None`` inherits the model path.

    The agent loads its tokenizer from here unless :attr:`tokenizer_path` is set.
    """

    tokenizer_path: Path | str | None = None
    """Where the agent loads the policy's tokenizer; ``None`` uses the checkpoint."""

    vllm_base_url: str = ""
    """The learner's rollout server; empty derives it from host and port.

    Agents reach this from the host, not from inside a sandbox: the sandbox has
    no network while an episode runs, which is one of the reasons the dataset's
    grader cannot be reached from inside a proof.
    """

    served_model: str = "policy"
    """The model name the rollout server reports."""

    vllm_command: tuple[str, ...] = ("uvx", f"vllm=={VLLM_VERSION}")
    """How the rollout server's ``vllm`` CLI is invoked, before ``serve``.

    The default runs the pinned release in its own ``uvx`` environment; a node
    with vLLM already installed can name its executable instead.
    """

    vllm_host: str = "127.0.0.1"
    """Host the rollout server binds, as agents reach it."""

    vllm_port: int = 8000
    """Port the rollout server binds."""

    rollout_gpus: tuple[int, ...] = ()
    """CUDA devices the rollout server runs on, as the learner's process numbers them.

    They must be disjoint from the learner's own devices on that node (its local
    ranks), because the learner keeps its weights and optimizer state resident
    while the server samples. Must hold exactly ``tensor_parallel_size *
    data_parallel_size`` devices; empty leaves the server on whatever devices the
    learner's environment makes visible, which only a standalone tool should do.
    """

    tensor_parallel_size: int = 1
    """GPUs one rollout engine spans."""

    data_parallel_size: int = 1
    """Rollout engines the server load-balances requests across."""

    gpu_memory_utilization: float = 0.9
    """Fraction of each rollout GPU's memory the server may use."""

    vllm_extra_args: tuple[str, ...] = ()
    """Extra rollout-server arguments, passed verbatim."""

    server_startup_timeout_s: float = 3_600.0
    """How long a restarted rollout server may take to answer."""

    num_updates: int | None = None
    """Updates to collect, counting from :attr:`start_step`.

    ``None`` inherits the learner's ``train_budget_steps``.
    """

    start_step: int = 0
    """First update to collect."""

    num_prompts: int = 8
    """Prompt groups per update; the TMax 4B recipe draws eight."""

    samples_per_prompt: int = 32
    """Attempts per prompt; the TMax 4B recipe draws thirty-two."""

    n_concurrent: int = 32
    """Trials Harbor runs at once, per prompt job."""

    concurrent_prompts: int = 8
    """Prompt jobs in flight at once; at most ``n_concurrent`` sandboxes each.

    An update's groups are independent, so running them one after another only
    multiplies its wall clock. The product with :attr:`n_concurrent` is the
    sandbox count the Daytona account must allow.
    """

    max_group_attempts: int = 3
    """Times to redraw a prompt whose group came back unusable.

    A group is unusable when a trial produced no verdict or no transcript, which
    the dataset says happens when grading itself fails. Redrawing is not the same
    as scoring zero: a failed grading run says nothing about the proof.
    """

    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    """Prompt budget, passed to the agent."""

    response_length: int = RESPONSE_LENGTH
    """Per-episode response budget, policy and sandbox tokens, passed to the agent."""

    per_turn_max_tokens: int = PER_TURN_MAX_TOKENS
    """Per-turn token budget, passed to the agent."""

    max_turns: int = MAX_TURNS
    """Turns per episode, passed to the agent."""

    max_observation_tokens: int = MAX_OBSERVATION_TOKENS
    """Per-observation token budget, passed to the agent."""

    temperature: float = 1.0
    """Sampling temperature, the TMax 4B recipe's value.

    Must equal the learner's teacher-forcing temperature, or the behavior
    logprobs and the scored logprobs come from different distributions; the
    live dataset refuses a mismatch when it binds the train step.
    """

    top_p: float = 1.0
    """Nucleus cutoff, the TMax 4B recipe's value."""

    seed: int | None = None
    """Seed of the task draw order; ``None`` inherits the experiment's seed.

    When both are ``None`` the order is seeded from OS entropy, and the seed is
    saved with the task cursor, so a resumed run continues the same order.
    """

    sample_seed: int | None = None
    """Sampling seed, or ``None`` to let the server sample freely.

    Left unset on purpose. One seed shared by all thirty-two attempts of a prompt
    would make them identical, and a group with no variation carries no advantage.
    """

    filter_zero_std_samples: bool = True
    """Drop a prompt group whose rewards are all equal, as the TMax recipe does.

    Such a group has zero centered advantage, so it carries no gradient. Lemma
    tasks range from one-line to out of reach, so most draws should be mixed
    for a 4B policy; :attr:`max_draws_per_update` bounds the cost when not.
    """

    active_sampling: bool = True
    """Refill a slot whose group was dropped, from a fresh draw, as TMax does.

    Only meaningful with :attr:`filter_zero_std_samples`, which the constructor
    enforces: a dropped group leaves a slot empty otherwise, and an update with
    fewer records than it promised would change the objective's denominator
    silently.
    """

    max_draws_per_update: int = 64
    """Most prompt groups one update may draw, kept or dropped.

    Eight times the update's eight slots. Past it the update fails rather than
    keep spending sandboxes: fewer than one draw in eight producing a mixed
    group means the task pool no longer matches the policy, which is a finding
    to report, not something to wait out.
    """

    task_names: tuple[str, ...] = ()
    """An explicit task subset to draw from; empty draws from every staged task."""

    job_timeout_s: float = 86_400.0
    """Wall-clock bound for one prompt's Harbor job."""

    @override
    def finalize(self) -> Self:
        """Resolve every set path against the experiment's directories."""
        if self.working_dir is not None:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.output_dir is not None:
                self.output_dir = resolve_working_dir(
                    self.working_dir,
                    self.output_dir,
                )
        self.tasks_dir = resolve_working_dir(self.base_dir, self.tasks_dir)
        if self.checkpoint is not None:
            self.checkpoint = resolve_working_dir(self.base_dir, self.checkpoint)
        if self.tokenizer_path is not None:
            self.tokenizer_path = resolve_working_dir(
                self.base_dir,
                self.tokenizer_path,
            )
        return super().finalize()

    @property
    def records_per_update(self) -> int:
        """Return the update's batch size: prompts times samples per prompt."""
        return self.num_prompts * self.samples_per_prompt

    @property
    def resolved_tokenizer_path(self) -> Path:
        """Return where the agent loads the policy's tokenizer.

        Raises:
          ValueError: Neither a tokenizer path nor a checkpoint is set.

        """
        path = self.tokenizer_path or self.checkpoint
        if path is None:
            raise ValueError("Set tokenizer_path or checkpoint for the agent.")
        return Path(path)

    @property
    def resolved_base_url(self) -> str:
        """Return the rollout server's base URL."""
        return self.vllm_base_url or f"http://{self.vllm_host}:{self.vllm_port}"


class OpenMathRolloutRunner:
    """Own the finalized, validated rollout configuration."""

    Config = OpenMathRunnerConfig

    def __init__(self, config: OpenMathRunnerConfig) -> None:
        """Keep Configgle's finalized copy and reject an unusable configuration.

        Raises:
          ValueError: A per-run setting is unset, or the settings cannot produce
            a well-formed update.

        """
        self.config = config
        unset = [
            name
            for name, value in (
                ("working_dir", config.working_dir),
                ("output_dir", config.output_dir),
                ("num_updates", config.num_updates),
            )
            if value is None
        ]
        if unset:
            raise ValueError(
                f"Rollout runner settings {unset} are unset; an experiment fills "
                "them from its own values, and a standalone tool must set them.",
            )
        self.working_dir = Path(cast("Path | str", config.working_dir))
        self.output_dir = Path(cast("Path | str", config.output_dir))
        self.num_updates = cast(int, config.num_updates)
        if self.num_updates <= 0:
            raise ValueError("num_updates must be positive.")
        if config.start_step < 0:
            raise ValueError("start_step must be non-negative.")
        if config.num_prompts <= 0:
            raise ValueError("num_prompts must be positive.")
        if config.samples_per_prompt <= 1:
            raise ValueError("samples_per_prompt must be greater than one.")
        if config.n_concurrent <= 0 or config.concurrent_prompts <= 0:
            raise ValueError("n_concurrent and concurrent_prompts must be positive.")
        if config.max_group_attempts <= 0:
            raise ValueError("max_group_attempts must be positive.")
        if config.max_draws_per_update < config.num_prompts:
            raise ValueError("max_draws_per_update must cover num_prompts.")
        if not config.snapshot:
            raise ValueError(
                "snapshot must name the dataset's Daytona snapshot; build it "
                "once with the dataset's generator, which Harbor's own "
                "auto-snapshot timeout cannot cover.",
            )
        if config.filter_zero_std_samples and not config.active_sampling:
            raise ValueError(
                "filter_zero_std_samples needs active_sampling: a dropped group "
                "would otherwise leave its slot empty.",
            )
        engines = config.tensor_parallel_size * config.data_parallel_size
        if config.tensor_parallel_size <= 0 or config.data_parallel_size <= 0:
            raise ValueError("Rollout parallel sizes must be positive.")
        if config.rollout_gpus and (
            len(config.rollout_gpus) != engines
            or len(set(config.rollout_gpus)) != len(config.rollout_gpus)
            or min(config.rollout_gpus) < 0
        ):
            raise ValueError(
                f"rollout_gpus {config.rollout_gpus} must name {engines} distinct "
                "devices, tensor_parallel_size times data_parallel_size.",
            )
        self.tasks = TaskStore(
            Path(config.tasks_dir),
            seed=config.seed,
            task_names=config.task_names,
        )
        self._attempt_log_lock = threading.Lock()

    @property
    def records_per_update(self) -> int:
        """Return the update's batch size: prompts times samples per prompt."""
        return self.config.records_per_update

    def job_dir(self, step: int, slot: int, draw: int, attempt: int) -> Path:
        """Return one prompt's Harbor job directory.

        Every draw gets its own directory, so a dropped group's trials stay on
        disk after its slot is refilled: they are graded attempts like any other.

        Args:
          step: The learner step the update trains.
          slot: The prompt's slot in the update.
          draw: How many earlier draws this slot made in this update.
          attempt: Which try of this draw, counting reruns of an unusable group.

        Returns:
          directory: The job directory, inside the run's output directory.

        """
        directory = self.output_dir / (
            f"jobs/{self.config.run_name}_step_{step:08d}"
            f"_slot_{slot:02d}_draw_{draw:02d}_try_{attempt:02d}"
        )
        if not directory.exists():
            return directory
        # An interrupted update may be recollected. Its earlier attempts are
        # still named by the append-only log, so never reuse their directories.
        retry = 1
        while (
            candidate := directory.with_name(
                f"{directory.name}_retry_{retry:02d}",
            )
        ).exists():
            retry += 1
        return candidate

    @property
    def attempt_log_path(self) -> Path:
        """Return the attempt log: one line per graded attempt, kept or not."""
        return self.output_dir / f"{self.config.run_name}_{ATTEMPT_LOG_SUFFIX}"

    def log_attempts(self, lines: Sequence[Mapping[str, object]]) -> None:
        """Append one job's attempts to the attempt log.

        Called from the update's worker threads, so the append is serialized.
        """
        if not lines:
            return
        text = "".join(json.dumps(dict(line)) + "\n" for line in lines)
        with self._attempt_log_lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.attempt_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def harbor_command(self, task_name: str, job_dir: Path) -> list[str]:
        """Build the Harbor CLI invocation for one prompt's attempts.

        One prompt per job, with ``-k`` attempts: Harbor runs that many trials of
        the same task, which is exactly one prompt group. The agent is told where
        the policy lives and what its budgets are through ``--ak``, because the
        agent runs in Harbor's process and cannot read this configuration.

        Args:
          task_name: The staged task to run.
          job_dir: Where Harbor writes the job's trials.

        Returns:
          command: The argument vector, ready for :mod:`subprocess`.

        """
        return harbor_command(self.config, task_name, job_dir)

    def run_job(self, task_name: str, job_dir: Path) -> list[TrialOutcome]:
        """Run one prompt's Harbor job and read its trials back.

        Args:
          task_name: The staged task to run.
          job_dir: Where Harbor writes the job's trials.

        Returns:
          trials: One outcome per attempt.

        Raises:
          RuntimeError: Harbor exited nonzero, which means the job did not run
            rather than that a proof was rejected.

        """
        return run_job(self.config, task_name, job_dir)


def agent_options(config: OpenMathRunnerConfig) -> list[str]:
    """Return the agent options Harbor forwards to the Lean agent.

    Args:
      config: The finalized rollout settings.

    Returns:
      options: ``key=value`` pairs, one per ``--ak`` flag.

    """
    options = [
        f"base_url={config.resolved_base_url}",
        f"model={config.served_model}",
        f"tokenizer_path={config.resolved_tokenizer_path}",
        f"temperature={config.temperature}",
        f"top_p={config.top_p}",
        f"max_turns={config.max_turns}",
        f"max_prompt_tokens={config.max_prompt_tokens}",
        f"response_length={config.response_length}",
        f"per_turn_max_tokens={config.per_turn_max_tokens}",
        f"max_observation_tokens={config.max_observation_tokens}",
    ]
    if config.sample_seed is not None:
        options.append(f"seed={config.sample_seed}")
    return options


def harbor_command(
    config: OpenMathRunnerConfig,
    task_name: str,
    job_dir: Path,
    *,
    agent_args: Sequence[str] | None = None,
    options: Sequence[str] | None = None,
    attempts: int | None = None,
) -> list[str]:
    """Build a Harbor CLI invocation for one task's attempts.

    See :meth:`OpenMathRolloutRunner.harbor_command` for why a job holds one
    prompt rather than a whole update. The overrides exist for evaluation, which
    runs the same tasks through Harbor's own oracle agent, with no policy server
    to point at and no sampling options to pass.

    Args:
      config: The finalized rollout settings.
      task_name: The staged task to run.
      job_dir: Where Harbor writes the job's trials.
      agent_args: How to select the agent; ``None`` uses this baseline's agent.
      options: The ``--ak`` values; ``None`` uses :func:`agent_options`.
      attempts: Trials to run; ``None`` uses the configured samples per prompt.

    Returns:
      command: The argument vector, ready for :mod:`subprocess`.

    """
    command = [
        config.harbor_executable,
        "run",
        "-p",
        str(config.tasks_dir),
        "-e",
        config.environment,
        "--ek",
        f"snapshot_template_name={config.snapshot}",
        *(AGENT_ARGS if agent_args is None else agent_args),
    ]
    for option in agent_options(config) if options is None else options:
        command.extend(["--ak", option])
    command.extend(
        [
            "-k",
            str(config.samples_per_prompt if attempts is None else attempts),
            "-n",
            str(config.n_concurrent),
            "-o",
            str(job_dir),
            "-i",
            task_name,
        ],
    )
    return command


def run_job(
    config: OpenMathRunnerConfig,
    task_name: str,
    job_dir: Path,
    *,
    agent_args: Sequence[str] | None = None,
    options: Sequence[str] | None = None,
    attempts: int | None = None,
    kind: str = "step",
) -> list[TrialOutcome]:
    """Run one task's Harbor job in a fresh directory and read its trials back.

    Args:
      config: The finalized rollout settings.
      task_name: The staged task to run.
      job_dir: Where Harbor writes the job's trials.
      agent_args: How to select the agent; ``None`` uses this baseline's agent.
      options: The ``--ak`` values; ``None`` uses :func:`agent_options`.
      attempts: Trials to run; ``None`` uses the configured samples per prompt.
      kind: The job directory's kind, checked by :func:`reset_job_dir`.

    Returns:
      trials: One outcome per attempt.

    Raises:
      RuntimeError: Harbor exited nonzero, which means the job did not run
        rather than that a proof was rejected.

    """
    reset_job_dir(job_dir, config.run_name, kind=kind)
    command = harbor_command(
        config,
        task_name,
        job_dir,
        agent_args=agent_args,
        options=options,
        attempts=attempts,
    )
    logger.info("harbor: %s -> %s", task_name, job_dir)
    completed = subprocess.run(  # noqa: S603 -- The command is built above, not interpolated from input.
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=config.job_timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Harbor job for task {task_name} exited "
            f"{completed.returncode}: {completed.stderr[-2_000:]}",
        )
    return read_job(job_dir)


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """One trial's verdict and transcript, as read back from disk."""

    trial: str
    """The trial directory's name, which orders the group deterministically."""

    reward: float | None
    """The verifier's verdict, or ``None`` when grading did not report one."""

    record: Mapping[str, object] | None
    """The episode's flattened transcript, or ``None`` when the agent wrote none."""

    verdict: str | None = None
    """The grader's reason for its verdict, or ``None`` when it wrote none."""

    grade_seconds: float | None = None
    """Seconds the grader took, from its own timestamps."""

    grade_started_at: float | None = None
    """When the grader started, as its ``grading.json`` records it."""

    grade_finished_at: float | None = None
    """When the grader finished, as its ``grading.json`` records it."""


def read_job(job_dir: Path) -> list[TrialOutcome]:
    """Read one Harbor job's trials.

    Trials are found by their verdict file and by their transcript, so a trial
    that crashed before grading still appears and can be reported as a failed
    attempt instead of silently shrinking the group.

    Args:
      job_dir: The job directory Harbor wrote.

    Returns:
      trials: One outcome per trial found, ordered by trial name.

    Raises:
      FileNotFoundError: The job directory holds no trials at all.

    """
    directory = Path(job_dir)
    trial_dirs = {
        path.parent.parent
        for pattern in (REWARD_NAME, f"{AGENT_LOG_DIR}/{TRANSCRIPT_NAME}")
        for path in directory.rglob(pattern)
    }
    if not trial_dirs:
        raise FileNotFoundError(f"No trials under {directory}.")
    return [_read_trial(path) for path in sorted(trial_dirs)]


def _read_trial(trial_dir: Path) -> TrialOutcome:
    """Read one trial's verdict and transcript."""
    reward_path = trial_dir / REWARD_NAME
    reward: float | None = None
    if reward_path.exists():
        text = reward_path.read_text(encoding="utf-8").strip()
        reward = float(text) if text else None
    verdict, started, finished = _read_grading(trial_dir / GRADING_NAME)
    return TrialOutcome(
        trial=trial_dir.name,
        reward=reward,
        record=_read_transcript(trial_dir / AGENT_LOG_DIR / TRANSCRIPT_NAME),
        verdict=verdict,
        grade_seconds=(
            finished - started if started is not None and finished is not None else None
        ),
        grade_started_at=started,
        grade_finished_at=finished,
    )


def _read_grading(path: Path) -> tuple[str | None, float | None, float | None]:
    """Read the grader's reason and timestamps out of its ``grading.json``.

    The file is the dataset grader's, not this baseline's, so a missing or
    differently shaped one yields ``None`` rather than failing the group: the
    verdict that trains is ``reward.txt``.
    """
    if not path.exists():
        return None, None, None
    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(payload, dict):
        return None, None, None
    grading = cast(dict[str, object], payload)
    reason = grading.get("reason")
    started, finished = (
        float(value) if isinstance(value, (int, float)) else None
        for value in (grading.get("started_at"), grading.get("finished_at"))
    )
    return (str(reason) if reason is not None else None), started, finished


def _read_transcript(path: Path) -> Mapping[str, object] | None:
    """Read an episode transcript, keeping the last line when a trial resumed.

    Args:
      path: The transcript file inside a trial's agent directory.

    Returns:
      record: The decoded transcript, or ``None`` when the file is absent.

    """
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return None
    return cast(Mapping[str, object], json.loads(lines[-1]))


def records_for_update(
    runner: OpenMathRolloutRunner,
    step: int,
) -> list[dict[str, object]]:
    """Collect one update's rollouts and return them as serialized records.

    Args:
      runner: The finalized rollout runner.
      step: The learner step these rollouts will train.

    Returns:
      records: ``num_prompts * samples_per_prompt`` records, grouped in prompt
        order, ready for the rollout writer and then for the learner.

    Groups are collected concurrently, up to ``concurrent_prompts`` jobs at a
    time, but drawn in slot order on this thread. A slot whose group is dropped
    is redrawn in the next round, again in slot order, so the tasks an update
    draws depend only on the seed, the cursor, and which groups were dropped,
    never on which job happened to finish first.

    Raises:
      RuntimeError: Filling the update would take more than
        ``max_draws_per_update`` draws.

    """
    config = runner.config
    groups: dict[int, list[dict[str, object]]] = {}
    empty = list(range(config.num_prompts))
    draws = 0
    # Every empty slot is redrawn each round, so the round is each slot's draw.
    round_index = 0
    with ThreadPoolExecutor(
        max_workers=min(config.concurrent_prompts, config.num_prompts),
    ) as pool:
        while empty:
            if draws + len(empty) > config.max_draws_per_update:
                raise RuntimeError(
                    f"Step {step}: {len(empty)} of {config.num_prompts} slots are "
                    f"still empty after {draws} draws, and the cap is "
                    f"{config.max_draws_per_update}. Almost every group had zero "
                    "reward spread: the tasks are all solved or all out of reach "
                    "for this policy.",
                )
            draws += len(empty)
            drawn = [(slot, runner.tasks.draw()) for slot in empty]
            futures = [
                pool.submit(
                    collect_group,
                    runner,
                    step,
                    slot,
                    task,
                    draw=round_index,
                )
                for slot, task in drawn
            ]
            for (slot, _), future in zip(drawn, futures, strict=True):
                group = future.result()
                if group is not None:
                    groups[slot] = group
            empty = [slot for slot, _ in drawn if slot not in groups]
            round_index += 1
    return [record for slot in sorted(groups) for record in groups[slot]]


def collect_group(
    runner: OpenMathRolloutRunner,
    step: int,
    slot: int,
    task: OpenMathTask,
    *,
    draw: int = 0,
) -> list[dict[str, object]] | None:
    """Run one prompt's attempts and turn them into a prompt group.

    Every try's attempts go to the attempt log, whatever becomes of the group.

    Args:
      runner: The finalized rollout runner.
      step: The learner step these rollouts will train.
      slot: The prompt's slot in the update.
      task: The drawn task.
      draw: How many earlier draws this slot made in this update.

    Returns:
      records: The group's records, or ``None`` when the group was dropped and
        the caller should draw again.

    Raises:
      RuntimeError: Every attempt produced an unusable group.

    """
    config = runner.config
    for attempt in range(config.max_group_attempts):
        job_dir = runner.job_dir(step, slot, draw, attempt)
        outcomes = runner.run_job(task.name, job_dir)
        group = _group_records(
            outcomes,
            task=task.name,
            step=step,
            slot=slot,
            size=config.samples_per_prompt,
        )
        if group is None:
            fate = "unusable"
        elif config.filter_zero_std_samples and _zero_spread(group):
            fate = "zero_spread"
        else:
            fate = "kept"
        runner.log_attempts(
            [
                attempt_line(
                    outcome,
                    task=task.name,
                    step=step,
                    slot=slot,
                    draw=draw,
                    attempt=attempt,
                    job=job_dir.name,
                    group=fate,
                )
                for outcome in outcomes
            ],
        )
        if group is None:
            logger.warning(
                "step %d slot %d: discarding the group for %s and rerunning it "
                "(attempt %d of %d)",
                step,
                slot,
                task.name,
                attempt + 1,
                config.max_group_attempts,
            )
            continue
        if fate == "zero_spread":
            logger.info(
                "step %d slot %d: dropping the zero-spread group for %s",
                step,
                slot,
                task.name,
            )
            return None
        return group
    raise RuntimeError(
        f"Step {step} slot {slot} produced no usable group for task "
        f"{task.name} in {config.max_group_attempts} attempts.",
    )


def attempt_line(
    outcome: TrialOutcome,
    *,
    task: str,
    step: int,
    slot: int,
    draw: int,
    attempt: int,
    job: str,
    group: str,
) -> dict[str, object]:
    """Build one attempt log line: a graded attempt's outcome, cost, and place.

    The tokens stay in the trial directory, named by ``job`` and ``trial``;
    the line carries only what joins, counts, and labels the attempt. Times are
    seconds since the epoch.

    Args:
      outcome: The attempt as read back from its trial directory.
      task: The task the group attempted.
      step: The learner step the update trains.
      slot: The prompt's slot in the update.
      draw: How many earlier draws this slot made in this update.
      attempt: Which try of this draw.
      job: The job directory's name, under the output directory's ``jobs/``.
      group: What became of the group: ``"kept"`` for training,
        ``"zero_spread"`` when dropped for equal rewards, or ``"unusable"``
        when a trial had no verdict or no transcript.

    Returns:
      line: A JSON-serializable record.

    """
    record: Mapping[str, object] = outcome.record or {}
    tool_mask = record.get("tool_mask")
    prompt_tokens = record.get("prompt_tokens")
    return {
        "step": step,
        "slot": slot,
        "draw": draw,
        "try": attempt,
        "task": task,
        "job": job,
        "trial": outcome.trial,
        "group": group,
        "reward": outcome.reward,
        "verdict": outcome.verdict,
        "has_transcript": outcome.record is not None,
        "finish_reason": record.get("finish_reason"),
        "submitted": record.get("submitted"),
        "turns": record.get("turns"),
        "prompt_tokens": (
            len(cast(Sequence[object], prompt_tokens))
            if isinstance(prompt_tokens, list)
            else None
        ),
        "policy_tokens": (
            sum(cast(Sequence[int], tool_mask)) if isinstance(tool_mask, list) else None
        ),
        "sandbox_seconds": record.get("sandbox_seconds"),
        "grade_seconds": outcome.grade_seconds,
        "episode_started_at": record.get("started_at"),
        "episode_finished_at": record.get("finished_at"),
        "grade_started_at": outcome.grade_started_at,
        "grade_finished_at": outcome.grade_finished_at,
        "submission_sha256": record.get("submission_sha256"),
    }


def _group_records(
    outcomes: Sequence[TrialOutcome],
    *,
    task: str,
    step: int,
    slot: int,
    size: int,
) -> list[dict[str, object]] | None:
    """Assemble one prompt group's records, or ``None`` when it is unusable.

    Besides what the learner reads, each record keeps the Lean side of its
    attempt: the task, the grader's reason, and the seconds spent in sandbox
    commands and in grading. The learner ignores these; they are the label log
    a verifier trained on Lean's own verdicts would start from, and the Lean
    half of the run's cost.

    Args:
      outcomes: The attempts' outcomes.
      task: The task the group attempted.
      step: The learner step these rollouts will train.
      slot: The prompt's slot in the update.
      size: Attempts a complete group holds.

    Returns:
      records: One record per attempt, ordered by trial name, or ``None`` when
        the group is short, ungraded, or missing transcripts.

    """
    if len(outcomes) != size:
        logger.warning(
            "step %d slot %d: %d trials for a group of %d",
            step,
            slot,
            len(outcomes),
            size,
        )
        return None
    ungraded = [outcome.trial for outcome in outcomes if outcome.reward is None]
    if ungraded:
        logger.warning(
            "step %d slot %d: no verdict for %s",
            step,
            slot,
            ungraded[:4],
        )
        return None
    missing = [outcome.trial for outcome in outcomes if outcome.record is None]
    if missing:
        logger.warning(
            "step %d slot %d: no transcript for %s",
            step,
            slot,
            missing[:4],
        )
        return None
    records: list[dict[str, object]] = []
    for index, outcome in enumerate(sorted(outcomes, key=lambda item: item.trial)):
        record = outcome.record
        if (
            record is None or outcome.reward is None
        ):  # Narrowed above; kept for the type.
            return None
        _require_record_shape(record, trial=outcome.trial)
        records.append(
            {
                "step": step,
                # Unique within the update, as TMax numbers it: the rollout
                # writer keys records by (step, sample_idx).
                "sample_idx": slot * size + index,
                "prompt_idx": slot,
                "prompt_tokens": record["prompt_tokens"],
                "response_tokens": record["response_tokens"],
                "logprobs": record["logprobs"],
                "reward": float(outcome.reward),
                "finish_reason": record["finish_reason"],
                "tool_mask": record["tool_mask"],
                "task": task,
                "verdict": outcome.verdict,
                "sandbox_seconds": record.get("sandbox_seconds"),
                "grade_seconds": outcome.grade_seconds,
            },
        )
    return records


def _require_record_shape(record: Mapping[str, object], *, trial: str) -> None:
    """Reject a transcript whose token arrays do not line up.

    Args:
      record: One decoded transcript.
      trial: The trial's name, for the error message.

    Raises:
      ValueError: A field is missing or the arrays disagree, which is a bug in
        the agent rather than a failed sandbox.

    """
    missing = [
        name
        for name in (
            "prompt_tokens",
            "response_tokens",
            "logprobs",
            "tool_mask",
            "finish_reason",
        )
        if name not in record
    ]
    if missing:
        raise ValueError(f"Trial {trial}: transcript is missing {missing}.")
    response = cast(Sequence[object], record["response_tokens"])
    for name in ("logprobs", "tool_mask"):
        if len(cast(Sequence[object], record[name])) != len(response):
            raise ValueError(
                f"Trial {trial}: {name} does not match "
                f"{len(response)} response tokens.",
            )


def _zero_spread(records: Sequence[Mapping[str, object]]) -> bool:
    """Report whether every reward in a group is the same."""
    first = records[0]["reward"]
    return all(record["reward"] == first for record in records[1:])


def reset_job_dir(job_dir: Path, run_name: str, *, kind: str = "step") -> None:
    """Remove a previous job directory before a fresh run writes into it.

    Without this, a redraw would read the earlier attempt's trials alongside the
    new ones and assemble a group out of two different draws.

    Args:
      job_dir: The job directory this run owns.
      run_name: The run's name, required to appear in the directory's name.
      kind: The job directory's middle word, ``step`` for training draws and
        ``eval`` for evaluation runs.

    Raises:
      ValueError: The path is not one of this run's job directories, so removing
        it would not be scoped to this runner's own output.

    """
    directory = Path(job_dir)
    if not directory.name.startswith(f"{run_name}_{kind}_"):
        raise ValueError(f"Refusing to reset a foreign job directory: {directory}.")
    if directory.exists():
        shutil.rmtree(directory)


def rollout_server_command(
    config: OpenMathRunnerConfig,
    checkpoint: Path,
) -> list[str]:
    """Build the rollout server's command line for one exported policy.

    The server is started from an exported checkpoint rather than handed tensors
    in place, so the policy it serves is the same artifact the evaluation serves
    and the same one a checkpoint on disk holds. The context bound is the
    episode's own: a whole prompt plus a whole response must fit, or the server
    would refuse a turn the agent is already half way through.

    Args:
      config: The finalized rollout settings.
      checkpoint: The exported Hugging Face checkpoint to serve.

    Returns:
      command: The argument vector, ready for :mod:`subprocess`.

    """
    return [
        *config.vllm_command,
        "serve",
        str(checkpoint),
        "--served-model-name",
        config.served_model,
        "--host",
        config.vllm_host,
        "--port",
        str(config.vllm_port),
        "--max-model-len",
        str(config.max_prompt_tokens + config.response_length),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--data-parallel-size",
        str(config.data_parallel_size),
        # Every turn of an episode resends the whole conversation, so the
        # prompt prefix is almost entirely shared between turns.
        "--enable-prefix-caching",
        *config.vllm_extra_args,
    ]


def rollout_server_environment(
    config: OpenMathRunnerConfig,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Return the rollout server's environment, restricted to its own GPUs.

    ``rollout_gpus`` numbers devices the way the learner's process sees them,
    so when that process already has ``CUDA_VISIBLE_DEVICES`` set, each index is
    mapped through it to the physical device the server must be given.

    Args:
      config: The finalized rollout settings.
      environ: The learner process's environment.

    Returns:
      environment: A copy of ``environ`` with the server's devices pinned.

    Raises:
      ValueError: A rollout GPU index is not visible to the learner's process.

    """
    environment = dict(environ)
    if not config.rollout_gpus:
        return environment
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        devices = [str(index) for index in config.rollout_gpus]
    else:
        physical = [device.strip() for device in visible.split(",") if device.strip()]
        if max(config.rollout_gpus) >= len(physical):
            raise ValueError(
                f"rollout_gpus {config.rollout_gpus} index past the learner's "
                f"visible devices CUDA_VISIBLE_DEVICES={visible}.",
            )
        devices = [physical[index] for index in config.rollout_gpus]
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    return environment


def check_rollout_gpus(
    rollout_gpus: Sequence[int],
    *,
    learner_devices: int,
    available_devices: int,
) -> None:
    """Refuse rollout GPUs the learner on this node also uses, or that do not exist.

    The learner binds local rank ``r`` to ``cuda:r``, so on the node that starts
    the server its devices are ``0 .. learner_devices - 1``. They hold the
    learner's weights and optimizer state for the whole run, so a server sharing
    one of them either fails to reserve its memory or starves the learner.

    Args:
      rollout_gpus: The configured rollout devices.
      learner_devices: Learner processes on this node (its local world size).
      available_devices: CUDA devices visible to this process.

    Raises:
      ValueError: The sets overlap, or a rollout device is not visible.

    """
    shared = sorted(set(rollout_gpus) & set(range(learner_devices)))
    if shared:
        raise ValueError(
            f"rollout_gpus {tuple(rollout_gpus)} overlap the learner's devices "
            f"0..{learner_devices - 1} on this node at {shared}. Start fewer "
            "learner processes per node or move the rollout server.",
        )
    missing = [index for index in rollout_gpus if index >= available_devices]
    if missing:
        raise ValueError(
            f"rollout_gpus {missing} are not among this node's "
            f"{available_devices} visible CUDA devices.",
        )


class _Flags(Protocol):
    """The parsed command line."""

    base_dir: Path
    working_dir: Path
    output_dir: Path
    tasks_dir: Path | None
    snapshot: str | None
    checkpoint: Path
    tokenizer_path: Path | None
    seed: int
    num_updates: int
    start_step: int
    vllm_base_url: str | None


def run_updates(runner: OpenMathRolloutRunner) -> list[Path]:
    """Collect a FIXED corpus of updates; train nothing.

    Every update here is drawn from the same unchanging policy, so only the first
    is on-policy. This is a corpus tool, not the published recipe;
    ``experiments.exp000`` interleaves collection with training through
    :mod:`priml.baselines.openaimath.live`.
    """
    writer = RolloutWriter(runner.output_dir, runner.config.run_name)
    # The old fixed writer stored no cursor, so its shards cannot be resumed
    # without repeating tasks. Refuse them rather than discard that corpus.
    shards = list(runner.output_dir.glob(f"{runner.config.run_name}_rollouts_*.jsonl"))
    markers = list(runner.output_dir.glob(f"{runner.config.run_name}_live_step_*.json"))
    pending = list(
        runner.output_dir.glob(f"{runner.config.run_name}_live_step_*.pending.json"),
    )
    if shards and not markers and not pending:
        raise ValueError(
            "Fixed corpus shards have no task-cursor markers and cannot be "
            "resumed safely; use a new output_dir to start a marked corpus.",
        )
    written: list[Path] = []
    complete = writer.committed_live_steps(runner.records_per_update)
    start = runner.config.start_step
    end = start + runner.num_updates
    earlier = [step for step in complete if step < start]
    if earlier:
        runner.tasks.load_state_dict(
            writer.live_cursor_after(max(earlier), runner.records_per_update),
        )
    missing = next((step for step in range(start, end) if step not in complete), end)
    later = sorted(step for step in complete if step > missing and step < end)
    if later:
        raise ValueError(
            f"Fixed corpus has a gap at step {missing} before committed steps {later}.",
        )
    writer.recover_uncommitted_live_tail(missing, runner.records_per_update)
    for step in range(
        start,
        end,
    ):
        if step in complete:
            runner.tasks.load_state_dict(
                writer.live_cursor_after(step, runner.records_per_update),
            )
            continue
        records = records_for_update(runner, step)
        written.append(
            writer.write_live(
                step,
                records,
                prompt_loader_state_after=runner.tasks.state_dict(),
            ),
        )
    return written


def main() -> int:
    """Collect a fixed rollout corpus and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/scratch"),
        help="Root for every default asset directory.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/runs/openaimath/corpus"),
        help="Run directory, under base-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rollouts"),
        help="Rollout shards and Harbor jobs, under working-dir.",
    )
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Daytona snapshot template name from the dataset's generator.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Qwen3.5-4B"),
        help="The served policy's checkpoint, for the agent's tokenizer.",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the task draw order.",
    )
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--num-updates", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=0)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = OpenMathRunnerConfig()
    config.base_dir = flags.base_dir
    config.working_dir = flags.working_dir
    config.output_dir = flags.output_dir
    config.checkpoint = flags.checkpoint
    config.tokenizer_path = flags.tokenizer_path
    config.seed = flags.seed
    if flags.tasks_dir is not None:
        config.tasks_dir = flags.tasks_dir
    if flags.snapshot is not None:
        config.snapshot = flags.snapshot
    if flags.vllm_base_url is not None:
        config.vllm_base_url = flags.vllm_base_url
    config.num_updates = flags.num_updates
    config.start_step = flags.start_step
    runner = OpenMathRolloutRunner(config.finalize())
    written = run_updates(runner)
    for path in written:
        logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
