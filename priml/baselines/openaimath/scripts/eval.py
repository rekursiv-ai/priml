r"""Grade the openai/math baseline on the dataset's own tasks.

Harbor runs the staged tasks against a served policy, one job per task with
``-k`` attempts, and the dataset's own verifier decides each attempt inside a
separate sandbox. This module only chooses the tasks, invokes Harbor, reads the
verdicts back, and reports pass@k. It trains nothing and writes no rollouts.

Two task sets matter (see ``lemma_tasks``). The progress metric is pass@1 on a
fixed seeded sample of the held-out lemma tasks, the default. The protected
score is the held-out families' published challenges, listed in
``lemmas/official_test_tasks.txt``.

``--oracle`` grades the reference solutions through Harbor's own oracle agent
instead of the policy. Every published task must then be accepted, because the
dataset published only tasks whose reference proof the same verifier accepted.
That run is the positive control: it separates "the harness, the snapshot, or
the grader is broken" from "the policy cannot prove this theorem". Lemma tasks
were cut by this baseline and never graded by the dataset, so an oracle run on
a sample of them is also how their cutting is checked.

Examples:
  eval.py --base-url http://127.0.0.1:8000 --limit 256
  eval.py --base-url http://127.0.0.1:8000 --attempts 8 \
    --tasks-dir /datasets/openaimath/openai-math/tasks \
    --task-list /datasets/openaimath/openai-math/lemmas/official_test_tasks.txt
  eval.py --oracle --limit 64
  eval.py --oracle --tasks-dir /datasets/openaimath/openai-math/tasks --limit 4

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json
import logging
import math

from priml.baselines.openaimath.scripts.runner import (
    OpenMathRunnerConfig,
    TaskStore,
    run_job,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


ORACLE_AGENT_ARGS: Final[tuple[str, ...]] = (
    "-a",
    "oracle",
    "--allow-agent-host",
    "github.com",
)
"""Harbor's own agent, which applies the dataset's reference solution.

The host allowance is what lets that agent reach the challenge repository the
reference solution is published in; the policy's own agent never gets one,
because an episode has no network.
"""

ORACLE_ATTEMPTS: Final = 1
"""The reference solution is deterministic, so one attempt per task suffices."""

REPORT_NAME: Final = "eval_report.json"
"""Where the per-task verdicts and the pass@k summary are written."""


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task's attempts and their verdicts."""

    task: str
    """The staged task's name."""

    rewards: tuple[float | None, ...]
    """One verdict per attempt; ``None`` when grading reported nothing."""

    @property
    def attempts(self) -> int:
        """Return how many attempts were made."""
        return len(self.rewards)

    @property
    def passes(self) -> int:
        """Return how many attempts the verifier accepted."""
        return sum(1 for reward in self.rewards if reward == 1.0)

    @property
    def missing(self) -> int:
        """Return how many attempts produced no verdict at all."""
        return sum(1 for reward in self.rewards if reward is None)

    def to_payload(self) -> dict[str, object]:
        """Return the task's record for the report file."""
        return {
            "task": self.task,
            "rewards": list(self.rewards),
            "passes": self.passes,
            "missing": self.missing,
        }


def pass_at_k(passes: int, attempts: int, k: int) -> float:
    """Return the unbiased pass@k estimate for one task.

    The estimator is the standard one: one minus the probability that a random
    sample of ``k`` of the ``attempts`` verdicts holds none of the ``passes``
    accepted ones. It is unbiased for the pass rate a fresh set of ``k``
    attempts would see, which is why it is preferred over reporting the best of
    ``k`` attempts, a number that rises with ``k`` by construction.

    Args:
      passes: How many attempts the verifier accepted.
      attempts: How many attempts were made.
      k: The number of attempts to estimate the pass rate for.

    Returns:
      estimate: A probability in ``[0, 1]``.

    Raises:
      ValueError: The counts are inconsistent or ``k`` exceeds ``attempts``.

    """
    if not 0 <= passes <= attempts:
        raise ValueError(f"passes={passes} is not between 0 and {attempts}.")
    if not 0 < k <= attempts:
        raise ValueError(f"k={k} is not between 1 and {attempts}.")
    if passes == 0:
        return 0.0
    if attempts - passes < k:
        return 1.0
    return 1.0 - math.comb(attempts - passes, k) / math.comb(attempts, k)


def summarize(results: Sequence[TaskResult], *, k: int) -> dict[str, object]:
    """Return the aggregate pass@1 and pass@k figures for a run.

    Args:
      results: One result per graded task.
      k: The number of attempts to estimate the pass rate for.

    Returns:
      summary: Task count, both estimates, the mean reward, and the number of
        attempts that produced no verdict.

    Raises:
      ValueError: No task was graded, or the results disagree on ``k``.

    """
    if not results:
        raise ValueError("Cannot summarize a run that graded no task.")
    attempts = {result.attempts for result in results}
    if len(attempts) != 1:
        raise ValueError(f"Tasks were graded with different attempts: {attempts}.")
    total = attempts.pop()
    if k > total:
        raise ValueError(f"k={k} exceeds the {total} attempts per task.")
    rewards = [
        reward for result in results for reward in result.rewards if reward is not None
    ]
    return {
        "tasks": len(results),
        "attempts_per_task": total,
        "k": k,
        "pass@1": sum(result.passes / total for result in results) / len(results),
        "pass@k": sum(pass_at_k(result.passes, total, k) for result in results)
        / len(results),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "tasks_with_a_pass": sum(1 for result in results if result.passes),
        "missing_trials": sum(result.missing for result in results),
    }


def job_directory(config: OpenMathRunnerConfig, task_name: str) -> Path:
    """Return one task's evaluation job directory."""
    return _output_dir(config) / f"jobs/{config.run_name}_eval_{task_name}"


def _output_dir(config: OpenMathRunnerConfig) -> Path:
    """Return where the jobs and the report go.

    Raises:
      ValueError: The settings name no output directory.

    """
    if config.output_dir is None:
        raise ValueError("The evaluation needs an output_dir.")
    return Path(config.output_dir)


def evaluate(
    config: OpenMathRunnerConfig,
    task_names: Sequence[str],
    *,
    attempts: int,
    oracle: bool,
) -> list[TaskResult]:
    """Grade every named task and return one result each.

    Args:
      config: The finalized rollout settings.
      task_names: The staged tasks to grade, in order.
      attempts: Attempts per task.
      oracle: Whether to run Harbor's oracle agent instead of the policy.

    Returns:
      results: One result per task, in the order given.

    Raises:
      RuntimeError: Every attempt failed to produce a verdict, which means the
        harness never reached the grader.

    """
    results: list[TaskResult] = []
    for index, name in enumerate(task_names, start=1):
        outcomes = run_job(
            config,
            name,
            job_directory(config, name),
            agent_args=ORACLE_AGENT_ARGS if oracle else None,
            options=() if oracle else None,
            attempts=attempts,
            kind="eval",
        )
        result = TaskResult(name, tuple(outcome.reward for outcome in outcomes))
        results.append(result)
        logger.info(
            "task %d/%d: %s -> %s",
            index,
            len(task_names),
            name,
            list(result.rewards),
        )
    missing = sum(result.missing for result in results)
    if missing and missing == sum(result.attempts for result in results):
        raise RuntimeError(
            "No attempt produced a verdict, so the harness never reached the"
            " grader. Check the snapshot and the Harbor logs before reading"
            " these attempts as failures.",
        )
    return results


def main() -> int:
    """Grade the configured tasks and return the process exit code."""
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
        "--tasks-dir",
        type=Path,
        default=Path("/datasets/openaimath/openai-math/lemmas/test"),
        help=(
            "Task directories to grade, under base-dir: the held-out lemmas"
            " (progress metric) by default, or the dataset's tasks/ with"
            " --task-list for the protected score."
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Daytona snapshot template name from the dataset's generator.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/runs/openaimath/eval"),
        help="Evaluation run directory, under base-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Harbor jobs and the report, under working-dir.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("/models/Qwen3.5-4B"),
        help="The served policy's tokenizer, under base-dir.",
    )
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Attempts per task; pass@k is estimated at this k.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Grade a seeded subset of this many tasks; 0 grades every task.",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Grade exactly these tasks, in this order.",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help=(
            "Grade the tasks named in this file, one per line, in order (for"
            " example lemmas/official_test_tasks.txt), under base-dir."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the order --limit samples tasks in.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=None,
        help="Trials Harbor runs at once, within one task.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Grade the dataset's reference solutions as a positive control.",
    )
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = _config(flags)
    attempts = ORACLE_ATTEMPTS if flags.oracle else flags.attempts
    if attempts < 1:
        raise ValueError("attempts must be positive.")
    names = _task_names(config, flags)
    logger.info(
        "%s mode: %d tasks, %d attempts each, snapshot %s",
        "oracle" if flags.oracle else "policy",
        len(names),
        attempts,
        config.snapshot,
    )
    results = evaluate(config, names, attempts=attempts, oracle=flags.oracle)
    summary = summarize(results, k=attempts)
    report = _report_payload(config, results, summary, oracle=flags.oracle)
    target = flags.report or (_output_dir(config) / REPORT_NAME)
    _write_report(target, report)
    logger.info("summary: %s", json.dumps(summary, sort_keys=True))
    logger.info("report: %s", target)
    if flags.oracle:
        failed = [result.task for result in results if result.passes != result.attempts]
        if failed:
            logger.error(
                "The oracle was not accepted on %d tasks: %s. The harness is"
                " broken, not the policy: check the snapshot, the sandbox, and"
                " the verifier.",
                len(failed),
                failed[:8],
            )
            return 1
    return 0


def _config(flags: _Flags) -> OpenMathRunnerConfig:
    """Build and finalize the rollout settings the jobs run under."""
    config = OpenMathRunnerConfig()
    config.base_dir = flags.base_dir
    config.working_dir = flags.working_dir
    config.output_dir = flags.output_dir
    config.seed = flags.seed
    if flags.tasks_dir is not None:
        config.tasks_dir = flags.tasks_dir
    if flags.snapshot is not None:
        config.snapshot = flags.snapshot
    if flags.base_url is not None:
        config.vllm_base_url = flags.base_url
    config.tokenizer_path = flags.tokenizer_path
    if flags.served_model is not None:
        config.served_model = flags.served_model
    if flags.temperature is not None:
        config.temperature = flags.temperature
    if flags.top_p is not None:
        config.top_p = flags.top_p
    if flags.n_concurrent is not None:
        config.n_concurrent = flags.n_concurrent
    config = config.finalize()
    if not config.snapshot:
        raise ValueError(
            "snapshot must name the dataset's Daytona snapshot; build it once"
            " with the dataset's generator, then pass the name here.",
        )
    return config


def _task_names(config: OpenMathRunnerConfig, flags: _Flags) -> list[str]:
    """Return the tasks to grade, in order.

    An explicit list is graded in the order given. Otherwise the draw order is
    the runner's own, so a ``--limit`` takes a seeded sample of the staged tasks
    rather than the alphabetically first ones, which would be a topical subset.
    """
    tasks_dir = Path(config.tasks_dir)
    if flags.task_names is not None and flags.task_list is not None:
        raise ValueError("Pass --task-names or --task-list, not both.")
    named = list(flags.task_names or ())
    if flags.task_list is not None:
        task_list = resolve_working_dir(flags.base_dir, flags.task_list)
        named = [
            line.strip()
            for line in task_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if named:
        TaskStore(tasks_dir, seed=flags.seed, task_names=named)
        return named
    store = TaskStore(tasks_dir, seed=flags.seed)
    total = len(store.names) if flags.limit <= 0 else min(flags.limit, len(store.names))
    return [store.draw().name for _ in range(total)]


def _report_payload(
    config: OpenMathRunnerConfig,
    results: Sequence[TaskResult],
    summary: dict[str, object],
    *,
    oracle: bool,
) -> dict[str, object]:
    """Return the run's report, with the pins a reader needs to reproduce it."""
    return {
        "schema": 1,
        "mode": "oracle" if oracle else "policy",
        "snapshot": config.snapshot,
        "tasks_dir": str(config.tasks_dir),
        "output_dir": str(_output_dir(config)),
        "base_url": None if oracle else config.resolved_base_url,
        "tokenizer_path": None if oracle else str(config.resolved_tokenizer_path),
        "temperature": None if oracle else config.temperature,
        "top_p": None if oracle else config.top_p,
        "seed": config.seed,
        "summary": summary,
        "results": [result.to_payload() for result in results],
    }


def _write_report(target: Path, payload: dict[str, object]) -> None:
    """Write the report through a temporary file, so a partial one never lands."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(target)


class _Flags(Protocol):
    """The parsed command line."""

    base_dir: Path
    tasks_dir: Path | None
    snapshot: str | None
    working_dir: Path
    output_dir: Path
    report: Path | None
    base_url: str | None
    tokenizer_path: Path
    served_model: str | None
    temperature: float | None
    top_p: float | None
    attempts: int
    limit: int
    task_names: list[str] | None
    task_list: Path | None
    seed: int
    n_concurrent: int | None
    oracle: bool


if __name__ == "__main__":
    raise SystemExit(main())
