"""TrainLoop abstraction for step-based training.

Bundles TrainStep + dataset + metrics + training loop orchestration.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sized
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)
from typing_extensions import TypeVar

import contextlib
import faulthandler
import gc
import logging
import math
import sys
import threading
import time

from configgle import Fig, Makeable
from torch import Tensor

import torch
import torch.distributed

from priml.data.custom_types import DatasetProtocol
from priml.data.dummy import DummyDataset
from priml.lib.custom_json import IntCodec


if TYPE_CHECKING:
    from typing import Self

    from priml.timer import CheckpointableStepTimer

from priml.custom_types import (
    HasNormalizedWorkingDirPattern,
)
from priml.math.seed import (
    RngState,
    get_rng_state,
    salt,
    set_rng_state,
    set_seed_distributed,
    set_seed_local,
)
from priml.metrics.custom_types import MetricProtocol, RequiresDeviceTiming
from priml.paths import resolve_working_dir
from priml.runtime import (
    RuntimeProtocol,
    SingleProcess,
    global_device_mesh,
    is_rank_zero,
    runtime_initialized,
)
from priml.train.checkpointer import Checkpointer
from priml.train.custom_types import (
    CheckpointerProtocol,
    CudaEventProtocol,
    PhaseTimerProtocol,
    ProfilerProtocol,
    TrackerProtocol,
    TrainStepProtocol,
)
from priml.train.profiler import PhaseTimer
from priml.train.tracker import scalar_metrics
from priml.train.train_step import TrainStep


logger = logging.getLogger(__name__)


_StepConfigT_co = TypeVar(
    "_StepConfigT_co",
    bound=Makeable[TrainStepProtocol],
    default=Makeable[TrainStepProtocol],
    covariant=True,
)
_DatasetConfigT_co = TypeVar(
    "_DatasetConfigT_co",
    bound=Makeable[DatasetProtocol],
    default=Makeable[DatasetProtocol],
    covariant=True,
)


class EvalTimeLimitError(RuntimeError):
    """Raised when a single eval pass exceeds ``Config.max_eval_time``.

    A cooperative failure after metric code returns, before successful eval
    publication. It does not interrupt arbitrary code or undo its side effects.
    """


class TrainLoop:
    """Step-based training loop: step + dataset + metrics + checkpointing.

    Three counters, each owned elsewhere:

    - ``step.global_step`` -- optimizer updates, the authority for
      ``max_steps`` and every cadence. Survives resume via the step.
    - ``current_epoch`` -- the DATASET's timer, ticked here on each
      ``StopIteration``, since only the loader knows when the data ran out.
    - ``local_step`` -- steps since process start, for the GC cadence alone.
      NOT a checkpoint anchor.

    Example:
      cfg = TrainLoop.Config(
          step=TrainStep.Config(...),
          dataset=ImageNetDataset.Config(...),
          metrics_eval={"accuracy": TopK.Config(k_values=[1, 5])},
      )
      loop = cfg.make()
      loop.train()

    """

    class Config(Fig["TrainLoop"], Generic[_StepConfigT_co, _DatasetConfigT_co]):
        """TrainLoop configuration.

        Generic over the ``step`` and ``dataset`` config types. Both parameters
        default, so a bare ``TrainLoop.Config`` behaves as before. Parameterize
        with the concrete leaf configs to read ``cfg.step``/``cfg.dataset`` fields
        without an ``isinstance`` narrow.
        """

        study_name: str = ""
        """Run-family name; launch derives it when empty."""

        experiment_name: str = ""
        """Run name within the study; launch derives it when empty."""

        base_dir: Path | str | None = "/opt/scratch"
        """Resource root inherited by the dataset subtree."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """Logical run directory inherited by run-output children."""

        doc: str = ""
        """Experiment description propagated to empty tracker notes."""

        step: _StepConfigT_co = field(
            default_factory=lambda: cast(_StepConfigT_co, TrainStep.Config()),
        )
        """What one optimizer update does: model, loss, optimizer, schedule."""

        dataset: _DatasetConfigT_co = field(
            default_factory=lambda: cast(_DatasetConfigT_co, DummyDataset.Config()),
        )
        """Supplies the train and eval loaders, and owns the epoch count."""

        metrics_train: dict[str, Makeable[MetricProtocol]] = field(
            default_factory=dict[str, Makeable[MetricProtocol]],
        )
        """Train metrics receiving each batch and ``step_sec``."""

        metrics_eval: dict[str, Makeable[MetricProtocol]] = field(
            default_factory=dict[str, Makeable[MetricProtocol]],
        )
        """Eval metrics updated per batch and published under ``eval/``."""

        checkpointer: Makeable[CheckpointerProtocol] | None = field(
            default_factory=Checkpointer.Config,
        )
        """Save cadence, resume, and retention. ``None`` writes nothing."""

        profiler: Makeable[ProfilerProtocol] | None = None
        """Per-step profiler hooks; ``None`` (the default) adds no overhead."""

        phase_timer: Makeable[PhaseTimerProtocol] = field(
            default_factory=PhaseTimer.Config,
        )
        """Times named startup and eval phases, and beats while one is open."""

        tracker: Makeable[TrackerProtocol] | None = None
        """Where metrics are published (W&B, TensorBoard, a JSON file)."""

        max_steps: float = math.inf
        """Optimizer-step limit; independent of ``step.train_budget_steps``."""

        max_epochs: float = math.inf
        """Passes over the training data before stopping."""

        max_time: float = math.inf
        """Training time limit in seconds, using ``max_time_kind``."""

        max_time_kind: Literal["wall", "train"] = "wall"
        """Clock capped by ``max_time``; ``train`` excludes first-step compile and evals."""

        max_eval_time: float = math.inf
        """Cooperative wall-time cap checked between eval batches and metrics."""

        eval_stop_on_time_limit: bool = False
        """Publish partial metrics instead of raising when eval exceeds its limit."""

        num_steps_eval: float = 1_000
        """Eval cadence: positive plus final; ``-1`` final only; ``0``/``inf`` never."""

        num_steps_log: int = 10
        """Optimizer steps between train-metric logs, after startup."""

        early_train_log_steps: int = 100
        """Log every optimizer step up to this step for startup diagnostics."""

        eval_every_epoch: bool = True
        """Run eval at epoch boundaries. Disable to save time."""

        eval_extras_every_eval: bool = False
        """Forward non-scalar payloads every eval instead of only the final one."""

        eval_warmup_batches: int = 0
        """Eval batches to run once before training timers start."""

        eval_only: bool = False
        """Evaluate once and exit; checkpoint loading still follows resume policy."""

        restore_rng_state: bool = True
        """Restore checkpoint RNG state; cross-GPU-count eval may disable it."""

        seed: int | None = None
        """Base seed for every RNG; ``None`` draws one from OS entropy."""

        num_steps_garbage_collect: float = math.inf
        """Manual-GC cadence; finite values disable automatic collection."""

        mesh_dim_model_seed: str = "pp"
        """Mesh dimension across which model seeds differ."""

        mesh_dim_data_seed: str = "dp"
        """Mesh dimension across which data seeds differ."""

        runtime: Makeable[RuntimeProtocol] = field(default_factory=SingleProcess.Config)
        """Process-global device, distribution, and determinism setup."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.working_dir, str):
                working_dir = self.working_dir.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            else:
                working_dir = self.working_dir
            self.working_dir = resolve_working_dir(self.base_dir, working_dir)
            if (
                isinstance(self.dataset, HasNormalizedWorkingDirPattern)
                and self.dataset.base_dir is None
            ):
                self.dataset.base_dir = self.base_dir
            for part in (
                self.step,
                self.checkpointer,
                self.profiler,
                self.phase_timer,
                self.tracker,
            ):
                if (
                    isinstance(part, HasNormalizedWorkingDirPattern)
                    and part.base_dir is None
                ):
                    part.base_dir = self.working_dir
            placement = getattr(self.step, "parallelism", None)
            if (
                isinstance(self.runtime, _DeclaresDevice)
                and isinstance(placement, _DeclaresDevice)
                and placement.device is None
            ):
                placement.device = self.runtime.device
            for metric in (*self.metrics_train.values(), *self.metrics_eval.values()):
                if (
                    not isinstance(metric, HasNormalizedWorkingDirPattern)
                    or metric.base_dir is not None
                ):
                    continue
                logical = str(metric.working_dir)
                shared = logical == "/datasets" or logical.startswith("/datasets/")
                metric.base_dir = self.base_dir if shared else self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Initialize training job.

        Raises:
          ValueError: If ``config`` is a training run (not ``eval_only``) with no
            finite stop condition -- ``max_steps``, ``max_epochs``, and
            ``max_time`` all infinite -- which would loop forever. Checked before
            any expensive build so the failure is immediate.

        """
        if (
            not config.eval_only
            and config.max_steps == math.inf
            and config.max_epochs == math.inf
            and config.max_time == math.inf
        ):
            raise ValueError(
                "TrainLoop has no finite stop condition: max_steps, "
                "max_epochs, and max_time are all infinite, so training "
                "would never terminate. Set at least one on the config.",
            )

        self.runtime = config.runtime.make()
        self._owns_runtime = not runtime_initialized()
        self._runtime_destroyed = False
        if self._owns_runtime:
            self.runtime.initialize()

        self.checkpointer: CheckpointerProtocol | None = None
        self.tracker: TrackerProtocol | None = None
        self.profiler: ProfilerProtocol | None = None
        self._gc_disabled = False
        self._training = False
        self._terminal_epoch_evaluated = False
        self._last_boundary_epoch = 0
        try:
            # Broadcast once; derive later seeds locally to avoid collective skew.
            mesh = global_device_mesh()
            if mesh:
                base_seed, _ = set_seed_distributed(
                    config.seed,
                    mesh=mesh[config.mesh_dim_model_seed],
                    salt_by_rank=True,
                )
            else:
                base_seed = set_seed_local(config.seed)

            self.phase_timer = config.phase_timer.make()
            with self.phase_timer.phase("model_init"):
                self.step = config.step.make()
            if isinstance(self.step, _HasTimer):
                self.step.timer = self.phase_timer
            self.metrics_train = {
                name: cfg.make() for name, cfg in config.metrics_train.items()
            }
            self._requires_device_timing = self.runtime.device.type != "cpu" and any(
                isinstance(metric, RequiresDeviceTiming)
                and metric.requires_device_timing
                for metric in self.metrics_train.values()
            )
            self.metrics_eval = {
                name: cfg.make() for name, cfg in config.metrics_eval.items()
            }

            if mesh:
                data_local_rank = mesh[config.mesh_dim_data_seed].get_local_rank()
                set_seed_local(
                    salt(
                        "rank",
                        data_local_rank,
                        salt(config.mesh_dim_data_seed, base_seed),
                    ),
                )
            with self.phase_timer.phase("data_load"):
                self.dataset = config.dataset.make()
            _bind_dataset_step(self.dataset, self.step)
            # Dataset and step share one checkpointed epoch counter.
            step = self.step
            if isinstance(step, _SupportsBindEpochTimer):
                step.bind_epoch_timer(self.dataset.timer_epoch)
            logger.info("TrainLoop startup: dataset ready.")

            logger.info("TrainLoop startup: creating checkpointer.")
            self.checkpointer = (
                config.checkpointer.make() if config.checkpointer else None
            )
            logger.info("TrainLoop startup: checkpointer ready.")

            self.working_dir = config.working_dir
            self.local_step = 0
            self._last_cadence_step = -1
            self.max_epochs = config.max_epochs
            self.max_steps = config.max_steps
            self.max_time = config.max_time
            self.max_time_kind = config.max_time_kind
            self.max_eval_time = config.max_eval_time
            self.eval_stop_on_time_limit = config.eval_stop_on_time_limit
            self.num_steps_eval = config.num_steps_eval
            self.eval_every_epoch = config.eval_every_epoch
            self.eval_extras_every_eval = config.eval_extras_every_eval
            self.eval_warmup_batches = config.eval_warmup_batches
            self.eval_only = config.eval_only
            self.restore_rng_state = config.restore_rng_state
            self.num_steps_log = config.num_steps_log
            self.early_train_log_steps = config.early_train_log_steps
            self.num_steps_garbage_collect = config.num_steps_garbage_collect
            self._start_time = time.perf_counter()
            self._train_clock_base = self._start_time
            self._eval_sec = 0.0

            if self.checkpointer is not None:
                self.checkpointer.load(
                    self,
                    max_steps=self.max_steps,
                    guard=not self.eval_only,
                )
                self._last_cadence_step = self.step.global_step

            logger.info("TrainLoop startup: creating tracker.")
            self.tracker = config.tracker.make() if config.tracker else None
            if self.tracker is not None and config.doc:
                self.tracker.log_notes(config.doc)
            logger.info("TrainLoop startup: tracker ready.")
            _barrier_if_distributed("tracker startup")

            logger.info("TrainLoop startup: creating profiler.")
            self.profiler = config.profiler.make() if config.profiler else None
            logger.info("TrainLoop startup: profiler ready.")

            if math.isfinite(config.num_steps_garbage_collect):
                gc.disable()
                self._gc_disabled = True

            self._last_eval_step = self.step.global_step

            self.train_loader: Iterable[dict[str, object]] | None = None
            self.train_iter: Iterator[dict[str, object]] | None = None
            self._time_limit_latched = False
            logger.info(
                "TrainLoop startup: warm eval compile begin "
                "(batches=%d, global_step=%d).",
                self.eval_warmup_batches,
                self.step.global_step,
            )
            self._warm_eval_compile()
            logger.info("TrainLoop startup: warm eval compile complete.")
            self._start_time = time.perf_counter()
            self._train_clock_base = self._start_time
            self._eval_sec = 0.0
        except BaseException:
            self._cleanup()
            raise

    def train(self) -> None:
        """Run step-based training loop."""
        self._training = True
        self._last_boundary_epoch = self.current_epoch
        try:
            if self.eval_only:
                self._run_eval_only()
                return
            trained_any = False
            while (
                self.step.global_step < self.max_steps
                and self.current_epoch < self.max_epochs
                and not self._time_limit_reached()
                and not self._should_stop_early()
            ):
                # Save completed prefixes before fetching data for their successor.
                stepped = self.step.global_step != self._last_cadence_step
                if self.local_step > 0 and stepped:
                    self._last_cadence_step = self.step.global_step
                    try:
                        self._maybe_eval()
                    except BaseException:
                        self._save_after_evaluation_error(is_final=False)
                        raise
                    if self.checkpointer is not None:
                        self.checkpointer.maybe_save(self, self.step.global_step)
                try:
                    batch = self._get_next_batch()
                except StopIteration:
                    break
                if self.current_epoch != self._last_boundary_epoch:
                    self._on_epoch_boundary()
                    if self.current_epoch >= self.max_epochs:
                        break
                self._maybe_garbage_collect()
                self._do_train_step(batch)
                trained_any = True

            elapsed = self._max_time_elapsed()
            if elapsed >= self.max_time:
                logger.warning(
                    "Time limit reached (%.1fs >= %.1fs), stopping early at step %d.",
                    elapsed,
                    self.max_time,
                    self.step.global_step,
                )
            if not trained_any:
                self._warn_nothing_to_train()
                return
            try:
                if not self._terminal_epoch_evaluated:
                    self._maybe_eval(is_final=True)
            except BaseException:
                self._save_after_evaluation_error(is_final=True)
                raise
            if self.checkpointer is not None:
                self.checkpointer.save(self, self.step.global_step)
        finally:
            self._training = False
            _finish_resources(
                [
                    partial(
                        self.phase_timer.publish_summary,
                        self.tracker,
                        step=self.step.global_step,
                    ),
                    self.phase_timer.log_summary,
                    self._cleanup,
                ],
            )

    def _save_after_evaluation_error(self, *, is_final: bool) -> None:
        """Attempt the scheduled save without masking an evaluation error."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return
        if self.checkpointer is None:
            return
        try:
            if is_final:
                self.checkpointer.save(self, step=self.step.global_step)
            else:
                self.checkpointer.maybe_save(self, step=self.step.global_step)
        except BaseException:
            logger.exception("Failed to save checkpoint after evaluation error.")

    def _warn_nothing_to_train(self) -> None:
        """Explain a run that resumed a finished experiment and did nothing."""
        if not is_rank_zero() or self.step.global_step == 0:
            return
        logger.warning(
            "No training step ran: this experiment already completed at step "
            "%d (max_steps=%s) in %s. To train further, raise the stop "
            "condition; to train again from scratch, fork it with a new "
            "experiment_name or point working_dir elsewhere.",
            self.step.global_step,
            self.max_steps,
            self.working_dir,
        )

    @property
    def current_epoch(self) -> int:
        """Completed passes over the training data.

        The dataset's count, since only the loader knows when the data ran
        out -- and reading it rather than keeping a copy is what stops the two
        from drifting across a resume.
        """
        return self.dataset.timer_epoch.global_count

    # Rank zero broadcasts on log cadence to prevent skew-driven eval deadlocks.
    def _time_limit_reached(self) -> bool:
        """Whether the ``max_time`` cap has elapsed, agreed by all ranks."""
        if self.max_time == math.inf:
            return False
        if not torch.distributed.is_initialized():
            return self._max_time_elapsed() >= self.max_time
        if self._time_limit_latched:
            return True
        if self.step.global_step % self.num_steps_log != 0:
            return False
        over = is_rank_zero() and self._max_time_elapsed() >= self.max_time
        flag = torch.tensor(
            1.0 if over else 0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torch.distributed.broadcast(flag, src=0)
        reached = bool(flag.item() > 0.0)
        if reached:
            self._time_limit_latched = True
        return reached

    def _max_time_elapsed(self) -> float:
        """Seconds charged against ``max_time``, per ``max_time_kind``."""
        if self.max_time_kind == "train":
            return self._train_elapsed()
        return time.perf_counter() - self._start_time

    def _pure_train_sec(self) -> float:
        """Wall seconds spent training: first-step compile and evals excluded."""
        return time.perf_counter() - self._train_clock_base

    def _train_elapsed(self) -> float:
        """Pure-train seconds: first-step compile and mid-loop evals excluded."""
        return self._pure_train_sec()

    def _billed_train_sec(self) -> float:
        """Seconds the recipe's own schedule charged against its budget."""
        billed: object = getattr(self.step, "elapsed_sec", None)
        if billed is None:
            return self._pure_train_sec()
        assert isinstance(billed, (int, float))
        return float(billed)

    # Account for every wall second, including unbilled training.
    def _time_account(self, elapsed: float) -> list[str]:
        """Decompose wall time so no clock can be moved without showing."""
        train = self._billed_train_sec()
        pure = self._pure_train_sec()
        unbilled = pure - train
        other = elapsed - pure - self._eval_sec
        return [
            f"train_sec={train:.1f}s",
            f"train_unbilled_sec={unbilled:.1f}s",
            f"eval_sec={self._eval_sec:.1f}s",
            f"other_sec={other:.1f}s",
        ]

    # Overrides must return a rank-agreed verdict.
    def _should_stop_early(self) -> bool:
        """Whether training should stop before ``max_steps`` / ``max_time``."""
        return False

    def _eval_time_limit_reached(self, eval_start: float) -> bool:
        """Whether eval's wall-clock cap elapsed, agreed by all ranks."""
        if self.max_eval_time == math.inf:
            return False
        if not torch.distributed.is_initialized():
            return time.perf_counter() - eval_start > self.max_eval_time
        over = is_rank_zero() and (
            time.perf_counter() - eval_start > self.max_eval_time
        )
        flag = torch.tensor(
            1.0 if over else 0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torch.distributed.broadcast(flag, src=0)
        return bool(flag.item() > 0.0)

    def _check_eval_deadline(self, eval_start: float) -> None:
        """Reject completed metric work over budget; never preempt metric code."""
        if (
            self._eval_time_limit_reached(eval_start)
            and not self.eval_stop_on_time_limit
        ):
            raise EvalTimeLimitError(
                f"eval exceeded max_eval_time ({self.max_eval_time}s) "
                "after metric work; reduce eval cost.",
            )

    def _run_eval_only(self) -> None:
        """Score the loaded checkpoint with one eval, then return."""
        if self.step.global_step == 0:
            logger.warning(
                "eval_only at global_step=0: no checkpoint was loaded "
                "(resume found none). Evaluating the freshly-initialized model.",
            )
        self._maybe_eval(is_final=True, force=True)

    # Cadence evals publish scalars unless full payload forwarding is enabled.
    def _publish_eval_metrics(
        self,
        eval_metrics: dict[str, object],
        *,
        eval_time: float,
        step: int,
        is_final: bool,
    ) -> dict[str, float]:
        """Publish eval metrics to the tracker; final-ness is expressed as data."""
        eval_scalar_metrics = scalar_metrics(eval_metrics)
        if self.tracker:
            payload: dict[str, object] = (
                dict(eval_metrics)
                if is_final or self.eval_extras_every_eval
                else dict(eval_scalar_metrics)
            )
            payload["time"] = eval_time
            payload.update(
                self._extra_eval_payload(eval_scalar_metrics, step, is_final=is_final),
            )
            self.tracker.log_metrics(payload, step, prefix="eval/")
        return eval_scalar_metrics

    def _extra_eval_payload(
        self,
        eval_scalar_metrics: dict[str, float],
        step: int,
        *,
        is_final: bool,
    ) -> dict[str, object]:
        """Extra ``eval/``-prefixed keys to merge into the eval payload."""
        del eval_scalar_metrics, step, is_final
        return {}

    def _get_next_batch(self) -> dict[str, object]:
        """Get a batch, stopping at the epoch budget while training."""
        batch_start = time.perf_counter()
        if self.train_loader is None:
            logger.info(
                "TrainLoop step %d: creating train dataloader.",
                self.step.global_step + 1,
            )
            self.train_loader = cast(
                Iterable[dict[str, object]],
                self.dataset.train_dataloader(),
            )
        if self.train_iter is None:
            logger.info(
                "TrainLoop step %d: creating train iterator for epoch %d.",
                self.step.global_step + 1,
                self.current_epoch,
            )
            _set_loader_epoch(self.train_loader, self.current_epoch)
            self.train_iter = iter(self.train_loader)
        for _ in range(2):
            try:
                logger.debug(
                    "TrainLoop step %d: fetching raw train batch.",
                    self.step.global_step + 1,
                )
                raw_batch = next(self.train_iter)
                batch = {str(key): value for key, value in raw_batch.items()}
                batch_time = time.perf_counter() - batch_start
                logger.debug(
                    "TrainLoop step %d: raw train batch ready (batch_time=%.3fs).",
                    self.step.global_step + 1,
                    batch_time,
                )
                if self.tracker:
                    self.tracker.log_metrics(
                        {"batch_time": batch_time},
                        self.step.global_step,
                        prefix="train/",
                    )
                logger.debug(
                    "TrainLoop step %d: preprocessing train batch.",
                    self.step.global_step + 1,
                )
                batch = self.step.preprocess_batch(batch)
                logger.debug(
                    "TrainLoop step %d: train batch preprocessed.",
                    self.step.global_step + 1,
                )
                self._on_batch_ready(time.perf_counter() - batch_start)
                return batch
            except StopIteration:
                self.dataset.timer_epoch.global_count += 1
                self.dataset.timer_epoch.local_count += 1
                logger.info(
                    "TrainLoop step %d: train iterator exhausted; "
                    "advancing to epoch %d.",
                    self.step.global_step + 1,
                    self.current_epoch,
                )
                self.train_iter = None
                if getattr(self, "_training", False):
                    self._on_epoch_boundary()
                    if self.current_epoch >= self.max_epochs:
                        raise
                _set_loader_epoch(self.train_loader, self.current_epoch)
                self.train_iter = iter(self.train_loader)
        raise RuntimeError("Failed to get next batch after epoch reset")

    def _on_epoch_boundary(self) -> None:
        """Finish the consumed epoch before constructing its successor iterator."""
        self._last_boundary_epoch = self.current_epoch
        self.step.on_epoch_end()
        if self.local_step == 0 or not self.eval_every_epoch:
            return
        is_final = self.current_epoch >= self.max_epochs
        try:
            self._maybe_eval(is_final=is_final, force=True)
        except BaseException:
            self._save_after_evaluation_error(is_final=is_final)
            raise
        self._terminal_epoch_evaluated = is_final

    def _on_batch_ready(self, fetch_time: float) -> None:
        """Record the seconds spent producing one training batch."""
        del fetch_time

    def _on_train_step_timed(self, step_time: float, *, is_first: bool) -> None:
        """Record each train step's wall-clock duration."""
        del step_time, is_first

    def _do_train_step(self, batch: dict[str, object]) -> None:
        """Execute one training step with profiling and logging."""
        if self.profiler:
            self.profiler.on_step_start(self.step.global_step)

        next_step = self.step.global_step + 1
        if is_rank_zero():
            logger.debug(
                "Entering train step %d/%s (elapsed=%.0fs)",
                next_step,
                self.max_steps,
                time.perf_counter() - self._start_time,
            )

        if self._requires_device_timing:
            torch.accelerator.synchronize(self.runtime.device)
        step_start = time.perf_counter()
        before = self.step.global_step
        if self.local_step < self.early_train_log_steps:
            with _compile_heartbeat(f"train step {next_step}"):
                step_results = self.step.train_step(**batch)
        else:
            step_results = self.step.train_step(**batch)
        if self._requires_device_timing:
            torch.accelerator.synchronize(self.runtime.device)
        step_time = time.perf_counter() - step_start
        self.local_step += 1
        self._on_train_step_timed(step_time, is_first=self.local_step == 1)
        if self.local_step == 1:
            self._train_clock_base = time.perf_counter()

        if self.profiler:
            self.profiler.on_step_end(self.step.global_step)

        for metric in self.metrics_train.values():
            metric.update(step_results["model"], **batch, step_sec=step_time)

        # Cadences count optimizer updates, but this method runs per microbatch.
        log_step = self.step.global_step != before and (
            self.step.global_step == 1
            or self.local_step <= self.early_train_log_steps
            or self.step.global_step <= self.early_train_log_steps
            or self.step.global_step % self.num_steps_log == 0
        )
        if not log_step:
            return
        self.phase_timer.publish_interval(
            self.tracker,
            step=self.step.global_step,
        )

        loss = step_results["loss"].mean().detach()
        raw_step_metrics = step_results.get("metrics", {})
        tensor_metrics = [
            (key, value)
            for key, value in raw_step_metrics.items()
            if isinstance(value, Tensor)
        ]
        step_metric_keys = [key for key, _ in tensor_metrics]
        reduced_values = torch.stack(
            [
                loss,
                *(value.mean().detach() for _, value in tensor_metrics),
            ],
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reduced_values)
            reduced_values = reduced_values / torch.distributed.get_world_size()
        # Metric computation may itself be collective.
        train_metrics = self._compute_train_metrics()
        if not is_rank_zero():
            return

        loss_value = float(reduced_values[0].item())
        elapsed = time.perf_counter() - self._start_time
        step_metric_tensors = {
            key: reduced_values[i + 1] for i, key in enumerate(step_metric_keys)
        }
        step_metrics: dict[str, float] = {
            key: float(
                step_metric_tensors[key].item()
                if key in step_metric_tensors
                else value,
            )
            for key, value in raw_step_metrics.items()
        }
        extra = " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items())
        line = (
            f"Step {self.step.global_step}/{self.max_steps}: "
            f"loss={loss_value:.4f} step_time={step_time:.3f}s "
            f"elapsed={elapsed:.0f}s"
        )
        extra = " ".join(
            [extra, *(f"{k}={v:.4f}" for k, v in train_metrics.items())],
        ).strip()
        logger.info(f"{line} {extra}" if extra else line)

        if self.tracker:
            metrics: dict[str, float] = {
                "total_loss": loss_value,
                "step_time": step_time,
                "time_since_start": elapsed,
                "elapsed": self._train_elapsed(),
                **step_metrics,
                **train_metrics,
            }
            if torch.cuda.is_available():
                metrics["gpu_mem_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / 1e9
                )
                metrics["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            self.tracker.log_metrics(metrics, self.step.global_step, prefix="train/")

    def _compute_train_metrics(self) -> dict[str, float]:
        """Compute, reset, and flatten every train metric under its name."""
        results: dict[str, object] = {}
        for name, metric in self.metrics_train.items():
            for key, value in metric.compute().items():
                results[f"{name}_{key}" if name else key] = value
            metric.reset()
        return scalar_metrics(results)

    def _maybe_garbage_collect(self) -> None:
        """Run manual garbage collection if configured."""
        if (
            not math.isfinite(self.num_steps_garbage_collect)
            or self.local_step <= 0
            or self.local_step % self.num_steps_garbage_collect != 0
        ):
            return
        gc_start = time.perf_counter()
        gc.collect()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        gc_time = time.perf_counter() - gc_start
        logger.info("GC at local_step %s (gc_time=%.3fs)", self.local_step, gc_time)

    def _maybe_eval(self, *, is_final: bool = False, force: bool = False) -> None:
        """Run validation and publish it; the final eval also writes metrics."""
        if not force:
            if self.num_steps_eval == 0 or not math.isfinite(self.num_steps_eval):
                return
            if self.num_steps_eval < 0:
                if not is_final:
                    return
            else:
                cadence_due = (
                    self.step.global_step != 0
                    and self.step.global_step % self.num_steps_eval == 0
                    and self.step.global_step != self._last_eval_step
                )
                if not is_final and not cadence_due:
                    return
        eval_start = time.perf_counter()
        eval_metrics = self.eval()
        eval_time = time.perf_counter() - eval_start
        self._eval_sec += eval_time
        self._train_clock_base += eval_time
        scalar_metrics = self._publish_eval_metrics(
            eval_metrics,
            eval_time=eval_time,
            step=self.step.global_step,
            is_final=is_final,
        )
        if self.checkpointer is not None and not self.eval_only:
            self.checkpointer.on_eval(self, self.step.global_step, scalar_metrics)
        if is_rank_zero():
            if is_final:
                elapsed = time.perf_counter() - self._start_time
                parts = [f"steps={self.step.global_step}", f"time={elapsed:.1f}s"]
                parts.extend(self._time_account(elapsed))
                for k, v in sorted(scalar_metrics.items()):
                    parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
                logger.info("RESULT: %s", " | ".join(parts))
            else:
                logger.info(
                    "Step %d: %s (eval_time=%.3fs)",
                    self.step.global_step,
                    scalar_metrics,
                    eval_time,
                )
        if not is_final:
            self._last_eval_step = self.step.global_step

    def _cleanup(self) -> None:
        """Cleanup resources after training."""
        actions: list[Callable[[], object]] = []
        if self.checkpointer is not None:
            actions.append(self.checkpointer.close)
        if self.tracker is not None:
            actions.append(self.tracker.close)
        if self.profiler is not None:
            actions.append(self.profiler.cleanup)
        if self._gc_disabled:
            actions.append(gc.enable)
            self._gc_disabled = False
        actions.append(self._destroy_runtime_once)
        _finish_resources(actions)

    def _destroy_runtime_once(self) -> None:
        """Tear down an owned runtime, at most once."""
        if self._owns_runtime and not self._runtime_destroyed:
            self.runtime.destroy()
            self._runtime_destroyed = True

    def _warm_eval_compile(self) -> None:
        """Populate eval-only compile caches before timed training/eval."""
        if self.eval_warmup_batches <= 0:
            logger.info("Warm eval compile skipped.")
            return
        logger.info("Warm eval compile: creating eval dataloader.")
        for batch_index, raw_batch in enumerate(self.dataset.eval_dataloader()):
            if batch_index >= self.eval_warmup_batches:
                break
            logger.info(
                "Warm eval compile: batch %d/%d raw batch ready.",
                batch_index + 1,
                self.eval_warmup_batches,
            )
            batch_start = time.perf_counter()
            raw_batch = cast(dict[str, object], raw_batch)
            batch = self.step.preprocess_batch(
                {str(key): value for key, value in raw_batch.items()},
            )
            if batch.get("metric_only", False):
                logger.info(
                    "Warm eval compile: batch %d/%d is metric-only; skipping.",
                    batch_index + 1,
                    self.eval_warmup_batches,
                )
                continue
            logger.info(
                "Warm eval compile: batch %d/%d preprocessed in %.3fs; "
                "running eval_loss.",
                batch_index + 1,
                self.eval_warmup_batches,
                time.perf_counter() - batch_start,
            )
            eval_start = time.perf_counter()
            self.step.eval_loss(**batch)
            logger.info(
                "Warm eval compile: batch %d/%d eval_loss complete in %.3fs.",
                batch_index + 1,
                self.eval_warmup_batches,
                time.perf_counter() - eval_start,
            )

    def _cuda_event_pair(self) -> tuple[CudaEventProtocol, CudaEventProtocol] | None:
        """Start a CUDA event pair when event timing is enabled."""
        if (
            not getattr(self.phase_timer, "cuda_events_enabled", False)
            or not torch.cuda.is_available()
        ):
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return cast(CudaEventProtocol, start), cast(CudaEventProtocol, end)

    def _record_cuda_timing(
        self,
        name: str,
        events: tuple[CudaEventProtocol, CudaEventProtocol] | None,
    ) -> None:
        """End and enqueue a CUDA event pair when event timing is enabled."""
        if events is None:
            return
        _, end = events
        end.record()
        self.phase_timer.record_cuda_events(name, *events)

    def eval(self) -> dict[str, object]:
        """Run validation.

        Returns:
          metrics: Computed validation metrics keyed by name.

        """
        for metric in self.metrics_eval.values():
            metric.reset()

        eval_loader = self.dataset.eval_dataloader()
        total_loss = 0.0
        total_batch_time = 0.0
        total_step_metrics: dict[str, float] = {}
        num_batches = 0
        total_weight = 0
        total_batches = len(eval_loader) if isinstance(eval_loader, Sized) else 0
        narrate = is_rank_zero()
        log_every = max(1, total_batches // 20) if total_batches else 50

        eval_start = time.perf_counter()
        batch_start = eval_start
        for raw_batch in eval_loader:
            elapsed_eval = time.perf_counter() - eval_start
            if self._eval_time_limit_reached(eval_start):
                if self.eval_stop_on_time_limit:
                    logger.warning(
                        "eval reached max_eval_time after %d batches; publishing "
                        "partial eval results.",
                        num_batches,
                    )
                    break
                raise EvalTimeLimitError(
                    f"eval exceeded max_eval_time "
                    f"({elapsed_eval:.0f}s > {self.max_eval_time:.0f}s) after "
                    f"{num_batches} batches; reduce eval cost.",
                )
            raw_batch = cast(dict[str, object], raw_batch)
            batch = self.step.preprocess_batch(
                {str(key): value for key, value in raw_batch.items()},
            )
            weight = IntCodec.coerce(batch.get("valid_count", 1))
            if weight == 0:
                batch_start = time.perf_counter()
                continue
            metric_only = bool(batch.pop("metric_only", False))
            extra_votes = None
            if metric_only:
                model_output: object = torch.empty(0)
            else:
                with _phase_heartbeat(
                    f"eval batch {num_batches + 1} eval_loss",
                    interval_sec=self.phase_timer.heartbeat_interval_sec,
                    fault_dump_interval_sec=self.phase_timer.fault_dump_interval_sec,
                ):
                    step_results = self.step.eval_loss(**batch)
                extra_votes = step_results.get("eval_extra_votes")
                loss = step_results["loss"]
                model_output = step_results["model"]
                total_weight += weight
                total_loss += loss.mean().item() * weight
                for key, value in step_results.get("metrics", {}).items():
                    total_step_metrics[key] = (
                        total_step_metrics.get(key, 0.0)
                        + float(
                            value.mean().item() if isinstance(value, Tensor) else value,
                        )
                        * weight
                    )
            batch_dt = time.perf_counter() - batch_start
            total_batch_time += batch_dt
            num_batches += 1

            for name, metric in self.metrics_eval.items():
                cuda_events = self._cuda_event_pair()
                with _phase_heartbeat(
                    f"eval batch {num_batches} metric[{name}].update",
                    interval_sec=self.phase_timer.heartbeat_interval_sec,
                    fault_dump_interval_sec=self.phase_timer.fault_dump_interval_sec,
                ):
                    metric.update(model_output, **batch, step_sec=batch_dt)
                    if extra_votes:
                        for extra_output, extra_batch in extra_votes:
                            metric.update(
                                extra_output,
                                **extra_batch,
                                step_sec=batch_dt,
                            )
                self._record_cuda_timing(f"eval_metric_{name}_update", cuda_events)

            if narrate and num_batches % log_every == 0:
                pct = (
                    f"{num_batches}/{total_batches}"
                    if total_batches
                    else f"{num_batches}"
                )
                logger.info(
                    "[eval] batch %s | running total_loss=%.4f | "
                    "batch_time=%.3fs mean=%.3fs elapsed=%.1fs",
                    pct,
                    total_loss / total_weight if total_weight else 0.0,
                    batch_dt,
                    total_batch_time / num_batches,
                    total_batch_time,
                )
            batch_start = time.perf_counter()

        results: dict[str, object] = {}

        if total_weight > 0:
            results["total_loss"] = total_loss / total_weight
            results["mean_batch_time"] = total_batch_time / num_batches
            for key, value in total_step_metrics.items():
                results[key] = value / total_weight

        self._check_eval_deadline(eval_start)
        for name, metric in self.metrics_eval.items():
            cuda_events = self._cuda_event_pair()
            metric_results = metric.compute()
            self._check_eval_deadline(eval_start)
            assert isinstance(metric_results, dict)
            self._record_cuda_timing(f"eval_metric_{name}_compute", cuda_events)
            for key, value in metric_results.items():
                results[f"{name}_{key}" if name else key] = value

        return results

    def run(self, *args: str) -> None:
        """Run training (entry point for experimental.lib.launch).

        Args:
          *args: Args.

        """
        del args
        self.train()

    class StateDict(TypedDict):
        """Checkpointed loop state: each owner's payload under its own key."""

        step: Mapping[str, object]
        dataset: Mapping[str, object]
        metrics_train: dict[str, Mapping[str, object]]
        metrics_eval: dict[str, Mapping[str, object]]
        metrics: NotRequired[dict[str, Mapping[str, object]]]
        rng: NotRequired[RngState]

    def state_dict(self) -> StateDict:
        """Get training state for checkpointing.

        Returns:
          state: All model, dataset, and RNG state for resume.

        """
        return {
            "step": self.step.state_dict(),
            "dataset": self.dataset.state_dict(),
            "metrics_train": {
                name: metric.state_dict() for name, metric in self.metrics_train.items()
            },
            "metrics_eval": {
                name: metric.state_dict() for name, metric in self.metrics_eval.items()
            },
            "rng": get_rng_state(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load training state from checkpoint (full restore for resume).

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(TrainLoop.StateDict, state_dict)
        self.step.load_state_dict(state["step"])
        self.dataset.load_state_dict(state["dataset"])
        # ``metrics`` preserves checkpoints written before train/eval metrics split.
        eval_state = state.get("metrics_eval", state.get("metrics", {}))
        for name, metric_state in eval_state.items():
            if name in self.metrics_eval:
                self.metrics_eval[name].load_state_dict(metric_state)
        for name, metric_state in state.get("metrics_train", {}).items():
            if name in self.metrics_train:
                self.metrics_train[name].load_state_dict(metric_state)
        self.local_step = 0

        if self.restore_rng_state and "rng" in state:
            set_rng_state(state["rng"])


def _finish_resources(actions: Iterable[Callable[[], object]]) -> None:
    """Attempt every finalizer without replacing an active primary exception."""
    primary = sys.exception()
    failure: BaseException | None = None
    for action in actions:
        try:
            action()
        except BaseException as error:
            logger.exception("Training resource finalization failed.")
            if failure is None:
                failure = error
    if primary is None and failure is not None:
        raise failure


@runtime_checkable
class _DeclaresDevice(Protocol):
    """A config naming the device its component uses.

    Both the runtime's and the placement strategy's, so the loop can hand the
    first's answer to the second without either importing the other.
    """

    device: torch.device | str | None


@runtime_checkable
class _HasTimer(Protocol):
    """A step that declares a slot for the loop's phase timer."""

    timer: PhaseTimerProtocol | None


@runtime_checkable
class _SupportsSetEpoch(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@runtime_checkable
class _SupportsBindStep(Protocol):
    def bind_step(self, step: TrainStepProtocol) -> None: ...


def _bind_dataset_step(dataset: DatasetProtocol, step: TrainStepProtocol) -> None:
    """Give a dataset that generates its own data the step that produces it."""
    if isinstance(dataset, _SupportsBindStep):
        dataset.bind_step(step)


@runtime_checkable
class _SupportsBindEpochTimer(Protocol):
    """A step that can anneal against passes over the data.

    Optional, because a step budgeted in steps or seconds has no use for the
    count -- and one written from scratch against ``TrainStepProtocol``,
    rather than by extending ``TrainStep``, should not have to declare a
    method it never reads.
    """

    def bind_epoch_timer(self, timer: CheckpointableStepTimer) -> None: ...


def _set_loader_epoch(loader: object, epoch: int) -> None:
    """Inform the loader's dataset of the current epoch before (re)iteration."""
    dataset = getattr(loader, "dataset", None)
    if isinstance(dataset, _SupportsSetEpoch):
        dataset.set_epoch(epoch)


def _barrier_if_distributed(stage: str) -> None:
    """Synchronize ranks after a startup stage that can be rank-skewed."""
    if not torch.distributed.is_initialized():
        return
    logger.info("TrainLoop startup: waiting after %s.", stage)
    torch.distributed.barrier()
    logger.info("TrainLoop startup: all ranks passed %s.", stage)


@contextlib.contextmanager
def _compile_heartbeat(label: str, *, interval_sec: float = 30.0) -> Generator[None]:
    """Log a periodic heartbeat while a (possibly long-compiling) block runs."""
    if not is_rank_zero():
        yield
        return
    done = threading.Event()
    start = time.perf_counter()

    def beat() -> None:
        while not done.wait(interval_sec):
            logger.info(
                "%s: still running after %.0fs (likely torch.compile; "
                "not hung unless this never stops)",
                label,
                time.perf_counter() - start,
            )

    thread = threading.Thread(target=beat, name="compile-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=interval_sec)


def _current_rank() -> int:
    """Global rank, or 0 when distributed is not initialized."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _arm_fault_dump(interval_sec: float) -> None:
    """Arm a finite, positive fault-dump timeout."""
    if interval_sec <= 0 or interval_sec == math.inf:
        return
    faulthandler.dump_traceback_later(
        interval_sec,
        repeat=True,
        file=sys.stderr,
        exit=False,
    )


@contextlib.contextmanager
def _phase_heartbeat(
    label: str,
    *,
    interval_sec: float = 20.0,
    fault_dump_interval_sec: float = 40.0,
) -> Generator[None]:
    """Log, on EVERY rank, which phase this rank is in while a block runs."""
    if interval_sec <= 0 or interval_sec == math.inf:
        yield
        return
    if (
        math.isfinite(fault_dump_interval_sec)
        and fault_dump_interval_sec > 0
        and fault_dump_interval_sec <= interval_sec
    ):
        raise ValueError("fault_dump_interval_sec must exceed interval_sec")
    rank = _current_rank()
    done = threading.Event()
    start = time.perf_counter()

    def beat() -> None:
        while not done.wait(interval_sec):
            _arm_fault_dump(fault_dump_interval_sec)
            logger.warning(
                "[rank %d] STILL IN PHASE %r after %.0fs "
                "(if this never advances, this rank is stuck HERE)",
                rank,
                label,
                time.perf_counter() - start,
            )

    _arm_fault_dump(fault_dump_interval_sec)
    thread = threading.Thread(target=beat, name="phase-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        done.set()
        thread.join(timeout=interval_sec)
