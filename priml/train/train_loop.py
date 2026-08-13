"""TrainLoop abstraction for step-based training.

Bundles TrainStep + dataset + metrics + training loop orchestration.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
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


if TYPE_CHECKING:
    from typing import Self

    from torch.utils.data import DataLoader

from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.math.seed import (
    get_rng_state,
    salt,
    set_rng_state,
    set_seed_distributed,
    set_seed_local,
)
from priml.metrics.custom_types import MetricProtocol
from priml.paths import resolve_working_dir
from priml.runtime import (
    RuntimeProtocol,
    SingleProcess,
    global_device_mesh,
    is_rank_zero,
    runtime_initialized,
)
from priml.train.checkpointing import Checkpointer
from priml.train.custom_types import (
    CheckpointingProtocol,
    PhaseTimerProtocol,
    ProfileProtocol,
    TrackerProtocol,
    TrainStepProtocol,
)
from priml.train.profiling import PhaseTimer
from priml.train.tracker import scalar_metrics
from priml.train.train_step import TrainStep


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _compile_heartbeat(label: str, *, interval_s: float = 30.0) -> Generator[None]:
    """Log a periodic heartbeat while a (possibly long-compiling) block runs.

    The first ``torch.compile`` of the train step can block for minutes with no
    output, which is indistinguishable from a true hang. This emits a rank-0
    heartbeat every ``interval_s`` seconds reporting elapsed wall time, so a slow
    compile (heartbeats then stops) is visibly distinct from a wedged process
    (heartbeats forever). The thread is a daemon and is always joined on exit, so
    it adds nothing once the block returns.
    """
    if not is_rank_zero():
        yield
        return
    done = threading.Event()
    start = time.perf_counter()

    def beat() -> None:
        while not done.wait(interval_s):
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
        thread.join(timeout=interval_s)


def _current_rank() -> int:
    """Global rank, or 0 when distributed is not initialized."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


@contextlib.contextmanager
def _phase_heartbeat(label: str, *, interval_s: float = 20.0) -> Generator[None]:
    """Log, on EVERY rank, which phase this rank is in while a block runs.

    Unlike :func:`_compile_heartbeat` (rank-0 only), this beats on all ranks so a
    distributed stall is self-diagnosing: a hang where ranks sit in DIFFERENT
    phases (e.g. one rank in the compiled core, another already waiting in a
    metric all-gather) shows each rank's last phase in its own torchrun-captured
    log, pinpointing the divergence WITHOUT an external py-spy. The label should
    name the phase AND the batch (e.g. ``"eval batch 54 eval_loss"``).

    Two complementary signals:

    * a daemon thread logs ``STILL IN PHASE`` every ``interval_s`` -- catches a
      stall whose main thread still releases the GIL (e.g. waiting on a Python
      ``Event`` or a collective that periodically yields);
    * a :mod:`faulthandler` watchdog (``dump_traceback_later``) dumps EVERY
      thread's stack -- including the C frames of a wedged CUDA/NCCL kernel --
      to ``stderr``. This is the part that survives a GIL-HOLDING native hang,
      where the daemon thread can never run: it printed nothing in a real eval
      wedge (rank spinning in a hung CUDA kernel held the GIL), so the Python
      beat alone is insufficient. The beat thread pushes the watchdog deadline
      forward on every beat, so the dump fires ONLY when the beat thread has
      been starved for ``2 * interval_s`` -- exactly the GIL-holding wedge it
      diagnoses, whose frame stacks are static and safe to walk. It must never
      fire against a RUNNING interpreter: ``dump_traceback_later`` walks every
      thread's frames from a C thread WITHOUT the GIL, and doing that against
      mutating frames read garbage (``<invalid frame>``) and finally
      segfaulted a healthy full-set eval whose batches merely outlived the
      timeout (exp9010 r1: SIGSEGV mid-dump number 67).

    Always cancelled/joined on exit; near-zero cost once the block returns.
    """
    rank = _current_rank()
    done = threading.Event()
    start = time.perf_counter()

    def beat() -> None:
        while not done.wait(interval_s):
            # Re-arming resets the watchdog deadline to now + 2*interval_s:
            # while this pure-Python thread can run, the process is not
            # GIL-wedged and the dump (unsafe against running threads, see
            # docstring) stays disarmed.
            faulthandler.dump_traceback_later(
                2.0 * interval_s, repeat=True, file=sys.stderr, exit=False
            )
            logger.warning(
                "[rank %d] STILL IN PHASE %r after %.0fs "
                "(if this never advances, this rank is stuck HERE)",
                rank,
                label,
                time.perf_counter() - start,
            )

    # faulthandler watchdog: dumps ALL thread stacks (C + Python) to stderr if
    # the beat thread stalls for 2*interval_s (a GIL-holding native hang the
    # Python beat cannot observe). ``repeat=True`` keeps a long wedge emitting.
    sys.stderr.write(f"[rank {rank}] >>> ENTER PHASE {label!r}\n")
    sys.stderr.flush()
    faulthandler.dump_traceback_later(
        2.0 * interval_s, repeat=True, file=sys.stderr, exit=False
    )
    thread = threading.Thread(target=beat, name="phase-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        done.set()
        thread.join(timeout=interval_s)


# Defaults are ``Any`` so a bare, unparameterized ``TrainLoop.Config`` accepts
# any specialization -- existing call sites that don't care about the concrete
# step/dataset types keep working unchanged. Code that wants the concrete fields
# (no ``isinstance`` narrow) parameterizes explicitly,
# e.g. ``TrainLoop.Config[TRMTrainStep.Config, PuzzleDataset.Config]``.
_StepConfigT = TypeVar(
    "_StepConfigT",
    bound=Makeable[TrainStepProtocol],
    default=Any,
)
_DatasetConfigT = TypeVar(
    "_DatasetConfigT",
    bound=Makeable[DatasetProtocol],
    default=Any,
)


class EvalTimeLimitError(RuntimeError):
    """Raised when a single eval pass exceeds ``Config.max_eval_time``.

    Surfaced as a hard failure (non-zero exit, no metrics written) so an
    over-budget eval fails the experiment instead of producing a slow,
    incomparable score.
    """


class TrainLoop:
    """Step-based training loop orchestration.

    Bundles TrainStep + dataset + metrics + validation + checkpointing.

    Features:
    - Step-based training with validation
    - Metric tracking and logging
    - Configurable validation frequency
    - Checkpointing and profiling support

    Counters:
    - ``self.step.global_step`` is the authoritative optimizer-step counter
      (drives ``max_steps``, eval/checkpoint cadence, logging). It survives
      resume via the step's own state_dict.
    - ``self.current_epoch`` counts completed dataset passes; it increments
      lazily on each dataloader ``StopIteration`` (see ``_get_next_batch``)
      and drives ``max_epochs`` + epoch-boundary eval. It is checkpointed
      separately under the ``"epoch"`` key.
    - ``self.local_step`` counts steps since process start / last resume
      (used only for the manual GC cadence), and is NOT a checkpoint anchor.

    Example:
      cfg = TrainLoop.Config(
          step=TrainStep.Config(...),
          dataset=ImageNetDataset.Config(...),
          metrics={"accuracy": TopK.Config(k_values=[1, 5])},
      )
      loop = cfg.make()
      loop.train()

    """

    class Config(Fig["TrainLoop"], Generic[_StepConfigT, _DatasetConfigT]):
        """TrainLoop configuration.

        Generic over the ``step`` and ``dataset`` config types. Both parameters
        default, so a bare ``TrainLoop.Config`` behaves as before. Parameterize
        with the concrete leaf configs to read ``cfg.step``/``cfg.dataset`` fields
        without an ``isinstance`` narrow.
        """

        study_name: str = ""
        """Grouping prefix shared across a family of runs (e.g. ``"arcagi1"``).

        Set once on the root experiment; forks inherit it. Combined with
        ``experiment_name`` it forms the run identity that the trainer injects
        into each part's run context (so paths key off ``{study_name}`` /
        ``{experiment_name}``). Auto-derived by launch if empty."""

        experiment_name: str = ""
        """Per-run name within a study (e.g. ``"exp002"``). Auto-derived by
        launch from the experiment function name if empty."""

        base_dir: Path | str | None = "/opt/scratch"
        """Resource root inherited by the dataset subtree."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """Logical run directory inherited by run-output children."""

        doc: str = ""
        """Free-text description of the experiment (hypothesis/changes/outcome).

        Populated by the launcher with the experiment function's docstring, so
        the run says WHAT it is and WHY -- not only its metrics. ``finalize``
        propagates this into the tracker's ``notes`` (e.g. the W&B run overview)
        when a tracker is configured and the experiment left tracker notes
        empty. Set it directly to override the docstring."""

        step: _StepConfigT = field(default_factory=TrainStep.Config)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment] -- default_factory yields the TypeVar default; safe by construction
        dataset: _DatasetConfigT = field(default_factory=DummyDataset.Config)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment] -- default_factory yields the TypeVar default; safe by construction
        metrics: dict[str, Makeable[MetricProtocol]] = field(
            default_factory=dict[str, Makeable[MetricProtocol]],
        )
        checkpointing: Makeable[CheckpointingProtocol] | None = field(
            default_factory=Checkpointer.Config,
        )
        profiling: Makeable[ProfileProtocol] | None = None
        phase_timer: Makeable[PhaseTimerProtocol] = field(
            default_factory=PhaseTimer.Config,
        )
        tracker: Makeable[TrackerProtocol] | None = None

        max_steps: float = math.inf
        """Optimizer-step limit. Defaults to no cap, uniform with the other stop
        conditions (``max_epochs``/``max_time``); every experiment is expected to
        set an explicit bound. The LR schedule horizon is separate
        (``step.total_train_steps``), so this does not affect LR defaults."""
        max_epochs: float = math.inf
        max_time: float = math.inf
        """Time limit in seconds (clock chosen by ``max_time_kind``). Training
        stops if exceeded."""
        max_time_kind: Literal["wall", "train"] = "wall"
        """Which clock ``max_time`` caps.

        ``"wall"`` (the long-standing behavior): wall-clock measured from after
        the warm eval compile -- the first train step's torch.compile and every
        periodic eval count against the cap. ``"train"``: pure training time --
        the clock starts once this process's FIRST train step completes, so that
        whole first step (which carries the one-time train-step torch.compile) is
        excluded, and every mid-loop eval's duration is excluded too. The warm
        eval compile is excluded in both. The budget is per-process -- it is not
        checkpointed, so a resumed run starts a fresh cap."""
        max_eval_time: float = math.inf
        """Wall-clock cap in seconds for a single eval pass. A run that exceeds
        it fails (``EvalTimeLimitError``) rather than silently producing an
        over-budget score."""
        eval_stop_on_time_limit: bool = False
        """If True, stop eval at ``max_eval_time`` and publish partial metrics.

        Default False preserves score integrity: over-budget eval raises. Enable
        only for data-generation jobs where partial artifacts are useful."""
        num_steps_eval: float = 1_000
        num_steps_log: int = 10
        early_train_log_steps: int = 100
        """Log every optimizer step up to this step for startup diagnostics."""
        eval_every_epoch: bool = True
        """Run eval at epoch boundaries. Disable to save time."""
        eval_extras_every_eval: bool = False
        """Forward the full eval payload (non-scalar ``extras``) on every eval.

        Default False keeps the long-standing contract: cadence evals forward
        scalars only, and only the final eval carries ``extras`` -- so a
        payload-consuming tracker (e.g. the ARC ``SignalDumpTracker``) acts
        once per run. Enable so every published eval forwards the payload,
        e.g. per-eval signal dumps for offline ensembling of mid-training
        checkpoints. A consumer that writes a file per eval then needs its own
        retention (e.g. ``SignalDumpTracker.keep_last_n``, mirroring
        ``Checkpointer.keep_last_n``), or per-eval artifacts grow unboundedly
        at eval cadence."""
        eval_warmup_batches: int = 0
        """Eval batches to run once before training timers start."""
        eval_only: bool = False
        """Run a single full-dataset eval on a loaded checkpoint, then exit.

        Skips training; the checkpoint is loaded by the checkpointer's own resume
        policy (``checkpointing.resume`` defaults on, selecting ``resume_step``)
        -- eval_only does not dictate checkpoint reads. The loop evaluates over
        the full eval set, logs ``eval/*`` at the checkpoint's ``global_step``,
        and returns. With ``resume`` off, eval scores fresh weights (warned)."""
        restore_rng_state: bool = True
        """Restore checkpoint RNG state when loading.

        Keep this enabled for training resumes. Eval-only jobs that deliberately
        score an N-GPU checkpoint on a different GPU count may disable it because
        RNG state is irrelevant to deterministic eval and CUDA device counts can
        differ.
        """
        seed: int | None = None
        num_steps_garbage_collect: float = math.inf  # inf for never (auto GC)

        # Mesh dimensions for seeding (only used in distributed mode)
        mesh_dim_model_seed: str = "pp"  # Model weights same across this dimension
        mesh_dim_data_seed: str = "dp"  # Data loading different across this dimension

        # Only used by `run`.
        runtime: Makeable[RuntimeProtocol] = field(default_factory=SingleProcess.Config)

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
            # base_dir injection, on the ``base_dir is None`` children:
            #   - The dataset is always a SHARED corpus: it inherits the bare
            #     resource root (e.g. /opt/scratch) so every run reads the same
            #     data, whatever its logical ``working_dir``.
            #   - Run-output children (checkpoints, profiling, tracker, per-run
            #     artifact/dump metrics) inherit THIS run's resolved directory.
            #   - A metric can be EITHER: one that reads a shared corpus (its
            #     logical ``working_dir`` names a ``/datasets/...`` tree, e.g. an
            #     ARC pass@k metric reading identifiers/test_puzzles) inherits the
            #     root; every other metric (an artifact dump) inherits the run dir.
            # The dataset always inherits the bare resource root.
            if (
                isinstance(self.dataset, HasNormalizedWorkingDirPattern)
                and self.dataset.base_dir is None
            ):
                self.dataset.base_dir = self.base_dir
            # Run-output children inherit this run's resolved directory.
            for part in (
                self.step,
                self.checkpointing,
                self.profiling,
                self.phase_timer,
                self.tracker,
            ):
                if (
                    isinstance(part, HasNormalizedWorkingDirPattern)
                    and part.base_dir is None
                ):
                    part.base_dir = self.working_dir
            # A metric reading a shared ``/datasets/...`` corpus inherits the
            # bare root; every other (artifact/dump) metric inherits the run dir.
            for metric in self.metrics.values():
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
        # A training run with no finite stop condition would loop forever. Fail
        # before building the runtime/model/dataset. eval_only never trains, so
        # the all-infinite case is fine there.
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

        # Strategy for seeding:
        #   runtime.initialize() →
        #   → Salt by mesh_dim_model_seed → Load model
        #   → Salt by mesh_dim_data_seed → Load dataset

        self.runtime = config.runtime.make()
        self._owns_runtime = not runtime_initialized()
        self._runtime_destroyed = False
        if self._owns_runtime:
            self.runtime.initialize()

        # If anything in the body below raises, the runtime must be torn
        # down -- otherwise ``MultiProcess`` leaks the device mesh /
        # process group for the lifetime of the python process.
        try:
            # Setup seed suitable for model loading.
            #
            # Distributed path: one broadcast establishes ``base_seed``;
            # all subsequent derivation happens locally via ``salt``.
            # Two broadcasts bracketing arbitrary user code (model_init,
            # dataset.make()) would deadlock if any rank skipped the
            # second one due to an earlier exception.
            mesh = global_device_mesh()
            if mesh:
                base_seed, _ = set_seed_distributed(
                    config.seed,
                    mesh=mesh[config.mesh_dim_model_seed],
                    salt_by_rank=True,
                )
            else:
                base_seed = set_seed_local(config.seed)

            # Setup components.
            self.phase_timer = config.phase_timer.make()
            with self.phase_timer.phase("model_init"):
                self.step = config.step.make()
            cast(_HasTimer, self.step).timer = self.phase_timer
            self.metrics = {name: cfg.make() for name, cfg in config.metrics.items()}

            if mesh:
                # Derive the data seed locally -- no collective.
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
            logger.info("TrainLoop startup: dataset ready.")

            # Setup checkpointing. The checkpointer is driven against this
            # TrainLoop (passed as the target to each call).
            logger.info("TrainLoop startup: creating checkpointer.")
            self.checkpointing: CheckpointingProtocol | None = (
                config.checkpointing.make() if config.checkpointing else None
            )
            logger.info("TrainLoop startup: checkpointer ready.")

            # Store training hyperparameters
            self.current_epoch = 0
            self.local_step = 0
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
            # Pure-train clock base: ``train/elapsed`` (and the
            # ``max_time_kind="train"`` budget) read ``perf_counter() - base``.
            # Rebased once the first train step's compile finishes and advanced
            # past each mid-loop eval, so neither counts as training time.
            self._train_clock_base = self._start_time

            # Setup tracker. W&B runs only on rank 0; the barrier keeps other
            # ranks from racing ahead into train/eval collectives while rank 0
            # is still initializing or falling back to a no-op tracker.
            logger.info("TrainLoop startup: creating tracker.")
            self.tracker: TrackerProtocol | None = (
                config.tracker.make() if config.tracker else None
            )
            # Record the experiment description as run notes via the tracker's
            # own log_notes (W&B sets its run overview; notes-less trackers
            # ignore). Done post-make so the launcher stays tracker-agnostic --
            # it only sets ``doc`` -- and no caller reaches into tracker config
            # internals. An explicitly-configured note wins (log_notes skips a
            # non-empty existing note).
            if self.tracker is not None and config.doc:
                self.tracker.log_notes(config.doc)
            logger.info("TrainLoop startup: tracker ready.")
            _barrier_if_distributed("tracker startup")

            # Setup profiling
            logger.info("TrainLoop startup: creating profiler.")
            self.profiling: ProfileProtocol | None = (
                config.profiling.make() if config.profiling else None
            )
            logger.info("TrainLoop startup: profiler ready.")

            # Garbage collection control
            if math.isfinite(config.num_steps_garbage_collect):
                gc.disable()

            # Resume + overwrite-guard, before the first training step: the
            # checkpointer restores per its config, then refuses to start if a
            # future save would clobber an existing checkpoint -- caught at
            # startup, not thousands of steps in. eval_only writes nothing, so
            # its guard is skipped.
            if self.checkpointing is not None:
                logger.info(
                    "TrainLoop startup: loading checkpoint (max_steps=%s, guard=%s).",
                    self.max_steps,
                    not self.eval_only,
                )
                self.checkpointing.load(
                    self, max_steps=self.max_steps, guard=not self.eval_only
                )
                logger.info(
                    "TrainLoop startup: checkpoint load complete (global_step=%d).",
                    self.step.global_step,
                )

            self.train_loader = None
            self.train_iter = None
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
        except BaseException:
            if self._owns_runtime and not self._runtime_destroyed:
                self.runtime.destroy()
                self._runtime_destroyed = True
            raise

    def train(self) -> None:
        """Run step-based training loop."""
        try:
            if self.eval_only:
                self._run_eval_only()
                return
            trained_any = False
            while (
                self.step.global_step < self.max_steps
                and (n := self.current_epoch) < self.max_epochs
                and not self._time_limit_reached()
                and not self._should_stop_early()
            ):
                batch = self._get_next_batch()
                if self.current_epoch == n + 1:
                    # Flush/discard any partial gradient accumulation from the
                    # epoch that just ended before the new epoch's first
                    # micro-batch is processed (no cross-epoch grad mixing).
                    self.step.on_epoch_end()
                    if self.local_step > 0 and self.eval_every_epoch:
                        eval_start = time.perf_counter()
                        eval_metrics = self.eval()
                        eval_time = time.perf_counter() - eval_start
                        # Eval is not training time: pause the pure-train clock.
                        self._train_clock_base += eval_time
                        self._publish_eval_metrics(
                            eval_metrics,
                            eval_time=eval_time,
                            step=self.current_epoch,
                            is_final=False,
                        )
                self._maybe_garbage_collect()
                # A resumed loop may start exactly on a checkpoint/eval cadence
                # step. Do not re-save or re-score that restored state before
                # this process has advanced training at least once.
                if self.local_step > 0 and self.checkpointing is not None:
                    self.checkpointing.maybe_save(self, self.step.global_step)
                if self.local_step > 0:
                    self._maybe_eval()
                self._do_train_step(batch)
                trained_any = True

            # Final checkpoint + evaluation.
            elapsed = self._max_time_elapsed()
            if elapsed >= self.max_time:
                logger.warning(
                    "Time limit reached (%.1fs >= %.1fs), stopping early at step %d.",
                    elapsed,
                    self.max_time,
                    self.step.global_step,
                )
            # Skip the end-of-training checkpoint / eval entirely when no step
            # ran (e.g. max_steps already reached on resume); there is nothing
            # new to persist or measure.
            if not trained_any:
                return
            if self.checkpointing is not None:
                self.checkpointing.save(self, self.step.global_step)
            self.phase_timer.log_summary()
            # The cadence eval inside the loop never lands on the final step (the
            # while-loop exits once global_step reaches max_steps), so run one
            # last eval here and mark it final: it emits the RESULT line and
            # runs final artifacts. ``num_steps_eval=inf`` disables eval entirely
            # (e.g. a profiling harness timing only train steps), so the final
            # eval is skipped too -- no pass holds the GPU after the last step.
            # (A subclass may suppress this final eval via ``_maybe_eval`` when its
            # last cadence eval already produced the run's terminal metrics.)
            self._maybe_eval(is_final=True)
        finally:
            self.phase_timer.log_summary()
            self._cleanup()

    def _time_limit_reached(self) -> bool:
        """Whether the ``max_time`` cap has elapsed, agreed by all ranks.

        ``max_time_kind`` selects the measured clock: wall seconds since
        ``_start_time``, or the pure-train clock (first-step compile and
        mid-loop evals excluded). Ranks have slightly different clocks and step
        pace, so a purely local check lets them disagree on which step is the
        last: one rank exits the loop and enters the collective final ``eval()``
        while another keeps training, and the eval's all-reduce deadlocks (GPUs
        hang, no timeout).

        Rank 0 is authoritative -- it reads the clock and broadcasts the stop
        flag, so every rank adopts the same decision regardless of skew. We
        broadcast only every ``num_steps_log`` steps (and latch true once set),
        not every step: ``max_time`` is a soft cap, so overshooting by up to a
        few steps is fine, and this keeps the extra collective off the hot path.
        """
        if self.max_time == math.inf:
            return False
        if not torch.distributed.is_initialized():
            return self._max_time_elapsed() >= self.max_time
        if self._time_limit_latched:
            return True
        # Only sync on the log cadence; between checks, keep training.
        if self.step.global_step % self.num_steps_log != 0:
            return False
        # Only rank 0 reads the clock; broadcast makes its decision authoritative
        # so every rank stops on the same step (no skew-driven eval deadlock).
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

    def _train_elapsed(self) -> float:
        """Pure-train seconds: first-step compile and mid-loop evals excluded."""
        return time.perf_counter() - self._train_clock_base

    def _should_stop_early(self) -> bool:
        """Whether training should stop before ``max_steps`` / ``max_time``.

        No-op in the base loop (always ``False``). A subclass overrides it to end
        the run once its objective is met -- e.g. a time-to-target loop stops once
        the watched metric has crossed its target, since further training cannot
        improve the already-latched score and only burns compute. An override MUST
        return a rank-agreed verdict (broadcast rank 0's decision, like
        :meth:`_time_limit_reached`): the latch may be computed on rank 0 only
        (the eval payload that sets it runs under the rank-0 tracker), so a purely
        local read would desync ranks and deadlock the next collective eval.
        """
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

    def _run_eval_only(self) -> None:
        """Score the loaded checkpoint with one eval, then return.

        The checkpoint is loaded in ``__init__`` by the checkpointer's own resume
        policy (``resume`` defaults on); eval_only does not dictate checkpoint
        reads. This runs a single eval over the eval set as the run's final eval
        -- logging ``eval/*`` and emitting the ``RESULT`` line -- via the same
        path as the end-of-training eval. No training step, optimizer step, or
        checkpoint write occurs.
        """
        if self.step.global_step == 0:
            logger.warning(
                "eval_only at global_step=0: no checkpoint was loaded "
                "(resume found none). Evaluating the freshly-initialized model.",
            )
        self._maybe_eval(is_final=True, force=True)

    def _publish_eval_metrics(
        self,
        eval_metrics: dict[str, Any],
        *,
        eval_time: float,
        step: int,
        is_final: bool,
    ) -> dict[str, float]:
        """Publish eval metrics to the tracker; final-ness is expressed as data.

        On the final eval the full ``eval_metrics`` (including any non-scalar
        ``extras`` payload) is forwarded so payload-consuming child trackers see
        it; cadence evals forward scalars only, unless ``eval_extras_every_eval``
        opts the run into per-eval payload forwarding. Each child tracker keeps
        only the keys it understands.
        """
        eval_scalar_metrics = scalar_metrics(eval_metrics)
        if self.tracker:
            payload: dict[str, Any] = (
                dict(eval_metrics)
                if is_final or self.eval_extras_every_eval
                else dict(eval_scalar_metrics)
            )
            payload["time"] = eval_time
            payload.update(
                self._extra_eval_payload(eval_scalar_metrics, step, is_final=is_final)
            )
            self.tracker.log_metrics(payload, step, prefix="eval/")
        return eval_scalar_metrics

    def _extra_eval_payload(
        self,
        eval_scalar_metrics: dict[str, float],
        step: int,
        *,
        is_final: bool,
    ) -> dict[str, Any]:
        """Extra ``eval/``-prefixed keys to merge into the eval payload.

        Empty in the base loop. Subclasses (e.g. a time-to-target loop) override
        this to inject objective-specific scalars derived from the eval; it is
        called once per published eval, after the scalar metrics are computed.
        """
        del eval_scalar_metrics, step, is_final
        return {}

    def _get_next_batch(self) -> dict[str, Any]:
        """Get next batch, handle epoch boundary and eval at end of epoch.

        Returns:
            batch: Next training batch

        """
        batch_start = time.perf_counter()
        if self.train_loader is None:
            logger.info(
                "TrainLoop step %d: creating train dataloader.",
                self.step.global_step + 1,
            )
            self.train_loader = self.dataset.train_dataloader()
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
                if self.local_step < self.early_train_log_steps:
                    logger.info(
                        "TrainLoop step %d: fetching raw train batch.",
                        self.step.global_step + 1,
                    )
                batch = next(self.train_iter)
                batch_time = time.perf_counter() - batch_start
                if self.local_step < self.early_train_log_steps:
                    logger.info(
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
                if self.local_step < self.early_train_log_steps:
                    logger.info(
                        "TrainLoop step %d: preprocessing train batch.",
                        self.step.global_step + 1,
                    )
                batch = self.step.preprocess_batch(batch)
                if self.local_step < self.early_train_log_steps:
                    logger.info(
                        "TrainLoop step %d: train batch preprocessed.",
                        self.step.global_step + 1,
                    )
                return batch
            except StopIteration:
                self.current_epoch += 1
                logger.info(
                    "TrainLoop step %d: train iterator exhausted; "
                    "advancing to epoch %d.",
                    self.step.global_step + 1,
                    self.current_epoch,
                )
                _set_loader_epoch(self.train_loader, self.current_epoch)
                self.train_iter = iter(self.train_loader)
        raise RuntimeError("Failed to get next batch after epoch reset")

    def _on_train_step_timed(self, step_time: float, *, is_first: bool) -> None:
        """Hook called after each train step with its wall-clock duration.

        No-op in the base loop. Subclasses override it to accumulate training
        wall-clock for a time-to-target objective. ``is_first`` marks this
        process's first step, whose ``step_time`` includes the backward-graph
        compile and is therefore excluded from "training time".
        """
        del step_time, is_first

    def _do_train_step(self, batch: dict[str, Any]) -> None:
        """Execute one training step with profiling and logging."""
        if self.profiling:
            self.profiling.on_step_start(self.step.global_step)

        next_step = self.step.global_step + 1
        if is_rank_zero() and self.local_step < self.early_train_log_steps:
            elapsed = time.perf_counter() - self._start_time
            logger.info(
                "Entering train step %d/%s (elapsed=%.0fs)",
                next_step,
                self.max_steps,
                elapsed,
            )

        step_start = time.perf_counter()
        # The first train step compiles the backward graph and can block for
        # minutes with no output (looks like a hang). Wrap the early steps in a
        # heartbeat so a slow compile is distinguishable from a wedged process.
        if self.local_step < self.early_train_log_steps:
            with _compile_heartbeat(f"train step {next_step}"):
                step_results = self.step.train_step(**batch)
        else:
            step_results = self.step.train_step(**batch)
        step_time = time.perf_counter() - step_start
        self.local_step += 1
        # Hook for subclasses that accumulate train wall-clock (e.g. a
        # time-to-target loop). ``is_first`` flags this process's first step,
        # which carries the one-time train-step compile. Eval is excluded
        # automatically (it runs outside this method).
        self._on_train_step_timed(step_time, is_first=self.local_step == 1)
        if self.local_step == 1:
            # Rebase the pure-train clock AFTER the first step: that whole step
            # (which carries the one-time compile) is excluded from "train" time.
            self._train_clock_base = time.perf_counter()

        if self.profiling:
            self.profiling.on_step_end(self.step.global_step)

        # Log every startup step, then on the ``num_steps_log`` cadence. This
        # gates BOTH the console line and the tracker upload: after startup,
        # emitting train metrics every step floods the tracker's history stream
        # (hundreds of keys/step at sub-second cadence), whose server-side
        # ingestion then lags the live run by tens of thousands of steps.
        log_step = (
            self.step.global_step == 1
            or self.local_step <= self.early_train_log_steps
            or self.step.global_step <= self.early_train_log_steps
            or self.step.global_step % self.num_steps_log == 0
        )
        if not log_step:
            return

        loss = step_results["loss"].mean().detach()
        raw_step_metrics: dict[str, float | Tensor] = step_results.get("metrics", {})
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
        if not is_rank_zero():
            return

        loss_value = float(reduced_values[0].item())
        elapsed = time.perf_counter() - self._start_time
        step_metric_tensors = {
            key: reduced_values[i + 1] for i, key in enumerate(step_metric_keys)
        }
        step_metrics: dict[str, float] = {
            key: float(
                step_metric_tensors[key].item() if key in step_metric_tensors else value
            )
            for key, value in raw_step_metrics.items()
        }
        # Append the step's scalar metrics to the console line so training
        # accuracy is greppable from the job log alone (e.g. when the tracker
        # backend is unavailable), not only on the dashboard.
        extra = " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items())
        line = (
            f"Step {self.step.global_step}/{self.max_steps}: "
            f"loss={loss_value:.4f} step_time={step_time:.3f}s "
            f"elapsed={elapsed:.0f}s"
        )
        logger.info(f"{line} {extra}" if extra else line)

        if self.tracker:
            metrics: dict[str, float] = {
                "total_loss": loss_value,
                "step_time": step_time,
                "time_since_start": elapsed,
                # Pure-train seconds (vs wall ``time_since_start``): the clock
                # a ``max_time_kind="train"`` budget charges against.
                "elapsed": self._train_elapsed(),
                **step_metrics,
            }
            if torch.cuda.is_available():
                # Cumulative peaks since the last torch CUDA memory-stat reset.
                metrics["gpu_mem_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / 1e9
                )
                metrics["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            self.tracker.log_metrics(metrics, self.step.global_step, prefix="train/")

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
        logger.info(f"GC at local_step {self.local_step} (gc_time={gc_time:.3f}s)")

    def _maybe_eval(self, *, is_final: bool = False, force: bool = False) -> None:
        """Run validation and publish it; the final eval also writes metrics.

        The single eval-and-report path. In the training loop it fires on the
        step cadence (``global_step % num_steps_eval == 0``, before the train
        step for online-learning semantics). At the end of training it is called
        once with ``is_final=True``: that eval emits the ``RESULT`` line and
        forwards the full payload (including any ``extras``), since the cadence
        eval never lands on the final step and forwards scalars only (unless
        ``eval_extras_every_eval`` opts cadence evals into the full payload).
        ``num_steps_eval=inf`` disables eval entirely (cadence and final);
        ``force`` overrides that gate for the ``eval_only`` path, whose single
        eval is always the final one regardless of step or cadence.

        Args:
            is_final: This is the run's last eval; emit RESULT and write metrics.
            force: Run regardless of the step/cadence gate (``eval_only``).

        """
        # eval_only (force) always runs. Otherwise num_steps_eval=inf disables
        # eval entirely; within an enabled run, the final eval always fires and
        # a cadence eval fires only on its step boundary (never step 0).
        if not force:
            if not math.isfinite(self.num_steps_eval):
                return
            cadence_due = (
                self.step.global_step != 0
                and self.step.global_step % self.num_steps_eval == 0
            )
            if not is_final and not cadence_due:
                return
        eval_start = time.perf_counter()
        eval_metrics = self.eval()
        eval_time = time.perf_counter() - eval_start
        # A cadence eval is not training time: pause the pure-train clock. The
        # final eval runs after training ended, so there is nothing to pause.
        if not is_final:
            self._train_clock_base += eval_time
        scalar_metrics = self._publish_eval_metrics(
            eval_metrics,
            eval_time=eval_time,
            step=self.step.global_step,
            is_final=is_final,
        )
        if is_rank_zero():
            if is_final:
                elapsed = time.perf_counter() - self._start_time
                parts = [f"steps={self.step.global_step}", f"time={elapsed:.1f}s"]
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

    def _cleanup(self) -> None:
        """Cleanup resources after training."""
        if self.checkpointing is not None:
            self.checkpointing.close()
        if self.tracker:
            self.tracker.close()
        if self.profiling:
            self.profiling.cleanup()
        if math.isfinite(self.num_steps_garbage_collect):
            gc.enable()
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
            batch = self.step.preprocess_batch(raw_batch)
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

    def _cuda_event_pair(self) -> tuple[Any, Any] | None:
        """Start a CUDA event pair when event timing is enabled."""
        if (
            not getattr(self.phase_timer, "cuda_events_enabled", False)
            or not torch.cuda.is_available()
        ):
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _record_cuda_timing(
        self,
        name: str,
        events: tuple[Any, Any] | None,
    ) -> None:
        """End and enqueue a CUDA event pair when event timing is enabled."""
        if events is None:
            return
        _, end = events
        end.record()
        self.phase_timer.record_cuda_events(name, *events)

    def eval(self) -> dict[str, Any]:
        """Run validation."""
        for metric in self.metrics.values():
            metric.reset()

        eval_loader = self.dataset.eval_dataloader()
        total_loss = 0.0
        total_batch_time = 0.0
        total_step_metrics: dict[str, float] = {}
        num_batches = 0
        total_weight = 0
        try:
            total_batches = len(eval_loader)
        except TypeError:
            total_batches = 0
        narrate = is_rank_zero()
        log_every = max(1, total_batches // 20) if total_batches else 50

        eval_start = time.perf_counter()
        batch_start = eval_start
        for raw_batch in eval_loader:
            # Fail fast once the eval pass blows its wall-clock budget, rather
            # than letting a runaway test-time-scaling config (e.g. very large
            # PTRM K over the full eval set) run for hours before failing.
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
            batch = self.step.preprocess_batch(raw_batch)
            weight = int(batch.get("valid_count", 1))
            if weight == 0:
                batch_start = time.perf_counter()
                continue
            # Phase heartbeat (all ranks): names the exact phase + batch this rank
            # is in, so a distributed eval stall reports WHERE each rank is stuck
            # (compiled forward / ACT loop vs a downstream metric all-gather)
            # without an external py-spy. eval_loss is the per-rank compute phase;
            # metric.update below is the cross-rank gather phase.
            with _phase_heartbeat(f"eval batch {num_batches + 1} eval_loss"):
                step_results = self.step.eval_loss(**batch)
            loss = step_results["loss"]
            model_output = step_results["model"]
            total_weight += weight
            total_loss += loss.mean().item() * weight
            for key, value in step_results.get("metrics", {}).items():
                total_step_metrics[key] = (
                    total_step_metrics.get(key, 0.0)
                    + float(
                        value.mean().item()
                        if isinstance(value, torch.Tensor)
                        else value,
                    )
                    * weight
                )
            # Per-batch eval step time, mirroring train/step_time. Summed here
            # and reported as a mean per eval (eval runs many batches at one
            # global_step, so a single mean keeps the W&B step axis monotonic).
            batch_dt = time.perf_counter() - batch_start
            total_batch_time += batch_dt
            num_batches += 1

            extra_votes = step_results.get("eval_extra_votes")
            for name, metric in self.metrics.items():
                cuda_events = self._cuda_event_pair()
                # A metric ``update`` that all-gathers across ranks is a classic
                # deadlock point: if one rank entered it while another is still in
                # eval_loss, the collective never completes. The heartbeat names
                # the metric + batch so the stuck phase is attributable.
                with _phase_heartbeat(
                    f"eval batch {num_batches} metric[{name}].update"
                ):
                    metric.update(model_output, **batch)
                    # Extra candidates (e.g. WTA's K heads) are fed as additional
                    # update() calls so accumulating metrics (pass@K voting) gain
                    # extra votes; scalar metrics already came from the primary
                    # path.
                    if extra_votes:
                        for extra_output, extra_batch in extra_votes:
                            metric.update(extra_output, **extra_batch)
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
                    total_loss / total_weight,
                    batch_dt,
                    total_batch_time / num_batches,
                    total_batch_time,
                )
            batch_start = time.perf_counter()

        # Collect results with metric name prefix
        results: dict[str, Any] = {}

        # Only add eval total_loss if we processed any batches
        if total_weight > 0:
            results["total_loss"] = total_loss / total_weight
            results["mean_batch_time"] = total_batch_time / num_batches
            for key, value in total_step_metrics.items():
                results[key] = value / total_weight

        for name, metric in self.metrics.items():
            cuda_events = self._cuda_event_pair()
            metric_results = metric.compute()
            self._record_cuda_timing(f"eval_metric_{name}_compute", cuda_events)
            for key, value in metric_results.items():
                results[f"{name}_{key}" if name else key] = value

        return results

    def run(self, *args: str) -> None:
        """Run training (entry point for experimental.lib.launch)."""
        del args
        self.train()

    def state_dict(self) -> dict[str, Any]:
        """Get training state for checkpointing."""
        state: dict[str, Any] = {
            "step": self.step.state_dict(),
            "dataset": self.dataset.state_dict(),
            "metrics": {
                name: metric.state_dict() for name, metric in self.metrics.items()
            },
            "epoch": self.current_epoch,
            "rng": get_rng_state(),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load training state from checkpoint (full restore for resume)."""
        self.step.load_state_dict(state_dict["step"])
        self.dataset.load_state_dict(state_dict["dataset"])
        for name, metric_state in state_dict["metrics"].items():
            if name in self.metrics:
                self.metrics[name].load_state_dict(metric_state)
        self.current_epoch = state_dict["epoch"]
        self.local_step = 0

        # Restore RNG states
        if self.restore_rng_state and "rng" in state_dict:
            set_rng_state(state_dict["rng"])


class _HasTimer(Protocol):
    timer: PhaseTimerProtocol


@runtime_checkable
class _SupportsSetEpoch(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


def _set_loader_epoch(loader: DataLoader[Any], epoch: int) -> None:
    """Inform the loader's dataset of the current epoch before (re)iteration.

    Epoch state must originate in the main process: a ``num_workers>0`` loader
    re-forks fresh per-worker sources each epoch, so the dataset wrapper folds
    this epoch into the per-worker shuffle seed. Loaders whose dataset does not
    support ``set_epoch`` (cached lists, third-party datasets) are left as-is.

    Args:
      loader: The training DataLoader about to be (re)iterated.
      epoch: Zero-based epoch index.

    """
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
