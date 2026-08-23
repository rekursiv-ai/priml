"""TrainStep: the model, its optimizer, and one training step.

What every recipe shares -- the model, the optimizer, device placement, the
timers a budget and a schedule read -- plus a supervised implementation of the
step itself: forward, loss, backward, accumulate, update.

A recipe whose step is that sequence inherits the whole thing and configures
the ``loss`` slot. A recipe that computes its loss inline, runs several
optimizer updates per batch, or must act between the backward and the update
overrides :meth:`train_step` and :meth:`eval_loss`; everything else -- the
optimizer build, the counters, the checkpoint, the learning-rate scaling --
still comes from here.

Features:
- Gradient clipping and gradient accumulation
- Model and data parallelism (TP, FSDP, DDP)
- Activation checkpointing
- torch.compile support
- EMA
- Budgets in steps, seconds, or passes over the data
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Any, Literal, cast

import contextlib
import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn

import torch
import torch.amp
import torch.distributed as dist

from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss
from priml.math.schedules import Schedule, constant
from priml.model.special import Identity
from priml.runtime import global_device_mesh
from priml.timer import CheckpointableStepTimer
from priml.train.activation import DefaultActivationStorage
from priml.train.custom_types import (
    ActivationMemoizationProtocol,
    EMAProtocol,
    ModelOutput,
    ModelQuantizationProtocol,
    OptimizerProtocol,
    ParallelStrategyProtocol,
    TrainStepOutput,
)
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel
from priml.train.quantization import NoModelQuantization


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class TrainStep:
    """Model plus optimization, and one step of training.

    Example:
      cfg = TrainStep.Config(
          model=Mamba2DClassifier.Config(),
          loss=CrossEntropyLoss.Config(),
          parallelism=FullySharded.Config(mesh_dim="dp"),
          gradient_clip_norm=1.0,
          accumulate_grad_batches=4,
      )
      step = cfg.make()
      result = step.train_step(media=images, label=targets)
      grad_norm = step.last_grad_norm  # Track gradient norm

    A subclass overrides :meth:`train_step` and :meth:`eval_loss` -- the two
    genuinely per-recipe members -- and drives the update through :meth:`step`,
    or through :meth:`apply_learning_rate` plus its own optimizer call.

    """

    class Config(Fig["TrainStep"], kw_only=False):
        """Configuration for TrainStep."""

        model: Makeable[nn.Module] = field(default_factory=Identity.Config)
        """The network being trained; every other slot serves it."""

        _: KW_ONLY

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.AdamW,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            ),
        )
        """Builds the optimizer from the model.

        Called with the MODEL, not a parameter list, because a split recipe
        routes by parameter NAME -- ``CompositeOptimizer`` reads
        ``named_parameters()`` -- and a list of tensors has no names. A torch
        optimizer, which wants parameters, is adapted by
        :meth:`TrainStep.build_optimizer`; anything wanting the module gets it
        untouched. That is what lets one slot hold both conventions.

        Typed on torch's concrete class rather than ``OptimizerProtocol``:
        ``Makeable`` is covariant in what it builds, so a
        ``PartialConfig(torch.optim.Adam)`` -- the ordinary spelling -- fits
        the protocol slot only if ``Adam`` is declared to implement it, which
        torch's own class is not."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=SimpleLoss.Config,
        )
        """Maps the model's output and the batch to an unreduced loss.

        Read by the supervised :meth:`train_step` alone. A subclass computing
        its loss inline -- because the terms depend on state the batch does not
        carry -- leaves this at its default and never calls it."""

        accumulate_grad_batches: int = 1
        """Micro-batches accumulated before one optimizer update."""

        drop_partial_accumulation_on_epoch_end: bool = True
        """Discard a partial gradient accumulation at each epoch boundary.

        Default ``True``: when an epoch ends mid-accumulation the pending
        micro-batch gradients are zeroed and the counters reset, so gradients
        never mix across epochs. This matches recursion-style training (e.g.
        TRM) where epoch boundaries are semantically real. Set ``False`` for
        pure IID-shuffled data, where carrying the partial accumulation across
        the boundary is harmless and avoids wasting micro-batches.
        """

        train_budget_steps: float = math.inf
        """Optimizer steps the schedule anneals over; infinite leaves it out.

        A HORIZON, not a stop condition: the loop's ``max_steps`` ends the run,
        and this is what the learning rate is annealed against. They are
        normally equal, and ``TrainLoop`` derives the stop from this when the
        experiment states only one -- a schedule whose horizon differs from the
        run's length anneals past the end or short of it."""

        train_budget_sec: float = math.inf
        """Training seconds the schedule anneals over; infinite leaves it out.

        A run budgeted in TIME cannot know its step count in advance, so a
        step-indexed schedule could not be written for it -- and a change that
        makes a step cheaper is then rewarded with more steps at the same
        schedule shape, which is the comparison a time budget exists to
        make."""

        train_budget_epochs: float = math.inf
        """Passes over the data the run covers; infinite leaves it out.

        A third axis rather than a step count in disguise: how many times the
        data has been seen is a statement about COVERAGE, and converting it
        needs a loader length that the config cannot know and a stream does not
        have. Counted by the dataset, which owns the only boundary that can
        say a pass ended."""

        learning_rate_scheduler: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(constant),
        )
        """Maps the learning schedule's progress to a rate multiplier.

        A plain function, not a ``torch.optim.lr_scheduler``: those hold the
        optimizer and mutate its groups from a counter they own, so their state
        must be checkpointed and can desync from the run they schedule. A
        function of progress is recomputed from the budget every step and
        cannot. :mod:`priml.math.schedules` carries the usual curves.

        The multiplier scales each group's ``initial_lr``, so a recipe running
        several rates at once keeps their ratios."""

        parallelism: Makeable[ParallelStrategyProtocol] = field(
            default_factory=NoParallel.Config,
        )
        """Owns device placement, sharding, and meta materialization."""

        model_quantization: Makeable[ModelQuantizationProtocol] = field(
            default_factory=NoModelQuantization.Config,
        )
        """Rewrites modules before sharding, so the strategy sees the final graph."""

        activation_memoization: Makeable[ActivationMemoizationProtocol] = field(
            default_factory=DefaultActivationStorage.Config,
        )
        """How activations are kept for backward: stored, recomputed, or quantized."""

        compile: Makeable[Callable[[Callable[..., Any]], Callable[..., Any]]] | None = (
            field(
                default_factory=lambda: PartialConfig(
                    torch.compile,
                    fullgraph=True,
                ),
            )
        )
        """Wraps the model before its first forward; ``None`` runs eager."""

        ema: Makeable[EMAProtocol] = field(default_factory=NoEMA.Config)
        """Weight-averaging shadow, applied after each optimizer update."""

        gradient_clip_norm: float = math.inf
        """Global gradient-norm ceiling; infinite disables clipping."""

        device_init: Literal["meta", "eager"] = "eager"
        """HOW the model's storage is allocated -- never WHERE, which is
        ``parallelism``'s alone.

        ``"eager"`` allocates during construction, then the parallel strategy
        moves the result. ``"meta"`` constructs without storage and lets that
        strategy materialize it instead: allocation happens once, already on
        the target device and already sharded, so init draws from THAT device's
        generator and no parameter crosses the bus.

        ``"meta"`` is the better default and is NOT yet the default, because it
        requires every container module to implement a ``reset_parameters``
        that recurses into what it built (see ``materialize_meta``'s ownership
        contract). Most models here inherit torch's leaf implementations
        without declaring the root one, so materializing leaves them NaN and
        the audit refuses the run. Fixing those models is what makes the flip
        safe.

        A device here would be a second answer to a question ``parallelism``
        already answers, and the two would agree only by coincidence."""

        dtype_autocast: torch.dtype | None = None
        """Autocast dtype for forward and loss (e.g. ``torch.bfloat16``);
        ``None`` disables autocast entirely."""
        autocast_cache_enabled: bool = False
        """Enable autocast's weight cache. Default False preserves exact
        numerics across forward calls; True trades a small numeric difference
        for reusing cast weights within a forward (perf)."""

    def __init__(self, config: Config) -> None:
        self.config = config

        if config.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {config.gradient_clip_norm}",
            )
        if config.accumulate_grad_batches <= 0:
            raise ValueError(
                "accumulate_grad_batches must be positive, got "
                f"{config.accumulate_grad_batches}",
            )
        for name, budget in (
            ("train_budget_steps", config.train_budget_steps),
            ("train_budget_sec", config.train_budget_sec),
            ("train_budget_epochs", config.train_budget_epochs),
        ):
            # NaN is excluded explicitly rather than covered by ``<= 0``: every
            # comparison against it is False, so a NaN here would not fail --
            # it would divide into a NaN progress and leave the rate undefined.
            if math.isnan(budget) or budget <= 0:
                raise ValueError(f"{name} must be positive; got {budget}.")

        self.gradient_clip_norm = config.gradient_clip_norm
        self.timer_forward = CheckpointableStepTimer()
        """Training forward passes: how many, and how long they took."""

        self.timer_eval = CheckpointableStepTimer()
        """Evaluation forward passes: how many, and how long they took."""

        self.timer_step = CheckpointableStepTimer()
        """Optimizer updates: how many, and how long they took.

        What both progress properties read, so what this timer wraps IS what
        the budget bounds -- compilation before the first update and
        evaluation between them fall outside it and are never charged."""

        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the training data: how many, and how long each took.

        Replaced by the loader's own via :meth:`bind_epoch_timer`, since only
        the loader knows when the data ran out. The dataset checkpoints it;
        this class does not, because it does not own it. Left at this
        standalone default it simply never ticks, which is the right reading
        for a step trained on a stream that has no pass -- and what makes
        ``train_budget_epochs`` unreachable rather than wrong there."""

        # Checked rather than trusted to the annotation: a ``--override`` or a
        # deserialized config carries unchecked text, and a device name here
        # would silently build somewhere ``parallelism`` never chose.
        if config.device_init not in ("meta", "eager"):
            raise ValueError(
                f"device_init must be 'meta' or 'eager'; got "
                f"{config.device_init!r}. It names how storage is allocated, "
                "not where -- set the device on ``parallelism``.",
            )
        if config.device_init == "eager":
            model = self.config.model.make()
        else:
            with torch.device("meta"):
                model = self.config.model.make()

        # Quantization rewrites modules BEFORE the parallel strategy so the
        # strategy shards/materializes the final module graph.
        model_quantization = self.config.model_quantization.make()
        model = model_quantization(model)

        # Single strategy owns device assignment, sharding, meta->real
        # materialization, and post-shard reset_parameters.
        self.parallelism: ParallelStrategyProtocol = self.config.parallelism.make()
        model = self.parallelism(model)

        activation_strategy = self.config.activation_memoization.make()
        activation_strategy(model)

        self._model = model

        self.optimizer: OptimizerProtocol = self.build_optimizer(self.model)

        # Recorded before any schedule runs, so the multiplier scales the rate
        # the recipe was tuned at rather than compounding on the last step's.
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        self.learning_rate_scheduler: Schedule[float] = (
            self.config.learning_rate_scheduler.make()
        )

        self.ema: EMAProtocol = self.config.ema.make()

        self._compile_fn: Callable[[Callable[..., Any]], Callable[..., Any]] | None = (
            self.config.compile.make() if self.config.compile else None
        )
        self._compiled_model: Any = None

        self.last_grad_norm: Tensor | None = None

        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.drop_partial_accumulation_on_epoch_end = (
            config.drop_partial_accumulation_on_epoch_end
        )
        self.accumulation_steps: int = 0
        self.accumulated_samples: int = 0
        self.last_microbatch_grads: list[Tensor] = []
        """Per-parameter gradients at the last optimizer step, after the
        grand-total division and before the optimizer zeroes them. Snapshotted
        for accumulation-correctness assertions; empty until the first step."""

        self.loss: Callable[..., LossOutput] = config.loss.make()

    @property
    def model(self) -> nn.Module:
        """The placed, sharded, materialized module this step trains.

        A read-only property rather than a plain attribute so a subclass whose
        config pins a concrete model class can override it with that narrower
        return type. A mutable attribute is invariant and cannot be narrowed,
        which forced every such subclass to cast its own model at each use.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """Device for model and data."""
        return self.parallelism.device

    @property
    def global_step(self) -> int:
        """Optimizer updates across the whole run, resumes included.

        The step timer's count under the name the loop, the cadences, and
        every logged metric already use for it.
        """
        return self.timer_step.global_count

    @property
    def local_step(self) -> int:
        """Optimizer updates since this process started.

        The step timer's session count. What a warmup exclusion or a
        first-step branch reads, since after a resume those are questions
        about THIS job and the lifetime count cannot answer them.
        """
        return self.timer_step.local_count

    def build_optimizer(self, model: nn.Module) -> OptimizerProtocol:
        """Build the optimizer, bridging the two calling conventions.

        A name-routing builder (``CompositeOptimizer``) needs the MODULE, since
        the split is by parameter name and a list of tensors carries none. A
        torch optimizer needs the parameters. Both are offered the module
        first, and the ``TypeError`` torch raises when handed one is what
        selects the fallback -- so a caller supplies either without saying
        which, and a builder that raises ``TypeError`` for its OWN reasons
        surfaces that error from the second call rather than being swallowed.

        Args:
          model: The placed, sharded module whose parameters are optimized.

        Returns:
          optimizer: The built optimizer.

        """
        build = self.config.optimizer.make()
        try:
            return build(model)
        except TypeError:
            return build([{"params": model.parameters()}])

    def bind_epoch_timer(self, timer: CheckpointableStepTimer) -> None:
        """Anneal against the loader's pass count rather than a private one.

        Called once by whatever holds both, before the first batch. Taking the
        loader's OBJECT rather than its number is what keeps the count a
        schedule reads and the count a checkpoint restored from being one --
        and the dataset owns the checkpointing, since only it can say when a
        pass ended.

        Args:
          timer: The dataset's own epoch timer.

        """
        self.timer_epoch = timer

    @property
    def progress_complete(self) -> float:
        """Fraction of the train budget spent, in ``[0, 1]``; how DONE the job is.

        The LARGEST of the budgets' fractions, so a run declaring several ends
        as the first of them binds. An unset budget is infinite and contributes
        nothing, which is what lets one formula serve a run bounded by steps,
        by seconds, by passes over the data, or by any mix.

        Monotone and always a bare fraction -- what a checkpoint cadence, an
        ETA, or a progress bar needs.
        """
        step = self.timer_step
        return min(
            1.0,
            max(
                step.global_count / self.config.train_budget_steps,
                step.global_sec / self.config.train_budget_sec,
                self.timer_epoch.global_count / self.config.train_budget_epochs,
            ),
        )

    @property
    def progress_learning_schedule(self) -> float:
        """What the learning-rate schedule reads.

        How done the job is, by default. Overridden where the recipe anneals
        against something else -- a clock excluding warmup steps, or passes
        over the data alone -- which is why the schedule reads this rather
        than :attr:`progress_complete` directly.
        """
        return self.progress_complete

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Training forward pass (sets train mode, applies autocast, optionally compiles)."""
        self.model.train()
        if self._compile_fn is not None and self._compiled_model is None:
            self._compiled_model = self._compile_fn(self.model)
        forward_model = (
            self._compiled_model if self._compiled_model is not None else self.model
        )
        autocast_ctx = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=self.config.autocast_cache_enabled,
            )
            if self.config.dtype_autocast is not None
            else contextlib.nullcontext()
        )
        with self.timer_forward, autocast_ctx:
            return forward_model(*args, **kwargs)

    def call_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluation forward pass (uses EMA if available, applies inference_mode and autocast).

        Runs the live model with EMA-averaged weights swapped in via
        ``ema.apply_to``. This is the single eval path for every shadow kind:
        the ``"param_dict"`` shadow keeps ``shadow_model is None`` (FSDP-safe),
        so a ``shadow_model`` truthiness fallback would silently evaluate LIVE
        un-averaged weights. NoEMA's ``apply_to`` is a no-op.
        """
        was_training = self.model.training
        self.model.eval()

        autocast_ctx = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=self.config.autocast_cache_enabled,
            )
            if self.config.dtype_autocast is not None
            else contextlib.nullcontext()
        )

        with (
            self.timer_eval,
            torch.inference_mode(),
            self.ema.apply_to(self.model),
            autocast_ctx,
        ):
            output = self.model(*args, **kwargs)

        self.model.train(was_training)
        return output

    def step(self, closure: Callable[[], Tensor | float] | None = None) -> None:
        """Optimization step (clip grads, scale the rate, optimizer.step, EMA).

        Args:
          closure: Loss-recomputing closure for closure-based optimizers (e.g.
            exact-Hessian Newton). It is forwarded to ``optimizer.step`` ONLY
            when the optimizer sets ``requires_closure``: a torch first-order
            optimizer executes any closure it is handed, running a wasteful
            second forward that would also double-count BatchNorm stats.

        """
        # The tally is what advances ``global_step`` and the budget clock, so
        # the whole update -- clip, schedule, step, EMA -- is inside it. A
        # counter incremented at the end instead would leave the schedule
        # reading a progress one step stale.
        with self.timer_step:
            self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
                foreach=True,
            )
            # Scaled BEFORE the optimizer, so the rate this update uses is the
            # one the schedule names for this progress; scaled afterwards it
            # would take effect a step late and the first update would land at
            # the unscheduled rate.
            self.apply_learning_rate()
            if closure is not None and getattr(
                self.optimizer,
                "requires_closure",
                False,
            ):
                self.optimizer.step(closure)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.ema(self.model)

    def apply_learning_rate(self) -> None:
        """Scale every group's rate by the schedule at the current progress.

        Each group is scaled from its own ``initial_lr``, never from its
        current one: a recipe running several rates at once (a token table at
        0.6 beside an unembedding at 0.004) keeps their ratios, and repeated
        application cannot compound.
        """
        multiplier = self.learning_rate_scheduler(self.progress_learning_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move every tensor in the batch to this step's device."""
        non_blocking = self.device.type == "cuda"
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Forward + loss + backprop with gradient accumulation.

        The supervised shape. A recipe that augments before the forward, runs
        several optimizer updates per batch, or must act between the backward
        and the update overrides this and drives :meth:`step` itself.

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (autocast applied in __call__). The output may be a single
        # Tensor or a multi-output container; the loss consumes it via the
        # ModelOutput contract rather than a blind ``cast(Tensor, ...)``.
        forward_output: Any = self(**preprocessed_batch)
        if not isinstance(forward_output, ModelOutput):
            raise TypeError(
                "Model forward output does not satisfy ModelOutput "
                f"(got {type(forward_output).__name__}); it must be a Tensor or "
                "an indexable multi-output container the loss can consume.",
            )
        output: ModelOutput = forward_output

        # Loss computation (inherits autocast from forward)
        loss_result = {**self.loss(output, **preprocessed_batch)}
        loss: Tensor = loss_result["loss"]

        # Validate loss is unreduced (per-element)
        if loss.ndim == 0:
            raise ValueError(
                "Loss must be unreduced (per-element), got scalar. "
                "Use reduction='none' in your loss function.",
            )

        # Accumulate the SUM of per-element losses. Backward is linear, so
        # summing per micro-batch and dividing once by the grand-total element
        # count reproduces a single big-batch gradient exactly across this
        # rank's micro-batches -- including UNEQUAL micro-batch sizes within
        # the rank (a mean-of-means would not). Across DATA-PARALLEL ranks the
        # exactness holds only when every rank accumulates the SAME total
        # element count: DDP/FSDP reduce the gradient as a MEAN over ranks, so
        # dividing each rank by its LOCAL count yields
        # ``mean_ranks(local_sum / local_N)`` rather than the true big batch
        # ``sum_ranks(local_sum) / sum_ranks(local_N)``; the two coincide iff
        # ``local_N`` is equal across ranks. ``_assert_uniform_microbatch_count``
        # enforces that precondition at the optimizer-step boundary.
        loss.sum().backward()
        self.accumulated_samples += loss.numel()
        self.accumulation_steps += 1

        # Update weights if accumulation complete.
        if self.accumulation_steps >= self.accumulate_grad_batches:
            # Cross-rank exactness precondition (see the backward comment):
            # every data-parallel rank must accumulate the same element count.
            _assert_uniform_microbatch_count(self.accumulated_samples)

            grad_snapshot: list[Tensor] = []
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.accumulated_samples)
                    grad_snapshot.append(param.grad.detach().clone())
            self.last_microbatch_grads = grad_snapshot

            self.step(lambda: self._recompute_loss(preprocessed_batch))
            self.accumulation_steps = 0
            self.accumulated_samples = 0

        loss_result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, loss_result)

    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute loss in train mode (no backprop).

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (train mode + autocast via __call__)
        output: ModelOutput = self(**preprocessed_batch)

        # Loss computation (inherits autocast)
        result = {**self.loss(output, **preprocessed_batch)}
        result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, result)

    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute loss in eval mode (uses EMA if available).

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (eval mode + autocast via call_eval)
        output: ModelOutput = self.call_eval(**preprocessed_batch)

        # Loss computation (inherits autocast)
        result = {**self.loss(output, **preprocessed_batch)}
        result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, result)

    def on_epoch_end(self) -> None:
        """Flush a partial gradient accumulation at an epoch boundary.

        When ``drop_partial_accumulation_on_epoch_end`` is True (default) and
        an accumulation is pending, zero the accumulated micro-batch gradients
        and reset the counters so no gradient mixes across the boundary. When
        False, the partial accumulation is left intact to carry into the next
        epoch. A no-op when nothing is pending -- which is every step whose
        accumulation completes inside one ``train_step``.
        """
        if (
            not self.drop_partial_accumulation_on_epoch_end
            or self.accumulation_steps == 0
        ):
            return
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_steps = 0
        self.accumulated_samples = 0

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict.

        Only what this class OWNS. ``timer_epoch`` is absent because the
        dataset owns and saves it -- a second copy here would be restored in
        whatever order the two loads happen to run, and the two would agree
        only by luck.

        The accumulation counters are recorded so a mid-accumulation
        checkpoint is auditable; restoring resets them (see
        :meth:`load_state_dict`), because per-microbatch gradients cannot be
        persisted.

        A subclass adding a timer of its own saves it by extending this, which
        is one line and visible where a reader looks for what a checkpoint
        holds.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "timer_forward": self.timer_forward.state_dict(),
            "timer_eval": self.timer_eval.state_dict(),
            "timer_step": self.timer_step.state_dict(),
            "ema": self.ema.state_dict(),
            "accumulation_steps": self.accumulation_steps,
            "accumulated_samples": self.accumulated_samples,
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        """Load checkpoint state dict.

        ``remap`` (if set) transforms the saved model dict first; ``strict``
        controls whether the resulting keys must match the model exactly;
        ``load_optimizer`` gates optimizer / EMA restore -- skipped when
        finetuning a changed architecture, whose saved optimizer state would
        mismatch. Defaults are a strict, full load (ordinary resume).

        Gradient accumulation resets: per-microbatch gradients are not saved,
        so a pending accumulation cannot be resumed.
        """
        model_state = state_dict["model"]
        if remap is not None:
            model_state = remap(model_state)
        self.model.load_state_dict(model_state, strict=strict)
        # Both halves of progress ride in the step timer -- the update count
        # and the seconds they took -- so a resume anneals from where the run
        # left off. A restarted clock would replay decay the run had already
        # applied, training its tail at a rate the recipe places in its
        # opening.
        #
        # A timer the checkpoint does not name keeps its fresh zero, so a
        # checkpoint written before a timer existed still loads.
        for name, timer in (
            ("timer_forward", self.timer_forward),
            ("timer_eval", self.timer_eval),
            ("timer_step", self.timer_step),
        ):
            if name in state_dict:
                timer.load_state_dict(state_dict[name])

        self.accumulation_steps = 0
        self.accumulated_samples = 0

        if not load_optimizer:
            return  # finetuning: keep the fresh optimizer/EMA

        self.optimizer.load_state_dict(state_dict["optimizer"])
        if "ema" in state_dict:
            self.ema.load_state_dict(state_dict["ema"])

    def _recompute_loss(self, preprocessed_batch: dict[str, Any]) -> Tensor:
        """Recompute the scalar training loss on ``preprocessed_batch``.

        The optimizer closure for closure-based optimizers (e.g. exact-Hessian
        Newton, which differentiates this via ``autograd.grad``). First-order
        optimizers never call it. Returns a graph-bearing scalar so the caller
        can take further derivatives.
        """
        output: ModelOutput = self(**preprocessed_batch)
        loss = {**self.loss(output, **preprocessed_batch)}
        return loss["loss"].sum()


def _collective_device(group: dist.ProcessGroup | None) -> torch.device:
    """The device this group's backend can reduce on: NCCL CUDA, gloo CPU."""
    if dist.get_backend(group) == "nccl":
        # The CURRENT device, not index 0: a shared index would put every
        # rank's reduction on one GPU.
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _assert_uniform_microbatch_count(accumulated_samples: int) -> None:
    """Verify every data-parallel rank accumulated the same element count.

    The precondition for the ``train_step`` division (derived in its backward
    comment). Uses the ``"dp"`` mesh group when one exposes it, else WORLD --
    correct when the world is a pure data-parallel group.

    Args:
      accumulated_samples: This rank's count for the completed window.

    Raises:
      ValueError: Ranks accumulated differing counts, which would make the
        per-rank division diverge from the true big-batch gradient.

    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    mesh = global_device_mesh()
    group = (
        mesh.get_group("dp")
        if mesh is not None
        and mesh.mesh_dim_names is not None
        and "dp" in mesh.mesh_dim_names
        else None
    )
    if group is not None and dist.get_world_size(group) <= 1:
        return

    # all_reduce MIN and MAX of the local count; they differ iff some rank
    # accumulated a different number of elements.
    #
    # Built on the BACKEND's device, not the default CPU one: NCCL -- the
    # backend of every multi-GPU run -- reduces only CUDA tensors and rejects
    # a CPU one outright ("No backend type associated with device type cpu").
    # Since this fires on every optimizer step that closes an accumulation
    # window, a CPU tensor here failed the first step of any NCCL run.
    extremes = torch.tensor(
        [accumulated_samples, -accumulated_samples],
        dtype=torch.long,
        device=_collective_device(group),
    )
    dist.all_reduce(extremes, op=dist.ReduceOp.MAX, group=group)
    count_max = int(extremes[0].item())
    count_min = -int(extremes[1].item())
    if count_max != count_min:
        raise ValueError(
            "Gradient accumulation requires an equal per-element count across "
            "data-parallel ranks (DDP/FSDP reduce gradients as a MEAN over "
            "ranks, so dividing each rank by its local count only reproduces "
            "the single big-batch gradient when the counts match). Ranks "
            f"accumulated between {count_min} and {count_max} elements; "
            "equalize the per-rank micro-batch element counts.",
        )
