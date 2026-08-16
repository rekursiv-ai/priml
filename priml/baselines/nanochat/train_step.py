"""Training step for a budgeted language-model run.

Trains to a wall-clock BUDGET, not a step count, so every schedule reads
elapsed training seconds over the budget rather than a step index -- a cheaper
step buys more steps at the same schedule shape, which is the comparison the
budget exists to make. The clock excludes ``budget_warmup_steps`` and
everything outside :meth:`train_step`, so compile time cannot decide how much
training a run gets. Gradient accumulation targets a fixed TOKEN count, the
quantity the recipe is tuned against.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import field
from typing import Any, Self, override

import math
import time

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.nanochat.model import NanoChatLM
from priml.loss.custom_types import LossOutput
from priml.math.schedules import Schedule, trapezoidal
from priml.optimizers import (
    CompositeOptimizer,
    FusedAdamW,
    HasParamGroups,
    NorMuon,
    apply_lr_scale,
)
from priml.optimizers.composite import Selector, excluding, matching
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


class TokenCrossEntropy:
    """Per-token cross-entropy over ``[B, S, V]`` logits, unreduced."""

    class Config(Fig["TokenCrossEntropy"]):
        """Which target value marks a position that is not data."""

        ignore_index: int = -1
        """Target marking a padded position; excluded from the loss.

        The value :data:`~priml.baselines.nanochat.data.IGNORED_TARGET`
        pads a short batch with."""

    def __init__(self, config: Config) -> None:
        self.ignore_index = config.ignore_index

    def __call__(self, prediction: Tensor, **batch: Any) -> LossOutput:
        """Score every target position.

        Args:
          prediction: Logits, ``[B, S, V]``.
          **batch: Must contain ``label``, ``[B, S]``.

        Returns:
          result: ``loss``, ``[B, S]`` cross-entropy in nats.

        """
        label: Tensor = batch["label"]
        # Upcast: the caller's reduction runs over thousands of terms, and the
        # reference measured this loss in fp32.
        return {
            "loss": functional.cross_entropy(
                prediction.reshape(-1, prediction.shape[-1]).float(),
                label.reshape(-1).long(),
                ignore_index=self.ignore_index,
                reduction="none",
            ).reshape(label.shape),
        }


def matrix_parameters() -> Selector:
    """Select the reasoning matrices: rank >= 2, and not a lookup table.

    Returns:
      select: The predicate routing parameters to the orthogonalizing member.

    """
    return excluding(NorMuon.eligible_tensor, "embed", "lm_head")


def nanochat_optimizer(*, compile: bool = True) -> CompositeOptimizer.Config:
    """NorMuon on the reasoning matrices, AdamW on everything else, by class.

    Five AdamW members rather than one: the classes want rates two orders of
    magnitude apart, and collapsing them trains a different model at the same
    nominal hyperparameters. Rates are stated AS TUNED at 768 channels_in; the
    step's ``finalize`` rescales them to the model's actual width.

    Args:
      compile: Whether each member fuses its step into one compiled graph. Off
        is for a caller that steps a handful of times and would otherwise pay
        Dynamo's tracing, which is never cached, for every one.

    Returns:
      config: The optimizer recipe: five AdamW members and one NorMuon.

    """
    on_matrices = matrix_parameters()
    config = CompositeOptimizer.Config()
    unembedding, embedding, value_embedding, residual, skip = (
        PartialConfig(
            # The fused kernel, not torch's: the two are the same algorithm and
            # round differently, so a score measured under one is not
            # comparable with a score measured under the other. This is the one
            # the reference recipe was measured with.
            FusedAdamW,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
            # Declared on every member, not only the exempt ones: the step's
            # finalize reads it to decide whether to rescale, and a member
            # that simply omitted it would have to be handled by a default
            # that silently applies to typos as well.
            width_scaled=True,
            compile=compile,
        )
        for _ in range(5)
    )
    unembedding.lr = 0.004
    embedding.lr = 0.6
    value_embedding.lr = 0.6
    # A hundredth of the scalar rate: this one multiplies the residual stream
    # itself, so it moves the whole stack's scale rather than one path's.
    #
    # Neither scalar group scales with width, so both are marked exempt: they
    # step a per-layer number rather than a projection, and the 1/sqrt(width)
    # rule follows from a matrix's fan-in.
    residual.lr = 0.5 * 0.01
    residual.width_scaled = False
    skip.lr = 0.5
    skip.width_scaled = False
    # Beta1 raised only here: the skip weights start at 0.1 and must travel,
    # and a longer memory keeps that trip from being driven by one batch.
    skip.betas = (0.96, 0.95)
    config.optimizers = [
        unembedding,
        embedding,
        value_embedding,
        residual,
        skip,
        NorMuon.Config(compile=compile),
    ]
    # Ordered most specific first: ``value_embeds`` also contains ``embed``, so
    # the token table's selector must exclude it rather than claim it.
    config.select = [
        matching("lm_head"),
        excluding(matching("embed"), "value_embeds"),
        matching("value_embeds"),
        matching("mix.running"),
        matching("mix.original"),
        on_matrices,
    ]
    # The value-embedding member is dropped when the model has no such tables.
    # A selector claiming nothing is REJECTED, which is the right default -- it
    # catches a misspelled fragment -- but a rung that switches the mechanism
    # off is not a typo, and it would otherwise be unable to use this recipe.
    config.drop_empty = True
    return config


class NanoChatTrainStep(TrainStep):
    """Model plus optimization for one budgeted language-model experiment.

    Takes the model, optimizer, loss, and device placement from
    :class:`~priml.train.train_step.TrainStep`. What stays here is the
    budgeted recipe: accumulation to a fixed TOKEN count, a clock excluding
    warmup, and momentum and weight decay annealed against that clock.
    """

    class Config(Makes["NanoChatTrainStep"], TrainStep.Config, kw_only=False):
        """Model, optimization, the budget, and the schedules it drives."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: NanoChatLM.Config = field(default_factory=NanoChatLM.Config)  # pyright: ignore[reportIncompatibleVariableOverride] -- narrowing a Makeable slot to its concrete Config is the priml idiom; finalize reaches this model's own fields
        """Network to train."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=nanochat_optimizer,
        )
        """Builds the optimizer from the model."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=TokenCrossEntropy.Config,
        )
        """Maps logits and the batch to an unreduced per-token loss."""

        train_budget_sec: float = 300.0
        """Training seconds the schedules anneal over.

        The base's field, on this baseline's own CLOCK: warmup and everything
        outside :meth:`train_step` are excluded. The loop's ``max_time`` stops
        the run and is set alongside it."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype; ``None`` trains in full precision."""

        compile: Makeable[Callable[[Callable[..., Any]], Callable[..., Any]]] | None = (
            field(default_factory=lambda: PartialConfig(torch.compile))
        )
        """Compile the model AND the loss with ``torch.compile``; ``None`` runs
        them eagerly.

        Both together, not the model alone: the loss reads the ``[B, S, V]``
        logits and upcasts them to float32, so a boundary between the two
        materializes that tensor -- 2 GiB at this recipe's geometry -- and runs
        the reduction and its backward unfused. Inside one graph inductor fuses
        them into the head and never writes it. Measured against the reference
        on a 5090: +7.2% per step with the boundary, +0.6% without.

        Bare, without the base's ``fullgraph=True``, which this graph does not
        compile under. Covers the forward only -- each optimizer member carries
        its own ``compile``, since a compiled step and an eager one differ
        numerically (measured, 2.9e-2 on one update)."""

        # ---- This recipe's own. ----

        schedule: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(trapezoidal, flat=0.5),
        )
        """Maps budget progress in ``[0, 1]`` to a learning-rate multiplier.

        The reference's flat-then-decay shape, verified equal to
        ``trapezoidal`` at every one of 100,001 sampled progresses. Read
        instead of the base's ``learning_rate_scheduler``, since this recipe
        drives the update itself rather than through ``TrainStep.step``."""

        budget_warmup_steps: int = 10
        """Leading optimizer STEPS excluded from the clock.

        Not passes: the reference excludes ten of its own steps
        (``train.py:576``), and each of those is a whole accumulation. Counting
        passes here would hand this recipe ``10 / accumulate_passes`` steps of
        warmup against the reference's ten -- 1.25 at the default geometry --
        and charge the budget for compilation the comparison excludes."""

        tokens_per_optimizer_step: int = 524_288
        """Tokens per optimizer step, reached by gradient accumulation.

        Held fixed while ``rows_per_pass`` follows whatever fits in memory."""

        rows_per_pass: int = 32
        """Rows per forward/backward pass. Reduce on out-of-memory."""

        momentum_start: float = 0.85
        """Orthogonalizing member's momentum at the first step."""

        momentum_end: float = 0.95
        """Momentum it reaches after ``momentum_warmup_steps``."""

        momentum_warmup_steps: int = 300
        """Steps over which momentum ramps. Indexed by STEP, not progress: it
        corrects a transient of early training, which is a step-count effect."""

        decay_to_zero: bool = True
        """Anneal weight decay to zero over the budget."""

        divergence_threshold: float = 100.0
        """Loss above which the run raises rather than continues."""

        adam_lr_tuned_at_channels: int = 768
        """Width the Adam rates in ``optimizer`` were fitted at.

        ``finalize`` rescales every Adam member by
        ``sqrt(tuned_at / channels_in)``; set this to the model's own width to
        disable that."""

        @override
        def finalize(self) -> Self:
            if self.train_budget_sec <= 0 or not math.isfinite(self.train_budget_sec):
                raise ValueError(
                    "train_budget_sec must be finite and positive; got "
                    f"{self.train_budget_sec}.",
                )
            if self.budget_warmup_steps < 0:
                raise ValueError(
                    "budget_warmup_steps must be nonnegative; got "
                    f"{self.budget_warmup_steps}.",
                )
            if self.momentum_warmup_steps <= 0:
                raise ValueError(
                    "momentum_warmup_steps must be positive; got "
                    f"{self.momentum_warmup_steps}.",
                )
            if self.rows_per_pass <= 0:
                raise ValueError(
                    f"rows_per_pass must be positive; got {self.rows_per_pass}.",
                )
            if self.tokens_per_optimizer_step <= 0:
                raise ValueError(
                    "tokens_per_optimizer_step must be positive; got "
                    f"{self.tokens_per_optimizer_step}.",
                )
            # NaN is excluded explicitly, not covered by ``<= 0``: every
            # comparison against it is False, so a NaN here does not fail --
            # it silently DISABLES the guard. ``math.isfinite`` is false for
            # NaN, so clipping would be skipped; ``loss > NaN`` is false, so
            # divergence would never be detected.
            if math.isnan(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
                raise ValueError(
                    f"gradient_clip_norm must be positive; got "
                    f"{self.gradient_clip_norm}. Infinite disables clipping.",
                )
            if math.isnan(self.divergence_threshold) or self.divergence_threshold <= 0:
                raise ValueError(
                    "divergence_threshold must be positive; got "
                    f"{self.divergence_threshold}.",
                )
            for name, momentum in (
                ("momentum_start", self.momentum_start),
                ("momentum_end", self.momentum_end),
            ):
                # The schedule writes these straight into the optimizer's
                # groups every step, past the constructor that would have
                # rejected them. NaN needs no separate check here: it fails
                # this comparison rather than slipping through it.
                if not 0.0 <= momentum < 1.0:
                    raise ValueError(
                        f"{name} must lie in [0, 1); got {momentum}.",
                    )
            tokens_per_pass = self.rows_per_pass * self.model.max_seq_len
            if self.tokens_per_optimizer_step % tokens_per_pass:
                raise ValueError(
                    f"tokens_per_optimizer_step={self.tokens_per_optimizer_step} "
                    f"is not divisible by rows_per_pass * max_seq_len="
                    f"{tokens_per_pass}, so no whole number of passes reaches "
                    "the token batch.",
                )
            # Rescaled here, not in the factory: the factory runs before the
            # caller has chosen a width, so a rate baked there is right for one
            # model and wrong for every fork that changes ``channels_in``. The
            # field is then set to the model's width, so a second finalize is a
            # no-op rather than a second rescale.
            if isinstance(self.optimizer, CompositeOptimizer.Config):
                for member in self.optimizer.optimizers:
                    if not isinstance(member, PartialConfig):
                        continue
                    # Popped, not read: the flag tells THIS method whether the
                    # rate scales, and the optimizer it is attached to would
                    # reject it as an unexpected keyword.
                    kwargs = member._kwargs  # noqa: SLF001 -- PartialConfig exposes no accessor for its keywords
                    scales = kwargs.pop("width_scaled", False)
                    rate = kwargs.get("lr")
                    # An exempt member steps a per-layer scalar rather than a
                    # projection, and the 1/sqrt(width) rule follows from
                    # fan-in, which a scalar does not have.
                    if not isinstance(rate, float) or not scales:
                        continue
                    # Divide by ``sqrt(width / tuned_at)``, never multiply by
                    # ``sqrt(tuned_at / width)``: the two round differently
                    # (0.6 at width 512 gives ...9535 against ...9533), and
                    # this is the spelling the rates were measured under.
                    member.lr = rate / math.sqrt(
                        self.model.channels_in / self.adam_lr_tuned_at_channels
                    )
            self.adam_lr_tuned_at_channels = self.model.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: NanoChatTrainStep.Config = config
        self.accumulate_passes = config.tokens_per_optimizer_step // (
            config.rows_per_pass * config.model.max_seq_len
        )
        self._pending_passes = 0
        # The worst loss of the passes accumulated so far, held on DEVICE so
        # the guard costs no synchronization until the batch is whole.
        self._pending_worst: Tensor | None = None
        # Budget-counted seconds, excluding warmup and everything outside
        # train_step. The loop reads it through NanoChatTrainLoop.
        self.elapsed_sec = 0.0
        # Optimizer steps this PROCESS has completed, restored across a resume
        # so a restarted run does not spend the warmup twice. Not the timer's
        # ``local_step``, which a fresh process zeroes.
        self._steps_this_process = 0
        # ``model`` stays the UNCOMPILED module, as the base assumes: it is what
        # the optimizer partition routed over and what the checkpoint holds,
        # and a compiled wrapper prefixes every ``state_dict`` key with
        # ``_orig_mod``. What is compiled is the forward BELOW, model and loss
        # together.
        if self._compile_fn is not None:
            self._compiled_model = self._compile_fn(self._forward)
        for group in self.optimizer.param_groups:
            group.setdefault("initial_weight_decay", group.get("weight_decay", 0.0))
        self.schedule = config.schedule.make()

    def _forward(self, media: Tensor, label: Tensor) -> Tensor:
        """Score one batch, returning the per-token loss.

        Model and loss in ONE function so the two compile together; see
        :attr:`Config.compile`. Unreduced because the metric weights each token
        by its byte length, and measured to fuse exactly as well as a reduced
        one (54.44 against 54.49 ms/pass).
        """
        logits = self.model(media)
        assert isinstance(logits, Tensor)
        return self.loss(logits, media=media, label=label)["loss"]

    def _per_token_loss(self, batch: dict[str, Any]) -> Tensor:
        """Score one batch through the compiled forward when there is one."""
        forward = self._compiled_model if self._compiled_model is not None else None
        if forward is None:
            return self._forward(batch["media"], batch["label"])
        result = forward(batch["media"], batch["label"])
        assert isinstance(result, Tensor)
        return result

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Fraction of the budget spent, clamped to ``[0, 1]``."""
        spent = self.elapsed_sec / float(self.config.train_budget_sec)
        return min(spent, 1.0)

    @override
    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch to the training device."""
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

    @override
    def train_step(self, **batch: Any) -> TrainStepOutput:
        """Forward, backward, and step the optimizer once the batch is full.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss``, the per-token loss as ``model``, and the schedule
            values this step ran at.

        Raises:
          RuntimeError: The loss became non-finite or exceeded
            ``divergence_threshold``.

        """
        config = self.config
        # Drain first, THEN start the clock: queued device work from an
        # evaluation would otherwise be waited for inside this step's timing
        # and charged to the budget. Measured: 422 steps against 836.
        #
        # Only at an accumulation BOUNDARY, which is where the reference drains
        # too (train.py:542). Draining mid-accumulation stops the CPU running
        # ahead of a queue it is about to extend, and buys nothing: the clock
        # charges whole optimizer steps, so what happens between the passes of
        # one is already inside the bracket. Measured on a 5090 at 32 passes,
        # 1786 against 1770 ms/step -- and 1770 with no drain at all, so this
        # cadence gives up none of the accuracy.
        if self._pending_passes == 0:
            self._synchronize()
        started = time.perf_counter()
        self.model.train()
        with self._autocast():
            per_token = self._per_token_loss(batch)
            loss = per_token.mean()
        # Each pass contributes its share, so the accumulated gradient is the
        # mean over the whole token batch rather than over the last pass.
        (loss / self.accumulate_passes).backward()
        # Kept ON DEVICE and reduced with the passes before it. Reading it here
        # instead would stall the CPU on the forward before the backward could
        # be enqueued, draining the pipeline once per PASS: measured at 85.5
        # against 58.6 ms/pass on a 5090 at eight rows, a third of the step.
        # ``maximum`` rather than a sum because the threshold is stated per
        # pass, and it propagates NaN, which is the other half of the guard.
        worst = loss.detach()
        if self._pending_worst is not None:
            worst = torch.maximum(worst, self._pending_worst)
        self._pending_worst = worst

        self._pending_passes += 1
        metrics: dict[str, float | Tensor] = {}
        if self._pending_passes >= self.accumulate_passes:
            # Read ONCE, at the boundary, and before the update: a diverged
            # batch must not reach the optimizer, and every pass of this token
            # batch has now been seen.
            self._assert_not_diverged()
            # The step timer brackets the update, so ``global_step`` advances
            # as it does for every other recipe; ``elapsed_sec`` below stays
            # the clock this baseline's schedules read.
            with self.timer_step:
                metrics = self._apply_update()
            self._pending_passes = 0
            self._steps_this_process += 1
        # Charged after the update so a step's own optimizer time counts, and
        # only past warmup so compilation does not consume the budget. Counted
        # in optimizer STEPS, matching the reference (train.py:576): in passes
        # the exclusion would be ``budget_warmup_steps / accumulate_passes``
        # steps -- 1.25 at the default geometry against the reference's ten --
        # and the budget would pay for the compilation it excludes.
        if self._steps_this_process > config.budget_warmup_steps:
            # Drain again, at the same boundary: the backward and the optimizer
            # are queued, not finished, so a CPU-side reading would undercharge
            # the step. ``_pending_passes`` is zeroed by the update above, so
            # this fires on the pass that completed one -- and the two drains
            # together bracket the whole accumulation, since each pass starts
            # where the last ended. What each individual pass is charged is
            # then enqueue time, but their SUM is the work, and the budget is
            # spent in optimizer steps.
            if self._pending_passes == 0:
                self._synchronize()
            self.elapsed_sec += time.perf_counter() - started
        return {
            "loss": loss.detach().reshape(1),
            "model": per_token.detach(),
            "metrics": metrics,
        }

    @override
    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        self.model.train()
        with torch.no_grad(), self._autocast():
            per_token = self._per_token_loss(batch)
        return {"loss": per_token.mean().reshape(1), "model": per_token}

    @override
    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Score one batch, returning the per-token loss the metric consumes.

        The reported ``loss`` is the mean over REAL rows only: the loop weights
        it by ``valid_count``, so averaging over the padded width would report
        a loss diluted by rows that are not data.
        """
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            per_token = self._per_token_loss(batch)
        valid = int(batch.get("valid_count", per_token.shape[0]))
        return {
            "loss": per_token[:valid].mean().reshape(1),
            "model": per_token,
        }

    @override
    def call_eval(self, **batch: Any) -> Tensor:
        """Return evaluation logits for one batch.

        Eager: the compiled forward returns a per-token LOSS, having fused the
        logits away, and materializing them again is the whole cost that
        fusion removes. Callers wanting a score use :meth:`eval_loss`.
        """
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            logits = self.model(batch["media"])
        assert isinstance(logits, Tensor)
        return logits

    @override
    def on_epoch_end(self) -> None:
        """Discard a partial accumulation so gradients never cross a pass."""
        if self._pending_passes:
            self.model.zero_grad(set_to_none=True)
            self._pending_passes = 0
            # The losses those gradients came from go with them: carried into
            # the next batch, a discarded pass could abort a healthy one.
            self._pending_worst = None

    @override
    def state_dict(self) -> dict[str, Any]:
        """Extend the base state with this baseline's own budget clock.

        The clock drives every schedule and the step count gates it, so a
        resume that dropped either would re-anneal from the top or grant
        another warmup costing no budget.
        """
        state = super().state_dict()
        state["elapsed_sec"] = self.elapsed_sec
        state["local_step"] = self._steps_this_process
        return state

    @override
    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> None:
        """Restore state produced by :meth:`state_dict`."""
        # Checked BEFORE anything is assigned, so a caller that catches this is
        # not left holding the checkpoint's clock beside its own step counter.
        if "local_step" not in state_dict:
            raise ValueError(
                "this checkpoint records no 'local_step', so it predates the "
                "budget-clock fix and its warmup accounting cannot be "
                "reconstructed; resuming would grant uncharged training. "
                "Start a fresh run.",
            )
        super().load_state_dict(state_dict, **kwargs)
        self.elapsed_sec = float(state_dict["elapsed_sec"])
        self._steps_this_process = int(state_dict["local_step"])
        self._pending_passes = 0
        self._pending_worst = None

    def _assert_not_diverged(self) -> None:
        """Refuse a token batch whose worst pass diverged, discarding it.

        Raises:
          RuntimeError: A pass was non-finite or above ``divergence_threshold``.
            The gradients are zeroed first, so the passes accumulated behind it
            are gone -- leaving their COUNT would make the next update fire on
            a short token batch, which is the one thing accumulation exists to
            prevent.

        """
        if self._pending_worst is None:
            return
        worst = float(self._pending_worst)
        self._pending_worst = None
        if math.isfinite(worst) and worst <= self.config.divergence_threshold:
            return
        self.model.zero_grad(set_to_none=True)
        self._pending_passes = 0
        raise RuntimeError(
            f"training diverged at step {self.global_step}: loss={worst}.",
        )

    def _apply_update(self) -> dict[str, float | Tensor]:
        """Set every schedule for this step, then step the optimizer."""
        config = self.config
        metrics: dict[str, float | Tensor] = {}
        if math.isfinite(config.gradient_clip_norm):
            metrics["grad_norm"] = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.gradient_clip_norm,
            ).detach()
        progress = self.progress_learning_schedule
        multiplier = self.schedule(progress)
        apply_lr_scale([self.optimizer], multiplier)
        share = min(self.global_step / config.momentum_warmup_steps, 1.0)
        momentum = (1 - share) * config.momentum_start + share * config.momentum_end
        for group in self.optimizer.param_groups:
            if "momentum" in group:
                group["momentum"] = momentum
            if config.decay_to_zero and "weight_decay" in group:
                group["weight_decay"] = group["initial_weight_decay"] * (1 - progress)
        self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        # Per MEMBER, not ``param_groups[0]``: the recipe runs two optimizers
        # at rates two orders of magnitude apart, and the first group belongs
        # to whichever the composite lists first -- so a single ``lr`` reports
        # one algorithm's rate while the other's is invisible.
        for name, rate in _learning_rates(self.optimizer).items():
            metrics[f"lr_{name}"] = rate
        metrics["progress"] = progress
        metrics["momentum"] = momentum
        return metrics

    def _synchronize(self) -> None:
        """Wait for queued device work, so the clock measures this step alone."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def _autocast(self) -> Generator[None]:
        """Enter autocast when a mixed-precision dtype is configured."""
        dtype = self.config.dtype_autocast
        if dtype is None:
            yield
            return
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=dtype,
            cache_enabled=False,
        ):
            yield


def _learning_rates(optimizer: HasParamGroups) -> dict[str, float]:
    """Return one rate per optimizer member, keyed by its class name.

    A composite holds every member's groups in one flat list, so a group's
    position says nothing about which algorithm owns it.

    Args:
      optimizer: The step's optimizer, composite or not.

    Returns:
      rates: Class name (lowercased) to that member's first group rate.

    """
    if not isinstance(optimizer, CompositeOptimizer):
        return {"all": float(optimizer.param_groups[0]["lr"])}
    return {
        type(member).__name__.lower(): float(member.param_groups[0]["lr"])
        for member in optimizer.optimizers
        if member.param_groups
    }
