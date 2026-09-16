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

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import field
from typing import Protocol, Self, cast, override

import math
import time

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch
import torch.distributed as dist

from priml.baselines.nanochat.model import (
    MemoryNanoChatLM,
    NanoChatLM,
    ScaledSoftCap,
)
from priml.baselines.nanochat.ngram import HashedNgramTables
from priml.baselines.nanochat.optimizers import BiasCorrectedRMSProp
from priml.lib.custom_json import FloatCodec
from priml.loss.custom_types import LossOutput
from priml.math.schedules import Schedule, trapezoidal
from priml.model.softcap import SoftCap
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


class _Compile(Protocol):
    def __call__[**P, R](self, model: Callable[P, R], /) -> Callable[P, R]: ...


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

    def __call__(self, prediction: Tensor, **batch: object) -> LossOutput:
        """Score every target position.

        Args:
          prediction: Logits, ``[B, S, V]``.
          **batch: Must contain ``label``, ``[B, S]``.

        Returns:
          result: ``loss``, ``[B, S]`` cross-entropy in nats.

        """
        label = batch["label"]
        assert isinstance(label, Tensor)
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

    class Config(
        Makes["NanoChatTrainStep"],
        TrainStep.Config[NanoChatLM.Config],
        kw_only=True,
    ):
        """Model, optimization, the budget, and the schedules it drives."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: NanoChatLM.Config = field(default_factory=NanoChatLM.Config)
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

        compile: Makeable[_Compile] | None = field(
            default_factory=lambda: PartialConfig(torch.compile),
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
        instead of the base's ``lr_schedule``, since this recipe
        drives the update itself rather than through ``TrainStep.step``."""

        budget_warmup_steps: int = 11
        """Leading optimizer STEPS excluded from the clock.

        Using 11 to match the Karpathy baseline."""

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
                # rejected them.
                if math.isnan(momentum) or momentum < 0.0 or momentum >= 1.0:
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
            if isinstance(self.loss, BoundedTokenCrossEntropy.Config):
                head = self.model.lm_head
                cap = (
                    abs(head.output_cap)
                    if isinstance(head, ScaledSoftCap.Config)
                    else head.cap
                    if isinstance(head, SoftCap.Config)
                    else math.inf
                )
                if (
                    not math.isfinite(cap)
                    or cap <= 0
                    or cap > self.loss.logit_upper_bound
                ):
                    raise ValueError(
                        "BoundedTokenCrossEntropy requires a symmetric readout "
                        "bound no larger than logit_upper_bound. Use "
                        "TokenCrossEntropy for an unbounded readout.",
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
                    kwargs = member._kwargs  # noqa: SLF001 -- The training step reuses the model's private timing seam.
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
                    member.lr = (
                        rate
                        / (self.model.channels_in / self.adam_lr_tuned_at_channels)
                        ** 0.5
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
        self._steps_this_process: int = 0
        # ``model`` stays the UNCOMPILED module, as the base assumes: it is what
        # the optimizer partition routed over and what the checkpoint holds,
        # and a compiled wrapper prefixes every ``state_dict`` key with
        # ``_orig_mod``. What is compiled is the forward BELOW, model and loss
        # together.
        if self._compile_fn is not None:
            self._compiled_model = self._compile_fn(self._forward)
        for group in self.optimizer.param_groups:
            group.setdefault(
                "initial_weight_decay",
                FloatCodec.coerce(group.get("weight_decay"), 0.0),
            )
        self.schedule = config.schedule.make()

    # Model and loss in ONE function so the two compile together; see
    # :attr:`Config.compile`. Unreduced because the metric weights each token by its
    # byte length, and measured to fuse exactly as well as a reduced one (54.44 against
    # 54.49 ms/pass).
    def _forward(self, media: Tensor, label: Tensor) -> Tensor:
        """Score one batch, returning the per-token loss."""
        logits = cast(object, self.model(media))
        assert isinstance(logits, Tensor)
        return self.loss(logits, media=media, label=label)["loss"]

    def _per_token_loss(self, batch: Mapping[str, object]) -> Tensor:
        """Score one batch through the compiled forward when there is one."""
        media, label = batch["media"], batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        forward = self._compiled_model if self._compiled_model is not None else None
        if forward is None:
            return self._forward(media, label)
        result = forward(media, label)
        assert isinstance(result, Tensor)
        return result

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Fraction of the budget spent, clamped to ``[0, 1]``."""
        spent = self.elapsed_sec / float(self.config.train_budget_sec)
        return min(spent, 1.0)

    @override
    def preprocess_batch(self, batch: Mapping[str, object]) -> dict[str, object]:
        """Move a batch to the training device."""
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
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
            # The step timer brackets the update, so ``global_step`` advances
            # as it does for every other recipe; ``elapsed_sec`` below stays
            # the clock this baseline's schedules read.
            with self.timer_step:
                metrics = self._apply_update()
            self._pending_passes = 0
            self._steps_this_process += 1
        # Charged after the update so a step's own optimizer time counts, and
        # only past warmup so compilation does not consume the budget. Counted
        # in optimizer STEPS, matching the reference (train.py:576). The
        # counter is incremented above, so this reads one higher than the
        # reference's does at the same update -- see ``budget_warmup_steps``,
        # whose default absorbs the difference.
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
        # Read LAST, behind the drain, and outside it: both wait for the same
        # queued work, so once the drain has run the read is free -- ahead of
        # it, it blocks the CPU on the whole step and was measured at 160
        # ms/step of stall. The reference is in the same order (train.py:565,
        # below the synchronize at 542 that drained the step before it).
        #
        # A step later than the batch that diverged, which is the same
        # guarantee: that update ran on gradients whose loss was finite, and
        # the diverged batch never reaches a second one.
        if self._pending_passes == 0:
            self._assert_not_diverged()
        return {
            "loss": loss.detach().reshape(1),
            "model": per_token.detach(),
            "metrics": metrics,
        }

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        self.model.train()
        with torch.no_grad(), self._autocast():
            per_token = self._per_token_loss(batch)
        return {"loss": per_token.mean().reshape(1), "model": per_token}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score one batch, returning the per-token loss the metric consumes.

        The reported ``loss`` is the mean over REAL rows only: the loop weights
        it by ``valid_count``, so averaging over the padded width would report
        a loss diluted by rows that are not data.
        """
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            per_token = self._per_token_loss(batch)
        raw_count = batch.get("valid_count", per_token.shape[0])
        assert isinstance(raw_count, int)
        valid = raw_count
        return {
            "loss": per_token[:valid].mean().reshape(1),
            "model": per_token,
        }

    @override
    def call_eval(self, *args: object, **batch: object) -> Tensor:
        """Return evaluation logits for one batch.

        Eager: the compiled forward returns a per-token LOSS, having fused the
        logits away, and materializing them again is the whole cost that
        fusion removes. Callers wanting a score use :meth:`eval_loss`.
        """
        del args
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            logits = cast(object, self.model(batch["media"]))
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

    class StateDict(TrainStep.StateDict):
        """The base state plus this baseline's own budget clock."""

        elapsed_sec: float
        local_step: int

    @override
    def state_dict(self) -> StateDict:
        """Extend the base state with this baseline's own budget clock.

        The clock drives every schedule and the step count gates it, so a
        resume that dropped either would re-anneal from the top or grant
        another warmup costing no budget.
        """
        if self._pending_passes:
            raise RuntimeError(
                "cannot checkpoint with incomplete gradient accumulation; "
                "per-pass gradients are not serializable",
            )
        return {
            **super().state_dict(),
            "elapsed_sec": self.elapsed_sec,
            "local_step": self._steps_this_process,
        }

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
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
        state = cast(NanoChatTrainStep.StateDict, state_dict)
        super().load_state_dict(
            state,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        self.elapsed_sec = float(state["elapsed_sec"])
        self._steps_this_process = state["local_step"]
        self._pending_passes = 0
        self._pending_worst = None

    def charge_budget(self, seconds: float) -> None:
        """Add loop-side work to the budget, once warmup is past.

        Called with the time spent fetching and staging a batch, which happens
        in the loop rather than here. The reference charges it -- its
        ``next(train_loader)`` sits at ``train.py:550``, between the ``t0`` at
        543 and the ``t1`` at 573 -- so a budget that skipped it would buy this
        recipe extra steps the comparison never granted. Measured on a 5090 at
        the shipped geometry: 0.160 of 1.683 s/step, a tenth of the run.

        Args:
          seconds: Wall-clock seconds to charge.

        """
        if self._steps_this_process > self.config.budget_warmup_steps:
            self.elapsed_sec += seconds

    def _assert_not_diverged(self) -> None:
        """Refuse a token batch whose worst pass diverged, discarding it."""
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
        metrics = self._clip_gradients()
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

    def _clip_gradients(self) -> dict[str, float | Tensor]:
        """Clip ordinary parameter gradients when the configured bound is finite."""
        if not math.isfinite(self.config.gradient_clip_norm):
            return {}
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            ).detach(),
        }

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


def _memory_model_config() -> NanoChatLM.Config:
    """Build the memory-model default through the inherited model-config interface."""
    config = MemoryNanoChatLM.Config()
    assert isinstance(config, NanoChatLM.Config)
    return config


class NgramTrainStep(NanoChatTrainStep):
    """Train n-gram memory with persistent gradient sinks and configurable updates."""

    class Config(NanoChatTrainStep.Config):
        """Extend the existing training configuration without duplicating its fields."""

        model: NanoChatLM.Config = field(default_factory=_memory_model_config)
        """Model with optional n-gram memory, pooling and attention-source reuse."""

        optimizer_update: (
            Makeable[Callable[[NgramTrainStep], dict[str, float | Tensor]]] | None
        ) = None
        """Injected update policy; None preserves the original optimizer schedules."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._optimizer_update = (
            config.optimizer_update.make()
            if config.optimizer_update is not None
            else None
        )
        model = self.model
        if isinstance(model, MemoryNanoChatLM) and model.fused_ngram:
            self._bind_ngram_gradients(model)
        if self._optimizer_update is not None:
            for group in self.optimizer.param_groups:
                group["initial_lr"] = group["lr"]
                if "betas" in group:
                    group["initial_betas"] = group["betas"]
                if "beta2" in group:
                    group["initial_beta2"] = group["beta2"]

    @property
    def completed_updates(self) -> int:
        """Return the checkpointed optimizer-update count."""
        return self._steps_this_process

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Accumulate gradients and charge the update receiving this batch.

        Args:
          **batch: Preprocessed media and labels.

        Returns:
          result: Detached loss, per-token losses and optimizer metrics.

        """
        update = self._steps_this_process + 1
        if self._pending_passes == 0:
            self._synchronize()
        started = time.perf_counter()
        self.model.train()
        with self._autocast():
            per_token = self._per_token_loss(batch)
            loss = per_token.mean()
        (loss / self.accumulate_passes).backward()
        worst = loss.detach()
        if self._pending_worst is not None:
            worst = torch.maximum(worst, self._pending_worst)
        self._pending_worst = worst
        self._pending_passes += 1
        metrics: dict[str, float | Tensor] = {}
        if self._pending_passes >= self.accumulate_passes:
            with self.timer_step:
                metrics = self._apply_update()
            self._pending_passes = 0
            self._steps_this_process += 1
        if update > self.config.budget_warmup_steps:
            if self._pending_passes == 0:
                self._synchronize()
            self.elapsed_sec += time.perf_counter() - started
        if self._pending_passes == 0:
            self._assert_not_diverged()
        return {
            "loss": loss.detach().reshape(1),
            "model": per_token.detach(),
            "metrics": metrics,
        }

    @override
    def charge_budget(self, seconds: float) -> None:
        """Charge loading after warmup using the receiving update's index."""
        if self._steps_this_process + 1 > self.config.budget_warmup_steps:
            self.elapsed_sec += seconds

    @override
    def _apply_update(self) -> dict[str, float | Tensor]:
        """Apply the injected policy or the unchanged original update."""
        if self._optimizer_update is None:
            return super()._apply_update()
        metrics = self._clip_gradients()
        metrics.update(self._optimizer_update(self))
        return metrics

    @override
    @torch.no_grad()
    def _clip_gradients(self) -> dict[str, float | Tensor]:
        """Clip the gradients actually consumed, including persistent FP32 sinks."""
        if not math.isfinite(self.config.gradient_clip_norm):
            return {}
        members = (
            self.optimizer.optimizers
            if isinstance(self.optimizer, CompositeOptimizer)
            else [self.optimizer]
        )
        sinks = {
            parameter: sink
            for member in members
            if isinstance(member, BiasCorrectedRMSProp)
            for parameter, sink in member.gradient_sinks.items()
        }
        gradients = [
            gradient
            for parameter in self.model.parameters()
            if (gradient := sinks.get(parameter, parameter.grad)) is not None
        ]
        total = nn.utils.get_total_norm(gradients)
        coefficient = torch.clamp(
            self.config.gradient_clip_norm / (total + 1e-6),
            max=1.0,
        )
        for gradient in gradients:
            gradient.mul_(coefficient.to(gradient.device))
        return {"grad_norm": total.detach()}

    def _bind_ngram_gradients(self, model: MemoryNanoChatLM) -> None:
        """Bind each persistent table sink to exactly its RMSProp owner."""
        tables = cast(
            "Mapping[str, HashedNgramTables]",
            cast(object, model.bigrams),
        )
        trigram_tables = cast(
            "Mapping[str, HashedNgramTables]",
            cast(object, model.trigrams),
        )
        sinks = {
            part.weight: sink
            for table in (*tables.values(), *trigram_tables.values())
            for part, sink in zip(table.tables, table.gradient_sinks, strict=True)
        }
        marks = {
            part.weight: bitmap
            for table in (*tables.values(), *trigram_tables.values())
            for part, bitmap in zip(table.tables, table.gradient_bitmaps, strict=False)
        }
        assert isinstance(self.optimizer, CompositeOptimizer)
        bound: set[Tensor] = set()
        for member in self.optimizer.optimizers:
            if isinstance(member, BiasCorrectedRMSProp):
                member.gradient_sinks = {
                    parameter: sinks[parameter]
                    for group in member.param_groups
                    for parameter in cast("list[Tensor]", group["params"])
                }
                member.gradient_bitmaps = {
                    parameter: marks[parameter]
                    for group in member.param_groups
                    for parameter in cast("list[Tensor]", group["params"])
                    if parameter in marks
                }
                bound.update(member.gradient_sinks)
        if bound != set(sinks):
            raise ValueError("Every fused n-gram table must route to RMSProp.")


class ReferenceBitsPerByte:
    """Use the reference reduction and denominator with explicit scoring masks."""

    class Config(Fig["ReferenceBitsPerByte"]):
        """Receive all evaluation accounting through batch metadata."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Start an empty evaluation."""
        self.nats = 0.0
        self.bytes = 0
        self.literal_bytes = 0
        self.batches = 0
        self.expected_batches = 0

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate the next complete reference batch.

        Args:
          logits: Per-token losses from the unchanged training-step evaluation API.
          **batch: Prepared mask, denominators, and ordered batch counters.

        """
        mask = batch["score_mask"]
        index = batch["evaluation_batch"]
        count = batch["evaluation_batches"]
        reference_bytes = batch["reference_bytes"]
        literal_bytes = batch["literal_bytes"]
        assert isinstance(mask, Tensor)
        assert isinstance(index, int)
        assert isinstance(count, int)
        assert isinstance(reference_bytes, int)
        assert isinstance(literal_bytes, int)
        if (
            index != self.batches
            or count <= index
            or (self.expected_batches and self.expected_batches != count)
        ):
            raise ValueError("Reference evaluation batch order or extent changed.")
        if mask.shape != logits.shape or mask.dtype != torch.bool:
            raise ValueError("Reference scoring mask differs from loss geometry.")
        # Preserve prepare.py's multiplication and native-dtype reduction. Selecting
        # masked elements or upcasting changes the reference's floating-point sum.
        self.nats += (logits.detach().view(-1) * mask.view(-1)).sum().item()
        self.bytes += reference_bytes
        self.literal_bytes += literal_bytes
        self.batches += 1
        self.expected_batches = count

    def compute(self) -> dict[str, float]:
        """Report BPB after complete evaluation coverage.

        Returns:
          metrics: Reference-denominator BPB and literal-byte BPB.

        """
        if self.batches != self.expected_batches or self.bytes <= 0:
            raise ValueError("Reference evaluation is incomplete.")
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Exact reference evaluation requires one process.")
        return {
            "bpb": self.nats / (math.log(2) * self.bytes),
            "literal_bpb": self.nats / (math.log(2) * self.literal_bytes),
        }

    def state_dict(self) -> dict[str, int | float]:
        """Snapshot evaluation progress.

        Returns:
          state: Accumulated loss, denominators and batch counters.

        """
        return {
            "nats": self.nats,
            "bytes": self.bytes,
            "literal_bytes": self.literal_bytes,
            "batches": self.batches,
            "expected_batches": self.expected_batches,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore evaluation progress.

        Args:
          state_dict: Accumulated loss, denominators and batch counters.

        """
        nats = state_dict["nats"]
        assert isinstance(nats, float)
        self.nats = nats
        for name in ("bytes", "literal_bytes", "batches", "expected_batches"):
            value = state_dict[name]
            assert isinstance(value, int)
            setattr(self, name, value)


class BoundedTokenCrossEntropy:
    """Cross-entropy for symmetrically bounded logits, saving narrow logits and LSE.

    All logits, including ignored positions, must lie in ``[-B, B]`` where B is
    logit_upper_bound. NanoChatTrainStep checks the readout config before training;
    direct callers own this precondition. Use TokenCrossEntropy for arbitrary logits.
    """

    class Config(Fig["BoundedTokenCrossEntropy"]):
        """The logit bound, saved precision, and ignored target."""

        logit_upper_bound: float = 15.0
        """Symmetric logit bound B in (0, 40]; the fixed exponential shift is B."""

        dtype: torch.dtype = torch.float32
        """Logit storage precision used in both the forward and backward."""

        ignore_index: int = -1
        """Target marking a position excluded from the loss."""

    def __init__(self, config: Config) -> None:
        # With logits in [-B, B], the smallest exponential is exp(-2B). Keeping B
        # at most 40 stays above FP32's normal floor even after narrow-logit rounding.
        if (
            not math.isfinite(config.logit_upper_bound)
            or config.logit_upper_bound <= 0
            or config.logit_upper_bound > 40
        ):
            raise ValueError("logit_upper_bound must lie in (0, 40].")
        if config.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("dtype must be bfloat16, float16, or float32.")
        self.logit_upper_bound = config.logit_upper_bound
        self.dtype = config.dtype
        self.ignore_index = config.ignore_index

    def __call__(self, prediction: Tensor, **batch: object) -> LossOutput:
        """Compute per-token cross-entropy for bounded logits.

        Args:
          prediction: Finite logits with vocabulary last, all in ``[-B, B]``.
          **batch: Integer targets under ``label``.

        Returns:
          result: Float32 losses shaped like the targets, zero at ignored positions.

        """
        label = batch["label"]
        assert isinstance(label, Tensor)
        # Save logits at the configured precision to avoid retaining a vocabulary-sized
        # FP32 tensor.
        return {
            "loss": _BoundedCrossEntropy.apply(
                prediction.to(self.dtype),
                label,
                self.logit_upper_bound,
                self.ignore_index,
            ).reshape(label.shape),
        }


# A composite holds every member's groups in one flat list, so a group's position says
# nothing about which algorithm owns it.
def _learning_rates(optimizer: HasParamGroups) -> dict[str, float]:
    """Return one rate per optimizer member, keyed by its class name."""
    if not isinstance(optimizer, CompositeOptimizer):
        return {
            "all": FloatCodec.coerce(
                cast(object, optimizer.param_groups[0]["lr"]),
                None,
            ),
        }
    return {
        type(member).__name__.lower(): FloatCodec.coerce(
            cast(object, member.param_groups[0]["lr"]),
            None,
        )
        for member in optimizer.optimizers
        if member.param_groups
    }


class _LossContext(Protocol):
    saved_tensors: tuple[Tensor, Tensor, Tensor, Tensor]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _bounded_forward(
    ctx: _LossContext,
    logits: Tensor,
    targets: Tensor,
    logit_upper_bound: float,
    ignore_index: int,
) -> Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).long()
    kept = flat_targets != ignore_index
    safe = torch.where(kept, flat_targets, torch.zeros_like(flat_targets))
    sumexp = torch.exp(flat.float() - logit_upper_bound).sum(dim=-1)
    lse = torch.log(sumexp) + logit_upper_bound
    picked = flat.gather(1, safe.unsqueeze(1)).squeeze(1).float()
    ctx.save_for_backward(logits, flat_targets, lse, kept)
    return torch.where(kept, lse - picked, torch.zeros_like(lse))


def _bounded_backward(
    ctx: _LossContext,
    /,
    *grad_outputs: Tensor,
) -> tuple[Tensor, None, None, None]:
    (grad_output,) = grad_outputs
    logits, flat_targets, lse, kept = ctx.saved_tensors
    flat = logits.reshape(-1, logits.shape[-1])
    scale = torch.where(kept, grad_output, torch.zeros_like(grad_output))
    probability = torch.exp(flat.float() - lse.unsqueeze(1))
    columns = torch.arange(flat.shape[-1], device=flat.device)
    selected = columns.unsqueeze(0) == flat_targets.unsqueeze(1)
    # Subtract and scale in FP32 before casting; earlier rounding changes gradients.
    gradient = (probability - selected.to(probability.dtype)) * scale.unsqueeze(1)
    return gradient.to(flat.dtype).reshape_as(logits), None, None, None


class _BoundedCrossEntropy(torch.autograd.Function):
    # Dynamo requires unbound callbacks; classmethods bind the class twice.
    forward = staticmethod(_bounded_forward)
    backward = staticmethod(_bounded_backward)
