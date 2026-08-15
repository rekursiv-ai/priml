"""Training step for a budgeted language-model run.

Two things distinguish it from an ordinary supervised step, and both follow
from training to a wall-clock BUDGET rather than to a step count.

**Progress, not steps, drives every schedule.** The step measures its own
elapsed training seconds and divides by the budget; the learning rate, the
optimizer's momentum, and its weight decay are all functions of that fraction.
A change that makes a step cheaper therefore buys more steps at the same
schedule shape, which is the comparison the budget exists to make. Step count
is not known in advance, so a step-indexed schedule could not be written.

**The clock excludes warmup.** The first steps compile kernels and touch cold
caches, so charging them to the budget would let compile time decide how much
training a run gets. ``budget_warmup_steps`` are excluded, after which the
clock accumulates only the time spent inside :meth:`train_step` -- never
evaluation, never data preparation.

Gradient accumulation is priml's, configured to reach a fixed TOKEN count per
optimizer step: the token batch is the quantity a language-model recipe is
tuned against, so it stays constant while the per-pass row count follows
whatever fits in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import field
from typing import Any, Self, override

import math
import time

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.nanochat.data import IGNORED_TARGET
from priml.baselines.nanochat.model import NanoChatLM
from priml.baselines.nanochat.optimizer import NorMuon
from priml.optimizers import CompositeOptimizer, FusedAdamW, apply_lr_scale
from priml.optimizers.composite import Selector, excluding, matching
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput


def trapezoid(progress: float, *, flat_fraction: float = 0.5) -> float:
    """Hold the learning rate flat, then decay it linearly to zero.

    A budgeted run cannot warm up on a step index it does not know, and the
    flat-then-decay shape spends most of the budget at the full rate while
    still landing at zero -- which is what makes the final weights an average
    over a low-noise tail rather than a snapshot mid-oscillation.

    Args:
      progress: Fraction of the budget spent, in ``[0, 1]``.
      flat_fraction: Share of the budget held at the full rate.

    Returns:
      multiplier: Learning-rate multiplier in ``[0, 1]``.

    """
    if progress < flat_fraction:
        return 1.0
    return max(0.0, (1.0 - progress) / (1.0 - flat_fraction))


def matrix_parameters() -> Selector:
    """Select the reasoning matrices: rank >= 2, and not a lookup table.

    Returns:
      select: The predicate routing parameters to the orthogonalizing member.

    """
    return excluding(NorMuon.eligible_tensor, "embed", "lm_head")


def width_scaled_lr(rate: float, *, channels: int, tuned_at: int = 768) -> float:
    """Scale a learning rate by ``1/sqrt(width)`` away from where it was tuned.

    The Adam rates in this recipe were fitted on a 768-wide model. A wider one
    takes smaller steps per unit of loss, so carrying the same number to
    another width silently trains a different recipe -- the rate is a property
    of the pair, not of the optimizer.

    Only the Adam rates scale. NorMuon's step is orthogonalized and therefore
    already scale-free, which is why the reference applies this to four groups
    and not to the matrices.

    Args:
      rate: Rate as tuned at ``tuned_at`` channels.
      channels: This model's width.
      tuned_at: Width the rate was fitted at.

    Returns:
      scaled: The rate this width should use.

    """
    # ``math.sqrt`` rather than ``** 0.5``: the operator types as ``Any``
    # because a fractional power of a negative base is complex, and widths are
    # positive by construction.
    return rate / math.sqrt(channels / tuned_at)


def _default_optimizer() -> CompositeOptimizer.Config:
    """NorMuon on the reasoning matrices, AdamW on everything else, by class.

    Orthogonalizing an update suits the square-ish matrices inside the blocks
    and not a lookup table, whose rows are independent and mostly untouched by
    any one batch -- so the model is partitioned by name and the selectors
    cover it exactly, which ``CompositeOptimizer`` verifies.

    The AdamW side is FIVE members rather than one because the classes it
    holds want rates two orders of magnitude apart: a token table is read once
    per occurrence and needs a large step, an unembedding projection sees every
    position and needs a small one, and the per-layer scalars sit between them.
    Collapsing them to a single rate trains a different model at the same
    nominal hyperparameters -- which is exactly the kind of difference a
    reproduction is supposed to exclude.

    The Adam rates are stated AS TUNED, at 768 channels; the step's
    ``finalize`` rescales them to the model's actual width. Baking a width in
    here would make the default silently wrong for every model that is not
    768 wide.

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
        NorMuon.Config(),
    ]
    # Ordered most specific first: ``value_embeds`` also contains ``embed``, so
    # the token table's selector must exclude it rather than claim it.
    config.select = [
        matching("lm_head"),
        excluding(matching("embed"), "value_embeds"),
        matching("value_embeds"),
        matching("residual_scale"),
        matching("skip_scale"),
        on_matrices,
    ]
    # The value-embedding member is dropped when the model has no such tables.
    # A selector claiming nothing is REJECTED, which is the right default -- it
    # catches a misspelled fragment -- but a rung that switches the mechanism
    # off is not a typo, and it would otherwise be unable to use this recipe.
    config.drop_empty = True
    return config


class NanoChatTrainStep:
    """Model plus optimization for one budgeted language-model experiment.

    Implements ``TrainStepProtocol``: the training loop calls
    :meth:`train_step` per batch and :meth:`eval_loss` per evaluation, and
    persists everything through :meth:`state_dict`.
    """

    class Config(Fig["NanoChatTrainStep"]):
        """Model, optimization, the budget, and the schedules it drives."""

        model: NanoChatLM.Config = field(default_factory=NanoChatLM.Config)
        """Network to train.

        Narrowed to the concrete config rather than ``Makeable[nn.Module]``:
        every experiment in this directory trains this model and reaches its
        fields directly, so the narrow belongs here once instead of as an
        ``isinstance`` in each factory."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=_default_optimizer,
        )
        """Builds the optimizer from the model.

        Injected, not selected from a fixed set: a new optimizer is a config a
        caller supplies. A split recipe routes parameters with
        ``CompositeOptimizer.Config.select`` and still presents as one."""

        schedule: Makeable[Callable[[float], float]] = field(
            default_factory=lambda: PartialConfig(trapezoid),
        )
        """Maps budget progress in ``[0, 1]`` to a learning-rate multiplier.

        Takes PROGRESS rather than ``(step, total_steps)`` like priml's
        schedules: a budgeted run does not know its step count in advance, so
        there is no horizon to divide by."""

        time_budget_sec: float = 300.0
        """Training seconds the schedules anneal over.

        The loop's own ``max_time`` stops the run; this is the horizon the
        schedules use, and the two are set together for the same reason a
        step-based run matches ``total_train_steps`` to ``max_steps``."""

        budget_warmup_steps: int = 10
        """Leading steps excluded from the clock (compilation, cold caches)."""

        tokens_per_optimizer_step: int = 524_288
        """Tokens per optimizer step, reached by gradient accumulation.

        The quantity the recipe is tuned against, held fixed while
        ``rows_per_pass`` follows whatever fits in memory."""

        rows_per_pass: int = 32
        """Rows per forward/backward pass. Reduce on out-of-memory."""

        momentum_start: float = 0.85
        """Orthogonalizing member's momentum at the first step."""

        momentum_end: float = 0.95
        """Momentum it reaches after ``momentum_warmup_steps``.

        Ramped rather than fixed: momentum averages over past gradients, and
        early ones come from a model changing fast enough that averaging them
        is averaging over different models."""

        momentum_warmup_steps: int = 300
        """Steps over which momentum ramps. Indexed by STEP, not progress: it
        corrects a transient of early training, which is a step-count effect."""

        decay_to_zero: bool = True
        """Anneal weight decay to zero over the budget.

        Decay pulls toward the origin at a rate the loss no longer balances
        once the learning rate has decayed, so holding it fixed shrinks the
        final weights for no reason."""

        gradient_clip_norm: float = math.inf
        """Global gradient-norm cap; infinite disables clipping."""

        divergence_threshold: float = 100.0
        """Loss above which the run raises rather than continues.

        A diverged language-model run does not recover, and a budgeted one
        would otherwise spend its whole budget proving it."""

        device: str = "auto"
        """Device to train on ("auto" picks the best available)."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype; ``None`` trains in full precision."""

        compile: bool = True
        """Compile the MODEL with ``torch.compile``.

        The optimizer's kernels are not covered by this and must not be: a
        compiled step and an eager one differ numerically (measured, 2.9e-2 on
        one update), so the reference's compiled kernel is part of the recipe
        rather than a speed switch. Each optimizer member carries its own
        ``compiled`` field for that reason -- turning this off to skip the
        model's compile would otherwise silently change the arithmetic."""

        adam_lr_tuned_at_channels: int = 768
        """Width the Adam rates in ``optimizer`` were fitted at.

        A rate is a property of the (rate, width) pair: a wider model takes
        smaller steps per unit of loss, so carrying one number across widths
        trains a different recipe under the same nominal hyperparameters.
        ``finalize`` therefore rescales every Adam member by
        ``sqrt(tuned_at / channels)``.

        Set it to the model's own width to disable the rescale -- which is
        what a caller supplying already-scaled rates wants."""

        @override
        def finalize(self) -> Self:
            if self.time_budget_sec <= 0 or not math.isfinite(self.time_budget_sec):
                raise ValueError(
                    "time_budget_sec must be finite and positive; got "
                    f"{self.time_budget_sec}.",
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
            # model and wrong for every fork that changes ``channels``. The
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
                    member.lr = width_scaled_lr(
                        rate,
                        channels=self.model.channels,
                        tuned_at=self.adam_lr_tuned_at_channels,
                    )
            self.adam_lr_tuned_at_channels = self.model.channels
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = get_device(config.device)
        self.global_step: int = 0
        self.local_step: int = 0
        self.accumulate_passes = config.tokens_per_optimizer_step // (
            config.rows_per_pass * config.model.max_seq_len
        )
        self._pending_passes = 0
        # Budget-counted seconds, excluding warmup and everything outside
        # train_step. The loop reads it through NanoChatTrainLoop.
        self.elapsed_sec = 0.0
        model: nn.Module = config.model.make()
        self.raw_model = model.to(self.device)
        self.model: nn.Module = (
            torch.compile(self.raw_model) if config.compile else self.raw_model
        )
        self.optimizer: torch.optim.Optimizer = config.optimizer.make()(self.raw_model)
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
            group.setdefault("initial_weight_decay", group.get("weight_decay", 0.0))
        self.schedule = config.schedule.make()

    @property
    def progress(self) -> float:
        """Fraction of the budget spent, clamped to ``[0, 1]``."""
        return min(1.0, self.elapsed_sec / self.config.time_budget_sec)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch to the training device."""
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

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
        # Drain first, THEN start the clock. Device work is asynchronous, so
        # anything still queued -- an evaluation that just ran -- would be
        # waited for inside this step's timing and charged to the budget as
        # training. Measured: with evaluation on the cadence, the budget bought
        # 422 steps against 836 with it off, for identical training work.
        self._synchronize()
        started = time.perf_counter()
        self.model.train()
        with self._autocast():
            per_token = self._per_token_loss(self.model, batch)
            loss = per_token.mean()
        # Each pass contributes its share, so the accumulated gradient is the
        # mean over the whole token batch rather than over the last pass.
        (loss / self.accumulate_passes).backward()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value) or loss_value > config.divergence_threshold:
            self.raw_model.zero_grad(set_to_none=True)
            # The gradients this count refers to were just discarded, so the
            # count goes with them: a caller that caught this and continued
            # would otherwise step on a short token batch.
            self._pending_passes = 0
            raise RuntimeError(
                f"training diverged at step {self.global_step}: loss={loss_value}.",
            )

        self._pending_passes += 1
        metrics: dict[str, float | Tensor] = {}
        if self._pending_passes >= self.accumulate_passes:
            metrics = self._apply_update()
            self._pending_passes = 0
            self.global_step += 1
        self.local_step += 1
        # Charged after the update so a step's own optimizer time counts, and
        # only past warmup so compilation does not consume the budget.
        if self.local_step > config.budget_warmup_steps:
            # Drain again before stopping: the backward and the optimizer are
            # queued, not finished, so a CPU-side reading would charge this
            # step for less than it used and the next one for the remainder.
            self._synchronize()
            self.elapsed_sec += time.perf_counter() - started
        return {
            "loss": loss.detach().reshape(1),
            "model": per_token.detach(),
            "metrics": metrics,
        }

    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        self.model.train()
        with torch.no_grad(), self._autocast():
            per_token = self._per_token_loss(self.model, batch)
        return {"loss": per_token.mean().reshape(1), "model": per_token}

    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Score one batch, returning the per-token loss the metric consumes.

        The reported ``loss`` is the mean over REAL rows only. Padding a short
        batch leaves ignored positions contributing zero, and the loop weights
        this mean by ``valid_count`` -- so averaging over the padded width
        would report a loss diluted by rows that are not data.
        """
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            per_token = self._per_token_loss(self.model, batch)
        valid = int(batch.get("valid_count", per_token.shape[0]))
        return {
            "loss": per_token[:valid].mean().reshape(1),
            "model": per_token,
        }

    def call_eval(self, **batch: Any) -> Tensor:
        """Return evaluation logits for one batch."""
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            logits = self.model(batch["media"])
        assert isinstance(logits, Tensor)
        return logits

    def on_epoch_end(self) -> None:
        """Discard a partial accumulation so gradients never cross a pass."""
        if self._pending_passes:
            self.raw_model.zero_grad(set_to_none=True)
            self._pending_passes = 0

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, and budget state.

        The elapsed clock is checkpointed because it drives every schedule: a
        resumed run that restarted it would re-anneal the learning rate from
        the top and undo the decay it had already applied.

        ``local_step`` travels with it because it GATES that clock: the warmup
        exclusion is "the first N steps of the process", and a resume that
        reset the counter would grant N more steps costing no budget, so a
        frequently-resumed run would train unboundedly on a fixed budget.
        """
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "local_step": self.local_step,
            "elapsed_sec": self.elapsed_sec,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        # Checked BEFORE anything is assigned: a caller that catches this must
        # not be left holding the checkpoint's weights and clock beside its own
        # warmup counter, which is the uncharged-training state being refused.
        if "local_step" not in state_dict:
            raise ValueError(
                "this checkpoint records no 'local_step', so it predates the "
                "budget-clock fix and its warmup accounting cannot be "
                "reconstructed; resuming would grant uncharged training. "
                "Start a fresh run.",
            )
        self.raw_model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = int(state_dict["global_step"])
        self.elapsed_sec = float(state_dict["elapsed_sec"])
        self.local_step = int(state_dict["local_step"])
        self._pending_passes = 0

    def _apply_update(self) -> dict[str, float | Tensor]:
        """Set every schedule for this step, then step the optimizer."""
        config = self.config
        metrics: dict[str, float | Tensor] = {}
        if math.isfinite(config.gradient_clip_norm):
            metrics["grad_norm"] = nn.utils.clip_grad_norm_(
                self.raw_model.parameters(),
                config.gradient_clip_norm,
            ).detach()
        progress = self.progress
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
        self.raw_model.zero_grad(set_to_none=True)
        # Per MEMBER, not ``param_groups[0]``: the recipe runs two optimizers
        # at rates two orders of magnitude apart, and the first group belongs
        # to whichever the composite lists first -- so a single ``lr`` reports
        # one algorithm's rate while the other's is invisible.
        for name, rate in _learning_rates(self.optimizer).items():
            metrics[f"lr_{name}"] = rate
        metrics["progress"] = progress
        metrics["momentum"] = momentum
        return metrics

    def _per_token_loss(self, model: nn.Module, batch: dict[str, Any]) -> Tensor:
        """Return ``[B, S]`` cross-entropy in nats, one entry per target.

        Rows padding a short evaluation batch carry :data:`IGNORED_TARGET`,
        which is not a class index -- so it is named here rather than left to
        reach the logits, where it raises rather than scoring.
        """
        media: Tensor = batch["media"]
        labels: Tensor = batch["label"]
        logits = model(media)
        assert isinstance(logits, Tensor)
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1).long(),
            ignore_index=IGNORED_TARGET,
            reduction="none",
        ).reshape(labels.shape)

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


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return one rate per optimizer member, keyed by its class name.

    A composite holds every member's groups in one flat list, so the position
    of a group says nothing about which algorithm owns it. Reading the member
    directly keeps the reported rate attributable when a recipe runs two.

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
