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

from priml.baselines.nanochat.model import NanoChatLM
from priml.baselines.nanochat.optimizer import NorMuon
from priml.optimizers import CompositeOptimizer, apply_lr_scale
from priml.optimizers.composite import Selector, complement, excluding
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


def _default_optimizer() -> CompositeOptimizer.Config:
    """NorMuon on the reasoning matrices, AdamW on the tables and the head.

    Orthogonalizing an update suits the square-ish matrices inside the blocks
    and not a lookup table, whose rows are independent and mostly untouched by
    any one batch -- so the model is partitioned by name and the two selectors
    cover it exactly, which ``CompositeOptimizer`` verifies.

    The learning rates differ by two orders of magnitude because the two
    algorithms normalize differently: NorMuon's step is scale-free, while
    AdamW's is in units of the parameter itself.

    Returns:
      config: The default two-member optimizer recipe.

    """
    on_matrices = matrix_parameters()
    config = CompositeOptimizer.Config()
    config.optimizers = [
        PartialConfig(
            torch.optim.AdamW,
            lr=0.004,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
        ),
        NorMuon.Config(),
    ]
    config.select = [complement(on_matrices), on_matrices]
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
        """Compile the model with ``torch.compile``."""

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
            if self.gradient_clip_norm <= 0:
                raise ValueError(
                    f"gradient_clip_norm must be positive; got "
                    f"{self.gradient_clip_norm}. Infinite disables clipping.",
                )
            if self.divergence_threshold <= 0:
                raise ValueError(
                    "divergence_threshold must be positive; got "
                    f"{self.divergence_threshold}.",
                )
            tokens_per_pass = self.rows_per_pass * self.model.max_seq_len
            if self.tokens_per_optimizer_step % tokens_per_pass:
                raise ValueError(
                    f"tokens_per_optimizer_step={self.tokens_per_optimizer_step} "
                    f"is not divisible by rows_per_pass * max_seq_len="
                    f"{tokens_per_pass}, so no whole number of passes reaches "
                    "the token batch.",
                )
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
        """Score one batch, returning the per-token loss the metric consumes."""
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            per_token = self._per_token_loss(self.model, batch)
        return {"loss": per_token.mean().reshape(1), "model": per_token}

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
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["progress"] = progress
        metrics["momentum"] = momentum
        return metrics

    def _per_token_loss(self, model: nn.Module, batch: dict[str, Any]) -> Tensor:
        """Return ``[B, S]`` cross-entropy in nats, one entry per target."""
        media: Tensor = batch["media"]
        labels: Tensor = batch["label"]
        logits = model(media)
        assert isinstance(logits, Tensor)
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1).long(),
            reduction="none",
        ).reshape(labels.shape)

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
