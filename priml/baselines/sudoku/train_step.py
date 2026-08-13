"""Training step for a grid-puzzle solver.

Owns the model, its optimizer, the learning-rate schedule, and the loss. The
optimizer and schedule are INJECTED rather than chosen from a fixed set, so
trying a new optimizer is a config a caller supplies, never a branch added
here.

Adaptive computation time is injected the same way. Without it a step is one
forward and one backward over the batch: the plain supervised recipe. With it,
each puzzle occupies a slot in a persistent pool and takes ONE reasoning step
per call, carrying its latent state forward until the model's halt head says it
is done -- so an easy puzzle leaves after a few steps and a hard one keeps its
slot. The pool is what makes that affordable at fixed batch shape, and it is
also the thing a non-recurrent architecture has no use for, which is why it
lives on the injected piece rather than as fields nobody else can set.

The loss is cross-entropy over grid cells. With ACT attached a second term
trains the halt head to predict whether the current grid is already correct, so
halting is a learned decision rather than a fixed step count.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import field
from typing import Any, Self, override

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.model import SudokuNet
from priml.optimizers import CompositeOptimizer, apply_lr_scale, lr_scale
from priml.optimizers.composite import complement, excluding
from priml.optimizers.muon import Muon
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput
from priml.train.ema import EMA, NoEMA


def _prefix_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    """The batch fields a prefix module consumes, if any.

    A per-puzzle prefix needs to know WHICH puzzle each row is; the grid alone
    cannot say. Passing the whole batch would instead hand the model its own
    labels.
    """
    identifiers = batch.get("puzzle_identifiers")
    return {} if identifiers is None else {"puzzle_identifiers": identifiers}


def _default_optimizer() -> CompositeOptimizer.Config:
    """Muon on the reasoning matrices, AdamW on embeddings and heads.

    Muon orthogonalizes each update, which suits the square-ish weight matrices
    inside the reasoning blocks but not lookup tables or the output projection,
    so the model is partitioned by name: everything Muon declares eligible
    EXCEPT the embeddings and heads goes to Muon, and the complement to AdamW.
    The two selectors partition the model exactly, which
    ``CompositeOptimizer`` verifies.

    Returns:
      config: The default two-member optimizer recipe.

    """
    on_muon = excluding(Muon.eligible_tensor, "embed", "head")
    config = CompositeOptimizer.Config()
    config.optimizers = [
        PartialConfig(
            torch.optim.AdamW,
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=1.0,
        ),
        Muon.Config(lr=0.02, momentum=0.6, nesterov=True, ns_steps=3),
    ]
    config.select = [complement(on_muon), on_muon]
    return config


class SudokuTrainStep:
    """Model plus optimization for one puzzle experiment.

    Implements ``TrainStepProtocol``: the training loop calls
    :meth:`train_step` per batch and :meth:`eval_loss` per evaluation, and
    persists everything through :meth:`state_dict`.
    """

    class Config(Fig["SudokuTrainStep"]):
        """Model, optimization, schedule, loss, and the optional ACT pool."""

        model: SudokuNet.Config = field(default_factory=SudokuNet.Config)
        """Network to train.

        Narrowed to the concrete config rather than ``Makeable[SudokuNet]``:
        every experiment in this directory trains this model and reaches its
        fields directly, so the narrow belongs here once instead of as an
        ``isinstance`` in each factory. A different network is a different
        baseline, not a different value here."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=_default_optimizer,
        )
        """Builds the optimizer from the model.

        Injected, not selected from a fixed set: a new optimizer is a config a
        caller supplies. A split recipe routes parameters with
        ``CompositeOptimizer.Config.select`` and still presents as one."""

        total_train_steps: int = 19_500
        """Schedule horizon. Set it to the run's step budget, or the learning
        rate anneals past the end of training or short of it."""

        warmup_steps: int = 0
        """Linear warmup steps before the cosine decay begins."""

        lr_min_ratio: float = 0.0
        """Floor of the cosine decay, as a fraction of the base rate."""

        act: ActPool.Config | None = None
        """Adaptive computation time. ``None`` trains one forward per batch.

        Every knob that only means something under ACT -- pool width, step cap,
        halt exploration -- lives on this piece, so a plain run's config does
        not carry them.

        Narrowed to the concrete config for the same reason as ``model``:
        experiments in this directory reach its fields directly."""

        label_smoothing: float = 0.0
        """Cross-entropy label smoothing."""

        ignore_label_id: int = -100
        """Label value excluded from the loss and from halt correctness."""

        gradient_clip_norm: float = 1.0
        """Global gradient-norm cap; infinite disables clipping."""

        use_ema: bool = True
        """Track an exponential moving average of the weights for evaluation."""

        ema_decay: float = 0.999
        """EMA decay applied each optimizer step."""

        ema_warmup_steps: int = 0
        """Steps before EMA tracking begins; live weights seed the shadow."""

        device: str = "auto"
        """Device to train on ("auto" picks the best available)."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype; ``None`` trains in full precision."""

        @override
        def finalize(self) -> Self:
            if self.act is not None:
                # The pool holds one latent state per slot, so it must be built
                # to the model's shape; pushing it down here keeps the two from
                # being set independently and silently disagreeing.
                self.act.grid_len = self.model.grid_len
                self.act.seq_len = self.model.total_seq_len
                self.act.hidden_size = self.model.hidden_size
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.total_train_steps <= 0:
            raise ValueError(
                f"total_train_steps must be positive; got {config.total_train_steps}.",
            )
        self.config = config
        self.device = get_device(config.device)
        self.global_step: int = 0
        self.local_step: int = 0
        self.model: SudokuNet = config.model.make()
        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = config.optimizer.make()(self.model)
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        self.act: ActPool | None = config.act.make() if config.act is not None else None
        if self.act is not None:
            self.act.to(self.device)
        self._ema: EMA | NoEMA
        if config.use_ema:
            self._ema = EMA.Config(
                decay=config.ema_decay,
                update_after_step=config.ema_warmup_steps,
                warmup_seed=True,
                track_buffers=False,
                shadow_kind="param_dict",
            ).make()
        else:
            self._ema = NoEMA()

    @property
    def ema_shadow(self) -> dict[str, Tensor] | None:
        """Name-keyed EMA weights, or None when EMA is disabled."""
        if isinstance(self._ema, NoEMA):
            return None
        return self._ema.shadow_params

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch to the training device."""
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

    def train_step(self, **batch: Any) -> TrainStepOutput:
        """Forward, backward, and step the optimizer.

        With an ACT pool attached the batch first refills halted slots, and the
        forward runs over the pool rather than the incoming batch; the pool
        then advances its halt state. Without one this is a plain supervised
        step over the batch as given.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss``, a ``model`` probe slice, and scalar metrics.

        """
        self.model.train()
        media, labels, active = self._ingest(batch)
        with self._autocast():
            out = self.model(media, *self._carry(), **_prefix_kwargs(batch))
            loss, metrics = self._loss(
                out.logits, labels=labels, halt=out.halt, active=active
            )
        loss.backward()

        if math.isfinite(self.config.gradient_clip_norm):
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            metrics["grad_norm"] = grad_norm.detach()
        scale = lr_scale(
            self.global_step,
            self.config.total_train_steps,
            self.config.warmup_steps,
            self.config.lr_min_ratio,
        )
        apply_lr_scale([self.optimizer], scale)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        self._ema(self.model)
        if self.act is not None:
            self.act.advance(
                out.z_slow,
                z_fast=out.z_fast,
                logits=out.logits,
                halt=out.halt,
                media=media,
            )

        self.global_step += 1
        self.local_step += 1
        return {
            "loss": loss.detach().reshape(1),
            "model": out.logits[:1, :1].detach(),
            "metrics": metrics,
        }

    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        self.model.train()
        media: Tensor = batch["media"]
        labels: Tensor = batch["label"]
        with torch.no_grad(), self._autocast():
            out = self.model(media, **_prefix_kwargs(batch))
            active = torch.ones(media.shape[0], dtype=torch.bool, device=media.device)
            loss, metrics = self._loss(
                out.logits, labels=labels, halt=out.halt, active=active
            )
        return {"loss": loss, "model": out.logits, "metrics": metrics}

    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Score one batch, packing predictions for the accuracy metric.

        Evaluation always scores the EMA weights once past warmup, and always
        runs the model's full reasoning depth -- a rollout carrying latents
        across steps, when ACT is attached, since that is the regime the model
        trained in.
        """
        media: Tensor = batch["media"]
        labels: Tensor = batch["label"]
        with self._eval_weights():
            self.model.eval()
            with torch.inference_mode(), self._autocast():
                logits, halt = self._eval_rollout(media, _prefix_kwargs(batch))
                active = torch.ones(
                    media.shape[0], dtype=torch.bool, device=media.device
                )
                loss, metrics = self._loss(
                    logits, labels=labels, halt=halt, active=active
                )
        predictions = logits.argmax(dim=-1)
        packed = torch.cat(
            [halt.reshape(-1, 1).float(), predictions.float()],
            dim=-1,
        )
        return {"loss": loss.detach().reshape(1), "model": packed, "metrics": metrics}

    def call_eval(self, **batch: Any) -> Tensor:
        """Return evaluation logits under the EMA weights."""
        media: Tensor = batch["media"]
        with self._eval_weights():
            self.model.eval()
            with torch.inference_mode(), self._autocast():
                logits, _ = self._eval_rollout(media, _prefix_kwargs(batch))
        return logits

    def on_epoch_end(self) -> None:
        """Do nothing: this step holds no partial state across epochs."""

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, EMA, and progress state.

        The ACT pool is deliberately excluded: it is in-flight state bound to
        the specific puzzles being solved, and a resumed run continues with the
        next batch rather than replaying interrupted ones.
        """
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        shadow = self.ema_shadow
        if shadow is not None:
            state["ema"] = dict(shadow)
        if self.act is not None:
            state["act"] = self.act.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = int(state_dict["global_step"])
        self.local_step = 0
        if not isinstance(self._ema, NoEMA) and "ema" in state_dict:
            self._ema.global_step = self.global_step
            self._ema.load_state_dict(
                {
                    "shadow_params": dict(state_dict["ema"]),
                    "global_step": self.global_step,
                },
            )
        if self.act is not None and "act" in state_dict:
            self.act.load_state_dict(state_dict["act"])

    def _ingest(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        """Return the ``(media, labels, active)`` this step trains on."""
        media: Tensor = batch["media"]
        labels: Tensor = batch["label"]
        valid_count = int(batch.get("valid_count", media.shape[0]))
        if self.act is None:
            active = torch.ones(media.shape[0], dtype=torch.bool, device=media.device)
            return media, labels, active
        return self.act.refill(
            media,
            labels=labels,
            valid_count=valid_count,
            ignore_label_id=self.config.ignore_label_id,
        )

    def _carry(self) -> tuple[Tensor, Tensor] | tuple[()]:
        """The latent state the pool carries, or nothing when ACT is off."""
        if self.act is None:
            return ()
        return self.act.latents()

    def _eval_rollout(
        self,
        media: Tensor,
        prefix_kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        """Run the model to its full depth, carrying latents when ACT is on."""
        if self.act is None:
            out = self.model(media, **prefix_kwargs)
            return out.logits, out.halt
        return self.act.rollout(self.model, media=media, prefix_kwargs=prefix_kwargs)

    def _loss(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        halt: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """Cross-entropy over cells, plus the halt term when ACT is attached.

        Only active rows contribute, and the mean is over those rows: a pool
        slot holding no puzzle must not dilute the gradient.
        """
        config = self.config
        ignore = config.ignore_label_id
        per_token = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1).long(),
            label_smoothing=config.label_smoothing,
            ignore_index=ignore,
            reduction="none",
        ).reshape(logits.shape[0], -1)
        counted = (labels != ignore).sum(dim=-1).clamp(min=1)
        per_sample = per_token.sum(dim=-1) / counted
        n_active = active.sum().clamp(min=1)
        lm_loss = torch.where(active, per_sample, torch.zeros_like(per_sample)).sum()
        lm_loss = lm_loss / n_active
        metrics: dict[str, float | Tensor] = {"lm_loss": lm_loss.detach()}
        if self.act is None:
            return lm_loss, metrics
        halt_loss, halt_metrics = self.act.halt_loss(
            logits,
            labels=labels,
            halt=halt,
            active=active,
            ignore_label_id=ignore,
        )
        metrics.update(halt_metrics)
        return lm_loss + halt_loss, metrics

    @contextmanager
    def _eval_weights(self) -> Generator[None]:
        """Swap in the EMA shadow for the duration, once past warmup.

        Strictly past: the shadow is seeded by the train step AT the warmup
        boundary, which runs after that step's evaluation, so an evaluation on
        the boundary itself would score an unseeded shadow.
        """
        if isinstance(self._ema, NoEMA) or self.global_step <= (
            self.config.ema_warmup_steps
        ):
            yield
            return
        with self._ema.apply_to(self.model):
            yield

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
