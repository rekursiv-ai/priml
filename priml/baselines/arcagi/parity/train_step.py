"""The ARC-AGI-2 reference step: atomic ACT, stablemax, sparse task-table updates.

A second implementation of the recipe
:class:`~priml.baselines.arcagi.train_step.PooledActTrainStep` runs, ported bit
for bit from the source as a subclass of the sudoku step. No rung trains on it;
its tests pin the source's single- and two-rank trajectories.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, override

import math

from configgle import Makeable, Makes, Makes2
from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.arcagi.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.sudoku.act import (
    AtomicPool,
    HaltTraining,
    PoolConfig,
    SampledMinimum,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.math.loss import stablemax_cross_entropy
from priml.model.custom_types import TensorModule
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import GridConfig, GridSolver
from priml.model.prefix import PrefixConfig, SparsePuzzleEmbedding
from priml.model.recurrence import DeepRecurrence, RecurrenceConfig
from priml.optimizers import (
    AdamATan2,
    SignSGD,
    apply_lr_scale,
    learning_rate,
    lr_scale,
)
from priml.train.grad_clip import clip_grad_norm_


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from priml.train.custom_types import TrainStepOutput


class ArcModelConfig(Makes["GridSolver"], GridSolver.Config):
    """Reference TRM assembled from the shared puzzle solver's slots."""

    embedding: GridConfig = field(default_factory=GridEmbedding.Config)
    """Token embeddings without additional position or feedback channels."""

    block: Makeable[TensorModule] = field(default_factory=RotaryBlock.Config)
    """Post-normalized rotary transformer."""

    recurrence: RecurrenceConfig | None = field(
        default_factory=lambda: DeepRecurrence.Config(slow_cycles=3, fast_cycles=4),
    )
    """Three slow cycles, each refining the fast state four times."""

    prefix: PrefixConfig | None = field(
        default_factory=lambda: PuzzleEmbedding.Config(
            num_puzzles=1_191_727,
            batch_size=256,
        ),
    )
    """Per-task sparse prefix; no register tokens."""


class ArcTrainStep(SudokuTrainStep):
    """Atomic ACT with a stablemax loss and a sparsely updated task table."""

    class Config(Makes2["ArcTrainStep"], SudokuTrainStep.Config):
        """Source TRM objective and injected dense/sparse optimizers."""

        model: GridSolver.Config = field(default_factory=ArcModelConfig)
        """Shared puzzle solver with ARC2-specific slot defaults."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: AdamATan2.Config(
                lr=1e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            ),
        )
        """Dense body optimizer."""

        sparse_optimizer: SignSGD.Config = field(
            default_factory=lambda: SignSGD.Config(lr=1e-2, weight_decay=0.1),
        )
        """Touched-row optimizer; its rate is constant during body warmup."""

        pool: PoolConfig | None = field(
            default_factory=lambda: AtomicPool.Config(
                batch_size=256,
                max_steps=16,
                halting=HaltTraining.Config(
                    weight=0.5,
                    exploration=SampledMinimum.Config(),
                ),
            ),
        )
        """Atomic ACT without prediction feedback."""

        total_train_steps: int = 541_580
        """Reference sample-scale training horizon."""

        warmup_steps: int = 2_000
        """Dense optimizer warmup."""

        lr_min_ratio: float = 1.0
        """Constant rate after warmup."""

        ema_warmup_steps: int = 2_000
        """Warmup before seeding the EMA shadow."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config.pool, AtomicPool.Config):
            raise TypeError("ARC2 training requires an atomic ACT pool.")
        super().__init__(config)
        self.sparse_optimizer = config.sparse_optimizer.make()(
            [
                {
                    "params": list(self.puzzle_embedding.buffers()),
                    "sparse_embedding": True,
                },
            ],
        )

    class StateDict(SudokuTrainStep.StateDict):
        """Dense trainer state plus the sparse optimizer's saved hyperparameters."""

        sparse_optimizer: Mapping[str, object]

    @override
    def state_dict(self) -> StateDict:
        """Persist sparse optimizer settings without persisting in-flight ACT slots."""
        return {  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            **super().state_dict(),
            "sparse_optimizer": self.sparse_optimizer.state_dict(),
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
        """Restore sparse hyperparameters together with the dense optimizer."""
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        if load_optimizer:
            self.sparse_optimizer.load_state_dict(
                from_plain(state_dict["sparse_optimizer"], dict[str, object]),
            )

    @property
    def puzzle_embedding(self) -> SparsePuzzleEmbedding:
        """The injected prefix's local-gradient and master-weight buffers."""
        prefix = self.net.prefix
        assert isinstance(prefix, SparsePuzzleEmbedding)
        return prefix

    @override
    def _loss(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        halt: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        pool = self.pool
        assert isinstance(pool, AtomicPool)
        ignore = self.config.ignore_label_id
        per_token = stablemax_cross_entropy(
            logits.double(),
            labels.long(),
            ignore_index=ignore,
        )
        counted = (labels != ignore).sum(dim=-1).clamp(min=1)
        per_sample = per_token.sum(dim=-1) / counted
        lm_loss = torch.where(
            active,
            per_sample,
            torch.zeros_like(per_sample),
        ).sum() / float(pool.config.batch_size)
        halt_loss, metrics = self._halt_loss(
            logits,
            labels=labels,
            halt=halt,
            active=active,
        )
        metrics["lm_loss"] = lm_loss.detach()
        return lm_loss + halt_loss, metrics

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score valid rows only, while retaining padded prediction alignment."""
        media, labels = batch["media"], batch["label"]
        identifiers = batch["puzzle_identifiers"]
        assert isinstance(media, Tensor)
        assert isinstance(labels, Tensor)
        assert isinstance(identifiers, Tensor)
        valid = batch.get("valid_count", len(media))
        assert isinstance(valid, int)
        if valid == 0:
            return {
                "loss": media.new_zeros(1, dtype=torch.float32),
                "model": media.new_zeros(
                    (len(media), 1 + media.shape[1]),
                    dtype=torch.float32,
                ),
            }
        with self._eval_weights():
            self.model.eval()
            with torch.inference_mode(), self._autocast():
                logits, halt = self._eval_rollout(
                    media,
                    {"puzzle_identifiers": identifiers},
                )
                target = labels[:valid]
                active = torch.ones(valid, dtype=torch.bool, device=self.device)
                per_token = stablemax_cross_entropy(
                    logits[:valid].double(),
                    target.long(),
                    ignore_index=self.config.ignore_label_id,
                )
                counted = (
                    (target != self.config.ignore_label_id).sum(dim=-1).clamp(min=1)
                )
                lm_loss = (per_token.sum(dim=-1) / counted).sum() / active.sum().clamp(
                    min=1,
                )
                halt_loss, metrics = self._halt_loss(
                    logits[:valid],
                    labels=target,
                    halt=halt[:valid],
                    active=active,
                )
                loss = lm_loss + halt_loss
        metrics["lm_loss"] = lm_loss.detach()
        packed = torch.cat(
            [halt.reshape(-1, 1).float(), logits.argmax(dim=-1).float()],
            dim=-1,
        )
        return {
            "loss": loss.detach().reshape(1).float(),
            "model": packed,
            "metrics": metrics,
        }

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Advance atomic ACT, then update the body and touched sparse rows."""
        self.model.train()
        media, labels, active = self._ingest(batch)
        self._stash_feedback()
        with self._autocast():
            output = self.net(media, *self._carry(), **self._prefix_kwargs(batch))
        loss, metrics = self._loss(
            output.logits,
            labels=labels,
            halt=output.halt,
            active=active,
        )
        self._advance_feedback(output.logits, metrics)
        loss.backward()
        if math.isfinite(self.config.gradient_clip_norm):
            self.last_grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            metrics["grad_norm"] = self.last_grad_norm.detach()
        with self.timer_step:
            scale = lr_scale(
                self.global_step,
                self.config.total_train_steps,
                self.config.warmup_steps,
                self.config.lr_min_ratio,
            )
            apply_lr_scale([self.optimizer], scale)
            metrics["lr"] = learning_rate(self.optimizer)
            self.optimizer.step()
            sparse = self.puzzle_embedding
            if sparse.local_weights.grad is not None:
                self.sparse_optimizer.step_sparse_embedding(
                    sparse.local_weights.grad.detach(),
                    sparse.local_ids.detach(),
                )
                sparse.local_weights.grad = None
            self.model.zero_grad(set_to_none=True)
            self._ema(self.model)
        self._advance_pool(
            output.z_slow,
            output.z_fast,
            halt=output.halt,
            active=active,
        )
        return {
            "loss": loss.detach().reshape(1),
            "model": output.logits[:1, :1].detach(),
            "metrics": metrics,
        }
