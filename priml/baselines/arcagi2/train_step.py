"""ARC2 atomic ACT with stablemax and sparse puzzle-embedding updates."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, override

import math

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch
import torch.distributed as dist

from priml.baselines.arcagi2.model import ArcModelConfig
from priml.baselines.sudoku.act import ActPool
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.lib.custom_json import DictCodec
from priml.math.loss import stablemax_cross_entropy
from priml.optimizers import (
    AdamATan2,
    SignSGD,
    apply_lr_scale,
    learning_rate,
    lr_scale,
)
from priml.train.grad_clip import clip_grad_norm_
from priml.train.parallelism import DataParallel, place


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from priml.baselines.sudoku.model import SudokuNet
    from priml.train.custom_types import TrainStepOutput


class ArcDataParallel(DataParallel):
    """Broadcast initial latents before ACT takes its first carried-state copy."""

    class Config(Makes["ArcDataParallel"], DataParallel.Config):
        """Source-compatible data parallelism with eager state synchronization."""

    @override
    def __call__(self, model: nn.Module) -> nn.Module:
        """Place and broadcast every parameter and buffer before replication."""
        model = place(model, self.device)
        source = dist.get_global_rank(self.process_group, 0)
        with torch.no_grad():
            for tensor in [*model.parameters(), *model.buffers()]:
                dist.broadcast(tensor, src=source, group=self.process_group)
        return super().__call__(model)


class ArcTrainStep(SudokuTrainStep):
    """Retain task identifiers and learned initial latents in atomic ACT slots."""

    class Config(SudokuTrainStep.Config):
        """Source TRM objective and injected dense/sparse optimizers."""

        if TYPE_CHECKING:

            @override
            def make(self) -> ArcTrainStep:
                """Build this nested configuration's ARC2 trainer."""
                ...

        model: SudokuNet.Config = field(default_factory=ArcModelConfig)
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

        act: ActPool.Config | None = field(
            default_factory=lambda: ActPool.Config(
                batch_size=256,
                max_steps=16,
                halt_weight=0.5,
                feedback=False,
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
        if config.act is None:
            raise ValueError("ARC2 training requires an atomic ACT pool.")
        super().__init__(config)
        self.puzzle_ids = torch.zeros(
            config.act.batch_size,
            dtype=torch.int32,
            device=self.device,
        )
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
        return {
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
                DictCodec.coerce(state_dict["sparse_optimizer"], default=None),
            )

    @property
    def puzzle_embedding(self) -> SparsePuzzleEmbedding:
        """The injected prefix's local-gradient and master-weight buffers."""
        prefix = self.net.prefix
        assert isinstance(prefix, SparsePuzzleEmbedding)
        return prefix

    @override
    def _ingest(self, batch: Mapping[str, object]) -> tuple[Tensor, Tensor, Tensor]:
        act = self.act
        assert isinstance(act, ActPool)
        identifiers = batch["puzzle_identifiers"]
        assert isinstance(identifiers, Tensor)
        identifiers = identifiers.to(torch.int32).clone()
        valid = batch.get("valid_count", len(identifiers))
        assert isinstance(valid, int)
        identifiers[valid:] = 0
        self.puzzle_ids = torch.where(act.halted, identifiers, self.puzzle_ids)
        result = super()._ingest(batch)
        initial_slow, initial_fast = self.net.init_latents(act.config.batch_size)
        halted = act.halted.view(-1, 1, 1)
        act.z_slow = torch.where(halted, initial_slow, act.z_slow)
        act.z_fast = torch.where(halted, initial_fast, act.z_fast)
        return result

    @override
    def _loss(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        halt: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        act = self.act
        assert isinstance(act, ActPool)
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
        ).sum() / float(act.config.batch_size)
        halt_loss, metrics = act.halt_loss(
            logits,
            labels=labels,
            halt=halt,
            active=active,
            ignore_label_id=ignore,
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
        act = self.act
        assert isinstance(act, ActPool)
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
                halt_loss, metrics = act.halt_loss(
                    logits[:valid],
                    labels=target,
                    halt=halt[:valid],
                    active=active,
                    ignore_label_id=self.config.ignore_label_id,
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
        with self._autocast():
            output = self.net(media, *self._carry(), puzzle_identifiers=self.puzzle_ids)
        loss, metrics = self._loss(
            output.logits,
            labels=labels,
            halt=output.halt,
            active=active,
        )
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
        assert isinstance(self.act, ActPool)
        self.act.advance(
            output.z_slow,
            z_fast=output.z_fast,
            logits=output.logits,
            halt=output.halt,
            media=media,
        )
        return {
            "loss": loss.detach().reshape(1),
            "model": output.logits[:1, :1].detach(),
            "metrics": metrics,
        }
