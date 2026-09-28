"""The reference TRM train step: pooled ACT, halt training, sparse task table.

Each call seats incoming puzzles in the pool, runs ONE forward over it, and
takes one optimizer step. Everything a recipe varies is an injected piece:
the seating policy and feedback carry (:mod:`~priml.baselines.arcagi1.act`),
the halt training and its exploration, the token loss and its reduction
(:mod:`~priml.baselines.arcagi1.loss`), the body optimizer, and the EMA.
The task table, when the model has one, trains through its own sparse SignSGD.

Evaluation runs every row to the step cap, carrying latents, under the EMA
weights once past warmup; :class:`EvalSignals` additionally packs the
label-free selection signals an offline reranker reads.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import field
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    NotRequired,
    Self,
    TypedDict,
    cast,
    override,
)

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch._inductor import config as inductor_config

import torch

from priml.baselines.arcagi1.act import AtomicPool, HaltTraining, TrmPool
from priml.baselines.arcagi1.loss import (
    CrossEntropyTokens,
    MeanOverActive,
    MeanOverBatch,
    Reduction,
    TokenLoss,
)
from priml.baselines.sudoku.model import SudokuNet
from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding
from priml.optimizers import AdamATan2, SignSGD, apply_lr_scale, lr_scale
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.train.ema import EMA
from priml.train.grad_clip import clip_grad_norm_, total_grad_norm
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.train.custom_types import OptimizerProtocol, TrainStepOutput


class _PrefixKwargs(TypedDict, total=False):
    puzzle_identifiers: Tensor


class EvalSignals:
    """Pack label-free selection signals ahead of the predicted grid.

    Columns ``[halt, logprob, stability, *preds]``; with ``per_step`` also
    ``converge_step, n_changes, *halt_step[K], *correct_step[K]`` before the
    grid, K the step cap. Column 0 stays the halt vote, so pass@K is unchanged.
    """

    class Config(Fig["EvalSignals"]):
        """Which signals to pack."""

        per_step: bool = False
        """Also pack the per-step halt and exact-match trajectory."""

        color_offset: int = 2
        """First token counted by the mean log-prob (the candidate's cells)."""

    def __init__(self, config: Config) -> None:
        self.per_step = config.per_step
        self.color_offset = config.color_offset

    def header_width(self, max_steps: int) -> int:
        """Return the columns packed ahead of the grid, halt vote included."""
        return 3 + (2 + 2 * max_steps if self.per_step else 0)

    def mean_token_logprob(self, logits: Tensor, tokens: Tensor) -> Tensor:
        """Return the ``[B]`` mean log-prob of ``tokens`` over their colored cells."""
        logp = logits.float().log_softmax(dim=-1)
        token_logp = logp.gather(-1, tokens.unsqueeze(-1).long()).squeeze(-1)
        colored = tokens >= self.color_offset
        counts = colored.sum(dim=-1).clamp(min=1)
        return (token_logp * colored).sum(dim=-1) / counts


class TrmTrainStep(TrainStep):
    """One pooled ACT step of the reference TRM recipe; see the module docstring."""

    class Config(
        Makes["TrmTrainStep"],
        TrainStep.Config[SudokuNet.Config],
        kw_only=True,
    ):
        """Pool, halting, loss, schedule, and the sparse table's optimizer."""

        model: SudokuNet.Config = field(default_factory=SudokuNet.Config)
        """Network to train."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: AdamATan2.Config(
                lr=1e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            ),
        )
        """Body optimizer; the task table is buffers, so it is never in it."""

        sparse_optimizer: SignSGD.Config = field(
            default_factory=lambda: SignSGD.Config(lr=1e-2, weight_decay=0.1),
        )
        """Updates the rows of a sparse task table; unused without one."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Must stay ``None``: the step calls ``net`` directly, never ``__call__``,
        so a whole-model compile would be ignored. ``model.compile_core``
        compiles the recurrent core instead."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype for forwards; ``None`` runs full precision."""

        pool: Makeable[TrmPool] = field(default_factory=AtomicPool.Config)
        """Slot seating and the optional feedback carry."""

        halting: HaltTraining.Config | None = field(
            default_factory=HaltTraining.Config,
        )
        """Halt-head training; ``None`` freezes the head and halts at the cap."""

        token_loss: Makeable[TokenLoss] = field(
            default_factory=CrossEntropyTokens.Config,
        )
        """Per-token loss."""

        reduction: Makeable[Reduction] = field(default_factory=MeanOverActive.Config)
        """How per-sample training losses become one number."""

        signals: EvalSignals.Config | None = None
        """Extra evaluation columns for offline reranking."""

        ignore_label_id: int = -100
        """Label excluded from the loss and from correctness."""

        total_train_steps: int = 36_000
        """Schedule horizon."""

        warmup_steps: int = 0
        """Linear warmup before the cosine decay."""

        lr_min_ratio: float = 0.0
        """Cosine floor as a fraction of each base rate; 1 holds it constant."""

        grad_clip_foreach: bool | None = True
        """Fused norm kernels; ``None`` or ``False`` on MPS, which has none."""

        emulate_precision_casts: bool = False
        """Keep the compiled core at evaluation, with Inductor emulating casts.

        Off, evaluation runs the model eagerly: a compiled bfloat16 autocast
        graph elides autocast's rounding casts and a long carried rollout
        compounds it. The flag is process-global in Inductor and must be set
        before the model is built, so the first trace captures it."""

        log_body_norms: bool = True
        """Report gradient and parameter norms."""

        norm_log_interval: int = 10
        """Steps between parameter-norm reports (and step 1)."""

        @override
        def finalize(self) -> Self:
            pool = self.pool
            if isinstance(pool, TrmPool.Config):
                pool.grid_len = self.model.grid_len
                pool.seq_len = self.model.total_seq_len
                pool.channels_hidden = self.model.channels_in
                pool.dtype = self.model.dtype
                if (
                    isinstance(self.reduction, MeanOverBatch.Config)
                    and self.reduction.batch_size == -1
                ):
                    self.reduction.batch_size = pool.batch_size
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.compile is not None:
            raise ValueError(
                "TrmTrainStep never calls the compiled model; set "
                "model.compile_core to compile the recurrent core instead.",
            )
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        super().__init__(config)
        self.config: TrmTrainStep.Config = config
        self.pool: TrmPool = config.pool.make()
        self.pool.to(self.device)
        self.halting = None if config.halting is None else config.halting.make()
        if self.halting is not None:
            self.halting.to(self.device)
        self.token_loss: TokenLoss = config.token_loss.make()
        self.reduction: Reduction = config.reduction.make()
        self.signals = None if config.signals is None else config.signals.make()
        table = self.puzzle_table
        self.sparse_optimizer = (
            None
            if table is None
            else config.sparse_optimizer.make()(
                [{"params": list(table.buffers()), "sparse_embedding": True}],
            )
        )

    @override
    def build_optimizer(self, model: nn.Module) -> OptimizerProtocol:
        """Freeze an untrained halt head first, so no optimizer claims it."""
        if self.config.halting is None:
            for parameter in cast(SudokuNet, model).halt_head.parameters():
                parameter.requires_grad_(False)
        return super().build_optimizer(model)

    @property
    def net(self) -> SudokuNet:
        """The model under its concrete type."""
        return cast(SudokuNet, self.model)

    @property
    def puzzle_table(self) -> SparsePuzzleEmbedding | None:
        """The sparse per-task table in the model's prefix, if any."""
        prefix = self.net.prefix
        parts = prefix.parts if isinstance(prefix, PrefixStack) else [prefix]
        for part in parts:
            if isinstance(part, SparsePuzzleEmbedding):
                return part
        return None

    @property
    def ema_shadow(self) -> dict[str, Tensor] | None:
        """Name-keyed EMA weights, or None when no EMA is configured."""
        return self.ema.shadow_params if isinstance(self.ema, EMA) else None

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Seat, forward, update, then carry the latents and decide halting.

        Args:
          **batch: ``media``, ``label``, and optionally ``valid_count`` and
            ``puzzle_identifiers``.

        Returns:
          result: ``loss``, a ``model`` probe slice, and scalar metrics.

        """
        media, labels = batch["media"], batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(labels, Tensor)
        valid = batch.get("valid_count", media.shape[0])
        assert isinstance(valid, int)
        ids = batch.get("puzzle_identifiers")
        if ids is not None and not isinstance(ids, Tensor):
            raise TypeError(f"puzzle_identifiers must be a Tensor; got {type(ids)}.")
        pool = self.pool
        active = pool.refill(
            self.net,
            media=media,
            labels=labels,
            valid_count=valid,
            puzzle_ids=ids,
            ignore_label_id=self.config.ignore_label_id,
        )
        self.model.train()
        if pool.carry is not None:
            pool.set_feedback(self.net, pool.feedback)
        with self._autocast():
            out = self.net(
                pool.inputs,
                pool.z_slow,
                pool.z_fast,
                collect_intermediates=False,
                **self._prefix_kwargs(pool.puzzle_ids),
            )
        loss, metrics, correctness = self._train_loss(out.logits, out.halt, active)
        if pool.carry is not None:
            with torch.no_grad():
                changed = pool.advance_feedback(pool.carry, out.logits)
            if changed is not None:
                metrics["feedback_corrupt_frac"] = changed
        loss.backward()
        self._update(metrics)
        pool.update_carry(z_slow=out.z_slow, z_fast=out.z_fast, active=active)
        pool.release(
            self.net,
            halt=pool.halt_mask(out.halt, halting=self.halting),
            active=active,
        )
        halted = pool.halted_this_step()
        if correctness is not None and halted is not None:
            # The reference definition: score only slots halting this step.
            scored = halted & active & (correctness.loss_counts > 0)
            metrics.update(
                _accuracy_metrics(
                    correctness,
                    halt=out.halt,
                    scored=scored,
                    steps=pool.steps,
                ),
            )
            metrics["halted_frac"] = scored.float().mean()
        return {
            "loss": loss.detach().unsqueeze(0),
            "model": out.logits[:1, :1].detach(),
            "metrics": metrics,
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Roll every row to the step cap and score the final prediction.

        Args:
          **batch: ``media``, ``label``, and optionally ``valid_count`` and
            ``puzzle_identifiers``.

        Returns:
          result: ``loss`` (token plus weighted halt term), the packed
            ``[halt, (signals), *preds]`` model output, and metrics.

        """
        media, labels = batch["media"], batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(labels, Tensor)
        valid = batch.get("valid_count", media.shape[0])
        assert isinstance(valid, int)
        rows = media.shape[0]
        if valid == 0:
            header = (
                1
                if self.signals is None
                else self.signals.header_width(self.pool.config.max_steps)
            )
            return _empty_eval(
                rows,
                header + self.net.config.grid_len,
                self.device,
                signals=self.signals is not None,
            )
        with self._eval_weights(), self._eval_compile():
            self.model.eval()
            with torch.inference_mode(), self._autocast():
                rollout = self._rollout(media, labels, batch.get("puzzle_identifiers"))
                logits, halt = rollout.logits, rollout.halt
                preds = logits.argmax(dim=-1)
                target = labels[:valid]
                counted = target != self.config.ignore_label_id
                cell_correct = (preds[:valid] == target) & counted
                loss_counts = counted.sum(dim=-1)
                solved = (cell_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)
                per_token = self.token_loss(
                    logits[:valid],
                    target,
                    ignore_index=self.config.ignore_label_id,
                )
                per_sample = per_token.sum(dim=-1) / loss_counts.clamp(min=1)
                every = torch.ones(valid, dtype=torch.bool, device=self.device)
                lm_loss = MeanOverActive(MeanOverActive.Config())(
                    per_sample,
                    active=every,
                )
                halt_v = halt[:valid]
                halt_loss = nn.functional.binary_cross_entropy_with_logits(
                    halt_v,
                    solved.to(halt_v.dtype),
                )
                per_sample_cell = cell_correct.float().sum(dim=-1) / loss_counts.clamp(
                    min=1,
                )
                metrics: dict[str, float | Tensor] = {
                    "lm_loss": lm_loss.detach(),
                    "q_halt_loss": halt_loss.detach(),
                    "cell_accuracy": per_sample_cell.mean().item(),
                    "cell_weighted_accuracy": cell_correct.sum().item()
                    / max(1, int(loss_counts.sum().item())),
                    "exact_accuracy": solved.float().mean().item(),
                    "q_halt_accuracy": ((halt_v >= 0) == solved).float().mean().item(),
                    "act_steps": float(self.pool.config.max_steps),
                    "q_continue_loss": 0.0,
                }
        total = lm_loss + self._halt_weight() * halt_loss
        columns = [halt.reshape(rows, 1).float()]
        if self.signals is not None:
            stability = (rollout.stable_run + 1.0) / self.pool.config.max_steps
            columns += [
                self.signals.mean_token_logprob(logits, preds).reshape(rows, 1).float(),
                stability.reshape(rows, 1).float(),
            ]
            if self.signals.per_step:
                columns += [
                    rollout.converge_step.reshape(rows, 1).float(),
                    rollout.n_changes.reshape(rows, 1).float(),
                    torch.stack(rollout.halt_steps, dim=1).float(),
                    torch.stack(rollout.correct_steps, dim=1).float(),
                ]
            metrics["grid_stable_frac"] = stability[:valid].mean().item()
        return {
            "loss": total.detach().reshape(1).float(),
            "model": torch.cat([*columns, preds.float()], dim=-1),
            "metrics": metrics,
        }

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Score a batch exactly as evaluation does."""
        return self.eval_loss(**batch)

    @override
    def call_eval(self, *args: object, **batch: object) -> Tensor:
        """Return the rollout's final logits under the live weights."""
        if args:
            raise ValueError("Expected no positional arguments.")
        media = batch["media"]
        assert isinstance(media, Tensor)
        self.model.eval()
        with self._eval_compile(), torch.inference_mode(), self._autocast():
            return self._rollout(media, None, batch.get("puzzle_identifiers")).logits

    @override
    def on_epoch_end(self) -> None:
        """Do nothing: this step accumulates nothing across a boundary."""

    class StateDict(TrainStep.StateDict):
        """Base state plus the dedicated generators and the sparse optimizer."""

        halt_rng: NotRequired[Tensor]
        corruption_rng: NotRequired[Tensor]
        sparse_optimizer: NotRequired[Mapping[str, object]]

    @override
    def state_dict(self) -> StateDict:
        """Extend the base state; in-flight pool slots are deliberately excluded."""
        state: TrmTrainStep.StateDict = {**super().state_dict()}
        if self.halting is not None:
            state["halt_rng"] = self.halting.generator.get_state()
        if self.pool.carry is not None:
            state["corruption_rng"] = self.pool.carry.generator.get_state()
        if self.sparse_optimizer is not None:
            state["sparse_optimizer"] = self.sparse_optimizer.state_dict()
        return state

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
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        state = cast(TrmTrainStep.StateDict, state_dict)
        # ``set_state`` accepts only a CPU byte tensor.
        if self.halting is not None and "halt_rng" in state:
            self.halting.generator.set_state(state["halt_rng"].cpu())
        if self.pool.carry is not None and "corruption_rng" in state:
            self.pool.carry.generator.set_state(state["corruption_rng"].cpu())
        if (
            load_optimizer
            and self.sparse_optimizer is not None
            and "sparse_optimizer" in state
        ):
            self.sparse_optimizer.load_state_dict(dict(state["sparse_optimizer"]))

    def _prefix_kwargs(self, identifiers: object) -> _PrefixKwargs:
        """Return ``identifiers`` as prefix arguments when the model has a table."""
        if self.puzzle_table is None:
            return {}
        if not isinstance(identifiers, Tensor):
            raise TypeError("A model with a task table needs puzzle_identifiers.")
        return {"puzzle_identifiers": identifiers}

    def _halt_weight(self) -> float:
        return 0.0 if self.halting is None else self.halting.weight

    def _train_loss(
        self,
        logits: Tensor,
        halt: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, float | Tensor], _Correctness | None]:
        """Token loss plus the halt term; returns the correctness it scored."""
        labels = self.pool.labels
        ignore = self.config.ignore_label_id
        per_token = self.token_loss(logits, labels, ignore_index=ignore)
        counted = (labels != ignore).sum(dim=-1).clamp(min=1)
        lm_loss = self.reduction(per_token.sum(dim=-1) / counted, active=active)
        metrics: dict[str, float | Tensor] = {
            "lm_loss": lm_loss.detach(),
            "q_continue_loss": 0.0,
        }
        if self.halting is None:
            return lm_loss, metrics, None
        with torch.no_grad():
            valid_mask = labels != ignore
            is_correct = (logits.argmax(dim=-1) == labels) & valid_mask
            loss_counts = valid_mask.sum(dim=-1)
            correct = (
                (is_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)
            ).float()
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            halt,
            correct,
            reduction="none",
        )
        per_sample = torch.where(active, per_sample, torch.zeros_like(per_sample))
        halt_loss = per_sample.sum() / active.sum().clamp(min=1)
        metrics["q_halt_loss"] = halt_loss.detach()
        correctness = _Correctness(is_correct, loss_counts, correct)
        # Every active slot, counting this step before the carry advances.
        scored = _accuracy_metrics(
            correctness,
            halt=halt,
            scored=active & (loss_counts > 0),
            steps=self.pool.steps + 1,
        )
        metrics.update(scored)
        metrics.update({f"active_{k}": v for k, v in scored.items()})
        loss = lm_loss + self.halting.weight * halt_loss
        return loss, metrics, correctness

    def _update(self, metrics: dict[str, float | Tensor]) -> None:
        """Clip, schedule, step both optimizers, zero, and advance the EMA."""
        config = self.config
        # The task table is buffers, so every trainable parameter is body.
        body = [p for p in self.model.parameters() if p.requires_grad]
        step = self.global_step + 1
        log_norms = config.log_body_norms and (
            step == 1 or step % config.norm_log_interval == 0
        )
        grad_norm: Tensor | None = None
        if math.isfinite(config.gradient_clip_norm):
            grad_norm = clip_grad_norm_(
                body,
                config.gradient_clip_norm,
                foreach=config.grad_clip_foreach,
            )
        elif log_norms:
            grad_norm = total_grad_norm(body, foreach=config.grad_clip_foreach)
        if config.log_body_norms and grad_norm is not None:
            metrics["grad_norm"] = grad_norm.detach()
        if log_norms:
            metrics["param_norm"] = (
                torch.stack([p.detach().norm(2.0) for p in body]).norm(2.0).detach()
            )
        with self.timer_step:
            scale = lr_scale(
                self.global_step,
                config.total_train_steps,
                config.warmup_steps,
                config.lr_min_ratio,
            )
            apply_lr_scale([self.optimizer], scale)
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            muon = _first_muon(self.optimizer)
            if muon is not None:
                metrics["muon_lr"] = muon.param_groups[0]["lr"]
            self.optimizer.step()
            table = self.puzzle_table
            if (
                self.sparse_optimizer is not None
                and table is not None
                and table.local_weights.grad is not None
            ):
                self.sparse_optimizer.step_sparse_embedding(
                    table.local_weights.grad.detach(),
                    table.local_ids.detach(),
                )
                table.local_weights.grad = None
            self.model.zero_grad(set_to_none=True)
            self.ema(self.model)

    def _rollout(
        self,
        media: Tensor,
        labels: Tensor | None,
        identifiers: object,
    ) -> _Rollout:
        """Run ``max_steps`` full forwards, carrying latents and feedback."""
        kwargs = self._prefix_kwargs(identifiers)
        rows = media.shape[0]
        z_slow, z_fast = self.net.init_latents(rows)
        feedback = media if self.pool.carry is not None else None
        rollout = _Rollout(rows=rows, device=self.device)
        # Per-step correctness needs labels; ``call_eval`` has none to give.
        trajectory = (
            labels if self.signals is not None and self.signals.per_step else None
        )
        for index in range(self.pool.config.max_steps):
            if feedback is not None:
                self.pool.set_feedback(self.net, feedback)
            out = self.net(
                media,
                z_slow,
                z_fast,
                collect_intermediates=False,
                **kwargs,
            )
            z_slow, z_fast = out.z_slow, out.z_fast
            rollout.record(
                out.logits,
                out.halt,
                index=index,
                trajectory_labels=trajectory,
                ignore_label_id=self.config.ignore_label_id,
            )
            if feedback is not None:
                feedback = self.pool.decode_feedback(out.logits, media=media)
        return rollout

    # Strictly past: the shadow is seeded by the train step AT the boundary,
    # which runs after that step's evaluation.
    def _eval_weights(self) -> AbstractContextManager[None]:
        ema = self.ema
        if isinstance(ema, EMA) and self.global_step > ema.update_after_step:
            return ema.apply_to(self.model)
        return nullcontext()

    def _eval_compile(self) -> AbstractContextManager[None]:
        return self.net.eager(enabled=not self.config.emulate_precision_casts)

    def _autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )


class _Rollout:
    """Final outputs and per-step signals of one evaluation rollout."""

    def __init__(self, *, rows: int, device: torch.device) -> None:
        self.logits = torch.empty(0, device=device)
        self.halt = torch.empty(0, device=device)
        self.previous: Tensor | None = None
        self.stable_run = torch.zeros(rows, device=device)
        self.n_changes = torch.zeros(rows, device=device)
        self.converge_step = torch.ones(rows, device=device)
        self.halt_steps: list[Tensor] = []
        self.correct_steps: list[Tensor] = []

    def record(
        self,
        logits: Tensor,
        halt: Tensor,
        *,
        index: int,
        trajectory_labels: Tensor | None,
        ignore_label_id: int,
    ) -> None:
        """Keep this step's outputs and update the stability counters.

        Args:
          logits: ``[B, grid_len, V]`` this step's predictions.
          halt: ``[B]`` halt logits.
          index: Zero-based step.
          trajectory_labels: Labels scoring the per-step trajectory; ``None``
            records stability only.
          ignore_label_id: Label that counts as correct anywhere.

        """
        self.logits, self.halt = logits, halt
        pred = logits.argmax(dim=-1)
        if self.previous is not None:
            same = (pred == self.previous).all(dim=-1)
            self.stable_run = torch.where(
                same,
                self.stable_run + 1.0,
                torch.zeros_like(self.stable_run),
            )
            if trajectory_labels is not None:
                self.n_changes = self.n_changes + (~same).float()
                self.converge_step = torch.where(
                    ~same,
                    float(index + 1),
                    self.converge_step,
                )
        self.previous = pred
        if trajectory_labels is not None:
            self.halt_steps.append(halt)
            self.correct_steps.append(
                (
                    (pred == trajectory_labels) | (trajectory_labels == ignore_label_id)
                ).all(dim=-1),
            )


class _Correctness(NamedTuple):
    """Per-cell and per-slot correctness the halt target was built from."""

    is_correct: Tensor
    """``[B, S]`` correct counted cells."""

    loss_counts: Tensor
    """``[B]`` counted cells per slot."""

    correct: Tensor
    """``[B]`` float 1 where every counted cell is correct."""


def _accuracy_metrics(
    correctness: _Correctness,
    *,
    halt: Tensor,
    scored: Tensor,
    steps: Tensor,
) -> dict[str, float | Tensor]:
    """Cell, exact, and halt accuracy plus depth over the ``scored`` slots."""
    is_correct, loss_counts, correct = correctness
    n = scored.sum().clamp(min=1)
    per_cell = (is_correct.float() / loss_counts.clamp_min(1).unsqueeze(-1)).sum(
        dim=-1,
    )
    return {
        "cell_accuracy": torch.where(scored, per_cell, torch.zeros_like(correct)).sum()
        / n,
        "exact_accuracy": (scored & correct.bool()).float().sum() / n,
        "q_halt_accuracy": (scored & ((halt >= 0) == correct.bool())).float().sum() / n,
        "act_steps": torch.where(scored, steps, torch.zeros_like(steps)).float().sum()
        / n,
    }


def _first_muon(optimizer: object) -> Muon | None:
    """Return a composite's first Muon member, whose rate the metrics report."""
    members = optimizer.optimizers if isinstance(optimizer, CompositeOptimizer) else []
    return next((m for m in members if isinstance(m, Muon)), None)


# Same packed width and metric keys as a scored batch: the metric infers the header
# from the column count and rejects any other, and every rank must report the same
# keys, so an all-padding tail is only distinguishable by its zero values.
def _empty_eval(
    rows: int,
    width: int,
    device: torch.device,
    *,
    signals: bool,
) -> TrainStepOutput:
    """Return the zero-filled contract for a batch with no valid rows."""
    metrics: dict[str, float | Tensor] = dict.fromkeys(
        (
            "lm_loss",
            "q_halt_loss",
            "cell_accuracy",
            "cell_weighted_accuracy",
            "exact_accuracy",
            "q_halt_accuracy",
            "act_steps",
            "q_continue_loss",
        ),
        0.0,
    )
    if signals:
        metrics["grid_stable_frac"] = 0.0
    return {
        "loss": torch.zeros(1, device=device),
        "model": torch.zeros(rows, width, device=device),
        "metrics": metrics,
    }
