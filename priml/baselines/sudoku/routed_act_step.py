"""The pooled ACT train step of the latent-recursion recipes, with a routing slot.

Each :meth:`RoutedActTrainStep.train_step` call (atomic ACT seating):

  1. Seats the incoming batch in the pool's halted slots; active slots keep
     their carried ``z_slow``/``z_fast`` and fed-back grid.
  2. Forwards the pool through the ``routing`` slot, which picks the latent
     chain the step trains and the pool carries: :class:`SingleChain` runs one
     multi-cycle forward; the hypothesis routings
     (:mod:`~priml.baselines.sudoku.hypotheses`) select one of several.
  3. Composes the loss, backpropagates, clips, applies the cosine rate, and
     steps the optimizers.
  4. Updates the EMA, the carried latents, and each slot's halt decision.

Order- and name-sensitive invariants (do not reorder):

* Loss composition: ``lm + halt_weight * halt_bce`` FIRST, then
  ``+ cardinality_weight * cardinality``.
* Step tail: backward -> clip (body params, foreach global norm) -> rate ->
  every optimizer member in order -> sparse puzzle-embedding SignSGD ->
  zero_grad -> EMA -> carry and halt update (halt-generator draws).
* Optimizer routing reads parameter NAMES (``embed``/``head`` stay off Muon),
  a contract with :mod:`~priml.baselines.sudoku.latent_recursion`.

arcagi's ``PooledActTrainStep`` runs the same pool on a different model
(``GridSolver``: attribute outputs, ``halt_head``, ``init_latents``), and this
baseline cannot import arcagi, which imports it; this step drives
:class:`~priml.baselines.sudoku.latent_recursion.LatentRecursion`, whose
flat reference keys every evaluation runner loads.

The ACT pool is not checkpointed: its slots are bound to the batches in flight,
and a resumed run continues with the next batch. Its dedicated generators are.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, NotRequired, Self, cast, override

import math

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch._inductor import config as inductor_config

import torch

from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.losses import CardinalityLoss, token_cross_entropy
from priml.baselines.sudoku.routing import Routing, SingleChain
from priml.optimizers import SignSGD, apply_lr_scale, learning_rate, lr_scale
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.train.ema import EMA
from priml.train.grad_clip import clip_grad_norm_, total_grad_norm
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from priml.train.custom_types import OptimizerProtocol, TrainStepOutput


class HaltRelease:
    """Release an evaluation row at the first ACT step its halt logit clears."""

    class Config(Fig["HaltRelease"]):
        """The release threshold."""

        threshold: float = 0.0
        """Halt-logit threshold; 0 is the trained halt decision boundary."""

    def __init__(self, config: Config) -> None:
        self.threshold = config.threshold

    def __call__(self, q_halt: Tensor) -> Tensor:
        """Return the ``[B]`` rows whose halt logit clears the threshold."""
        return q_halt >= self.threshold


class RoutedActTrainStep(TrainStep):
    """One pooled ACT step of the latent-recursion recipes; see the module docstring."""

    class Config(
        Makes["RoutedActTrainStep"],
        TrainStep.Config[LatentRecursion.Config],
        kw_only=True,
    ):
        """Pool, routing, losses, schedule, and evaluation release."""

        model: LatentRecursion.Config = field(default_factory=LatentRecursion.Config)
        """Network to train."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Must stay ``None``: the step calls the model directly, never through
        ``__call__``, so a whole-model compile would be ignored.
        ``model.compile`` compiles the recurrent core instead."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype for forwards; ``None`` runs full precision. Master
        weights and carried latents stay float32."""

        sparse_optimizer: SignSGD.Config = field(
            default_factory=lambda: SignSGD.Config(lr=1e-2, weight_decay=0.1),
        )
        """Updates the touched rows of the sparse puzzle table; unused without one."""

        pool: AtomicPool.Config = field(default_factory=AtomicPool.Config)
        """Slot seating, halt training, and the optional feedback carry."""

        routing: Makeable[Routing] = field(default_factory=SingleChain.Config)
        """Which latent chain each step trains and the pool carries."""

        token_loss: Makeable[Callable[..., Tensor]] = field(
            default_factory=lambda: PartialConfig(
                token_cross_entropy,
                label_smoothing=0.0,
            ),
        )
        """Per-token loss; called with ``ignore_index``."""

        cardinality: CardinalityLoss.Config | None = None
        """Row, column, and box cardinality term; ``None`` omits it."""

        release: HaltRelease.Config | None = None
        """Per-row early release in evaluation; ``None`` runs every row to the
        pool's step cap, the depth it trained at."""

        ignore_label_id: int = -100
        """Label excluded from the loss and from halt correctness."""

        total_train_steps: int = 19_500
        """Cosine horizon in optimizer steps."""

        warmup_steps: int = 0
        """Linear warmup steps before the cosine decay."""

        lr_min_ratio: float = 0.0
        """Cosine floor as a fraction of each base rate."""

        grad_clip_foreach: bool | None = True
        """Fused norm kernels; ``None`` or ``False`` on MPS, which has none."""

        emulate_precision_casts: bool = True
        """Keep evaluation compiled, with Inductor emulating autocast's casts.

        Off, evaluation runs the core eagerly: a compiled bfloat16 autocast
        graph elides autocast's rounding casts and the carried rollout
        compounds it. The flag is process-global in Inductor and must be set
        before the model is built, so the first trace captures it."""

        log_body_norms: bool = True
        """Report gradient and parameter norms."""

        norm_log_interval: int = 100
        """Steps between parameter-norm reports (and step 1)."""

        @override
        def finalize(self) -> Self:
            # The pool holds one latent state per slot, so it is laid out on the
            # model's shape rather than set beside it.
            self.pool.grid_len = self.model.num_puzzle_grid_tokens
            self.pool.seq_len = self.model.total_seq_len
            self.pool.channels_hidden = self.model.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.compile is not None:
            raise ValueError(
                "RoutedActTrainStep never calls the compiled model; set "
                "model.compile to compile the recurrent core instead.",
            )
        inductor_config.emulate_precision_casts = config.emulate_precision_casts
        # Built on the host, then placed: the recorded runs drew every initial
        # weight from the CPU generator, and the loop builds under its own
        # device, where the CUDA generator would draw different weights.
        with torch.device("cpu"):
            super().__init__(config)
        self.config: RoutedActTrainStep.Config = config
        self._body_params = [p for p in self.model.parameters() if p.requires_grad]
        self.pool: AtomicPool = config.pool.make()
        self.pool.to(self.device)
        self.routing: Routing = config.routing.make()
        self.token_loss: Callable[[Tensor, Tensor], Tensor] = partial(
            config.token_loss.make(),
            ignore_index=config.ignore_label_id,
        )
        self.cardinality: CardinalityLoss | None = (
            None if config.cardinality is None else config.cardinality.make()
        )
        if self.cardinality is not None:
            self.cardinality.to(self.device)
        self.release = None if config.release is None else config.release.make()
        table = self.net.puzzle_emb
        # The table is buffers, so the body optimizer never claims it.
        self.sparse_optimizer = (
            None
            if table is None
            else config.sparse_optimizer.make()([{"params": list(table.buffers())}])
        )

    @override
    def build_optimizer(self, model: nn.Module) -> OptimizerProtocol:
        """Freeze an untrained halt head first, so no optimizer claims it."""
        if self.config.pool.halting is None:
            for parameter in cast(LatentRecursion, model).q_head.parameters():
                parameter.requires_grad_(False)
        return super().build_optimizer(model)

    @property
    def net(self) -> LatentRecursion:
        """The model under its concrete type."""
        return cast(LatentRecursion, self.model)

    @property
    def ema_shadow(self) -> dict[str, Tensor] | None:
        """Name-keyed EMA weights, or None when no EMA is configured."""
        return self.ema.shadow_params if isinstance(self.ema, EMA) else None

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """One optimizer step over the ACT pool.

        Args:
          **batch: ``media`` and ``label`` ``[B, G]``, ``valid_count``, and
            ``puzzle_identifiers`` ``[B]``.

        Returns:
          out: ``loss`` [1], a tiny ``model`` probe slice, and ``metrics``.

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        raw_count = batch.get("valid_count", media.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        pid_raw = batch.get("puzzle_identifiers")
        pid = pid_raw if isinstance(pid_raw, Tensor) else None
        pool = self.pool
        active = pool.refill(
            self.net.init_z,
            media=media,
            labels=label,
            valid_count=valid_count,
            puzzle_ids=pid,
            ignore_label_id=self.config.ignore_label_id,
        )
        # Before the forward and outside autocast: seating state is drawn in
        # full precision, as the routed recipes were trained.
        self.routing.seat(self.net, pool, global_step=self.global_step)
        n_active = active.sum()
        fl = self._forward_and_loss(active=active, n_active=n_active)
        train_metrics = fl.train_metrics
        fl.loss.backward()
        self._update(train_metrics)
        pool.update_carry(z_slow=fl.z_slow_out, z_fast=fl.z_fast_out, active=active)
        pool.release(self.net.init_z, halt=pool.halt_mask(fl.q_halt), active=active)

        # Halted-only train metrics, the reference definition: score only slots
        # that HALT this step. These overwrite the primary names; the
        # all-active variants stay under ``active_*``.
        if fl.correctness is not None:
            is_correct, loss_counts, correct = fl.correctness
            with torch.no_grad():
                halted_valid = pool.halted & active & (loss_counts > 0)
                n_halted = halted_valid.sum().clamp(min=1)
                train_metrics.update(
                    {
                        "cell_accuracy": torch.where(
                            halted_valid,
                            (
                                is_correct.float()
                                / loss_counts.clamp_min(1).unsqueeze(-1)
                            ).sum(dim=-1),
                            torch.zeros_like(correct),
                        ).sum()
                        / n_halted,
                        "exact_accuracy": (halted_valid & correct.bool()).float().sum()
                        / n_halted,
                        "q_halt_accuracy": (
                            halted_valid & ((fl.q_halt >= 0) == correct.bool())
                        )
                        .float()
                        .sum()
                        / n_halted,
                        "act_steps": torch.where(
                            halted_valid,
                            pool.steps,
                            torch.zeros_like(pool.steps),
                        )
                        .float()
                        .sum()
                        / n_halted,
                        "halted_frac": halted_valid.float().mean(),
                    },
                )
        return {
            "loss": fl.loss.detach().unsqueeze(0),
            "model": fl.logits[:1, :1].detach(),
            "metrics": train_metrics,
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Full ACT evaluation of one batch: EMA swap, rollout, metric pack.

        Each ACT step runs the full ``slow_cycles`` forward via ``act_step``
        (the unit the model trains on), threading the decoded-grid feedback
        when the pool carries one. With ``release`` set, a row leaves at the
        first step whose halt logit clears the threshold; otherwise every row
        runs to the pool's step cap.

        Args:
          **batch: ``media`` and ``label`` ``[B, G]``, ``valid_count``, and
            ``puzzle_identifiers`` ``[B]``.

        Returns:
          out: ``loss`` [1]; ``model`` packed ``[B, 1 + G]``, column 0 the halt
            logit and the rest the predicted grid tokens; ``metrics`` scalars.

        """
        config = self.config
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        raw_count = batch.get("valid_count", media.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        batch_size = media.shape[0]
        if valid_count == 0:
            grid_len = config.model.num_puzzle_grid_tokens
            return {
                "loss": torch.zeros(1, device=self.device),
                "model": torch.zeros(batch_size, 1 + grid_len, device=self.device),
                "metrics": {
                    "lm_loss": 0.0,
                    "q_halt_loss": 0.0,
                    "cell_accuracy": 0.0,
                    "exact_accuracy": 0.0,
                    "q_halt_accuracy": 0.0,
                    "act_steps": 0.0,
                    "q_continue_loss": 0.0,
                },
            }

        step_kwargs: dict[str, Tensor] = {}
        if self.net.puzzle_emb is not None:
            puzzle_identifiers_value = batch["puzzle_identifiers"]
            assert isinstance(puzzle_identifiers_value, Tensor)
            step_kwargs["puzzle_identifiers"] = puzzle_identifiers_value
        self.model.eval()
        with (
            self._eval_weights(),
            self._eval_compile_disabled(),
            torch.inference_mode(),
            self._autocast(),
        ):
            logits, q_halt, act_used = self._eval_rollout(media, step_kwargs)

            logits_v = logits[:valid_count]
            labels_v = label[:valid_count]
            preds = logits.argmax(dim=-1)
            preds_v = preds[:valid_count]
            q_halt_v = q_halt[:valid_count]

            ignore = config.ignore_label_id
            valid_mask = labels_v != ignore
            cell_correct = (preds_v == labels_v) & valid_mask
            loss_counts = valid_mask.sum(dim=-1)
            sample_correct = (cell_correct.sum(dim=-1) == loss_counts) & (
                loss_counts > 0
            )
            active = torch.ones(logits_v.shape[0], dtype=torch.bool, device=self.device)
            lm_loss = self._compute_loss(
                logits=logits_v,
                labels=labels_v,
                active=active,
                n_active=active.sum(),
            )
            q_halt_loss = nn.functional.binary_cross_entropy_with_logits(
                q_halt_v,
                sample_correct.to(q_halt_v.dtype),
            )
            per_sample_cell = cell_correct.float().sum(dim=-1) / loss_counts.clamp(
                min=1,
            )
            cell_acc = per_sample_cell.mean().item()
            cell_weighted_acc = cell_correct.sum().item() / max(
                1,
                int(loss_counts.sum().item()),
            )
            exact_acc = sample_correct.float().mean().item()
            q_halt_acc = ((q_halt_v >= 0) == sample_correct).float().mean().item()
            act_used_v = act_used[:valid_count].float()
            depth_cap = self.pool.config.max_steps

        # eval/total_loss mirrors train/total_loss: lm + weighted halt loss.
        total_loss = lm_loss + self._halt_weight() * q_halt_loss
        model_output = torch.cat(
            [q_halt.reshape(batch_size, 1).float(), preds.float()],
            dim=-1,
        )
        metrics: dict[str, float | Tensor] = {
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "cell_accuracy": cell_acc,
            "cell_weighted_accuracy": cell_weighted_acc,
            "exact_accuracy": exact_acc,
            "q_halt_accuracy": q_halt_acc,
            "act_steps": act_used_v.mean().item(),
            "q_continue_loss": 0.0,
        }
        if self.release is not None:
            metrics["act_cap_frac"] = (act_used_v >= depth_cap).float().mean().item()
        return {
            "loss": total_loss.detach().reshape(1).float(),
            "model": model_output,
            "metrics": metrics,
        }

    class StateDict(TrainStep.StateDict):
        """Base state plus the pool's generators and the sparse optimizer."""

        halt_rng: NotRequired[Tensor]
        corruption_rng: NotRequired[Tensor]
        sparse_optimizer: NotRequired[Mapping[str, object]]

    @override
    def state_dict(self) -> StateDict:
        """Extend the base state; in-flight pool slots are deliberately excluded."""
        state: RoutedActTrainStep.StateDict = {**super().state_dict()}
        if self.pool.halting is not None:
            state["halt_rng"] = self.pool.halting.generator.get_state()
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
        state = cast(RoutedActTrainStep.StateDict, state_dict)
        # ``set_state`` accepts only a CPU byte tensor.
        if self.pool.halting is not None and "halt_rng" in state:
            self.pool.halting.generator.set_state(state["halt_rng"].cpu())
        if self.pool.carry is not None and "corruption_rng" in state:
            self.pool.carry.generator.set_state(state["corruption_rng"].cpu())
        if (
            load_optimizer
            and self.sparse_optimizer is not None
            and "sparse_optimizer" in state
        ):
            self.sparse_optimizer.load_state_dict(dict(state["sparse_optimizer"]))

    # Order: feedback stash -> forward -> token loss -> ``+ halt_weight * halt`` ->
    # ``+ cardinality_weight * cardinality`` -> feedback update (argmax and
    # corruption draws). The feedback update runs LAST.
    def _forward_and_loss(self, *, active: Tensor, n_active: Tensor) -> _ForwardLoss:
        """Forward the pool and compose the loss (the fixed-order contract)."""
        config = self.config
        self.model.train()
        pool = self.pool
        if pool.carry is not None:
            self.net.set_feedback(pool.feedback)
        with self._autocast():
            routed = self.routing(
                self.net,
                pool,
                token_loss=self.token_loss,
                ignore_index=config.ignore_label_id,
            )
        logits = routed.logits
        q_halt = routed.q_halt

        lm_loss = self._compute_loss(
            logits=logits,
            labels=pool.labels,
            active=active,
            n_active=n_active,
        )
        loss = lm_loss
        train_metrics: dict[str, float | Tensor] = {
            "lm_loss": lm_loss.detach(),
            "q_continue_loss": 0.0,
        }

        # Halt loss. Positions with the ignore label do not count toward
        # correctness (atomic tail padding).
        correctness: _Correctness | None = None
        halting = pool.halting
        if halting is not None:
            ignore = config.ignore_label_id
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid_mask = pool.labels != ignore
                is_correct = (preds == pool.labels) & valid_mask
                loss_counts = valid_mask.sum(dim=-1)
                correct = (
                    (is_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)
                ).float()
            q_loss = nn.functional.binary_cross_entropy_with_logits(
                q_halt,
                correct,
                reduction="none",
            )
            q_loss = torch.where(active, q_loss, torch.zeros_like(q_loss))
            q_halt_loss = q_loss.sum() / n_active.clamp(min=1)
            loss = loss + halting.weight * q_halt_loss
            active_valid = active & (loss_counts > 0)
            n_active_valid = active_valid.sum().clamp(min=1)
            # All-active metrics mix mid-ACT depths; they live under
            # ``active_*`` while the primary names are overwritten with
            # halted-only values in the step tail.
            active_metrics = {
                "cell_accuracy": torch.where(
                    active_valid,
                    (is_correct.float() / loss_counts.clamp_min(1).unsqueeze(-1)).sum(
                        dim=-1,
                    ),
                    torch.zeros_like(correct),
                ).sum()
                / n_active_valid,
                "exact_accuracy": (active_valid & correct.bool()).float().sum()
                / n_active_valid,
                "q_halt_accuracy": (active_valid & ((q_halt >= 0) == correct.bool()))
                .float()
                .sum()
                / n_active_valid,
                "act_steps": torch.where(
                    active_valid,
                    pool.steps + 1,
                    torch.zeros_like(pool.steps),
                )
                .float()
                .sum()
                / n_active_valid,
            }
            train_metrics["q_halt_loss"] = q_halt_loss.detach()
            train_metrics.update(active_metrics)
            train_metrics.update(
                {f"active_{k}": v for k, v in active_metrics.items()},
            )
            correctness = _Correctness(is_correct, loss_counts, correct)

        # Composition order contract: (lm + w_halt * halt) FIRST, then
        # + w_cardinality * cardinality.
        cardinality = self.cardinality
        if cardinality is not None:
            if self.global_step < cardinality.warmup_steps:
                train_metrics["csp_loss"] = 0.0
            else:
                csp_loss = cardinality(
                    logits,
                    pool.labels,
                    active=active,
                    ignore_index=config.ignore_label_id,
                )
                loss = loss + cardinality.weight * csp_loss
                train_metrics["csp_loss"] = csp_loss.detach()
        train_metrics.update(routed.metrics)

        # Advance the feedback recurrence AFTER the loss terms: the next
        # feedback is this forward's decoded grid (clues restored), optionally
        # corrupted for repair training.
        if pool.carry is not None:
            with torch.no_grad():
                pool.advance_feedback(logits)

        return _ForwardLoss(
            loss=loss,
            logits=logits,
            q_halt=q_halt,
            z_slow_out=routed.z_slow,
            z_fast_out=routed.z_fast,
            correctness=correctness,
            train_metrics=train_metrics,
        )

    def _compute_loss(
        self,
        *,
        logits: Tensor,
        labels: Tensor,
        active: Tensor,
        n_active: Tensor,
    ) -> Tensor:
        """Token loss -> per-sample mean over counted cells -> mean over active."""
        ignore = self.config.ignore_label_id
        loss_per_token = self.token_loss(logits, labels)
        nonpad = (labels != ignore).sum(dim=-1).clamp(min=1)
        loss_per_sample = loss_per_token.sum(dim=-1) / nonpad
        loss_per_sample = torch.where(
            active,
            loss_per_sample,
            torch.zeros_like(loss_per_sample),
        )
        return loss_per_sample.sum() / n_active.clamp(min=1)

    def _update(self, metrics: dict[str, float | Tensor]) -> None:
        """Clip, schedule, step every optimizer, zero, and advance the EMA."""
        config = self.config
        step = self.global_step + 1
        log_norms = config.log_body_norms and (
            step == 1 or step % config.norm_log_interval == 0
        )
        grad_norm: Tensor | None = None
        if math.isfinite(config.gradient_clip_norm):
            grad_norm = clip_grad_norm_(
                self._body_params,
                config.gradient_clip_norm,
                foreach=config.grad_clip_foreach,
            )
        elif log_norms:
            grad_norm = total_grad_norm(
                self._body_params,
                foreach=config.grad_clip_foreach,
            )
        if config.log_body_norms and grad_norm is not None:
            metrics["grad_norm"] = grad_norm.detach()
        if log_norms:
            metrics["param_norm"] = (
                torch.stack([p.detach().norm(2.0) for p in self._body_params]).norm(2.0)
            ).detach()
        with self.timer_step:
            scale = lr_scale(
                self.global_step,
                config.total_train_steps,
                config.warmup_steps,
                config.lr_min_ratio,
            )
            apply_lr_scale([self.optimizer], scale)
            metrics["lr"] = learning_rate(self.optimizer)
            muon = _first_muon(self.optimizer)
            if muon is not None:
                metrics["muon_lr"] = learning_rate(muon)
            self.optimizer.step()
            self._apply_sparse_puzzle_emb_step()
            self.model.zero_grad(set_to_none=True)
            self.ema(self.model)

    # No-op when the model has no puzzle table or no gradient reached its local
    # buffer this step.
    def _apply_sparse_puzzle_emb_step(self) -> None:
        """Apply SignSGD's sparse-row update for the puzzle table."""
        table = self.net.puzzle_emb
        if (
            self.sparse_optimizer is None
            or table is None
            or table.local_weights.grad is None
        ):
            return
        self.sparse_optimizer.step_sparse_embedding(
            table.local_weights.grad.detach(),
            table.local_ids.detach(),
        )
        table.local_weights.grad = None

    def _halt_weight(self) -> float:
        halting = self.pool.halting
        return 0.0 if halting is None else halting.weight

    def _eval_rollout(
        self,
        media: Tensor,
        step_kwargs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ACT rollout with optional per-row early release."""
        batch_size = media.shape[0]
        depth_cap = self.pool.config.max_steps
        halt_release = self.release

        rows = torch.arange(batch_size, device=self.device)
        boards = media
        kwargs = step_kwargs
        z_slow, z_fast = self.net.init_z(batch_size)
        carry = self.pool.carry
        feedback = boards if carry is not None else None
        final_logits: Tensor | None = None
        final_q_halt = torch.zeros(batch_size, device=self.device)
        act_used = torch.full(
            (batch_size,),
            depth_cap,
            dtype=torch.int64,
            device=self.device,
        )
        for step in range(1, depth_cap + 1):
            out = self.net.act_step(
                boards,
                z_slow,
                z_fast,
                feedback_ids=feedback,
                **kwargs,
            )
            z_slow = out["z_slow"]
            z_fast = out["z_fast"]
            if carry is not None:
                feedback = carry.decode(out["logits"], media=boards)
            if final_logits is None:
                final_logits = torch.empty(
                    (batch_size, *out["logits"].shape[1:]),
                    dtype=out["logits"].dtype,
                    device=self.device,
                )
            # Every surviving row's CURRENT outputs overwrite its finals, so
            # cap-reaching rows end with step-``depth_cap`` outputs exactly
            # like the fixed-depth evaluation; released rows stop being
            # overwritten.
            final_logits[rows] = out["logits"]
            final_q_halt[rows] = out["q_halt"].float()
            if halt_release is None or step == depth_cap:
                continue
            release = halt_release(out["q_halt"])
            if not bool(release.any()):
                continue
            act_used[rows[release]] = step
            keep = (~release).nonzero(as_tuple=True)[0]
            if not keep.shape[0]:
                break
            rows = rows[keep]
            boards = boards[keep]
            if feedback is not None:
                feedback = feedback[keep]
            z_slow = z_slow[keep]
            z_fast = z_fast[keep]
            kwargs = {key: value[keep] for key, value in kwargs.items()}
        if final_logits is None:
            raise ValueError("Expected final_logits is not None.")
        return final_logits, final_q_halt, act_used

    # Strictly past: the shadow is seeded by the train step AT the warmup boundary,
    # which runs after that boundary's evaluation, so an evaluation on the boundary
    # itself would score an unseeded shadow.
    def _eval_weights(self) -> AbstractContextManager[None]:
        ema = self.ema
        if isinstance(ema, EMA) and self.global_step > ema.update_after_step:
            return ema.apply_to(self.model)
        return nullcontext()

    # torch.compile with bfloat16 autocast corrupts the repeated-carry evaluation when
    # Inductor elides autocast's precision casts; ``emulate_precision_casts`` (set
    # before the model is built) is the opt-in that keeps evaluation compiled.
    @contextmanager
    def _eval_compile_disabled(self) -> Generator[None]:
        """Run evaluation eagerly unless precision-cast emulation is enabled."""
        if self.config.emulate_precision_casts:
            yield
            return
        model_config = self.net.config
        old = model_config.compile
        model_config.compile = False
        try:
            yield
        finally:
            model_config.compile = old

    def _autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
            cache_enabled=False,
        )


def _first_muon(optimizer: object) -> Muon | None:
    """Return a composite's first Muon member, whose rate the metrics report."""
    members = optimizer.optimizers if isinstance(optimizer, CompositeOptimizer) else []
    return next((m for m in members if isinstance(m, Muon)), None)


@dataclass(slots=True, kw_only=True)
class _ForwardLoss:
    """Result of the per-step forward and loss, the single train-step seam.

    Attributes:
      loss: Scalar loss to backpropagate (token, halt, and cardinality terms).
      logits: ``[B, G, V]`` logits for metrics and the probe slice.
      q_halt: ``[B]`` halt logit driving the halting decision.
      z_slow_out: ``[B, S, C]`` updated slow latent for the carry update.
      z_fast_out: ``[B, S, C]`` updated fast latent for the carry update.
      correctness: ``(is_correct, loss_counts, correct)`` when the halt head
        trains, else None; reused for the halted-only metrics.
      train_metrics: Scalar metrics accumulated this step.

    """

    loss: Tensor
    logits: Tensor
    q_halt: Tensor
    z_slow_out: Tensor
    z_fast_out: Tensor
    correctness: _Correctness | None
    train_metrics: dict[str, float | Tensor]


class _Correctness(NamedTuple):
    """Per-cell and per-slot correctness the halt target was built from."""

    is_correct: Tensor
    """``[B, G]`` correct counted cells."""

    loss_counts: Tensor
    """``[B]`` counted cells per slot."""

    correct: Tensor
    """``[B]`` float 1 where every counted cell is correct."""
