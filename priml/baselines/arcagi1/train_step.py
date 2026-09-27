"""Atomic adaptive-computation training for ARC grids."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

import math

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor
from torch._inductor import config
from torch.distributed.tensor import DTensor, Shard
from torch.nn.functional import binary_cross_entropy_with_logits

import torch

from priml.baselines.arcagi1.model import HPSURM
from priml.baselines.sudoku.embedding import PredictionFeedback
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.math.loss import stablemax_cross_entropy
from priml.math.schedules import warmup
from priml.optimizers import SignSGD
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding
from priml.runtime import global_device_mesh
from priml.train.custom_types import TrainStepOutput
from priml.train.ema import EMA, NoEMA
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.nn import Parameter


def _optimizer() -> CompositeOptimizer.Config:
    """Route the reasoning matrices to Muon and vectors/heads to AdamW."""
    eligible = excluding(Muon.eligible_tensor, "embed", "head")

    def matrices(name: str, parameter: Parameter) -> bool:
        return eligible(name, parameter) and parameter.ndim == 2

    def ensemble_matrices(name: str, parameter: Parameter) -> bool:
        return eligible(name, parameter) and parameter.ndim > 2

    config = CompositeOptimizer.Config()
    config.optimizers = [
        PartialConfig(torch.optim.AdamW, lr=1e-4, betas=(0.9, 0.95), weight_decay=1.0),
        Muon.Config(
            lr=5e-3,
            weight_decay=0.02,
            momentum=0.6,
            nesterov=True,
            ns_steps=3,
        ),
        Muon.Config(
            lr=5e-3,
            weight_decay=0.02,
            momentum=0.6,
            nesterov=True,
            ns_steps=3,
            ensemble_dims=1,
        ),
    ]
    config.select = [complement(eligible), matrices, ensemble_matrices]
    return config


class HPSFeedbackTrainStep(TrainStep):
    """One optimizer step over an atomic batch with persistent ACT slots."""

    class Config(Makes["HPSFeedbackTrainStep"], TrainStep.Config[HPSURM.Config]):
        """Model, batch, loss, corruption, and halting settings."""

        if TYPE_CHECKING:

            @override
            def make(self) -> HPSFeedbackTrainStep:
                """Build the configured HPS feedback train step."""
                ...

        model: HPSURM.Config = field(default_factory=HPSURM.Config)
        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=_optimizer,
        )
        batch_size: int = 96
        total_train_steps: int = 388_670
        warmup_steps: int = 2_000
        max_act_steps: int = 16
        halt_weight: float = 0.5
        halt_exploration_prob: float = 0.1
        halt_seed: int = 0
        feedback_corruption_rate: float = 0.075
        feedback_seed: int = 0
        ignore_label_id: int = 0
        sparse_embedding_lr: float = 1e-2
        sparse_embedding_weight_decay: float = 0.1
        use_ema: bool = True
        ema_decay: float = 0.999
        ema_warmup_steps: int = 2_000
        emulate_precision_casts: bool = True

        @override
        def finalize(self) -> Self:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if self.max_act_steps < 2:
                raise ValueError("max_act_steps must be at least 2")
            if (
                math.isnan(self.feedback_corruption_rate)
                or self.feedback_corruption_rate < 0
                or self.feedback_corruption_rate > 1
            ):
                raise ValueError("feedback_corruption_rate must be in [0, 1]")
            self.train_budget_steps = self.total_train_steps
            self.gradient_clip_norm = float("inf")
            self.lr_schedule = PartialConfig(warmup, fraction=1.0)
            self.ema = (
                EMA.Config(
                    decay=self.ema_decay,
                    update_after_step=self.ema_warmup_steps,
                    warmup_seed=True,
                    track_buffers=False,
                    shadow_kind="param_dict",
                )
                if self.use_ema
                else NoEMA.Config()
            )
            return super().finalize()

    def __init__(self, step_config: Config) -> None:
        with _precision_casts(step_config.emulate_precision_casts):
            super().__init__(step_config)
        self.config: HPSFeedbackTrainStep.Config = step_config
        model = self.model
        assert isinstance(model, HPSURM)
        batch = step_config.batch_size
        self._pool_inputs = torch.zeros(
            batch,
            model.config.grid_len,
            dtype=torch.long,
            device=self.device,
        )
        self._pool_labels = torch.zeros_like(self._pool_inputs)
        self._pool_ids = torch.zeros(batch, dtype=torch.long, device=self.device)
        self._pool_feedback = torch.zeros_like(self._pool_inputs)
        self._pool_z_slow, self._pool_z_fast = model.init_latents(batch)
        self._pool_halted = torch.ones(batch, dtype=torch.bool, device=self.device)
        self._pool_steps = torch.zeros(batch, dtype=torch.long, device=self.device)
        self._halt_rng = torch.Generator(device=self.device).manual_seed(
            step_config.halt_seed,
        )
        self._feedback_rng = torch.Generator(device=self.device).manual_seed(
            step_config.feedback_seed,
        )
        self._feedback = next(
            (
                layer
                for layer in model.modules()
                if isinstance(layer, PredictionFeedback)
            ),
            None,
        )
        self._sparse = next(
            (
                layer
                for layer in model.modules()
                if isinstance(layer, SparsePuzzleEmbedding)
            ),
            None,
        )
        sparse_params = (
            [self._sparse.weights, self._sparse.local_weights, self._sparse.local_ids]
            if self._sparse is not None
            else [torch.zeros(1, device=self.device)]
        )
        self._sparse_optimizer = SignSGD(
            [
                {
                    "params": sparse_params,
                    "sparse_embedding": self._sparse is not None,
                },
            ],
            lr=step_config.sparse_embedding_lr,
            weight_decay=step_config.sparse_embedding_weight_decay,
        )

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Linear warmup followed by the recipe's flat learning rate."""
        return float(min(1.0, self.global_step / max(1, self.config.warmup_steps)))

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Train one atomic ACT update while retaining unfinished slot state.

        Args:
          **batch: Input grids, labels, task ids, and valid row count.

        Returns:
          result: Loss, predictions, and training metrics.

        """
        media = batch["media"]
        labels = batch["label"]
        ids = batch["puzzle_identifiers"]
        assert isinstance(media, Tensor)
        assert isinstance(labels, Tensor)
        assert isinstance(ids, Tensor)
        valid_count = batch.get("valid_count", media.shape[0])
        assert isinstance(valid_count, int)
        self._refill(
            media,
            labels=labels,
            ids=ids,
            valid_count=valid_count,
        )
        model = cast(HPSURM, self.model)
        model.train()
        if self._feedback is not None:
            self._feedback.set_feedback(self._pool_feedback)
        autocast = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=False,
            )
            if self.config.dtype_autocast is not None
            else nullcontext()
        )
        with _precision_casts(self.config.emulate_precision_casts):
            with autocast:
                out = model(
                    self._pool_inputs,
                    self._pool_z_slow,
                    self._pool_z_fast,
                    puzzle_identifiers=self._pool_ids,
                )
                loss, lm_loss, q_loss, correct = self._loss(
                    out.logits,
                    halt=out.halt,
                    labels=self._pool_labels,
                )
            loss.backward()
        self._step_sparse_embedding()
        self.step()
        with torch.no_grad():
            self._pool_z_slow.copy_(out.z_slow)
            self._pool_z_fast.copy_(out.z_fast)
            prediction = out.logits.argmax(dim=-1)
            corruption = self._corrupt(prediction)
            self._pool_feedback.copy_(corruption)
            self._pool_steps.add_(1)
            at_max = self._pool_steps >= self.config.max_act_steps
            explore = (
                torch.rand(
                    self.config.batch_size,
                    device=self.device,
                    generator=self._halt_rng,
                )
                < self.config.halt_exploration_prob
            )
            min_steps = torch.randint(
                2,
                self.config.max_act_steps + 1,
                (self.config.batch_size,),
                device=self.device,
                generator=self._halt_rng,
            )
            min_done = self._pool_steps >= torch.where(explore, min_steps, 1)
            self._pool_halted.copy_(at_max | ((out.halt > 0) & min_done))
            halted_valid = self._pool_halted & (
                self._pool_labels != self.config.ignore_label_id
            ).any(dim=-1)
            halted_count = halted_valid.sum().clamp_min(1)
            exact = (correct & halted_valid).float().sum() / halted_count
        return cast(
            TrainStepOutput,
            {
                "loss": loss.detach().reshape(1),
                "model": out.logits[:1, :1].detach(),
                "metrics": {
                    "lm_loss": lm_loss.detach(),
                    "q_halt_loss": q_loss.detach(),
                    "exact_accuracy": exact,
                    "active_exact_accuracy": correct.float().mean(),
                },
            },
        )

    def _refill(
        self,
        media: Tensor,
        *,
        labels: Tensor,
        ids: Tensor,
        valid_count: int,
    ) -> None:
        """Replace halted slots and preserve every occupied slot's identity."""
        batch = self.config.batch_size
        if media.shape[0] != batch:
            raise ValueError(f"atomic ACT needs {batch} rows, got {media.shape[0]}")
        seat = self._pool_halted
        incoming_labels = labels.to(torch.long).clone()
        incoming_labels[valid_count:] = self.config.ignore_label_id
        self._pool_inputs = torch.where(
            seat[:, None],
            media.long(),
            self._pool_inputs,
        )
        self._pool_labels = torch.where(
            seat[:, None],
            incoming_labels,
            self._pool_labels,
        )
        incoming_ids = ids.long().clone()
        incoming_ids[valid_count:] = 0
        self._pool_ids = torch.where(seat, incoming_ids, self._pool_ids)
        self._pool_feedback = torch.where(
            seat[:, None],
            media.long(),
            self._pool_feedback,
        )
        model = cast(HPSURM, self.model)
        z_slow, z_fast = model.init_latents(batch)
        self._pool_z_slow = torch.where(
            seat[:, None, None],
            z_slow,
            self._pool_z_slow,
        )
        self._pool_z_fast = torch.where(
            seat[:, None, None],
            z_fast,
            self._pool_z_fast,
        )
        self._pool_steps = torch.where(
            seat,
            torch.zeros_like(self._pool_steps),
            self._pool_steps,
        )

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Run a clean 16-step feedback rollout under EMA weights.

        Args:
          **batch: Input grids, labels, task ids, and valid row count.

        Returns:
          result: Loss, predictions, and evaluation metrics.

        """
        media = batch["media"]
        labels = batch["label"]
        ids = batch["puzzle_identifiers"]
        assert isinstance(media, Tensor)
        assert isinstance(labels, Tensor)
        assert isinstance(ids, Tensor)
        valid_count = batch.get("valid_count", media.shape[0])
        assert isinstance(valid_count, int)
        if valid_count == 0:
            empty = torch.zeros((), device=self.device)
            return cast(
                TrainStepOutput,
                {
                    "loss": empty.reshape(1),
                    "model": torch.zeros(
                        media.shape[0],
                        1 + media.shape[-1],
                        device=self.device,
                    ),
                    "metrics": {
                        "lm_loss": empty,
                        "q_halt_loss": empty,
                        "exact_accuracy": empty,
                    },
                },
            )
        eval_labels = labels.long().clone()
        eval_labels[valid_count:] = self.config.ignore_label_id
        model = self.model
        assert isinstance(model, HPSURM)
        swap = (
            self.ema.apply_to(model)
            if self.global_step > self.config.ema_warmup_steps
            and not isinstance(self.ema, NoEMA)
            else nullcontext()
        )
        model.eval()
        context = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=False,
            )
            if self.config.dtype_autocast is not None
            else nullcontext()
        )
        with (
            swap,
            torch.inference_mode(),
            _precision_casts(self.config.emulate_precision_casts),
            context,
        ):
            z_slow, z_fast = model.init_latents(media.shape[0])
            feedback = media.long()
            out = None
            for _ in range(self.config.max_act_steps):
                if self._feedback is not None:
                    self._feedback.set_feedback(feedback)
                out = model(
                    media,
                    z_slow,
                    z_fast,
                    puzzle_identifiers=ids.long(),
                )
                z_slow, z_fast = out.z_slow, out.z_fast
                feedback = out.logits.argmax(dim=-1)
        if out is None:
            raise RuntimeError("evaluation rollout produced no model output")
        loss, lm_loss, q_loss, correct = self._loss(
            out.logits,
            halt=out.halt,
            labels=eval_labels,
            valid_count=valid_count,
        )
        packed = torch.cat(
            [out.halt[:, None].float(), out.logits.argmax(dim=-1).float()],
            dim=-1,
        )
        return cast(
            TrainStepOutput,
            {
                "loss": loss.detach().reshape(1),
                "model": packed,
                "metrics": {
                    "lm_loss": lm_loss.detach(),
                    "q_halt_loss": q_loss.detach(),
                    "exact_accuracy": (
                        correct[:valid_count].float().mean()
                        if valid_count
                        else torch.zeros((), device=self.device)
                    ),
                },
            },
        )

    def _loss(
        self,
        logits: Tensor,
        *,
        halt: Tensor,
        labels: Tensor,
        valid_count: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stablemax token loss and correctness supervision for the halt head."""
        token_loss = stablemax_cross_entropy(
            logits.double(),
            labels.long(),
            ignore_index=self.config.ignore_label_id,
        )
        valid = labels != self.config.ignore_label_id
        counts = valid.sum(dim=-1).clamp_min(1)
        per_sample = token_loss.sum(dim=-1) / counts
        active = valid.any(dim=-1)
        denominator = float(
            self.config.batch_size if valid_count is None else max(1, valid_count),
        )
        lm_loss = torch.as_tensor((per_sample * active).sum() / denominator)
        prediction = logits.argmax(dim=-1)
        correct = torch.as_tensor(
            ((prediction == labels) | ~valid).all(dim=-1) & valid.any(dim=-1),
        )
        per_halt_loss = binary_cross_entropy_with_logits(
            halt,
            correct.float(),
            reduction="none",
        )
        halt_loss = torch.as_tensor((per_halt_loss * active).sum() / denominator)
        total_loss = torch.as_tensor(
            lm_loss + self.config.halt_weight * halt_loss,
        )
        return (
            total_loss,
            lm_loss,
            halt_loss,
            correct,
        )

    def _corrupt(self, prediction: Tensor) -> Tensor:
        """Replace feedback cells using a dedicated, reproducible RNG."""
        mask = (
            torch.rand(
                prediction.shape,
                device=self.device,
                generator=self._feedback_rng,
            )
            < self.config.feedback_corruption_rate
        )
        colors = torch.randint(
            2,
            12,
            prediction.shape,
            device=self.device,
            generator=self._feedback_rng,
        )
        return torch.where(mask, colors, prediction)

    def _step_sparse_embedding(self) -> None:
        """Apply the constant-rate sparse table update before EMA handling."""
        if self._sparse is None or self._sparse.local_weights.grad is None:
            return
        self._sparse_optimizer.step_sparse_embedding(
            self._sparse.local_weights.grad,
            self._sparse.local_ids,
        )
        self._sparse.local_weights.grad = None

    @override
    def state_dict(self) -> TrainStep.StateDict:
        """Checkpoint in-flight ACT slots and their independent random streams."""
        mesh = global_device_mesh()
        dp_mesh = mesh["dp"] if mesh is not None else None
        state = {
            **super().state_dict(),
            "hps_dp_size": dp_mesh.size() if dp_mesh is not None else 1,
            "hps_pool": {
                "inputs": _rank_local_state(self._pool_inputs, dp_mesh, self.device),
                "labels": _rank_local_state(self._pool_labels, dp_mesh, self.device),
                "ids": _rank_local_state(self._pool_ids, dp_mesh, self.device),
                "feedback": _rank_local_state(
                    self._pool_feedback,
                    dp_mesh,
                    self.device,
                ),
                "z_slow": _rank_local_state(self._pool_z_slow, dp_mesh, self.device),
                "z_fast": _rank_local_state(self._pool_z_fast, dp_mesh, self.device),
                "halted": _rank_local_state(self._pool_halted, dp_mesh, self.device),
                "steps": _rank_local_state(self._pool_steps, dp_mesh, self.device),
            },
            "hps_rng": {
                "halt": _rank_local_state(
                    self._halt_rng.get_state(),
                    dp_mesh,
                    self.device,
                ),
                "feedback": _rank_local_state(
                    self._feedback_rng.get_state(),
                    dp_mesh,
                    self.device,
                ),
            },
        }
        return cast(TrainStep.StateDict, state)

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Restore shared state, in-flight ACT slots, and random streams."""
        state = dict(state_dict)
        saved_dp_size = state.pop("hps_dp_size", 1)
        pool_state = state.pop("hps_pool", None)
        random_state = state.pop("hps_rng", None)
        mesh = global_device_mesh()
        current_dp_size = mesh["dp"].size() if mesh is not None else 1
        if load_optimizer and saved_dp_size != current_dp_size:
            raise ValueError(
                f"ACT pool checkpoint DP size {saved_dp_size} differs from "
                f"current DP size {current_dp_size}",
            )
        super().load_state_dict(
            state,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        if load_optimizer and isinstance(pool_state, Mapping):
            pool = cast(Mapping[str, Tensor], pool_state)
            self._pool_inputs.copy_(_local_state(pool["inputs"]))
            self._pool_labels.copy_(_local_state(pool["labels"]))
            self._pool_ids.copy_(_local_state(pool["ids"]))
            self._pool_feedback.copy_(_local_state(pool["feedback"]))
            self._pool_z_slow.copy_(_local_state(pool["z_slow"]))
            self._pool_z_fast.copy_(_local_state(pool["z_fast"]))
            self._pool_halted.copy_(_local_state(pool["halted"]))
            self._pool_steps.copy_(_local_state(pool["steps"]))
        if isinstance(random_state, Mapping):
            halt = random_state.get("halt")
            feedback = random_state.get("feedback")
            if isinstance(halt, Tensor):
                self._halt_rng.set_state(_local_state(halt).cpu())
            if isinstance(feedback, Tensor):
                self._feedback_rng.set_state(_local_state(feedback).cpu())


def _rank_local_state(
    tensor: Tensor,
    mesh: DeviceMesh | None,
    device: torch.device,
) -> Tensor:
    """Preserve this rank's tensor in a distributed checkpoint."""
    local = tensor.clone()
    if mesh is None:
        return local
    return DTensor.from_local(local.to(device), mesh, [Shard(0)], run_check=False)


def _local_state(tensor: Tensor) -> Tensor:
    """Extract a checkpoint tensor's shard for this rank."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _precision_casts(enabled: bool) -> AbstractContextManager[None]:
    """Scope Inductor's dynamically typed precision-cast option."""
    patch = cast(Callable[[str, bool], AbstractContextManager[None]], config.patch)
    return patch("emulate_precision_casts", enabled)
