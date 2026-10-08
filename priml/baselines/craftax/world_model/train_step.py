"""Training step for the world model and the flat model of the validation ladder.

The model's forward returns a ``ModalityLoss``: the sum of modality means is
differentiated, so the 99 cells of a frame do not bury the action, reward, and
terminal. The optimizer is the GLM-4.5/Moonlight split, and the schedule
warms up, holds, and decays linearly to zero over the final tenth, as SmolLM3
does. Two injected functions keep the step model-agnostic: one scores every
validation target for the metric, one counts a micro-batch's FLOPs and
positions for ``train/mfu`` and ``train/tok_per_sec``.

Every optimizer update reports the series nanochat's ``base_train.py`` logs:
the smoothed ``loss``, the learning-rate multiplier ``lrm``, ``dt``,
``tok_per_sec``, ``mfu``, and the cumulative ``total_training_flops`` and
``total_training_time`` that comparisons plot against. Beside them are the
stability series a base run is judged by: ``grad_norm``, ``clip_fraction``
(the share of this process's updates that clipped), and the window's
``max_attention_logit`` over every global stack and every rank.
"""

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import field
from typing import Protocol, override

import dataclasses
import time

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch._dynamo.config import patch
from torch._dynamo.decorators import maybe_mark_dynamic

import torch
import torch.distributed as dist

from priml.baselines.craftax.world_model.batch import PackedBatch
from priml.baselines.craftax.world_model.loss import ModalityLoss
from priml.baselines.craftax.world_model.metric import (
    counted_nll,
    craftax_target_nll,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    GlobalTransformer,
    WorldModel,
)
from priml.lib.codec import from_plain
from priml.math import schedules
from priml.math.schedules import Schedule
from priml.model.attention.attention import Attention
from priml.optimizers import CompositeOptimizer, FusedAdamW, Muon
from priml.optimizers.muon import adjust_lr_match_rms_adamw
from priml.optimizers.parameter_filter import complement, excluding, matching
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ForwardCost:
    """What one forward pass over a micro-batch costs.

    Attributes:
      flops: Matmul FLOPs of the forward; attention scores are excluded, as the
        design's budget counts them.
      positions: Model positions processed, global plus local.

    """

    flops: float
    positions: int


class TargetNllFn(Protocol):
    """Scores every validation target of a batch, in the metric's layout."""

    def __call__(self, model: nn.Module, media: object, /) -> Tensor:
        """Return per-target NLL in nats."""
        ...


class CostFn(Protocol):
    """Counts the cost of one forward pass over a batch."""

    def __call__(self, model: nn.Module, media: object, /) -> ForwardCost:
        """Return the forward's cost."""
        ...


def wsd(progress: float, *, warmup: float, decay: float) -> float:
    """Warm up linearly over ``warmup``, hold, and decay to zero over ``decay``.

    Args:
      progress: Fraction of the budget spent.
      warmup: Share of the run ramping from zero.
      decay: Share of the run, at its end, decaying linearly to zero.

    Returns:
      multiplier: The rate's share of its peak.

    """
    return schedules.warmup(progress, fraction=warmup) * schedules.trapezoidal(
        progress,
        flat=1 - decay,
    )


def muon_adamw(*, adamw_only: Sequence[str] = ()) -> CompositeOptimizer.Config:
    """Muon on hidden matrices, AdamW on tables, heads, and scales.

    The GLM-4.5/Moonlight split. Muon's ``adjust_lr_match_rms_adamw`` scales its
    update RMS to ``0.2·√max(m, n)``, so both share the peak rate 3e-4, and Muon
    keeps its decoupled weight decay 0.1. Every attention's fused per-head
    projection ``proj_qkv`` is a ``[heads, head width, width]`` ensemble, so a
    second Muon member orthogonalizes each head's matrix on its own
    (``ensemble_dims=1``). AdamW takes every embedding table (the
    tied table, the action table, slot embeddings), the action head, and every
    vector: the start vector, the pooling token, the null memory, and norm
    scales.

    Args:
      adamw_only: Name fragments of further rank >= 2 weights for AdamW, whose
        update is not a map across channels, e.g. depthwise kernels.

    Returns:
      config: AdamW, Muon, and per-head Muon members with their selectors.

    """
    heads = matching("proj_qkv")
    matrices = excluding(
        Muon.eligible_tensor,
        "embedding",
        "table",
        "action_head",
        "proj_qkv",
        *adamw_only,
    )
    adamw = FusedAdamW.Config()
    adamw.lr = 3e-4
    adamw.betas = (0.9, 0.95)
    muon, muon_heads = Muon.Config(), Muon.Config()
    for member in (muon, muon_heads):
        member.lr = adamw.lr
        member.adjust_lr_fn = adjust_lr_match_rms_adamw
    muon_heads.ensemble_dims = 1
    config = CompositeOptimizer.Config()
    config.optimizers = [adamw, muon, muon_heads]
    config.select = [
        excluding(complement(matrices), "proj_qkv"),
        matrices,
        heads,
    ]
    return config


class WorldModelTrainStep(TrainStep):
    """Model plus optimization for a world-model or flat-model experiment.

    Takes the model, optimizer, schedule, clipping, and placement from
    :class:`~priml.train.train_step.TrainStep`. What stays here is the
    modality-mean objective with accumulation, the evaluation scorer, and the
    nanochat series.
    """

    class Config(Makes["WorldModelTrainStep"], TrainStep.Config, kw_only=True):
        """Model, optimizer, schedule, and how to score and cost a batch."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: Makeable[nn.Module] = field(default_factory=WorldModel.Config)
        """Network to train; its forward maps ``media`` to a ``ModalityLoss``."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=muon_adamw,
        )
        """Builds the optimizer from the model."""

        lr_schedule: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(wsd, warmup=0.1, decay=0.1),
        )
        """Warmup-stable-decay over ``train_budget_steps``."""

        gradient_clip_norm: float = 1.0
        """Global gradient-norm ceiling."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """BF16 activations over FP32 master weights."""

        # ---- This recipe's own. ----

        target_nll_fn: Makeable[TargetNllFn] = field(
            default_factory=lambda: PartialConfig(craftax_target_nll),
        )
        """Scores every validation target in the layout the metric reads."""

        cost_fn: Makeable[CostFn] = field(
            default_factory=lambda: PartialConfig(world_model_cost),
        )
        """Counts a micro-batch's forward FLOPs and positions."""

        device_peak_flops: float = 989e12
        """Dense BF16 peak of one device, the ``train/mfu`` denominator (H100)."""

        loss_smoothing: float = 0.9
        """EMA coefficient of ``train/loss``, bias-corrected as nanochat does."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: WorldModelTrainStep.Config = config
        self.target_nll: TargetNllFn = config.target_nll_fn.make()
        self.cost: CostFn = config.cost_fn.make()
        self.total_training_flops = 0.0
        """This process's FLOPs of every update so far times the world size,
        resumes included: every rank's when the ranks' frame and job counts
        agree, else an estimate. Three times each forward's ``cost``, so
        attention scores and recomputed activations are not counted."""
        self.total_training_time = 0.0
        """Seconds of every update after this process's first, resumes included."""
        self._loss_ema: Tensor | float = 0.0
        self._clipped_updates: Tensor | float = 0.0
        self._window_started = 0.0
        self._window_flops = 0.0
        self._window_positions = 0
        self._window_loss: Tensor | float = 0.0
        self._window_logits: list[Tensor] = []

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Forward, backward, and step the optimizer once the window is full.

        Args:
          **batch: ``media``, the model's input.

        Returns:
          result: ``loss`` and ``model`` (the objective), and the nanochat
            series when this call closed an accumulation window.

        """
        media = batch["media"]
        if self.accumulation_steps == 0:
            self._window_started = time.perf_counter()
            self._window_flops, self._window_positions = 0.0, 0
            self._window_loss, self._window_logits = 0.0, []
        if isinstance(media, PackedBatch):
            _mark_counts_dynamic(media)
        modality = self(media)
        assert isinstance(modality, ModalityLoss)
        (modality.loss / self.accumulate_grad_batches).backward()
        cost = self.cost(self.model, media)
        self._window_flops += 3 * cost.flops
        self._window_positions += cost.positions
        objective = modality.objective.detach()
        self._window_loss = self._window_loss + objective
        self._window_logits += [
            module.max_attention_logit()
            for module in self.model.modules()
            if isinstance(module, GlobalTransformer)
        ]
        self.accumulation_steps += 1
        result: TrainStepOutput = {
            "loss": modality.loss.detach().reshape(1),
            "model": objective.reshape(1),
        }
        if self.accumulation_steps == self.accumulate_grad_batches:
            self.step()
            self.accumulation_steps = 0
            result["metrics"] = self._update_metrics()
        return result

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Compute the training objective without a backward pass."""
        with torch.no_grad():
            modality = self(batch["media"])
        assert isinstance(modality, ModalityLoss)
        return {"loss": modality.loss.reshape(1), "model": modality.objective}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score every validation target of one micro-batch.

        priml's ``call_eval`` runs the model's forward, and the scorer reads
        ``logits`` and ``target_terms`` instead, so this enters the same
        context itself: eval mode, restored after, inference, the EMA weights,
        and autocast.

        Args:
          **batch: ``media``, and ``weight`` where the stream weighs positions,
            as ``data.ReplayStream`` does.

        Returns:
          result: ``model``, the per-target NLL the metric reads, and ``loss``,
            the NLL the metric counts (``metric.counted_nll``), or every
            target's without a ``weight``.

        """
        media, weight = batch["media"], batch.get("weight")
        training = self.model.training
        self.model.eval()
        with (
            self.timer_eval,
            torch.inference_mode(),
            self.ema.apply_to(self.model),
            self._autocast(),
        ):
            nll = self.target_nll(self.model, media)
            counted = nll.sum()
            if isinstance(weight, Tensor):
                assert isinstance(media, PackedBatch)
                counted = counted_nll(nll, media=media, weight=weight)
        self.model.train(training)
        return {"loss": counted.reshape(1), "model": nll}

    class StateDict(TrainStep.StateDict):
        """The base state plus the cumulative FLOPs and training time."""

        total_training_flops: float
        total_training_time: float

    @override
    def state_dict(self) -> StateDict:
        """Extend the base state with the cumulative FLOPs and training time."""
        return {
            **super().state_dict(),
            "total_training_flops": self.total_training_flops,
            "total_training_time": self.total_training_time,
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
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        self.total_training_flops = from_plain(
            state_dict["total_training_flops"],
            float,
        )
        self.total_training_time = from_plain(
            state_dict["total_training_time"],
            float,
        )

    # Synchronized first, so ``dt`` is the device's time for the whole window and not
    # the host's enqueue time; nanochat synchronizes at the same boundary.
    def _update_metrics(self) -> dict[str, float | Tensor]:
        """Return the nanochat series for the update just taken."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._window_started
        world = dist.get_world_size() if dist.is_initialized() else 1
        # nanochat's EMA: it starts at zero, so dividing by 1 - beta**n, n the
        # updates it has seen (this process's), removes the pull toward zero.
        beta = self.config.loss_smoothing
        loss = self._window_loss / self.accumulate_grad_batches
        self._loss_ema = beta * self._loss_ema + (1 - beta) * loss
        grad_norm = self._grad_norm()
        self._clipped_updates = (
            self._clipped_updates + (grad_norm > self.gradient_clip_norm).float()
        )
        if self.local_step > 1:
            self.total_training_time += dt
        self.total_training_flops += world * self._window_flops
        group = self.optimizer.param_groups[0]
        rate = self._window_flops / dt
        metrics: dict[str, float | Tensor] = {
            "loss": self._loss_ema / (1 - beta**self.local_step),
            "lrm": group["lr"] / group["initial_lr"],
            "dt": dt,
            # Tensors are averaged over ranks by the loop, so scaling each rank's
            # rate by the world size reports the global rate. The loop stacks them
            # with the loss, so they live on its device.
            "tok_per_sec": torch.tensor(
                world * self._window_positions / dt,
                device=self.device,
            ),
            "mfu": torch.tensor(
                rate / self.config.device_peak_flops,
                device=self.device,
            ),
            "grad_norm": grad_norm,
            "clip_fraction": self._clipped_updates / self.local_step,
            "total_training_flops": self.total_training_flops,
            "total_training_time": self.total_training_time,
        }
        if self._window_logits:
            largest = torch.stack(self._window_logits).amax()
            # The loop averages each tensor metric over ranks, which would dilute a
            # spike on one rank; after a MAX every rank holds the global maximum,
            # and the average of equal values is that value, to rounding.
            if dist.is_initialized():
                dist.all_reduce(largest, op=dist.ReduceOp.MAX)
            metrics["max_attention_logit"] = largest
        return metrics

    def _grad_norm(self) -> Tensor:
        """Return the norm the last update clipped."""
        if self.last_grad_norm is None:
            raise ValueError("Expected self.last_grad_norm is not None.")
        return self.last_grad_norm.detach()

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
            cache_enabled=self.config.autocast_cache_enabled,
        ):
            yield


def world_model_cost(model: nn.Module, media: object) -> ForwardCost:
    """Count the matmul FLOPs and positions of one world-model forward.

    Each module's matrices run once per position it processes: the encoder as
    its ``frame_macs`` counts a frame, the global stack and action head over
    every global position, and the local decoder and tied output table over
    every job's local slots, except the cross-attention's key-value projection,
    which runs over the job's memory slots. The frame embedding is a lookup
    whether it runs as a gather or as ``multi_hot_board``'s matmul, so the
    count, ``train/mfu`` and ``total_training_flops`` do not change with it.

    Args:
      model: A ``WorldModel``.
      media: The ``PackedBatch`` it runs on.

    Returns:
      cost: Forward FLOPs and global plus local positions.

    """
    assert isinstance(model, WorldModel)
    assert isinstance(media, PackedBatch)
    schema = model.schema
    frames, jobs, positions = len(media.aux), len(media.job_at), media.kind.numel()
    memory = _memory_matrices(model.decoder)
    local = _matrices(model.decoder) - memory + model.table.weight.numel()
    per_frame = model.encoder.frame_macs() + model.obs_proj.weight.numel()
    per_position = _matrices(model.transformer) + model.action_head.weight.numel()
    per_job = local * schema.local_slots + memory * schema.frame_slots
    per_job += model.cond_proj.weight.numel()
    return ForwardCost(
        flops=2.0 * (per_frame * frames + per_position * positions + per_job * jobs),
        positions=positions + jobs * schema.local_slots,
    )


def compile_forward(
    model: nn.Module,
    *,
    fullgraph: bool = True,
    backend: str | Callable[..., object] = "inductor",
) -> nn.Module:
    """Compile ``model``'s forward in place, leaving its call hooks eager.

    The compile slot for data parallel. ``torch.compile(model)`` traces
    ``Module.__call__``, hooks included, and priml's ``DataParallel`` installs
    composable ``replicate``'s: its pre-forward hook runs a lazy init that
    disables Dynamo, so under ``fullgraph=True`` every multi-GPU run failed at
    its first step. Here only ``forward`` is traced; the hooks
    run eagerly around the same graph, and a single-GPU run compiles what
    ``torch.compile(model)`` would. Use it as
    ``step.compile = PartialConfig(compile_forward, fullgraph=True)``.

    It also keeps the graph whole. Seeing replicate's reducer, Dynamo would
    split it at every gradient bucket to overlap the all-reduce with backward;
    the split subgraphs recompute the attention a local block was told to keep
    (``model.keep_attention``), and on 4 H200s the split
    update took 1.067 s against 1.015 s whole, its all-reduce overlapped but
    slowing the kernels beside it.

    Args:
      model: The placed module; its ``forward`` attribute is replaced.
      fullgraph: Refuse graph breaks, as the default compile slot does.
      backend: ``torch.compile``'s backend.

    Returns:
      model: ``model`` itself, whose calls now run the compiled forward.

    """
    compiled = torch.compile(model.forward, fullgraph=fullgraph, backend=backend)
    model.forward = patch(optimize_ddp=False)(compiled)
    return model


# Frame and job counts change with every micro-batch. Unmarked, the compiled
# model specializes on the first micro-batch's counts and recompiles for the
# second: 114 s to the first exp002 update with a warm cache, against 75 s
# marked (one H200).
def _mark_counts_dynamic(batch: PackedBatch) -> None:
    """Mark the frame and job axes of ``batch`` dynamic for ``torch.compile``."""
    for tensor in (
        batch.cells,
        batch.aux,
        batch.job_at,
        batch.job_memory,
        batch.job_next,
        batch.job_reward,
        batch.job_done,
        batch.job_is_start,
    ):
        maybe_mark_dynamic(tensor, 0)


def _matrices(module: nn.Module) -> int:
    """Count the matmul weights of ``module``."""
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if parameter.ndim >= 2 and "slot_embedding" not in name
    )


def _memory_matrices(decoder: nn.Module) -> int:
    """Count the cross-attention weights that project the memory: keys and values."""
    total = 0
    for block in decoder.modules():
        if isinstance(block, DecoderBlock):
            attention = block.cross_attn
            assert isinstance(attention, Attention)
            total += attention.proj_qkv.weight[attention.num_heads :].numel()
    return total
