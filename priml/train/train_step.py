"""TrainStep: Learnable with bundled loss for training convenience.

Extends Learnable (Forge-style wrapper) with loss bundling and training methods.
For simple supervised learning. Use Learnable directly for Forge-style flexibility.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Makes
from torch import Tensor

import torch
import torch.distributed as dist

from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss
from priml.runtime import global_device_mesh
from priml.train.custom_types import ModelOutput, TrainStepOutput
from priml.train.learnable import Learnable


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from configgle import Makeable


class TrainStep(Learnable):
    """Learnable with bundled loss for training convenience.

    Extends Learnable (Forge-style wrapper) with loss bundling and training loop helpers.
    Use Learnable directly for Forge-style flexibility without loss coupling.

    Example:
      cfg = TrainStep.Config(
          model=Mamba2DClassifier.Config(),
          loss=CrossEntropyLoss.Config(),
          parallelism=FullySharded.Config(mesh_dim="dp"),
          gradient_clip_norm=1.0,
          accumulate_grad_batches=4,
      )
      step = cfg.make()
      result = step.train_step(media=images, label=targets)
      grad_norm = step.last_grad_norm  # Track gradient norm

    """

    class Config(Makes["TrainStep"], Learnable.Config):
        """Configuration for TrainStep (adds loss to Learnable.Config)."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=SimpleLoss.Config,
        )
        accumulate_grad_batches: int = 1

        drop_partial_accumulation_on_epoch_end: bool = True
        """Discard a partial gradient accumulation at each epoch boundary.

        Default ``True``: when an epoch ends mid-accumulation the pending
        micro-batch gradients are zeroed and the counters reset, so gradients
        never mix across epochs. This matches recursion-style training (e.g.
        TRM) where epoch boundaries are semantically real. Set ``False`` for
        pure IID-shuffled data, where carrying the partial accumulation across
        the boundary is harmless and avoids wasting micro-batches.
        """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if config.accumulate_grad_batches <= 0:
            raise ValueError(
                f"accumulate_grad_batches must be positive, got {config.accumulate_grad_batches}",
            )

        self.accumulate_grad_batches = config.accumulate_grad_batches
        self.drop_partial_accumulation_on_epoch_end = (
            config.drop_partial_accumulation_on_epoch_end
        )
        self.accumulation_steps: int = 0
        self.accumulated_samples: int = 0
        self.last_microbatch_grads: list[Tensor] = []
        """Per-parameter gradients at the last optimizer step, after the
        grand-total division and before the optimizer zeroes them. Snapshotted
        for accumulation-correctness assertions; empty until the first step."""

        self.loss: Callable[..., LossOutput] = config.loss.make()

    def train_step(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Forward + loss + backprop with gradient accumulation.

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (autocast applied in Learnable.__call__). The output may be a
        # single Tensor or a multi-output container; the loss consumes it via
        # the ModelOutput contract rather than a blind ``cast(Tensor, ...)``.
        forward_output: Any = self(**preprocessed_batch)
        if not isinstance(forward_output, ModelOutput):
            raise TypeError(
                "Model forward output does not satisfy ModelOutput "
                f"(got {type(forward_output).__name__}); it must be a Tensor or "
                "an indexable multi-output container the loss can consume.",
            )
        output: ModelOutput = forward_output

        # Loss computation (inherits autocast from forward)
        loss_result = {**self.loss(output, **preprocessed_batch)}
        loss: Tensor = loss_result["loss"]

        # Validate loss is unreduced (per-element)
        if loss.ndim == 0:
            raise ValueError(
                "Loss must be unreduced (per-element), got scalar. "
                "Use reduction='none' in your loss function.",
            )

        # Accumulate the SUM of per-element losses. Backward is linear, so
        # summing per micro-batch and dividing once by the grand-total element
        # count reproduces a single big-batch gradient exactly across this
        # rank's micro-batches -- including UNEQUAL micro-batch sizes within
        # the rank (a mean-of-means would not). Across DATA-PARALLEL ranks the
        # exactness holds only when every rank accumulates the SAME total
        # element count: DDP/FSDP reduce the gradient as a MEAN over ranks, so
        # dividing each rank by its LOCAL count yields
        # ``mean_ranks(local_sum / local_N)`` rather than the true big batch
        # ``sum_ranks(local_sum) / sum_ranks(local_N)``; the two coincide iff
        # ``local_N`` is equal across ranks. ``_assert_uniform_microbatch_count``
        # enforces that precondition at the optimizer-step boundary.
        loss.sum().backward()
        self.accumulated_samples += loss.numel()
        self.accumulation_steps += 1

        # Update weights if accumulation complete.
        if self.accumulation_steps >= self.accumulate_grad_batches:
            # Cross-rank exactness precondition (see the backward comment):
            # every data-parallel rank must accumulate the same element count.
            _assert_uniform_microbatch_count(self.accumulated_samples)

            grad_snapshot: list[Tensor] = []
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.accumulated_samples)
                    grad_snapshot.append(param.grad.detach().clone())
            self.last_microbatch_grads = grad_snapshot

            self.step(lambda: self._recompute_loss(preprocessed_batch))
            self.accumulation_steps = 0
            self.accumulated_samples = 0

        loss_result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, loss_result)

    def _recompute_loss(self, preprocessed_batch: dict[str, Any]) -> Tensor:
        """Recompute the scalar training loss on ``preprocessed_batch``.

        The optimizer closure for closure-based optimizers (e.g. exact-Hessian
        Newton, which differentiates this via ``autograd.grad``). First-order
        optimizers never call it. Returns a graph-bearing scalar so the caller
        can take further derivatives.
        """
        output: ModelOutput = self(**preprocessed_batch)
        loss = {**self.loss(output, **preprocessed_batch)}
        return loss["loss"].sum()

    def on_epoch_end(self) -> None:
        """Flush a partial gradient accumulation at an epoch boundary.

        When ``drop_partial_accumulation_on_epoch_end`` is True (default) and
        an accumulation is pending, zero the accumulated micro-batch gradients
        and reset the counters so no gradient mixes across the boundary. When
        False, the partial accumulation is left intact to carry into the next
        epoch. A no-op when nothing is pending.
        """
        if (
            not self.drop_partial_accumulation_on_epoch_end
            or self.accumulation_steps == 0
        ):
            return
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_steps = 0
        self.accumulated_samples = 0

    def train_loss(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Compute loss in train mode (no backprop).

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (train mode + autocast via Learnable.__call__)
        output: ModelOutput = self(**preprocessed_batch)

        # Loss computation (inherits autocast)
        result = {**self.loss(output, **preprocessed_batch)}
        result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, result)

    def eval_loss(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Compute loss in eval mode (uses EMA if available).

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        # Forward (eval mode + autocast via Learnable.call_eval())
        output: ModelOutput = super().call_eval(**preprocessed_batch)

        # Loss computation (inherits autocast)
        result = {**self.loss(output, **preprocessed_batch)}
        result["model"] = cast(Tensor, output)
        return cast(TrainStepOutput, result)

    @override
    def call_eval(self, **preprocessed_batch: Any) -> Any:
        """Evaluation forward pass (uses EMA if available).

        Args:
          **preprocessed_batch: Preprocessed batch data as kwargs.

        """
        return cast(Tensor, super().call_eval(**preprocessed_batch))

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state including grad-accumulation counters.

        Recording the counters keeps a mid-accumulation checkpoint auditable.
        Restoring resets them (see ``load_state_dict``) because per-microbatch
        gradients cannot be persisted.
        """
        state = super().state_dict()
        state["accumulation_steps"] = self.accumulation_steps
        state["accumulated_samples"] = self.accumulated_samples
        return state

    @override
    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> None:
        """Load checkpoint (resets gradient accumulation since grads not saved).

        Forwards finetuning kwargs (``strict`` / ``load_optimizer`` / ``remap``)
        to ``Learnable.load_state_dict``.
        """
        super().load_state_dict(state_dict, **kwargs)
        # Reset accumulation - can't restore gradients from checkpoint
        self.accumulation_steps = 0
        self.accumulated_samples = 0

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch (move tensors to device, etc.)."""
        non_blocking = self.device.type == "cuda"
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }


def _assert_uniform_microbatch_count(accumulated_samples: int) -> None:
    """Verify every data-parallel rank accumulated the same element count.

    The grad-accumulation division reproduces a single big-batch gradient
    only when each rank's accumulated element count is equal (see the
    ``train_step`` backward comment for the derivation). Fires one collective
    per optimizer step, only under an initialized multi-rank process group.
    Uses the ``"dp"`` mesh group when a global device mesh exposes it; falls
    back to the default (WORLD) group otherwise -- correct when the world is a
    pure data-parallel group, and the documented assumption when no mesh is
    set.

    Args:
      accumulated_samples: This rank's accumulated per-element count for the
        completed accumulation window.

    Raises:
      ValueError: If ranks accumulated differing element counts, which would
        make the per-rank division diverge from the true big-batch gradient.

    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    mesh = global_device_mesh()
    group = (
        mesh.get_group("dp")
        if mesh is not None
        and mesh.mesh_dim_names is not None
        and "dp" in mesh.mesh_dim_names
        else None
    )
    if group is not None and dist.get_world_size(group) <= 1:
        return

    # all_reduce MIN and MAX of the local count; they differ iff some rank
    # accumulated a different number of elements.
    extremes = torch.tensor(
        [accumulated_samples, -accumulated_samples],
        dtype=torch.long,
    )
    dist.all_reduce(extremes, op=dist.ReduceOp.MAX, group=group)
    count_max = int(extremes[0].item())
    count_min = -int(extremes[1].item())
    if count_max != count_min:
        raise ValueError(
            "Gradient accumulation requires an equal per-element count across "
            "data-parallel ranks (DDP/FSDP reduce gradients as a MEAN over "
            "ranks, so dividing each rank by its local count only reproduces "
            "the single big-batch gradient when the counts match). Ranks "
            f"accumulated between {count_min} and {count_max} elements; "
            "equalize the per-rank micro-batch element counts.",
        )
