"""Custom types for train module."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from priml.custom_types import CheckpointableProtocol


if TYPE_CHECKING:
    from torch import Tensor, nn

    import torch


__all__ = [
    "ActivationMemoizationProtocol",
    "CheckpointableProtocol",
    "CheckpointingProtocol",
    "CudaEventProtocol",
    "EMAProtocol",
    "LearnableProtocol",
    "LearningRateSchedulerProtocol",
    "LossFn",
    "ModelOutput",
    "ModelQuantizationProtocol",
    "ModuleLike",
    "OptimizerProtocol",
    "ParallelStrategyProtocol",
    "PhaseTimerProtocol",
    "ProfileProtocol",
    "TrackerProtocol",
    "TrainStepOutput",
    "TrainStepProtocol",
]


@runtime_checkable
class ModelOutput(Protocol):
    """A model forward result that a loss function can consume.

    The structural contract is intentionally narrow: the output must support
    indexing (``__getitem__``), which a single-output ``Tensor`` satisfies
    trivially and a multi-output container (dict of named tensors, tuple of
    tensors, or a structured output object) satisfies by construction. Values
    that carry no model prediction -- ``None``, scalars, bare callables -- do
    not conform, letting ``train_step`` raise a clear contract error instead of
    silently mis-casting the output to ``Tensor``.
    """

    def __getitem__(self, key: Any) -> Any:
        """Index into the model output (per-key tensor or per-position tensor)."""
        ...


class TrainStepOutput(TypedDict):
    """Return type for TrainStep methods.

    Contains loss for backpropagation, model output for metrics, and optional
    scalar metrics for logging.

    ``eval_extra_votes`` (eval only): additional ``(model_output, batch)`` pairs
    the eval harness feeds to each metric via extra ``update`` calls, on top of
    the primary ``model`` + the harness batch. Metrics that accumulate
    candidates (e.g. pass@K voting) thereby see extra votes -- used by WTA eval
    to add each of the K heads' predictions as extra pass@K candidates while the
    primary metrics stay on head 0. Each ``model_output`` and its paired
    ``batch`` rows must align (same row count) like the primary path.
    """

    loss: Tensor
    model: Tensor
    metrics: NotRequired[dict[str, float | Tensor]]
    eval_extra_votes: NotRequired[list[tuple[Tensor, dict[str, Any]]]]


if TYPE_CHECKING:
    LossFn = Callable[..., Tensor | TrainStepOutput]


@runtime_checkable
class OptimizerProtocol(CheckpointableProtocol, Protocol):
    """Protocol for optimizers."""

    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Perform single optimization step.

        Args:
          closure: Loss-recomputing closure. The training loop forwards it
            ONLY to optimizers exposing a truthy ``requires_closure`` attribute
            (closure-based ones such as exact-Hessian Newton); a torch
            first-order optimizer *executes* any closure it receives, running a
            wasteful second forward, so the closure is withheld from it.

        Returns:
          loss: Loss value from the closure, or None.

        """
        ...

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clear gradients.

        Args:
          set_to_none: Set gradients to None instead of zero.

        """
        ...


@runtime_checkable
class LearningRateSchedulerProtocol(CheckpointableProtocol, Protocol):
    """Protocol for LR schedulers."""

    def step(self) -> None:
        """Update learning rate."""
        ...

    def get_last_lr(self) -> list[float]:
        """Get last computed learning rate."""
        ...


@runtime_checkable
class ModuleLike(CheckpointableProtocol, Protocol):
    """Protocol for modules with parameters and buffers."""

    def parameters(self) -> Any:
        """Get model parameters."""
        ...

    def buffers(self) -> Any:
        """Get model buffers."""
        ...

    def to(self, device: torch.device | str | None = None) -> Any:
        """Move module to device."""
        ...

    def eval(self) -> Any:
        """Set module to eval mode."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set module to train mode."""
        ...

    def requires_grad_(self, requires_grad: bool = True) -> Any:
        """Set requires_grad for all parameters."""
        ...


@runtime_checkable
class EMAProtocol(CheckpointableProtocol, Protocol):
    """Protocol for exponential moving average.

    Attributes:
        shadow_model: Shadow model with EMA'd parameters for evaluation (None if not initialized).
        global_step: Total number of EMA updates applied.
        local_step: Number of EMA updates since last checkpoint.

    """

    shadow_model: nn.Module | None
    global_step: int
    local_step: int

    def __call__(self, model: Any) -> None:
        """Update EMA with current model state."""
        ...

    def apply_to(self, model: nn.Module) -> AbstractContextManager[None]:
        """Swap shadow weights into ``model`` for the context; restore on exit.

        The canonical eval path for every shadow kind: ``"module"`` and
        ``"param_dict"`` both swap in place, so callers never branch on the
        representation (the ``"param_dict"`` kind keeps ``shadow_model`` None).
        NoEMA yields with the live model untouched.

        Args:
          model: Live model whose parameters receive the shadow values.

        Returns:
          context: Context manager whose body sees shadow weights in ``model``.

        """
        ...


@runtime_checkable
class LearnableProtocol(CheckpointableProtocol, Protocol):
    """Protocol for learnable models (model + optimizer + scheduler + optional EMA).

    Provides training forward, evaluation forward, and optimization steps.
    """

    model: nn.Module
    global_step: int
    local_step: int
    last_grad_norm: Tensor | None
    ema: EMAProtocol
    optimizer: OptimizerProtocol
    learning_rate_scheduler: LearningRateSchedulerProtocol

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Training forward pass (sets train mode, applies autocast, optionally compiles)."""
        ...

    def call_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluation forward pass (uses EMA if available, applies inference_mode and autocast)."""
        ...

    def step(self) -> None:
        """Optimization step (clip grads, optimizer.step, scheduler.step, EMA)."""
        ...


class CheckpointingProtocol(Protocol):
    """The stepped checkpoint engine: save cadence, resume, overwrite-guard, retention.

    The training loop drives this per step against a *target* (the ``TrainLoop``,
    a ``CheckpointableProtocol``) passed to each call. The engine never parses
    the blob -- the target owns its state-dict structure. The default
    ``Checkpointer`` implements this.
    """

    def maybe_save(self, target: CheckpointableProtocol, step: int) -> bool:
        """Save ``target`` at ``step`` iff on the save cadence; return whether saved."""
        ...

    def save(self, target: CheckpointableProtocol, step: int) -> None:
        """Force-save ``target`` at ``step`` (end-of-run) unless it already exists."""
        ...

    def load(
        self,
        target: CheckpointableProtocol,
        *,
        max_steps: float,
        guard: bool = True,
    ) -> bool:
        """Resume ``target`` per config, then guard overwrites; return whether loaded.

        Resume selection and the overwrite guard are atomic over one inventory
        read. ``guard=False`` skips the guard (e.g. an eval-only run that writes
        nothing). ``max_steps`` bounds the guard's collision prediction.
        """
        ...

    def available_steps(self) -> list[int]:
        """Ascending steps of all complete checkpoints on disk (for diagnostics)."""
        ...

    def close(self) -> None:
        """Finish any pending async write and its retention. Call once at run end."""
        ...


class TrackerProtocol(Protocol):
    """Protocol for experiment tracking (TensorBoard, W&B, file, etc.).

    Handles logging metrics and images at a global step.
    """

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Log metrics at a given step.

        Args:
            metrics: Mapping of metric name -> value. ``prefix`` is prepended to
                each key. The mapping MAY contain non-scalar values (e.g. an
                ``extras`` key carrying a payload); each tracker consumes only
                what it understands and ignores the rest.
            step: Global step number.
            prefix: String prepended to every metric key before logging.

        """
        ...

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        """Log images at given step.

        Args:
            key: Media key.
            images: Image-like objects accepted by the tracker backend.
            step: Global step number.

        """
        ...

    def log_notes(self, notes: str) -> None:
        """Set free-text run notes (e.g. the W&B run overview).

        Called once after tracker creation with the experiment's description.
        Trackers that have no notes concept (file, TensorBoard) ignore it; a
        composite forwards it to each child. An explicitly-set note is not
        overwritten.

        Args:
            notes: Free-text description of the run.

        """
        ...

    def close(self) -> None:
        """Cleanup tracker resources."""
        ...


class ProfileProtocol(Protocol):
    """Protocol for profiling training performance.

    Handles torch profiler (CPU/CUDA) and memory profiling.
    """

    def on_step_start(self, step: int) -> None:
        """Called at the start of each training step.

        Args:
            step: Current global step number.

        """
        ...

    def on_step_end(self, step: int) -> None:
        """Called at the end of each training step.

        Args:
            step: Current global step number.

        """
        ...

    def cleanup(self) -> None:
        """Called at end of training to cleanup profiler resources."""
        ...


class CudaEventProtocol(Protocol):
    """Minimal ``torch.cuda.Event`` surface used for deferred GPU timing."""

    def record(self) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, end_event: CudaEventProtocol) -> float: ...


class PhaseTimerProtocol(Protocol):
    """The phase-timer surface the train loop and steps consume.

    Satisfied by :class:`~priml.train.profiling.PhaseTimer`. Lets
    ``TrainLoop.Config.phase_timer`` be a ``Makeable[PhaseTimerProtocol]`` --
    uniform with the other component fields (step/dataset/tracker/...) -- instead
    of pinning the concrete config type, so an alternative timer (or a test fake)
    can be substituted.
    """

    @property
    def cuda_events_enabled(self) -> bool: ...

    def phase(self, name: str) -> AbstractContextManager[None]: ...

    def record(self, name: str, elapsed: float) -> None: ...

    def record_cuda_events(
        self,
        name: str,
        start: CudaEventProtocol,
        end: CudaEventProtocol,
    ) -> None: ...

    def summary(self) -> dict[str, float]: ...

    def log_summary(self) -> None: ...


class ActivationMemoizationProtocol(Protocol):
    """Protocol for activation memory management strategies.

    Strategies control how activations are stored during forward pass
    to reduce memory usage during training. Includes:
    - Full precision storage (no optimization)
    - Recomputation-based checkpointing (delete and recompute)
    - Quantized storage (compress to FP8, upcast for gradients)
    """

    def __call__(self, model: nn.Module) -> None:
        """Apply activation memory strategy to module.

        Args:
            model: Module to apply strategy to.

        """
        ...


@runtime_checkable
class ParallelStrategyProtocol(Protocol):
    """Protocol for parallelism strategies (single-device, FSDP, DDP, HSDP).

    A strategy owns the full placement lifecycle for a module in ``__call__``:
    device assignment, shard application, meta -> real materialization, and
    post-shard ``reset_parameters``. ``device`` is the concrete device this
    rank places parameters on.
    """

    device: torch.device

    def __call__(self, model: nn.Module) -> nn.Module:
        """Place + shard + materialize the model; return the prepared module."""
        ...


class ModelQuantizationProtocol(Protocol):
    """Protocol for model quantization."""

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        ...


@runtime_checkable
class TrainStepProtocol(CheckpointableProtocol, Protocol):
    """Protocol for trainable models.

    Defines interface for models with training logic.
    """

    global_step: int
    local_step: int

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch (move tensors to device, etc.)."""
        ...

    def train_loss(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Compute loss in train mode (no backprop).

        Args:
          **preprocessed_batch: Batch data as kwargs.

        Returns:
          result: Dict with 'loss' and 'model' keys.

        """
        ...

    def eval_loss(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Compute loss in eval mode (no backprop).

        Args:
          **preprocessed_batch: Batch data as kwargs.

        Returns:
          result: Dict with 'loss' and 'model' keys.

        """
        ...

    def train_step(self, **preprocessed_batch: Any) -> TrainStepOutput:
        """Train mode + loss + backprop + optimizer step.

        Args:
          **preprocessed_batch: Batch data as kwargs.

        Returns:
          result: Dict with 'loss' and 'model' keys.

        """
        ...

    def call_eval(self, **preprocessed_batch: Any) -> Any:
        """Evaluation forward pass (uses EMA if available, applies inference_mode and autocast)."""
        ...

    def on_epoch_end(self) -> None:
        """Called by the loop at each epoch boundary.

        Steps with gradient accumulation use this to flush or discard a partial
        accumulation so gradients do not mix across epochs. Steps without
        accumulation lifecycle treat it as a no-op.
        """
        ...
