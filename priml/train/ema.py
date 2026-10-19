"""Exponential Moving Average (EMA) for model parameters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast

import copy
import math

from configgle import Fig
from torch import Tensor

import torch

from priml.optimizers.parameter_filter import ParameterFilter, trainable


if TYPE_CHECKING:
    from torch import nn


class NoEMA:
    """No-op EMA for when EMA is disabled.

    Provides same interface as EMA but does nothing.
    """

    class Config(Fig["NoEMA"]):
        """NoEMA configuration (empty)."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize NoEMA."""
        del config
        self.shadow_model: nn.Module | None = None
        self.global_step = 0
        self.local_step = 0

    def __call__(self, model: nn.Module) -> None:
        """No-op update."""

    @contextmanager
    def apply_to(self, model: nn.Module) -> Generator[None]:
        """No-op swap: yields with the live model untouched.

        Args:
          model: PyTorch module (unused in no-op).

        Yields:
          context: Block in which ``model`` carries its live weights.

        """
        del model
        yield

    class StateDict(TypedDict):
        """Step counters; a no-op EMA has no shadow to persist."""

        global_step: int
        local_step: NotRequired[int]

    def state_dict(self) -> StateDict:
        """Get the step counters.

        Returns:
          state: The global and local step counters.

        """
        return {"global_step": self.global_step, "local_step": self.local_step}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load step counters only.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(NoEMA.StateDict, state_dict)
        self.global_step = state["global_step"]
        self.local_step = state.get("local_step", 0)


type DecaySchedule = Callable[[float, int], float]
"""Maps ``(decay, post_warmup_step)`` to the decay applied at that step.

A function rather than a named schedule: the two shipped curves are one line
each, and a recipe wanting a third writes it instead of editing this module.
"""

type EMAUpdateFn = Callable[[Tensor, Tensor, float], None]
"""Updates a shadow tensor from its live counterpart and effective decay."""


type BufferUpdateFn = Callable[[Mapping[str, Tensor], nn.Module, float], None]
"""Updates named shadow buffers from the live module and effective decay."""


def copy_buffers(shadow: Mapping[str, Tensor], model: nn.Module, decay: float) -> None:
    """Copy every matching named buffer; averaging decay does not apply.

    Args:
      shadow: Name-keyed buffers owned by the EMA shadow.
      model: Live model supplying the buffers.
      decay: Effective parameter decay; unused by this copy policy.

    """
    del decay
    live = dict(model.named_buffers())
    for name, buffer in shadow.items():
        if name in live:
            buffer.copy_(live[name].detach())


def average_floating_state(shadow: Tensor, live: Tensor, decay: float) -> None:
    """Average real floating state; copy integer and complex state verbatim.

    Args:
      shadow: Shadow tensor, mutated in the caller's no-grad context.
      live: Corresponding current state tensor.
      decay: Fraction of the previous floating state retained.

    """
    if shadow.is_floating_point():
        shadow.mul_(decay).add_(live, alpha=1 - decay)
    else:
        shadow.copy_(live)


def average_persistent_buffers(
    shadow: Mapping[str, Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    """Average persistent floating buffers while preserving transient snapshots.

    Args:
      shadow: Name-keyed buffers owned by the EMA shadow.
      model: Live model whose state dict defines persistence.
      decay: Fraction of the previous floating buffer retained.

    """
    live = model.state_dict()
    for name, buffer in shadow.items():
        if name in live:
            average_floating_state(buffer, live[name].detach(), decay)


def constant_decay(decay: float, step: int) -> float:
    """Hold ``decay`` flat for the whole run.

    Args:
      decay: Decay.
      step: Step.

    Returns:
      decay: The float.

    """
    del step
    return decay


def karras_decay(decay: float, step: int) -> float:
    """Grow the effective decay with the step, capped at ``decay``.

    Early averaging is faster, which keeps the shadow from being dominated by
    the first few steps' weights.

    Args:
      decay: Target decay rate (upper bound).
      step: Post-warmup step count (0-indexed).

    Returns:
      result: Effective decay at this step, ramping from 0 to decay.

    References:
      https://arxiv.org/abs/2312.02696
        Karras et al. 2023, "Analyzing and Improving the Training Dynamics of
        Diffusion Models."

    """
    return min(decay, (1 + step) / (10 + step))


class EMA:
    """Exponential Moving Average of model parameters.

    A single class covering two shadow representations and two access
    patterns.

    Shadow representations (``shadow_kind``):

    - ``"module"`` (default): a full clone of the source module with
      EMA-averaged parameters, exposed as ``ema.shadow_model``. Supports
      the parallel-forward path ``ema.shadow_model(x)``. NOT survivable
      under ``fully_shard`` -- ``copy.deepcopy`` of a sharded module
      breaks -- so use ``"param_dict"`` for DTensor/FSDP models.
    - ``"param_dict"``: a name-keyed dict of averaged tensors, exposed as
      ``ema.shadow_params``. Each entry is a clone of the live param's
      LOCAL shard (``param.detach().clone()``), never a module deepcopy,
      so DTensor/FSDP-sharded models survive. No ``shadow_model`` is built;
      evaluate via the ``apply_to`` swap.

    Access patterns:

    - **Parallel-forward** via ``ema.shadow_model`` (``"module"`` only):
      run the shadow module directly.
    - **In-place swap** via ``with ema.apply_to(model): ...``: temporarily
      swap shadow params into the live model for the context body so
      autocast / profiler / ``torch.compile`` hooks all work without
      re-wiring. Pre-allocated backup buffers keep eval-time memory O(1).
      Works for both shadow kinds.

    Configuration:

    - ``track_buffers`` (default True): also copy non-parameter buffers
      (BatchNorm running stats etc.) into the shadow during ``__call__``.
      Set False when buffers are derived (RoPE positions) or too big to
      shadow (sparse puzzle embeddings) -- buffers stay at live-model
      values. Note the asymmetry: ``track_buffers`` only affects the
      shadow updated in ``__call__``. The in-place ``apply_to`` swap leaves
      live-model buffers in place unless ``buffer_update_fn`` averages them.
    - ``select``: a :class:`~priml.optimizers.parameter_filter.ParameterFilter`
      choosing which params the shadow averages -- ``trainable`` (default)
      or ``everything``, which also takes frozen ones.
    - ``decay`` / ``update_after_step`` / ``update_every``: the usual
      lerp schedule controls.
    - ``warmup_seed``: copy live weights into the shadow at the warmup
      boundary (the first post-warmup call) before averaging begins.
    - ``decay_schedule``: a ``(decay, step) -> decay`` function.
      :func:`constant_decay` (default) ignores the step;
      :func:`karras_decay` ramps it in so early averaging is faster.

    Example::

        ema = EMA.Config(decay=0.999).make()
        for batch in loader:
            ...
            ema(model)                          # after optimizer.step
        with torch.no_grad():
            out = ema.shadow_model(x)           # parallel-forward path
        with ema.apply_to(model):               # in-place swap path
            metrics = compute_eval(model, x)
    """

    class Config(Fig["EMA"]):
        """EMA configuration."""

        decay: float = 0.9999
        """Exponential decay rate for shadow parameters."""

        update_after_step: int = 0
        """Warmup length: the first ``update_after_step`` calls only advance
        the step counter (no lerp). The first actual update happens AT
        ``global_step == update_after_step`` (i.e. on call index
        ``update_after_step``, 0-based), not after it."""

        update_every: int = 1
        """Update shadow parameters on calls where
        ``(global_step - update_after_step) % update_every == 0`` (so every
        Nth call past warmup, starting at the warmup boundary)."""

        track_buffers: bool = True
        """If True, copy non-parameter buffers verbatim each update.
        False (TRM-style) skips buffer copy entirely -- useful when
        buffers are derived/sentinel or too big to shadow."""

        shadow_kind: Literal["module", "param_dict"] = "module"
        """Shadow representation: ``"module"`` clones the source module
        (parallel-forward path); ``"param_dict"`` keeps a name-keyed dict of
        local-shard clones (FSDP/DTensor-survivable, swap-only eval)."""

        warmup_seed: bool = False
        """If True, copy live weights into the shadow at the warmup boundary
        before averaging begins (TRM-style). Default leaves the shadow at
        its first-call snapshot."""

        decay_schedule: DecaySchedule = constant_decay
        """Maps ``(decay, post-warmup step)`` to the decay used at that step.

        :func:`karras_decay` ramps it in; any ``(float, int) -> float`` works."""

        select: ParameterFilter = trainable
        """Chooses which parameters the shadow averages.

        A frozen tensor still drifts under the lerp, since
        ``p * d + p * (1 - d)`` rounds, so ``everything`` is not the default."""

        update_fn: EMAUpdateFn | None = None
        """Optional shadow-update expression. ``None`` retains the default
        in-place multiply/add operation order."""

        buffer_update_fn: BufferUpdateFn = copy_buffers
        """Buffer policy; default copies all named buffers. Use
        ``average_persistent_buffers`` to average persistent floating state."""

    def __init__(self, config: Config) -> None:
        """Initialize EMA.

        Args:
          config: EMA configuration.

        """
        if not math.isfinite(config.decay) or config.decay < 0 or config.decay > 1:
            raise ValueError(
                f"decay must be in [0, 1], got {config.decay}",
            )
        if config.update_every < 1:
            raise ValueError(
                f"update_every must be >= 1, got {config.update_every}",
            )
        if config.update_after_step < 0:
            raise ValueError(
                f"update_after_step must be >= 0, got {config.update_after_step}",
            )

        self.decay = config.decay
        self.update_after_step = config.update_after_step
        self.update_every = config.update_every
        self.track_buffers = config.track_buffers
        self.shadow_kind = config.shadow_kind
        self.warmup_seed = config.warmup_seed
        self.decay_schedule = config.decay_schedule
        self.select = config.select
        self.update_fn = config.update_fn
        self.buffer_update_fn = config.buffer_update_fn

        self.shadow_model: nn.Module | None = None
        self.shadow_params: dict[str, Tensor] = {}
        self._module_params: dict[str, nn.Parameter] = {}
        self._shadow_buffers: dict[str, Tensor] = {}
        self.global_step = 0
        self.local_step = 0
        self._pending_state: _StateDict | None = None
        self._tracked_names: set[str] = set()
        self._backup: dict[str, Tensor] = {}
        self._initialized = False
        # True once a param_dict shadow has been adopted from load_state_dict
        # before lazy init, so lazy init must not re-clone over it.
        self._loaded_shadow = False

    def effective_decay(self, step: int) -> float:
        """Return the decay applied at post-warmup step ``step`` (0-based).

        Args:
          step: Post-warmup averaging-step index (``local_step``).

        Returns:
          decay: Whatever ``decay_schedule`` returns at ``step``, bounded to
            ``[0, 1]``.

        Raises:
          ValueError: The schedule returned a value outside ``[0, 1]``.

        """
        decay = self.decay_schedule(self.decay, step)
        # Used directly as a lerp coefficient: outside [0, 1] it extrapolates
        # rather than averages.
        if not math.isfinite(decay) or decay < 0 or decay > 1:
            raise ValueError(
                f"decay_schedule returned {decay} at step {step}; the "
                "effective decay must lie in [0, 1].",
            )
        return decay

    def __call__(self, model: nn.Module) -> None:
        """Advance one EMA step.

        On first call, lazily seeds the shadow from ``model``. Returns
        immediately during warmup (``global_step < update_after_step``).
        At the warmup boundary, re-seeds the shadow from live weights when
        ``warmup_seed`` is set. Otherwise lerps tracked params with the
        scheduled decay and (when ``track_buffers``) copies buffers.

        Args:
          model: Live model providing the current parameters.

        """
        if not self._initialized:
            self.initialize(model)

        if self.global_step < self.update_after_step:
            self.global_step += 1
            return

        if (self.global_step - self.update_after_step) % self.update_every != 0:
            self.global_step += 1
            return

        at_boundary = self.global_step == self.update_after_step
        with torch.no_grad():
            live_params = dict(model.named_parameters())
            decay = self.decay
            if self.warmup_seed and at_boundary:
                for name in self._tracked_names:
                    self._shadow_param(name).copy_(live_params[name].data)
            else:
                decay = self.effective_decay(self.local_step)
                for name in self._tracked_names:
                    shadow = self._shadow_param(name)
                    live = live_params[name].data
                    if self.update_fn is None:
                        shadow.mul_(decay).add_(live, alpha=1 - decay)
                    else:
                        self.update_fn(shadow, live, decay)
            if self.track_buffers:
                self._copy_buffers(model, decay=decay)

        self.global_step += 1
        self.local_step += 1

    def initialize(self, model: nn.Module, *, reset: bool = False) -> None:
        """Seed the shadow from ``model`` without advancing the EMA schedule.

        Repeated calls are no-ops after the first initialization unless
        ``reset`` is true. Reset replaces shadow tensors without advancing the
        EMA schedule, as needed after weights-only model loading.

        Args:
          model: Live model providing the initial parameters and structure.
          reset: Replace an existing shadow while preserving both counters.

        """
        if reset:
            self._pending_state = None
        if reset or not self._initialized:
            self._lazy_initialize(model, reset=reset)

    @contextmanager
    def apply_to(self, model: nn.Module) -> Generator[None]:
        """Swap shadow params into ``model`` for the context; restore on exit.

        Uses pre-allocated backup buffers so eval-time memory is O(1).
        Persistent buffers are swapped only when ``buffer_update_fn`` averages
        them (not the default :func:`copy_buffers`); otherwise they keep
        live-model values. Works for both shadow kinds.

        Args:
          model: PyTorch module whose parameters to temporarily swap.

        Yields:
          context: Block in which ``model.parameters()`` carry shadow values.

        """
        if not self._initialized:
            if self._loaded_shadow or self._pending_state is not None:
                self._lazy_initialize(model)
            else:
                # Nothing to swap; act as a no-op.
                yield
                return
        live_params = dict(model.named_parameters())
        swapped: list[str] = []
        buffers = self._averaged_buffers(model)
        saved = {name: live.clone() for name, (live, _) in buffers.items()}
        try:
            # The swap mutates live params in place, so it must sit INSIDE the
            # try: a failure mid-loop (e.g. a desynced tracked param) would
            # otherwise leave the model partially EMA-swapped. ``finally``
            # restores exactly the params already swapped.
            for name in self._tracked_names:
                if name not in live_params:
                    # A tracked param missing from the live module is a genuine
                    # desync (e.g. model rewired after shadow init); a silent
                    # skip would eval against a half-swapped model.
                    raise RuntimeError(
                        f"EMA.apply_to(): tracked parameter {name!r} is absent "
                        "from the live or shadow model; the shadow is out of "
                        "sync.",
                    )
                live = live_params[name].data
                self._backup[name].copy_(live)
                live.copy_(self._shadow_param(name))
                swapped.append(name)
            for live, shadow in buffers.values():
                live.copy_(shadow)
            yield
        finally:
            for name in swapped:
                live_params[name].data.copy_(self._backup[name])
            for name, (live, _) in buffers.items():
                live.copy_(saved[name])

    class StateDict(TypedDict):
        """Checkpointed EMA state.

        Exactly one shadow kind is present once initialized: ``shadow_model``
        (a module ``state_dict`` carrying torch's ``_metadata``) or the
        ``shadow_params``/``shadow_buffers`` pair. Neither is present before
        the first update.
        """

        global_step: int
        local_step: NotRequired[int]
        shadow_model: NotRequired[dict[str, Tensor]]
        shadow_params: NotRequired[dict[str, Tensor]]
        shadow_buffers: NotRequired[dict[str, Tensor]]

    def state_dict(self) -> StateDict:
        """Get EMA state for checkpointing.

        Returns:
          state: The name-keyed shadow tensors (cloned for storage
            independence) and step counters. The ``"module"`` kind preserves
            the ``_metadata`` attribute torch attaches to a module
            ``state_dict`` for versioned load.

        """
        if not self._initialized:
            return {
                "global_step": self.global_step,
                "local_step": self.local_step,
            }
        if self.shadow_model is not None:
            return {
                "shadow_model": _clone_module_state(self.shadow_model.state_dict()),
                "global_step": self.global_step,
                "local_step": self.local_step,
            }
        return {
            "shadow_params": {
                name: t.detach().clone() for name, t in self.shadow_params.items()
            },
            "shadow_buffers": {
                name: t.detach().clone() for name, t in self._shadow_buffers.items()
            },
            "global_step": self.global_step,
            "local_step": self.local_step,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load EMA state from checkpoint.

        Clones tensors on load so two EMAs loaded from the same source
        dict have independent storage.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(EMA.StateDict, state_dict)
        self.global_step = state["global_step"]
        # A step-dependent schedule reads local_step, so resetting it would
        # restart the ramp on resume. Absent in older checkpoints.
        self.local_step = state.get("local_step", 0)
        if "shadow_model" in state:
            cloned = _clone_module_state(state["shadow_model"])
            if self.shadow_model is None:
                self._pending_state = cloned
            else:
                self.shadow_model.load_state_dict(cloned)
        elif "shadow_params" in state:
            params = {
                name: t.detach().clone() for name, t in state["shadow_params"].items()
            }
            buffers = {
                name: t.detach().clone()
                for name, t in state.get("shadow_buffers", {}).items()
            }
            if not self._initialized:
                # param_dict shadows are self-describing (name -> tensor), so
                # adopt them directly: ``shadow_params`` is queryable (via
                # ``ema_shadow``) immediately after load, before the first
                # ``__call__`` -- required for eval-on-resume. Lazy-init then
                # only fills in tracked-name bookkeeping without re-cloning.
                self.shadow_params = params
                self._shadow_buffers = buffers
                self._tracked_names = set(params)
                self._loaded_shadow = True
            else:
                for name, t in params.items():
                    self.shadow_params[name].copy_(t)
                for name, t in buffers.items():
                    self._shadow_buffers[name].copy_(t)

    # -- Helpers --------------------------------------------------------------

    def _shadow_param(self, name: str) -> Tensor:
        """Return the shadow tensor for ``name`` regardless of shadow kind."""
        if self.shadow_model is not None:
            return self._module_params[name].data
        return self.shadow_params[name]

    def _lazy_initialize(self, model: nn.Module, *, reset: bool = False) -> None:
        self._tracked_names = {
            name for name, param in model.named_parameters() if self.select(name, param)
        }
        if self.shadow_kind == "module":
            self.shadow_model = copy.deepcopy(model)
            self.shadow_model.eval()
            self.shadow_model.requires_grad_(False)
            self._module_params = dict(self.shadow_model.named_parameters())
        elif self._loaded_shadow and not reset:
            # A shadow already loaded from a checkpoint before init: keep its
            # values (and tracked-name set) rather than re-cloning live weights
            # over the restored moving average -- but re-device each tensor to
            # its live counterpart. A checkpoint is saved by rank 0, so a plain
            # ``torch.load`` restores every shadow onto rank 0's device on ALL
            # ranks; the first ``mul_/add_`` against a local-device param would
            # otherwise fail with a cross-device error.
            live = dict(model.named_parameters())
            self.shadow_params = {
                name: t.to(live[name].device) if name in live else t
                for name, t in self.shadow_params.items()
            }
            live_buffers = dict(model.named_buffers())
            self._shadow_buffers = {
                name: t.to(live_buffers[name].device) if name in live_buffers else t
                for name, t in self._shadow_buffers.items()
            }
            self._tracked_names = set(self.shadow_params)
        else:
            # Clone LOCAL shards (param.detach().clone()), never a module
            # deepcopy, so DTensor/FSDP-sharded params survive.
            self.shadow_params = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if name in self._tracked_names
            }
            self._shadow_buffers = (
                {name: buf.detach().clone() for name, buf in model.named_buffers()}
                if self.track_buffers
                else {}
            )
        # Pre-allocate backup buffers for ``apply_to`` so eval-time memory
        # is constant.
        self._backup = {
            name: torch.empty_like(param.data)
            for name, param in model.named_parameters()
            if name in self._tracked_names
        }
        self._initialized = True
        self._load_pending()

    # Only a module-kind shadow can be pending: the param-kind load adopts its
    # tensors immediately (see ``load_state_dict``), so there is nothing to defer.
    def _load_pending(self) -> None:
        if self._pending_state is None:
            return
        pending, self._pending_state = self._pending_state, None
        if self.shadow_model is None:
            raise ValueError("Expected self.shadow_model is not None.")
        self.shadow_model.load_state_dict(pending)

    def _averaged_buffers(self, model: nn.Module) -> dict[str, tuple[Tensor, Tensor]]:
        """Pair each persistent live buffer with its averaged shadow."""
        if not self.track_buffers or self.buffer_update_fn is copy_buffers:
            return {}
        shadow = (
            dict(self.shadow_model.named_buffers())
            if self.shadow_model is not None
            else self._shadow_buffers
        )
        live = model.state_dict()
        return {name: (live[name], buf) for name, buf in shadow.items() if name in live}

    def _copy_buffers(self, model: nn.Module, *, decay: float) -> None:
        buffers = (
            dict(self.shadow_model.named_buffers())
            if self.shadow_model is not None
            else self._shadow_buffers
        )
        self.buffer_update_fn(buffers, model, decay)


class _StateDict(OrderedDict[str, Tensor]):
    """The mapping ``nn.Module.state_dict`` actually returns.

    torch attaches ``_metadata`` to it and reads it back in ``load_state_dict``
    for module-version migration hooks. No stub declares the attribute, so a
    plain ``OrderedDict`` cannot carry it across a clone.
    """

    _metadata: OrderedDict[str, dict[str, object]]


def _clone_module_state(source: Mapping[str, Tensor]) -> _StateDict:
    """Clone a module ``state_dict`` for storage independence, keeping ``_metadata``."""
    cloned = _StateDict((name, v.detach().clone()) for name, v in source.items())
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        metadata_name = "_" + "metadata"
        setattr(cloned, metadata_name, metadata)
    return cloned
