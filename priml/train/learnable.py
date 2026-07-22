"""Learnable: Model wrapper with optimization (Forge-style).

Provides model + optimizer + scheduler + EMA + autocast without bundling loss.
For standalone use or as base class for training abstractions.

Features:
- Gradient clipping
- Model and data parallelism (TP, FSDP, DDP)
- Activation checkpointing
- torch.compile support
- EMA
- Gradient norm tracking
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING

import contextlib
import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn

import torch
import torch.amp

from priml.model.special import Identity
from priml.train.activation import DefaultActivationStorage
from priml.train.custom_types import (
    ActivationMemoizationProtocol,
    EMAProtocol,
    LearningRateSchedulerProtocol,
    ModelQuantizationProtocol,
    OptimizerProtocol,
    ParallelStrategyProtocol,
)
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel
from priml.train.quantization import NoModelQuantization


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any


class Learnable:
    """Model wrapper with optimization.

    Bundles model + optimizer + scheduler + optional EMA + parallelism.
    No loss bundled - user handles forward + loss + backward externally.

    Example:
      cfg = Learnable.Config(
          model=Mamba2DClassifier.Config(),
          parallelism=FullySharded.Config(mesh_dim="dp"),
          gradient_clip_norm=1.0,
          dtype_autocast=torch.bfloat16,
      )
      learnable = cfg.make()

      # Training loop
      for batch in loader:
          pred = learnable(**batch)  # Forward with autocast
          loss = my_loss_fn(pred, batch)
          loss.backward()
          learnable.step()  # Clip + optimizer.step + scheduler.step + EMA

    """

    class Config(Fig["Learnable"], kw_only=False):
        """Configuration for Learnable."""

        model: Makeable[nn.Module] = field(default_factory=Identity.Config)
        _: KW_ONLY

        optimizer: Makeable[Callable[[list[dict[str, Any]]], OptimizerProtocol]] = (
            field(
                default_factory=lambda: PartialConfig(
                    torch.optim.AdamW,
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                ),
            )
        )

        learning_rate_scheduler: Makeable[
            Callable[[OptimizerProtocol], LearningRateSchedulerProtocol]
        ] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.lr_scheduler.ConstantLR,
                factor=1.0,
            ),
        )

        parallelism: Makeable[ParallelStrategyProtocol] = field(
            default_factory=NoParallel.Config,
        )

        model_quantization: Makeable[ModelQuantizationProtocol] = field(
            default_factory=NoModelQuantization.Config,
        )

        activation_memoization: Makeable[ActivationMemoizationProtocol] = field(
            default_factory=DefaultActivationStorage.Config,
        )

        compile: Makeable[Callable[[Callable[..., Any]], Callable[..., Any]]] | None = (
            field(
                default_factory=lambda: PartialConfig(
                    torch.compile,
                    fullgraph=True,
                ),
            )
        )

        ema: Makeable[EMAProtocol] = field(default_factory=NoEMA.Config)

        gradient_clip_norm: float = math.inf

        device_init: torch.device | str | None = None
        dtype_autocast: torch.dtype | None = (
            None  # e.g., torch.bfloat16 for mixed precision
        )
        autocast_cache_enabled: bool = False
        """Enable autocast's weight cache. Default False preserves exact
        numerics across forward calls; True trades a small numeric difference
        for reusing cast weights within a forward (perf)."""

    def __init__(self, config: Config) -> None:
        self.config = config

        if config.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {config.gradient_clip_norm}",
            )

        self.gradient_clip_norm = config.gradient_clip_norm

        if config.device_init is None:
            model = self.config.model.make()
        else:
            with torch.device(config.device_init):
                model = self.config.model.make()

        # Quantization rewrites modules BEFORE the parallel strategy so the
        # strategy shards/materializes the final module graph.
        model_quantization = self.config.model_quantization.make()
        model = model_quantization(model)

        # Single strategy owns device assignment, sharding, meta->real
        # materialization, and post-shard reset_parameters.
        self.parallelism: ParallelStrategyProtocol = self.config.parallelism.make()
        model = self.parallelism(model)

        activation_strategy = self.config.activation_memoization.make()
        activation_strategy(model)

        self.model: nn.Module = model

        param_groups = [{"params": self.model.parameters()}]
        self.optimizer: OptimizerProtocol = self.config.optimizer.make()(param_groups)

        self.learning_rate_scheduler: LearningRateSchedulerProtocol = (
            self.config.learning_rate_scheduler.make()(self.optimizer)
        )

        self.ema: EMAProtocol = self.config.ema.make()

        self._compile_fn: Callable[[Callable[..., Any]], Callable[..., Any]] | None = (
            self.config.compile.make() if self.config.compile else None
        )
        self._compiled_model: Any = None

        self.global_step: int = 0
        self.local_step: int = 0
        self.last_grad_norm: Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Device for model and data."""
        return self.parallelism.device

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Training forward pass (sets train mode, applies autocast, optionally compiles)."""
        self.model.train()
        if self._compile_fn is not None and self._compiled_model is None:
            self._compiled_model = self._compile_fn(self.model)
        forward_model = (
            self._compiled_model if self._compiled_model is not None else self.model
        )
        autocast_ctx = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=self.config.autocast_cache_enabled,
            )
            if self.config.dtype_autocast is not None
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            return forward_model(*args, **kwargs)

    def call_eval(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluation forward pass (uses EMA if available, applies inference_mode and autocast).

        Runs the live model with EMA-averaged weights swapped in via
        ``ema.apply_to``. This is the single eval path for every shadow kind:
        the ``"param_dict"`` shadow keeps ``shadow_model is None`` (FSDP-safe),
        so a ``shadow_model`` truthiness fallback would silently evaluate LIVE
        un-averaged weights. NoEMA's ``apply_to`` is a no-op.
        """
        was_training = self.model.training
        self.model.eval()

        autocast_ctx = (
            torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast,
                cache_enabled=self.config.autocast_cache_enabled,
            )
            if self.config.dtype_autocast is not None
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), self.ema.apply_to(self.model), autocast_ctx:
            output = self.model(*args, **kwargs)

        self.model.train(was_training)
        return output

    def step(self, closure: Callable[[], Tensor | float] | None = None) -> None:
        """Optimization step (clip grads, optimizer.step, scheduler.step, EMA).

        Args:
          closure: Loss-recomputing closure for closure-based optimizers (e.g.
            exact-Hessian Newton). It is forwarded to ``optimizer.step`` ONLY
            when the optimizer sets ``requires_closure``: a torch first-order
            optimizer executes any closure it is handed, running a wasteful
            second forward that would also double-count BatchNorm stats.

        """
        self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm,
            foreach=True,
        )
        if closure is not None and getattr(self.optimizer, "requires_closure", False):
            self.optimizer.step(closure)
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.learning_rate_scheduler.step()
        self.ema(self.model)
        self.global_step += 1
        self.local_step += 1

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate_scheduler": self.learning_rate_scheduler.state_dict(),
            "global_step": self.global_step,
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        """Load checkpoint state dict.

        ``remap`` (if set) transforms the saved model dict first; ``strict``
        controls whether the resulting keys must match the model exactly;
        ``load_optimizer`` gates optimizer / LR-scheduler / EMA restore --
        skipped when finetuning a changed architecture, whose saved optimizer
        state would mismatch. Defaults are a strict, full load (ordinary resume).
        """
        model_state = state_dict["model"]
        if remap is not None:
            model_state = remap(model_state)
        self.model.load_state_dict(model_state, strict=strict)
        self.global_step = state_dict["global_step"]
        self.local_step = 0

        if not load_optimizer:
            return  # finetuning: keep the fresh optimizer/scheduler/EMA

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.learning_rate_scheduler.load_state_dict(
            state_dict["learning_rate_scheduler"],
        )
        if "ema" in state_dict:
            self.ema.load_state_dict(state_dict["ema"])
