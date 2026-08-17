"""Activation checkpointing strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, cast, override

import logging

from configgle import Fig
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  -- partial torch stub
)

import torch
import torch.nn.grad


if TYPE_CHECKING:
    from torch import nn


logger = logging.getLogger(__name__)


class DefaultActivationStorage:
    """Default activation storage (full precision, no memory optimization)."""

    class Config(Fig["DefaultActivationStorage"]):
        pass

    def __init__(self, config: Config):
        """Initialize default activation storage.

        Args:
            config: Configuration (unused).

        """

    def __call__(self, model: nn.Module) -> None:
        """No-op - use PyTorch default activation storage.

        Args:
            model: Module (unused).

        """


class LayerActivationCheckpointing:
    """Activation checkpointing for every N-th layer.

    Uses torch.distributed checkpoint API for FSDP compatibility.
    Applies checkpointing to leaf modules (modules with no children).
    """

    class Config(Fig["LayerActivationCheckpointing"]):
        interval: int = 1
        """Checkpoint every N-th leaf module."""
        reentrant: bool = False
        """Use reentrant checkpointing (legacy, less safe)."""

    def __init__(self, config: Config):
        """Initialize layer-based activation checkpointing.

        Args:
            config: Configuration for layer-based checkpointing.

        """
        if config.interval < 1:
            raise ValueError(
                f"interval must be >= 1, got {config.interval}",
            )
        self.interval = config.interval
        self.reentrant = config.reentrant

    def __call__(self, model: nn.Module) -> None:
        """Apply checkpointing to every N-th leaf module.

        Args:
            model: Module to apply checkpointing to.

        """
        layer_count = 0
        checkpoint_count = 0

        wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=(
                CheckpointImpl.REENTRANT
                if self.reentrant
                else CheckpointImpl.NO_REENTRANT
            ),
        )

        def check_fn(submodule: nn.Module) -> bool:
            nonlocal layer_count, checkpoint_count
            if len(list(submodule.children())) == 0:
                should_checkpoint = layer_count % self.interval == 0
                layer_count += 1
                if should_checkpoint:
                    checkpoint_count += 1
                return should_checkpoint
            return False

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper_fn,
            check_fn=check_fn,
        )

        logger.info(
            f"Applied LayerActivationCheckpointing: checkpointed {checkpoint_count} "
            f"of {layer_count} leaf modules (interval={self.interval})",
        )


class SelectiveActivationCheckpointing:
    """Selective activation checkpointing based on module types.

    Applies checkpointing to modules matching specified types, with optional
    sparse checkpointing via checkpoint_fraction (e.g., checkpoint_fraction=0.5
    checkpoints every other block).

    Uses torch.distributed checkpoint API for FSDP compatibility.
    """

    class Config(Fig["SelectiveActivationCheckpointing"]):
        module_types: Sequence[type[nn.Module]] = ()
        """Module classes to checkpoint (e.g., (TransformerBlock,))."""
        checkpoint_fraction: float = 1.0
        """Fraction of matching modules to checkpoint (1.0=all, 0.5=every other)."""
        reentrant: bool = False
        """Use reentrant checkpointing (legacy, less safe)."""

    def __init__(self, config: Config):
        """Initialize selective activation checkpointing.

        Args:
            config: Configuration for selective checkpointing.

        """
        if not config.module_types:
            raise ValueError(
                "SelectiveActivationCheckpointing requires module_types. "
                "Provide tuple of module classes to checkpoint, e.g., (TransformerBlock,)",
            )
        self.module_types = tuple(config.module_types)
        self.checkpoint_fraction = config.checkpoint_fraction
        self.reentrant = config.reentrant

        if self.checkpoint_fraction < 0 or self.checkpoint_fraction > 1:
            raise ValueError(
                f"checkpoint_fraction must be in [0, 1], got {self.checkpoint_fraction}",
            )

    def __call__(self, model: nn.Module) -> None:
        """Apply checkpointing to modules matching types.

        Args:
            model: Module to apply checkpointing to.

        """
        wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=(
                CheckpointImpl.REENTRANT
                if self.reentrant
                else CheckpointImpl.NO_REENTRANT
            ),
        )

        # Evenly space checkpointed blocks using modular arithmetic.
        # E.g., checkpoint_fraction=0.5 checkpoints every 2nd block.
        if self.checkpoint_fraction <= 0:
            logger.info("checkpoint_fraction=0, skipping activation checkpointing.")
            return
        interval = max(1, round(1.0 / self.checkpoint_fraction))
        block_count = 0
        checkpoint_count = 0

        def check_fn(submodule: nn.Module) -> bool:
            nonlocal block_count, checkpoint_count
            if isinstance(submodule, self.module_types):
                block_count += 1
                if block_count % interval == 0:
                    checkpoint_count += 1
                    return True
            return False

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper_fn,
            check_fn=check_fn,
        )

        logger.info(
            f"Applied SelectiveActivationCheckpointing: checkpointed {checkpoint_count} "
            f"of {block_count} modules matching {[t.__name__ for t in self.module_types]} "
            f"(checkpoint_fraction={self.checkpoint_fraction})",
        )


class QuantizedActivationStorage:
    """Store activations in quantized format (FP8) for memory reduction.

    Alternative to recomputation-based checkpointing. Stores activations
    in FP8 format during forward pass, then upcasts to higher precision
    during backward pass for gradient computation.
    """

    class Config(Fig["QuantizedActivationStorage"]):
        dtype_storage: torch.dtype = torch.float8_e4m3fn
        """FP8 dtype for storing activations."""
        dtype_compute: torch.dtype | None = None
        """Dtype for backward computation (None = original dtype)."""
        min_size: int = 4_096
        """Minimum tensor numel to quantize (smaller tensors stored as-is)."""

    def __init__(self, config: Config):
        """Initialize quantized activation storage.

        Args:
            config: Configuration for quantization strategy.

        """
        if not str(config.dtype_storage).startswith("torch.float8"):
            raise ValueError(
                f"dtype_storage must be FP8 type, got {config.dtype_storage}",
            )
        self.dtype_storage = config.dtype_storage
        self.dtype_compute = config.dtype_compute
        self.min_size = config.min_size

    def __call__(self, model: nn.Module) -> None:
        """Apply quantized storage hooks to model activations.

        Args:
            model: Module to apply quantization to.

        """
        _empty_marker = torch.tensor([], device="cpu")

        def _make_pack_hook(
            dtype_storage: torch.dtype,
            min_size: int,
        ) -> Callable[[Tensor], Any]:
            """Create pack hook with config captured in closure."""

            def pack_hook(tensor: Tensor) -> Any:
                """Quantize activation for storage (forward pass)."""
                if tensor.numel() < min_size or not tensor.is_floating_point():
                    return (tensor, _empty_marker, tensor.dtype)

                amax = tensor.abs().max()
                # An all-zero tensor has amax == 0; clamp the scale to 1 so the
                # division yields zeros rather than NaN (0/0). Dequant recovers
                # exact zeros since quantized values are all zero.
                scale = amax / torch.finfo(dtype_storage).max
                scale = torch.where(scale > 0, scale, 1.0)
                quantized = (tensor / scale).to(dtype_storage)
                return (quantized, scale.reshape(1), tensor.dtype)

            return pack_hook

        def _make_unpack_hook(
            dtype_compute: torch.dtype | None,
        ) -> Callable[[Any], Tensor]:
            """Create unpack hook with config captured in closure."""

            def unpack_hook(packed: Any) -> Tensor:
                """Dequantize activation for gradient computation (backward pass)."""
                quantized, scale, orig_dtype = packed
                assert isinstance(quantized, Tensor)

                if scale is _empty_marker:
                    return quantized

                target_dtype = dtype_compute or orig_dtype
                assert isinstance(scale, Tensor)
                dequantized = quantized.to(target_dtype) * scale[0]
                return dequantized.requires_grad_(quantized.requires_grad)

            return unpack_hook

        original_forward = model.forward

        pack_hook = _make_pack_hook(self.dtype_storage, self.min_size)
        unpack_hook = _make_unpack_hook(self.dtype_compute)

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            """Forward pass with quantized activation storage hooks."""
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                return original_forward(*args, **kwargs)

        model.forward = wrapped_forward  # ty: ignore[invalid-assignment]

        logger.info(
            f"Applied QuantizedActivationStorage: dtype_storage={self.dtype_storage}, "
            f"dtype_compute={self.dtype_compute}, min_size={self.min_size}",
        )


class QuantizedModuleActivationStorage:
    """Module-level activation quantization for reduced overhead.

    Instead of using saved_tensors_hooks (180% overhead), wraps specific
    module types with custom autograd.Function (20% overhead).

    Only quantizes module inputs, not all intermediate activations.
    Requires custom backward implementation per module type.
    """

    class Config(Fig["QuantizedModuleActivationStorage"]):
        module_types: Sequence[type[torch.nn.Module]] = ()
        """Module classes to quantize (e.g., (nn.Conv2d,))."""
        dtype_storage: torch.dtype = torch.float8_e4m3fn
        """FP8 dtype for storing activations."""
        min_size: int = 4_096
        """Minimum tensor numel to quantize (smaller tensors stored as-is)."""

    def __init__(self, config: Config):
        if not config.module_types:
            raise ValueError(
                "QuantizedModuleActivationStorage requires module_types. "
                "Provide tuple of module classes, e.g., (nn.Conv2d,)",
            )
        self.module_types = tuple(config.module_types)
        self.dtype_storage = config.dtype_storage
        self.min_size = config.min_size

        supported = {torch.nn.Conv2d}
        for mod_type in self.module_types:
            if mod_type not in supported:
                raise ValueError(
                    f"Module type {mod_type} not yet supported. "
                    f"Supported types: {supported}",
                )

    def __call__(self, model: torch.nn.Module) -> None:
        """Apply quantization to matching modules.

        Args:
            model: Module to apply quantization to.

        """
        quantized_count = 0

        for module in model.modules():
            if type(module) in self.module_types:
                self._wrap_module(module)
                quantized_count += 1

        logger.info(
            f"Applied QuantizedModuleActivationStorage: quantized {quantized_count} modules "
            f"of types {[t.__name__ for t in self.module_types]}",
        )

    def _wrap_module(self, module: torch.nn.Module) -> None:
        """Wrap a module with quantized forward."""
        if isinstance(module, torch.nn.Conv2d):
            self._wrap_conv2d(module)

    def _wrap_conv2d(self, module: torch.nn.Conv2d) -> None:
        """Wrap Conv2d with quantized activation storage."""

        class QuantizedConv2dFunction(torch.autograd.Function):
            @classmethod
            @override
            def forward(
                cls,
                ctx: Any,
                input: Tensor,
                weight: Tensor,
                bias: Tensor | None,
                stride: Any,
                padding: Any,
                dilation: Any,
                groups: int,
                dtype_storage: torch.dtype,
                min_size: int,
            ) -> Tensor:
                # Quantize input if large enough
                if input.numel() >= min_size:
                    amax = input.abs().max()
                    # Clamp scale to 1 when amax == 0 (all-zero input) to avoid
                    # a 0/0 NaN; quantized zeros dequantize back to zeros.
                    scale = amax / torch.finfo(dtype_storage).max
                    scale = torch.where(scale > 0, scale, 1.0)
                    input_q = (input / scale).to(dtype_storage)
                    ctx.save_for_backward(
                        input_q,
                        weight,
                        bias if bias is not None else torch.tensor([]),
                    )
                    ctx.scale = scale.reshape(1)
                    ctx.quantized = True
                else:
                    ctx.save_for_backward(
                        input,
                        weight,
                        bias if bias is not None else torch.tensor([]),
                    )
                    ctx.quantized = False

                ctx.stride = stride
                ctx.padding = padding
                ctx.dilation = dilation
                ctx.groups = groups
                ctx.input_dtype = input.dtype

                return torch.nn.functional.conv2d(
                    input,
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                )

            @classmethod
            @override
            def backward(  # ty: ignore[invalid-method-override]
                cls,
                ctx: Any,
                grad_output: Tensor,
            ) -> tuple[
                Tensor,
                Tensor,
                Tensor | None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]:
                saved = ctx.saved_tensors

                if ctx.quantized:
                    input_q, weight, bias = (
                        saved[0],
                        saved[1],
                        (saved[2] if saved[2].numel() > 0 else None),
                    )
                    # Dequantize to the original input dtype (not hardcoded
                    # fp32) so the backward matmuls stay dtype-consistent.
                    input = input_q.to(ctx.input_dtype) * ctx.scale[0].to(
                        ctx.input_dtype,
                    )
                else:
                    input, weight, bias = (
                        saved[0],
                        saved[1],
                        (saved[2] if saved[2].numel() > 0 else None),
                    )

                grad_input = torch.nn.grad.conv2d_input(
                    input.shape,
                    weight,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    ctx.groups,
                )
                grad_weight = torch.nn.grad.conv2d_weight(
                    input,
                    weight.shape,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    ctx.groups,
                )
                grad_bias = grad_output.sum([0, 2, 3]) if bias is not None else None

                return (
                    grad_input,
                    grad_weight,
                    grad_bias,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        # Replace _conv_forward to intercept before F.conv2d call
        def quantized_conv_forward(
            input: Tensor,
            weight: Tensor,
            bias: Tensor | None,
        ) -> Tensor:
            # Legacy `forward(ctx, ...)` convention: `apply` is typed against
            # the modern ctx-less static `forward`, so every argument lands one
            # position early. Converting would mean returning the quantized
            # intermediates as outputs, which changes the autograd signature.
            result = cast(  # ty: ignore[redundant-cast] -- pyright still infers Unknown here
                "Tensor",
                QuantizedConv2dFunction.apply(  # ty: ignore[missing-argument]  # pyright: ignore[reportCallIssue]
                    input,
                    weight,
                    bias,  # ty: ignore[invalid-argument-type]
                    module.stride,  # ty: ignore[invalid-argument-type]
                    module.padding,
                    module.dilation,
                    module.groups,
                    self.dtype_storage,  # ty: ignore[invalid-argument-type]
                    self.min_size,  # ty: ignore[invalid-argument-type]
                ),
            )
            return result

        module._conv_forward = quantized_conv_forward  # noqa: SLF001  # ty: ignore[invalid-assignment]
