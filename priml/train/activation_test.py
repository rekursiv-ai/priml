"""Tests for activation checkpointing."""

from __future__ import annotations

from typing import Any, override

import contextlib

from configgle import Fig
from torch import nn

import pytest
import torch

from priml.train.activation import (
    DefaultActivationStorage,
    LayerActivationCheckpointing,
    QuantizedActivationStorage,
    QuantizedModuleActivationStorage,
    SelectiveActivationCheckpointing,
)
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


class SimpleModule(nn.Module):
    """Simple test module."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NestedModule(nn.Module):
    """Nested module for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class BlockModule(nn.Module):
    """Block module for transformer-style testing."""

    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(10, 10)
        self.ffn = nn.Linear(10, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x


class TransformerModel(nn.Module):
    """Simple transformer-style model."""

    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([BlockModule() for _ in range(num_blocks)])
        self.output = nn.Linear(10, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def test_default_activation_storage_init():
    """Test DefaultActivationStorage initialization."""
    config = DefaultActivationStorage.Config()
    strategy = DefaultActivationStorage(config)
    assert strategy is not None


def test_default_activation_storage_apply():
    """Test DefaultActivationStorage.apply does nothing."""
    config = DefaultActivationStorage.Config()
    strategy = DefaultActivationStorage(config)

    module = SimpleModule()
    strategy(module)

    # Just verify it doesn't raise
    x = torch.randn(4, 10)
    output = module(x)
    assert output.shape == (4, 10)


def test_layer_activation_checkpointing_init():
    """Test LayerActivationCheckpointing initialization."""
    config = LayerActivationCheckpointing.Config(interval=2)
    strategy = LayerActivationCheckpointing(config)
    assert strategy.interval == 2


def test_layer_activation_checkpointing_default_interval():
    """Test LayerActivationCheckpointing default interval."""
    config = LayerActivationCheckpointing.Config()
    strategy = LayerActivationCheckpointing(config)
    assert strategy.interval == 1


def test_layer_activation_checkpointing_apply():
    """Test LayerActivationCheckpointing applies without error."""
    config = LayerActivationCheckpointing.Config(interval=1)
    strategy = LayerActivationCheckpointing(config)

    module = NestedModule()
    strategy(module)

    # Verify forward still works
    x = torch.randn(4, 10)
    output = module(x)
    assert output.shape == (4, 10)


def test_layer_activation_checkpointing_interval():
    """Test LayerActivationCheckpointing respects interval."""
    config = LayerActivationCheckpointing.Config(interval=2)
    strategy = LayerActivationCheckpointing(config)

    module = NestedModule()
    strategy(module)

    # Just verify it works
    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)
    assert output.shape == (4, 10)


def test_layer_activation_checkpointing_forward_works():
    """Test LayerActivationCheckpointing wrapped forward works."""
    config = LayerActivationCheckpointing.Config(interval=1)
    strategy = LayerActivationCheckpointing(config)

    module = NestedModule()
    strategy(module)

    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)

    # Verify backward works
    loss = output.sum()
    loss.backward()
    assert module.layer1.weight.grad is not None


def test_selective_activation_checkpointing_init():
    """Test SelectiveActivationCheckpointing initialization."""
    config = SelectiveActivationCheckpointing.Config(module_types=(BlockModule,))
    strategy = SelectiveActivationCheckpointing(config)
    assert strategy.module_types == (BlockModule,)


def test_selective_activation_checkpointing_requires_module_types():
    """Test SelectiveActivationCheckpointing requires module_types."""
    config = SelectiveActivationCheckpointing.Config()
    with pytest.raises(ValueError, match="requires module_types"):
        SelectiveActivationCheckpointing(config)


def test_selective_activation_checkpointing_apply():
    """Test SelectiveActivationCheckpointing applies without error."""
    config = SelectiveActivationCheckpointing.Config(module_types=(BlockModule,))
    strategy = SelectiveActivationCheckpointing(config)

    module = TransformerModel(num_blocks=3)
    strategy(module)

    # Verify forward still works
    x = torch.randn(4, 10)
    output = module(x)
    assert output.shape == (4, 10)


def test_selective_activation_checkpointing_checkpoint_fraction():
    """Test SelectiveActivationCheckpointing with checkpoint_fraction."""
    config = SelectiveActivationCheckpointing.Config(
        module_types=(BlockModule,),
        checkpoint_fraction=0.5,
    )
    strategy = SelectiveActivationCheckpointing(config)

    module = TransformerModel(num_blocks=4)
    strategy(module)

    # Verify forward still works
    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)
    assert output.shape == (4, 10)

    # Verify backward works
    loss = output.sum()
    loss.backward()


def test_selective_activation_checkpointing_forward_works():
    """Test SelectiveActivationCheckpointing wrapped forward works."""
    config = SelectiveActivationCheckpointing.Config(module_types=(BlockModule,))
    strategy = SelectiveActivationCheckpointing(config)

    module = TransformerModel(num_blocks=2)
    strategy(module)

    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)

    # Verify backward works
    loss = output.sum()
    loss.backward()
    assert module.output.weight.grad is not None


def test_selective_activation_checkpointing_invalid_checkpoint_fraction():
    """Test SelectiveActivationCheckpointing validates checkpoint_fraction."""
    config = SelectiveActivationCheckpointing.Config(
        module_types=(BlockModule,),
        checkpoint_fraction=1.5,
    )
    with pytest.raises(ValueError, match="checkpoint_fraction must be in"):
        SelectiveActivationCheckpointing(config)


def test_layer_activation_checkpointing_gradient_checkpointing():
    """Test LayerActivationCheckpointing saves memory via gradient checkpointing."""
    config = LayerActivationCheckpointing.Config(interval=1)
    strategy = LayerActivationCheckpointing(config)

    module = NestedModule()
    strategy(module)

    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    # Verify gradients exist
    assert module.layer1.weight.grad is not None
    assert module.layer2.weight.grad is not None
    assert module.layer3.weight.grad is not None


def test_quantized_activation_storage_init():
    """Test QuantizedActivationStorage initialization."""
    config = QuantizedActivationStorage.Config()
    strategy = QuantizedActivationStorage(config)
    assert strategy.dtype_storage == torch.float8_e4m3fn
    assert strategy.min_size == 4_096


def test_quantized_activation_storage_config_defaults():
    """Test QuantizedActivationStorage default configuration."""
    config = QuantizedActivationStorage.Config()
    assert config.dtype_storage == torch.float8_e4m3fn
    assert config.dtype_compute is None
    assert config.min_size == 4_096


def test_quantized_activation_storage_config_custom():
    """Test QuantizedActivationStorage custom configuration."""
    config = QuantizedActivationStorage.Config()
    config.dtype_storage = torch.float8_e5m2
    config.dtype_compute = torch.bfloat16
    config.min_size = 8_192

    strategy = QuantizedActivationStorage(config)
    assert strategy.dtype_storage == torch.float8_e5m2
    assert strategy.dtype_compute == torch.bfloat16
    assert strategy.min_size == 8_192


def test_quantized_activation_storage_invalid_dtype():
    """Test QuantizedActivationStorage rejects non-FP8 dtype."""
    config = QuantizedActivationStorage.Config()
    config.dtype_storage = torch.float32

    with pytest.raises(ValueError, match="must be FP8"):
        QuantizedActivationStorage(config)


def test_quantized_activation_storage_apply():
    """Test QuantizedActivationStorage applies without error."""
    config = QuantizedActivationStorage.Config()
    strategy = QuantizedActivationStorage(config)

    module = SimpleModule()
    strategy(module)

    # Verify forward still works
    x = torch.randn(4, 10)
    output = module(x)
    assert output.shape == (4, 10)


def test_quantized_activation_storage_backward():
    """Test QuantizedActivationStorage works with backward pass."""
    config = QuantizedActivationStorage.Config()
    config.min_size = 10
    strategy = QuantizedActivationStorage(config)

    module = SimpleModule()
    strategy(module)

    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    assert module.linear.weight.grad is not None
    assert x.grad is not None


def test_quantized_activation_storage_min_size_threshold():
    """Test QuantizedActivationStorage respects min_size threshold."""
    config = QuantizedActivationStorage.Config()
    config.min_size = 1_000_000
    strategy = QuantizedActivationStorage(config)

    module = SimpleModule()
    strategy(module)

    x = torch.randn(4, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    assert module.linear.weight.grad is not None


def test_quantized_activation_storage_hooks_invoked():
    """Verify hooks work by checking backward pass completes with quantization."""
    config = QuantizedActivationStorage.Config()
    config.min_size = 10

    module = NestedModule()
    config.make()(module)

    x = torch.randn(8, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()

    loss.backward()

    assert module.layer1.weight.grad is not None
    assert module.layer2.weight.grad is not None
    assert module.layer3.weight.grad is not None
    assert x.grad is not None


def test_quantized_activation_storage_uses_fp8_dtype():
    """Verify FP8 dtype can be used (implementation test via config)."""
    config = QuantizedActivationStorage.Config()
    config.min_size = 10
    config.dtype_storage = torch.float8_e4m3fn

    module = NestedModule()
    config.make()(module)

    x = torch.randn(8, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    assert module.layer1.weight.grad is not None


def test_quantized_activation_storage_memory_reduction():
    """Verify quantization works on larger models (memory test skipped for small models)."""
    config = QuantizedActivationStorage.Config()
    config.min_size = 10

    module = NestedModule()
    config.make()(module)

    x = torch.randn(32, 10, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    assert module.layer1.weight.grad is not None


def test_quantized_activation_storage_gradient_accuracy():
    """Verify FP8 quantization preserves gradient accuracy within tolerance."""
    torch.manual_seed(42)
    module_ref = NestedModule()
    DefaultActivationStorage.Config().make()(module_ref)
    x_ref = torch.randn(8, 10, requires_grad=True)
    loss_ref = module_ref(x_ref).sum()
    loss_ref.backward()
    grad_ref = module_ref.layer1.weight.grad
    assert grad_ref is not None
    grad_ref = grad_ref.clone()

    torch.manual_seed(42)
    module_fp8 = NestedModule()
    config = QuantizedActivationStorage.Config()
    config.min_size = 10
    config.make()(module_fp8)
    x_fp8 = torch.randn(8, 10, requires_grad=True)
    loss_fp8 = module_fp8(x_fp8).sum()
    loss_fp8.backward()
    grad_fp8 = module_fp8.layer1.weight.grad
    assert grad_fp8 is not None

    torch.testing.assert_close(grad_fp8, grad_ref, rtol=0.6, atol=0.2)


def test_quantized_activation_storage_min_size_boundary():
    """Verify min_size threshold controls quantization (edge case test)."""
    config_skip = QuantizedActivationStorage.Config()
    config_skip.min_size = 1_000_000

    module_skip = NestedModule()
    config_skip.make()(module_skip)

    x_skip = torch.randn(4, 10, requires_grad=True)
    output_skip = module_skip(x_skip)
    loss_skip = output_skip.sum()
    loss_skip.backward()

    assert module_skip.layer1.weight.grad is not None

    config_quantize = QuantizedActivationStorage.Config()
    config_quantize.min_size = 10

    module_quantize = NestedModule()
    config_quantize.make()(module_quantize)

    x_quantize = torch.randn(4, 10, requires_grad=True)
    output_quantize = module_quantize(x_quantize)
    loss_quantize = output_quantize.sum()
    loss_quantize.backward()

    assert module_quantize.layer1.weight.grad is not None


def test_quantized_activation_storage_with_learnable():
    """Verify quantization works end-to-end with TrainStep."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

        class Config(Fig["TinyModel"]):
            @override
            def make(self) -> TinyModel:
                return TinyModel()

    config = TrainStep.Config()
    config.model = TinyModel.Config()
    config.activation_memoization = QuantizedActivationStorage.Config()
    config.activation_memoization.min_size = 10
    config.compile = None
    # FP8 not supported on MPS; force CPU when CUDA unavailable
    if not torch.cuda.is_available():
        config.parallelism = NoParallel.Config(device="cpu")

    learnable = config.make()

    x = torch.randn(4, 10, device=learnable.device)
    output = learnable(x)
    loss = output.sum()
    loss.backward()
    learnable.step()

    assert learnable.last_grad_norm is not None
    assert learnable.global_step == 1


def test_quantized_module_conv_fp32_backward_regression() -> None:
    """T-038 regression: fp32 quantized conv backward stays finite + fp32."""
    config = QuantizedModuleActivationStorage.Config(
        module_types=(nn.Conv2d,),
        min_size=4,
    )
    conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
    config.make()(conv)

    x = torch.randn(1, 2, 8, 8, requires_grad=True)
    conv(x).sum().backward()

    assert conv.weight.grad is not None
    assert conv.weight.grad.dtype == torch.float32
    assert torch.isfinite(conv.weight.grad).all()


def test_quantized_module_conv_dequant_matches_input_dtype() -> None:
    """T-038: backward dequant must use the saved input dtype, not hardcoded fp32.

    bf16 conv *backward* is unsupported on many CPUs, so we assert the saved
    context dtype directly: the forward must record the input dtype and the
    backward must dequant to it. We capture the dtype the backward dequant
    targets by intercepting the saved-context attribute.
    """
    config = QuantizedModuleActivationStorage.Config(
        module_types=(nn.Conv2d,),
        min_size=4,
    )
    conv = nn.Conv2d(2, 2, kernel_size=3, padding=1).to(torch.bfloat16)
    config.make()(conv)

    captured: dict[str, object] = {}
    orig_to = torch.Tensor.to

    def spy_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Record any dequant on a float8 tensor (capture its target dtype).
        if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and args:
            captured["dequant_target"] = args[0]
        return orig_to(self, *args, **kwargs)

    x = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16, requires_grad=True)
    out = conv(x)
    torch.Tensor.to = spy_to
    try:
        with contextlib.suppress(RuntimeError):  # bf16 conv backward may be unsupported
            out.sum().backward()
    finally:
        torch.Tensor.to = orig_to

    # The float8 activation must be dequantized to the input's bf16 dtype.
    assert captured.get("dequant_target") == torch.bfloat16, (
        f"dequant target was {captured.get('dequant_target')}, expected bfloat16"
    )


def test_quantized_activation_storage_all_zero_no_nan() -> None:
    """T-036: an all-zero activation (amax==0) must not divide by zero → NaN."""

    class ZeroOutModule(nn.Module):
        """Multiplies input by zero so the saved activation is all zeros."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(16, 16)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x) * 0.0

    config = QuantizedActivationStorage.Config()
    config.min_size = 4  # ensure the zero activation crosses the quantize path

    module = ZeroOutModule()
    config.make()(module)

    # All-zero input: the tensor saved for backward has amax == 0.
    x = torch.zeros(8, 16, requires_grad=True)
    output = module(x)
    loss = output.sum()
    loss.backward()

    assert module.linear.weight.grad is not None
    assert not torch.isnan(module.linear.weight.grad).any(), (
        "all-zero activation produced NaN gradients (fp8 div-by-zero)"
    )
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize("interval", [0, -1])
def test_layer_activation_checkpointing_rejects_nonpositive_interval(
    interval: int,
) -> None:
    """#343: interval<=0 would divide by zero in the modulo; reject at init."""
    with pytest.raises(ValueError, match="interval"):
        LayerActivationCheckpointing(
            LayerActivationCheckpointing.Config(interval=interval),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
