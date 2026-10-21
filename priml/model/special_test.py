"""Tests for special module."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.attention.attention import Attention
from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsInOut,
    ChannelsOut,
    HasDepthIndex,
    TensorModule,
    has_weight,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import (
    BatchNorm,
    BatchNorm2d,
    BatchRenorm,
    CenteredRMSNorm,
    GroupNorm,
    GroupNorm2d,
    LayerNorm,
    RMSNorm,
)
from priml.model.sequential import Sequential
from priml.model.special import (
    Activation,
    Dropout,
    Dropout2d,
    Identity,
    Skip,
    TiedLinear,
    dropout_cost,
)
from priml.model.swiglu import relu, silu
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import MMDiTBlock, MMDiTStream
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from torch._ops import OpOverload

    from priml.math.custom_types import TensorFn


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_dropout_matches_torch_values_gradients_and_next_draw(
    probability: float,
    training: bool,
) -> None:
    config = Dropout.Config()
    config.p = probability
    config.channels_out = 3
    native = config.make().train(training)
    reference = nn.Dropout(probability).train(training)
    source = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 17
    left, right = source.clone().requires_grad_(), source.clone().requires_grad_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = reference(left)
        expected.square().sum().backward()
        after = torch.get_rng_state()
        torch.manual_seed(17)
        actual = native(right, ignored=True)
        actual.square().sum().backward()
        assert actual.dtype == expected.dtype
        assert right.grad is not None
        assert left.grad is not None
        assert torch.equal(actual, expected)
        assert torch.equal(right.grad, left.grad)
        assert torch.equal(torch.get_rng_state(), after)
    assert native.state_dict() == {}
    native.reset_parameters()
    finalized = config.copy_tree().finalize()
    assert finalized.channels_in == finalized.channels_out == 3
    assert finalized.output_grid((4, 5)) == (4, 5)


def test_dropout_cost_distinguishes_channel_masks_and_disabled_modes() -> None:
    dtype = torch.float64
    per_element = dropout_cost(0.25, elements=24, masks=24, dtype=dtype)
    per_channel = dropout_cost(0.25, elements=24, masks=6, dtype=dtype)
    assert per_element["flops", "primal"].sum() == 48
    assert per_channel["flops", "primal"].sum() == 30
    assert per_channel["flops", "adjoint"].sum() == 24
    assert per_channel["bytes", "primal"].sum() == 8 * (2 * 24 + 4 * 6)
    assert per_channel["bytes", "adjoint"].sum() == 8 * (2 * 24 + 6)
    assert dropout_cost(0.0, elements=24, masks=24, dtype=dtype) == Cost()
    assert (
        dropout_cost(0.25, elements=24, masks=24, dtype=dtype, training=False) == Cost()
    )
    zeros = dropout_cost(1.0, elements=24, masks=24, dtype=dtype)
    assert zeros["flops", "primal"].sum() == zeros["flops", "adjoint"].sum() == 24
    config = Dropout.Config()
    config.channels_in = 3
    config.p = 0.25
    assert config.cost(seq_len=4, batch_size=2, dtype=dtype) == per_element
    config.channels_in = -1
    assert config.cost(channels=3, seq_len=4, batch_size=2, dtype=dtype) == per_element


def test_dropout2d_matches_torch_and_costs_one_mask_per_channel() -> None:
    config = Dropout2d.Config()
    config.p = 0.25
    config.channels_out = 3
    source = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = nn.Dropout2d(0.25)(source)
        torch.manual_seed(17)
        actual = config.make()(source, ignored=True)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, nn.Dropout(0.25).train()(source))
    dtype = torch.float64
    assert config.finalize().cost(
        seq_len=20,
        batch_size=2,
        dtype=dtype,
    ) == dropout_cost(
        0.25,
        elements=120,
        masks=6,
        dtype=dtype,
    )


def test_inplace_dropout_preserves_storage_and_complete_gradient() -> None:
    config = Dropout.Config()
    config.inplace = True
    config.p = 0.25
    native = config.make()
    reference = nn.Dropout(0.25, inplace=True)
    left = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).requires_grad_()
    right = left.detach().clone().requires_grad_()
    first, second = left * 1, right * 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        actual = native(first)
        torch.manual_seed(23)
        expected = reference(second)
    assert actual.data_ptr() == first.data_ptr()
    assert expected.data_ptr() == second.data_ptr()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.sum(), left)[0],
        torch.autograd.grad(expected.sum(), right)[0],
    )


@pytest.mark.parametrize("width", [0, -2])
def test_dropout_rejects_inconsistent_widths(width: int) -> None:
    config = Dropout.Config()
    config.channels_in, config.channels_out = 3, width
    with pytest.raises(ValueError, match="must equal"):
        config.make()


@pytest.mark.parametrize("function", [relu, silu])
def test_activation_uses_its_injected_function_and_cost(function: TensorFn) -> None:
    config = Activation.Config()
    config.channels_in = 3
    config.act = function
    native = config.make()
    value = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 7 - 1
    left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
    actual, expected = native(left, ignored=True), function(right)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.sum(), left)[0],
        torch.autograd.grad(expected.sum(), right)[0],
    )
    assert config.cost(seq_len=4, batch_size=2, dtype=torch.float64) == cost(
        function,
        channels=24,
        dtype=torch.float64,
    )
    assert native.state_dict() == {}
    native.reset_parameters()
    assert config.copy_tree().finalize().output_grid((4, 5)) == (4, 5)


def test_activation_rejects_an_uncosted_function() -> None:
    config = Activation.Config()
    config.act = torch.neg
    with pytest.raises(TypeError, match="has no cost"):
        config.cost(channels=3, seq_len=4, batch_size=2, dtype=torch.float32)


class _MaskTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.elements: defaultdict[str, list[int]] = defaultdict(list)

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        output = func(*args, **(kwargs or {}))
        if isinstance(output, Tensor):
            self.elements[str(func)].append(output.numel())
        return output


@pytest.mark.parametrize("channelwise", [False, True])
def test_torch_dropout_scales_each_mask_then_multiplies_values_and_gradients(
    channelwise: bool,
) -> None:
    value = torch.ones(2, 3, 4, 5, requires_grad=True)
    with torch.random.fork_rng(devices=[]), _MaskTrace() as trace:
        output = (
            nn.functional.dropout2d(value, p=0.25)
            if channelwise
            else nn.functional.dropout(value, p=0.25)
        )
        output.backward(torch.ones_like(output))
    masks = 6 if channelwise else 120
    assert trace.elements["aten.bernoulli_.float"] == [masks]
    assert trace.elements["aten.div_.Scalar"] == [masks]
    assert trace.elements["aten.mul.Tensor"] == [120, 120]


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_dropout_disabled_and_constant_zero_modes_draw_no_mask(
    probability: float,
    training: bool,
) -> None:
    value = torch.ones(2, 3, 4, 5, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        before = torch.get_rng_state()
        with _MaskTrace() as trace:
            nn.functional.dropout(
                value,
                p=probability,
                training=training,
            ).sum().backward()
        assert not trace.elements["aten.bernoulli_.float"]
        assert not trace.elements["aten.div_.Scalar"]
        assert torch.equal(torch.get_rng_state(), before)


def test_identity_config_pprint() -> None:
    config = Identity.Config(channels_in=4)
    assert_pprint_golden(
        test_file=__file__,
        name="identity",
        config=config,
    )


def test_skip_config_pprint() -> None:
    config = Skip.Config(inner=Linear.Config(4, 4))
    assert_pprint_golden(
        test_file=__file__,
        name="skip",
        config=config,
    )


def test_identity_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="identity",
        build_module=lambda: Identity.Config(channels_in=4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_skip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="skip",
        build_module=lambda: Skip.Config(inner=Linear.Config(4, 4)).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_identity():
    m = Identity.Config(channels_in=64).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert m(x, key="val").shape == (2, 8, 64)
    with pytest.raises(TypeError):
        m(x, "extra")


def test_identity_channels_infer():
    cfg = Identity.Config(channels_in=64).finalize()
    assert cfg.channels_out == 64

    cfg2 = Identity.Config(channels_out=32).finalize()
    assert cfg2.channels_in == 32


def test_identity_invalid_widths_print_before_make_rejects() -> None:
    config = Identity.Config(channels_in=128, channels_out=64)

    with pytest.warns(UserWarning, match="must equal"):
        rendered = config.pformat(hide_default_values=False)

    assert "channels_in=128" in rendered
    assert "channels_out=64" in rendered
    with pytest.raises(ValueError, match="channels_in=128 must equal channels_out=64"):
        config.make()


def test_identity_preserves_the_spatial_extents() -> None:
    assert Identity.Config(channels_in=4).output_grid((5, 7)) == (5, 7)


def test_identity_reset():
    m = Identity.Config(channels_in=64).make()
    m.reset_parameters()


_SKIP_CONFIG_TYPES: Final = [
    Identity.Config,
    GatedDeltaNet.Config,
    MultiHeadLatentAttention.Config,
    MultiStreamAttention.Config,
    OutputGate.Config,
    Attention.Config,
    ValueGatedAttention.Config,
    MLPMixerBlock.Config,
    RMSNorm.Config,
    CenteredRMSNorm.Config,
    LayerNorm.Config,
    BatchNorm.Config,
    BatchRenorm.Config,
    BatchNorm2d.Config,
    GroupNorm2d.Config,
    GroupNorm.Config,
    TransformerBlock.Config,
    MMDiTBlock.Config,
]


@pytest.mark.parametrize(
    "config_type",
    _SKIP_CONFIG_TYPES,
    ids=[config_type.__qualname__ for config_type in _SKIP_CONFIG_TYPES],
)
def test_skip_preserving_configs_infer_either_channel_boundary(
    config_type: type[Fig[nn.Module]],
) -> None:
    from_output = config_type()
    assert isinstance(from_output, ChannelsInOut)
    from_output.channels_out = 128
    from_output.finalize()
    assert from_output.channels_in == 128

    from_input = config_type()
    assert isinstance(from_input, ChannelsInOut)
    from_input.channels_in = 128
    from_input.finalize()
    assert from_input.channels_out == 128


@pytest.mark.parametrize(
    "config_type",
    _SKIP_CONFIG_TYPES,
    ids=[config_type.__qualname__ for config_type in _SKIP_CONFIG_TYPES],
)
def test_skip_preserving_modules_reject_width_changes(
    config_type: type[Fig[nn.Module]],
) -> None:
    config = config_type()
    assert isinstance(config, ChannelsInOut)
    config.channels_in = 128
    config.channels_out = 64

    with pytest.raises(ValueError, match="channels_in=128 must equal channels_out=64"):
        config.make()


def test_skip_channels_proxy_inner() -> None:
    config = Skip.Config(inner=Identity.Config(channels_in=64)).finalize()

    assert isinstance(config, ChannelsInOut)  # pyrefly: ignore[unsafe-overlap] -- The passthrough descriptors answer at runtime, which is what this asserts.
    assert config.channels_in == 64
    assert config.channels_out == 64
    assert isinstance(config.inner, Identity.Config)
    assert "Skip.Config" in config.pformat(finalize=False)
    assert "inner=Identity.Config" in config.pformat(finalize=False)


def test_skip_forwards_direct_attributes_and_propagation_to_inner() -> None:
    config = Skip.Config(inner=Attention.Config())

    config.depth_index = ((3, 5),)
    assert config.depth_index == ((3, 5),)
    propagate_attr(config, "channels_out", 32, protocol=ChannelsOut)
    propagate_attr(config, "depth_index", ((4, 5),), protocol=HasDepthIndex)

    assert isinstance(config.inner, Attention.Config)
    assert config.inner.channels_out == 32
    assert config.inner.depth_index == ((4, 5),)
    with pytest.raises(AttributeError, match="typo"):
        config.typo = 1


def test_nested_skips_forward_protocol_gated_depth_to_the_leaf() -> None:
    inner = Skip.Config()
    inner.inner = Attention.Config()
    config = Skip.Config()
    config.inner = inner

    propagate_attr(config, "depth_index", ((2, 3),), protocol=HasDepthIndex)

    assert isinstance(config.inner, Skip.Config)
    assert isinstance(config.inner.inner, Attention.Config)
    assert config.inner.inner.depth_index == ((2, 3),)


def test_propagation_replaces_the_wrappers_own_inner_field() -> None:
    inner = Skip.Config()
    inner.inner = Attention.Config()
    config = Skip.Config()
    config.inner = inner
    replacement = Identity.Config()
    replacement.channels_in = 16

    propagate_attr(config, "inner", replacement, protocol=ChannelsIn)

    assert config.inner is replacement


@pytest.mark.parametrize("config_type", [TransformerBlock.Config, MMDiTStream.Config])
def test_block_depth_reaches_a_wrapped_branch(
    config_type: type[TransformerBlock.Config | MMDiTStream.Config],
) -> None:
    config = config_type()
    config.channels_in = 16
    config.depth_index = ((1, 3),)
    branch = Skip.Config()
    branch.inner = Linear.Config()
    config.ffn = branch

    config.finalize()

    assert isinstance(config.ffn, Skip.Config)
    assert isinstance(config.ffn.inner, Linear.Config)
    assert config.ffn.inner.depth_index == config.depth_index


@pytest.mark.parametrize("config_type", [TransformerBlock.Config, MMDiTStream.Config])
def test_depth_indexed_blocks_preserve_wrapped_identity_branches(
    config_type: type[TransformerBlock.Config | MMDiTStream.Config],
) -> None:
    config = config_type()
    config.channels_in = 16
    config.depth_index = ((1, 3),)
    branch = Skip.Config()
    branch.inner = Identity.Config()
    config.norm1 = branch

    config.finalize()

    assert isinstance(config.norm1, Skip.Config)
    assert isinstance(config.norm1.inner, Identity.Config)
    assert config.norm1.inner.channels_in == config.norm1.inner.channels_out == 16
    assert not hasattr(config.norm1, "depth_index")


def test_repeated_sequential_assigns_nested_depth_through_skip() -> None:
    inner = Linear.Config()
    inner.channels_in = 4
    inner.channels_out = 4
    branch = Skip.Config()
    branch.inner = inner
    config = Sequential.Config()
    config.elements = branch
    config.repeat = 3
    config.depth_index = ((1, 2),)

    config.finalize()

    assert isinstance(config.elements, list)
    for index, element in enumerate(config.elements):
        assert isinstance(element.inner, Linear.Config)
        assert element.inner.depth_index == ((1, 2), (index, 3))


def test_transformer_assigns_layer_depth_through_a_wrapped_branch() -> None:
    config = Transformer.Config()
    config.channels_in = 16
    config.num_layers = 3
    assert isinstance(config.block, TransformerBlock.Config)
    branch = Skip.Config()
    branch.inner = Linear.Config()
    config.block.ffn = branch

    config.finalize()

    assert isinstance(config.block, list)
    for index, block in enumerate(config.block):
        assert isinstance(block.ffn, Skip.Config)
        assert isinstance(block.ffn.inner, Linear.Config)
        assert block.ffn.inner.depth_index == ((index, 3),)


def test_skip_rejects_an_inner_width_change() -> None:
    """The residual add rejects it, naming both widths and the axis.

    Skip does not pre-check the inner module's widths: that is the inner
    module's invariant, and torch reports it with more detail than a
    parent-side check could.
    """
    model = Skip.Config(inner=Linear.Config(64, 32)).make()

    with pytest.raises(RuntimeError, match="must match the size of tensor"):
        model(torch.randn(2, 8, 64))


def test_skip():
    m = Skip.Config(inner=Linear.Config(64, 64)).make()
    x = torch.randn(2, 8, 64)
    assert m.in_features == 64
    assert m.out_features == 64
    assert m(x).shape == (2, 8, 64)


def test_skip_reset():
    m = Skip.Config(inner=Linear.Config(64, 64)).make()
    m.reset_parameters()


def test_skip_requires_inner():
    with pytest.raises(ValueError, match=r"^Must specify `inner`\.$"):
        Skip.Config().make()


class _LanguageModel(torch.nn.Module):
    """Embedding, head; one is a real table, the other borrows it."""

    class Config(Fig["_LanguageModel"]):
        embed: Makeable[TensorModule] = field(default_factory=Embedding.Config)
        head: Makeable[TensorModule] = field(default_factory=Linear.Config)

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Head first: a tie at ``embed`` points at a module not yet built.
        self.head = config.head.make()
        self.embed = config.embed.make()

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        return self.head(self.embed(tokens))


def test_tied_linear_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="tied_linear",
        config=TiedLinear.Config(8, 17, tied="embed"),
    )


def test_tied_linear_head_borrows_the_embedding() -> None:
    """The head reads the table transposed; only the table is a parameter."""
    config = _LanguageModel.Config()
    config.embed = Embedding.Config(channels_out=8, channels_in=17)
    config.head = TiedLinear.Config(tied="embed")
    model = config.make()
    assert isinstance(model.embed, Embedding)

    tokens = torch.tensor([[1, 2, 3]])
    logits = model(tokens)

    assert torch.equal(logits, model.embed(tokens) @ model.embed.weight.T)
    assert [name for name, _ in model.named_parameters()] == ["embed.weight"]
    logits.square().sum().backward()
    assert model.embed.weight.grad is not None


def test_tied_linear_table_borrows_the_head() -> None:
    """The reverse tie: integer input looks rows up in the head's weight."""
    config = _LanguageModel.Config()
    config.embed = TiedLinear.Config(tied="head", transpose=False)
    config.head = Linear.Config(8, 17)
    model = config.make()
    assert isinstance(model.head, Linear)

    tokens = torch.tensor([[1, 2, 3]])

    assert torch.equal(model.embed(tokens), model.head.weight[tokens])
    assert model(tokens).shape == (1, 3, 17)
    assert [name for name, _ in model.named_parameters()] == ["head.weight"]


def test_tied_linear_exposes_the_borrowed_weight_in_linear_layout() -> None:
    """``weight`` is ``[out, in]`` like ``nn.Linear``, so ``x @ weight.T`` is the map."""
    config = _LanguageModel.Config()
    config.embed = Embedding.Config(channels_out=8, channels_in=17)
    config.head = TiedLinear.Config(tied="embed")
    model = config.make()
    assert isinstance(model.embed, Embedding)
    assert isinstance(model.head, TiedLinear)
    assert has_weight(model.head)
    assert model.head.weight is model.embed.weight
    hidden = torch.randn(2, 3, 8)
    assert torch.equal(model.head(hidden), hidden @ model.head.weight.T)

    config = _LanguageModel.Config()
    config.embed = TiedLinear.Config(tied="head", transpose=False)
    config.head = Linear.Config(8, 17)
    model = config.make()
    assert isinstance(model.embed, TiedLinear)
    assert isinstance(model.head, Linear)
    assert torch.equal(model.embed.weight, model.head.weight.T)


def test_an_unbound_tied_linear_has_no_weight() -> None:
    """Unbound means no weight: ``has_weight`` answers False instead of raising."""
    head = TiedLinear(TiedLinear.Config(tied="embed"))
    assert not has_weight(head)
    with pytest.raises(AttributeError):
        _ = head.weight


def test_an_unbound_tied_linear_refuses_to_run() -> None:
    head = TiedLinear(TiedLinear.Config(tied="embed"))
    with pytest.raises(RuntimeError, match="unbound"):
        head(torch.randn(2, 3))


def test_tied_linear_rejects_a_source_without_a_weight() -> None:
    config = _LanguageModel.Config()
    config.embed = Identity.Config(channels_in=8)
    config.head = TiedLinear.Config(tied="embed")

    with pytest.raises(ValueError, match="tied='embed'"):
        config.make()


def test_tied_linear_requires_a_path() -> None:
    with pytest.raises(ValueError, match="tied"):
        TiedLinear.Config().make()


def test_tied_linear_built_alone_is_unbound() -> None:
    """A bare make binds against the leaf itself, which owns no weight."""
    with pytest.raises(AttributeError):
        TiedLinear.Config(tied="embed").make()


def test_identity_cost_is_free() -> None:
    free = assert_cost_matches_torch(
        Identity.Config(8),
        build_input=lambda: torch.randn(2, 8, requires_grad=True),
        seq_len=2,
        batch_size=1,
        dtype=None,
    )
    assert free == Cost()


def test_skip_cost_is_the_inner_cost_plus_residual_additions() -> None:
    inner = Linear.Config(4, 4)
    skip = assert_cost_matches_torch(
        Skip.Config(inner=inner),
        build_input=lambda: torch.randn(2, 4, requires_grad=True),
        seq_len=2,
        batch_size=1,
        dtype=None,
    )
    f32 = torch.float32
    assert skip == inner.copy_tree().finalize().cost(
        seq_len=2,
        batch_size=1,
        dtype=None,
    ) + Cost(
        cells={
            ("flops", "primal", "elementwise", f32): 4 * 2,
            ("flops", "adjoint", "elementwise", f32): 4 * 2,
            ("bytes", "primal", "elementwise", f32): 4 * 3 * 4 * 2,
            ("bytes", "adjoint", "elementwise", f32): 4 * 3 * 4 * 2,
        },
    )


def test_skip_cost_without_inner_raises() -> None:
    with pytest.raises(ValueError, match="inner"):
        Skip.Config().cost(seq_len=1, batch_size=1, dtype=None)


def test_tied_linear_cost_pays_flops_and_owns_nothing() -> None:
    tied = (
        TiedLinear.Config(4, 8, tied="embed")
        .copy_tree()
        .finalize()
        .cost(seq_len=1, batch_size=1, dtype=None)
    )
    owned = (
        Linear.Config(4, 8)
        .copy_tree()
        .finalize()
        .cost(seq_len=1, batch_size=1, dtype=None)
    )
    assert tied["flops", "matmul"].sum() == owned["flops", "matmul"].sum()
    assert tied.params == 0
    assert tied["bytes", "matmul"].sum() == owned["bytes", "matmul"].sum()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
