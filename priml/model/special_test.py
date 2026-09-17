"""Tests for special module."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Final, override

from configgle import Fig, Makeable
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.self_attention import SelfAttention
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.cost import Cost
from priml.model.custom_types import (
    ChannelsInOut,
    ChannelsOut,
    HasDepthIndex,
    TensorModule,
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
from priml.model.special import Identity, Skip, TiedLinear
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import MMDiTBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


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


def test_identity_reset():
    m = Identity.Config(channels_in=64).make()
    m.reset_parameters()


_SKIP_CONFIG_TYPES = [
    Identity.Config,
    GatedDeltaNet.Config,
    MultiHeadLatentAttention.Config,
    MultiStreamAttention.Config,
    OutputGate.Config,
    SelfAttention.Config,
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

    assert isinstance(config, ChannelsInOut)
    assert config.channels_in == 64
    assert config.channels_out == 64
    assert isinstance(config.inner, Identity.Config)
    assert "Skip.Config" in config.pformat(finalize=False)
    assert "inner=Identity.Config" in config.pformat(finalize=False)


def test_skip_forwards_direct_attributes_and_propagation_to_inner() -> None:
    config = Skip.Config(inner=SelfAttention.Config())

    config.depth_index = ((3, 5),)
    assert config.depth_index == ((3, 5),)
    propagate_attr(config, "channels_out", 32, protocol=ChannelsOut)
    propagate_attr(config, "depth_index", ((4, 5),), protocol=HasDepthIndex)

    assert isinstance(config.inner, SelfAttention.Config)
    assert config.inner.channels_out == 32
    assert config.inner.depth_index == ((4, 5),)
    with pytest.raises(AttributeError, match="typo"):
        config.typo = 1


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
    with pytest.raises(ValueError, match="inner"):
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
            ("flops", "primal", "elementwise", f32): 4,
            ("flops", "adjoint", "elementwise", f32): 4,
            ("bytes", "primal", "elementwise", f32): 4 * 3 * 4,
            ("bytes", "adjoint", "elementwise", f32): 4 * 3 * 4,
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
    assert tied["flops", :, "matmul"].sum() == owned["flops", :, "matmul"].sum()
    assert tied.params == 0
    assert tied["bytes", :, "matmul"].sum() == owned["bytes", :, "matmul"].sum()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
