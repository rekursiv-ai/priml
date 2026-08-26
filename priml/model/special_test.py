"""Tests for special module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.self_attention import SelfAttention
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.custom_types import (
    ChannelsInOut,
    ChannelsOut,
    HasDepthIndex,
    propagate_attr,
)
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
from priml.model.special import Identity, Skip
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import MMDiTBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_identity_config_pprint(request: pytest.FixtureRequest) -> None:
    config = Identity.Config(channels_in=4)
    assert_text_golden(
        request,
        test_file=__file__,
        name="identity",
        rendered=config.pformat(hide_default_values=False),
    )


def test_skip_config_pprint(request: pytest.FixtureRequest) -> None:
    config = Skip.Config(inner=Linear.Config(4, 4))
    assert_text_golden(
        request,
        test_file=__file__,
        name="skip",
        rendered=config.pformat(hide_default_values=False),
    )


def test_identity_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="identity",
        build_module=lambda: Identity.Config(channels_in=4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_skip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
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
    "config_type", _SKIP_CONFIG_TYPES, ids=lambda cls: cls.__qualname__
)
def test_skip_preserving_configs_infer_either_channel_boundary(
    config_type: type,
) -> None:
    from_output = config_type()
    from_output.channels_out = 128
    from_output.finalize()
    assert from_output.channels_in == 128

    from_input = config_type()
    from_input.channels_in = 128
    from_input.finalize()
    assert from_input.channels_out == 128


@pytest.mark.parametrize(
    "config_type", _SKIP_CONFIG_TYPES, ids=lambda cls: cls.__qualname__
)
def test_skip_preserving_modules_reject_width_changes(config_type: type) -> None:
    config = config_type()
    config.channels_in = 128
    config.channels_out = 64

    with pytest.raises(ValueError, match="channels_in=128 must equal channels_out=64"):
        config.finalize()


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
    with pytest.raises(ValueError, match="channels_in=64 must equal channels_out=32"):
        Skip.Config(inner=Linear.Config(64, 32)).make()


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
