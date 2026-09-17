"""Architecture, transferred weights, and language-preserving graft behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.lib.custom_json import DictCodec
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.self_attention import SelfAttention
from priml.model.cost import Cost, cost
from priml.model.linear import Linear
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import AdaLNZero, MMDiTStream
from priml.model.transformer.mmdit_graft import MMDiTGraft
from priml.model.transformer.qwen3 import Qwen3
from priml.model.transformer.qwen3_test import _canonical_config
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import (
    assert_bfb_against_golden,
    host_agnostic_numerics,
    randomize_parameters,
)
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def _backbone(*, depth: int = 1, tie: bool = False) -> Qwen3.Config:
    config = _canonical_config()
    config.num_layers = depth
    if tie:
        assert isinstance(config.proj_out, Sequential.Config)
        assert isinstance(config.proj_out.elements, list)
        config.proj_out.elements[1] = TiedLinear.Config(tied="proj_in")
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()
    return config


def _config(
    *,
    depth: int = 1,
    tie: bool = False,
    conditioned: bool = False,
) -> MMDiTGraft.Config:
    config = MMDiTGraft.Config()
    config.backbone = _backbone(depth=depth, tie=tie)
    stream = MMDiTStream.Config()
    stream.ffn = SwiGLU.Config(channels_hidden=24)
    if conditioned:
        stream.adaln = AdaLNZero.Config(cond_dim=4)
    config.streams = [stream]
    return config


@pytest.mark.compute_large_fixture
def test_graft_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="mmdit_graft",
        config=_config(conditioned=True),
    )


def _constructor_state(module: nn.Module, input: Tensor) -> Tensor:
    del module, input
    model = _config(conditioned=True).make()
    state = DictCodec.coerce(model.state_dict(), Tensor)
    return torch.cat(
        [
            *(value.detach().flatten().float() for value in state.values()),
            torch.get_rng_state().float(),
        ],
    )


def test_graft_constructor_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="mmdit_graft_constructor",
        build_module=nn.Identity,
        build_input=lambda: torch.empty(0),
        run=_constructor_state,
    )


def _assert_same_state(source: object, target: object) -> None:
    assert isinstance(source, nn.Module)
    assert isinstance(target, nn.Module)
    before = DictCodec.coerce(source.state_dict(), Tensor)
    after = DictCodec.coerce(target.state_dict(), Tensor)
    assert before.keys() == after.keys()
    for name, value in before.items():
        assert torch.equal(value, after[name]), name


def _assert_transferred(source: Transformer, graft: MMDiTGraft) -> None:
    _assert_same_state(source.proj_in, graft.proj_in)
    _assert_same_state(source.proj_out, graft.proj_out)
    assert len(source.blocks) == len(graft.blocks)
    for before, after in zip(source.blocks, graft.blocks, strict=True):
        assert isinstance(before, TransformerBlock)
        for old, new in (
            (before.attn, after.attn.streams[0]),
            (before.norm1, after.norms1[0]),
            (before.norm2, after.norms2[0]),
            (before.ffn, after.ffns[0]),
        ):
            _assert_same_state(old, new)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("tie", [False, True])
def test_weights_logits_and_stream_isolation(depth: int, tie: bool) -> None:
    source = _backbone(depth=depth, tie=tie).make().eval()
    graft = _config(depth=depth, tie=tie).make().eval()
    randomize_parameters(source, seed=7, std=0.2)
    other_state = DictCodec.coerce(graft.blocks[0].ffns[1].state_dict(), Tensor)
    other_before = {name: value.clone() for name, value in other_state.items()}
    graft.load_backbone(source)
    _assert_transferred(source, graft)
    other_after = DictCodec.coerce(graft.blocks[0].ffns[1].state_dict(), Tensor)
    for name, value in other_after.items():
        assert torch.equal(value, other_before[name])
    tokens = torch.tensor([[1, 2, 3]])
    other = torch.randn(1, 2, 16)
    masks = [
        torch.cat(
            (
                torch.full((3, 3), float("-inf")).triu(1),
                torch.full((3, 2), float("-inf")),
            ),
            -1,
        ),
        None,
    ]
    with torch.no_grad(), host_agnostic_numerics():
        expected = source(tokens)
        logits, streams = graft(tokens, [other], attn_mask=masks)
        assert torch.equal(logits, expected)
        assert streams[0].shape == other.shape
        assert torch.equal(graft(tokens, [other + 100], attn_mask=masks)[0], expected)
        assert not torch.equal(
            graft(tokens, [other], attn_mask=[None, None])[0],
            expected,
        )


@pytest.mark.parametrize("projected", [False, True])
def test_continuous_backbone_projections(projected: bool) -> None:
    backbone = Transformer.Config()
    backbone.channels_in = 4 if projected else 16
    backbone.num_layers = 1
    assert isinstance(backbone.block, TransformerBlock.Config)
    backbone.block.attn = SelfAttention.Config(
        num_heads=2,
        attn_kernel=SdpaNaive.Config(),
    )
    if projected:
        backbone.proj_in = Linear.Config(channels_out=16)
        backbone.proj_out = Linear.Config(channels_out=3)
    config = MMDiTGraft.Config()
    config.backbone = backbone
    source, graft = backbone.make(), config.make()
    graft.load_backbone(source)
    inputs = torch.randn(1, 3, backbone.channels_in)
    other = torch.randn(1, 2, 16)
    mask = torch.cat((torch.zeros(3, 3), torch.full((3, 2), float("-inf"))), -1)
    with host_agnostic_numerics():
        output, streams = graft(inputs, [other], attn_mask=[mask, None])
        assert torch.equal(output, source(inputs))
    assert output.shape == (1, 3, 3 if projected else 16)
    assert streams[0].shape == other.shape


def test_freezing_preserves_language_weights_while_new_stream_learns() -> None:
    source = _backbone().make()
    graft = _config(conditioned=True).make()
    randomize_parameters(graft, seed=11, std=0.2)
    graft.load_backbone(source)
    graft.freeze_backbone()
    frozen = {
        name: parameter.clone()
        for name, parameter in graft.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen
    tokens = torch.tensor([[1, 2, 3]])
    other = torch.randn(1, 2, 16)
    logits, streams = graft(
        tokens,
        [other],
        c=[None, torch.randn(1, 4)],
        attn_mask=[
            torch.cat(
                (
                    torch.full((3, 3), float("-inf")).triu(1),
                    torch.full((3, 2), float("-inf")),
                ),
                -1,
            ),
            None,
        ],
    )
    streams[0].square().sum().backward()
    ffn = graft.blocks[0].ffns[1]
    assert isinstance(ffn, SwiGLU)
    assert ffn.up_proj.weight.grad is not None
    assert torch.count_nonzero(ffn.up_proj.weight.grad) > 0
    assert all(
        parameter.grad is None
        for parameter in graft.parameters()
        if not parameter.requires_grad
    )
    before = ffn.up_proj.weight.clone()
    torch.optim.SGD(graft.parameters(), lr=0.01).step()
    assert not torch.equal(before, ffn.up_proj.weight)
    for name, parameter in graft.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    assert logits.shape == (1, 3, 32)


def test_loading_preflights_every_layer_before_copying() -> None:
    source = _backbone(depth=2).make()
    graft = _config(depth=2).make()
    assert isinstance(source.blocks[1], TransformerBlock)
    source.blocks[1].ffn = SwiGLU.Config(16, channels_hidden=48).make()
    graft_state = DictCodec.coerce(graft.state_dict(), Tensor)
    before = {name: value.clone() for name, value in graft_state.items()}
    with pytest.raises(ValueError, match="shape"):
        graft.load_backbone(source)
    after = DictCodec.coerce(graft.state_dict(), Tensor)
    assert all(torch.equal(before[name], value) for name, value in after.items())


def test_factory_keeps_caller_configuration_unchanged() -> None:
    source = _backbone()
    stream = MMDiTStream.Config()
    before = source.pformat(finalize=False)
    config = MMDiTGraft.Config()
    config.backbone = source
    config.streams = [stream]
    config.backbone.num_layers = 2
    model = config.make()
    assert len(model.blocks) == 2
    config.backbone.num_layers = 1
    assert source.pformat(finalize=False) == before
    assert stream.channels_in == -1


@pytest.mark.parametrize("invalid", ["postnorm", "nonattention", "no_streams"])
def test_factory_rejects_unsupported_backbones(invalid: str) -> None:
    config = _backbone()
    assert isinstance(config.block, TransformerBlock.Config)
    streams = [MMDiTStream.Config()]
    if invalid == "postnorm":
        config.block.prenorm = False
    elif invalid == "nonattention":
        config.block.attn = SwiGLU.Config()
    else:
        streams = []
    graft = MMDiTGraft.Config()
    graft.backbone = config
    graft.streams = streams
    error = TypeError if invalid == "nonattention" else ValueError
    with pytest.raises(
        error,
        match=r"Grafting requires|at least one additional stream",
    ):
        graft.make()


@pytest.mark.parametrize("tie", [False, True])
def test_graft_cost_prices_host_projections_and_joint_blocks_once(
    tie: bool,
) -> None:
    """The host's layers live inside the joint blocks, so they are not re-added."""
    config = _config(depth=2, tie=tie, conditioned=True)
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=8, batch_size=1, dtype=None)
    backbone = finalized.backbone
    projections = [p for p in (backbone.proj_in, backbone.proj_out) if p is not None]
    expected = sum(
        (
            cost(c, seq_len=8, batch_size=1, dtype=None)
            for c in (*projections, *finalized.block)
        ),
        Cost(),
    )
    assert model_cost == expected
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    host = cost(backbone, seq_len=8, batch_size=1, dtype=None)
    assert model_cost.params > host.params
    assert model_cost["flops", :, "matmul"].sum() > host["flops", :, "matmul"].sum()


@pytest.mark.parametrize("tie", [False, True])
def test_graft_cost_matches_torch(tie: bool) -> None:
    """Embedding, joint blocks over both streams, and the head are torch's count.

    A "token" is one POSITION holding one token per stream, as
    ``MultiStreamAttention`` defines it, so both streams run three positions
    and the joint key length is six. Unconditioned, since adaLN runs once per
    sequence while ``cost`` prices it per token (``mmdit_test``).
    """
    assert_cost_matches_torch(
        _config(depth=2, tie=tie),
        build_input=lambda: (
            torch.randint(0, 32, (1, 3)),
            torch.randn(1, 3, 16, requires_grad=True),
        ),
        seq_len=6,
        batch_size=1,
        dtype=None,
        rows=3,
        run=run_graft,
    )


def run_graft(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Run a graft on ``(tokens, stream)`` and reduce both outputs to a scalar."""
    assert isinstance(module, MMDiTGraft)
    tokens, other = inputs
    logits, streams = module(tokens, [other], attn_mask=[None, None])
    return logits.sum() + streams[0].sum()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
