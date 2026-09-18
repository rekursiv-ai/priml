"""Architecture, transferred weights, and language-preserving graft behavior.

Regenerate bit-for-bit goldens after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer/mmdit_graft_test.py

Run regeneration through pytest so priml's conftest establishes the required
math environment before torch imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.lib.custom_json import DictCodec
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.self_attention import SelfAttention
from priml.model.linear import Linear
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import AdaLNZero, MMDiTStream
from priml.model.transformer.mmdit_graft import MMDiTGraft
from priml.model.transformer.qwen3_test import _canonical_config
from priml.model.transformer.transformer import Transformer, head_is_tied
from priml.testing.bfb import (
    assert_bfb_against_golden,
    host_agnostic_numerics,
    randomize_parameters,
)
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.model.transformer.qwen3 import Qwen3


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


def test_graft_forward_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="mmdit_graft_forward",
        build_module=lambda: _config(conditioned=True).make(),
        build_input=_graft_batch,
        run=_run_graft_forward,
    )


def _graft_batch() -> tuple[Tensor, Tensor, Tensor]:
    """Draw tokens, one modality stream, and its conditioning from the seeded RNG."""
    return torch.tensor([[1, 2, 3]]), torch.randn(1, 2, 16), torch.randn(1, 4)


def _run_graft_forward(
    module: nn.Module,
    batch: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Concatenate the language logits and modality stream of one recipe forward."""
    assert isinstance(module, MMDiTGraft)
    tokens, modality, conditioning = batch
    logits, streams = module(
        tokens,
        [modality],
        c=[None, conditioning],
        attn_mask=_language_only_masks(3, modality=2),
    )
    return torch.cat([logits.flatten(), streams[0].flatten()])


def _language_only_masks(
    language: int,
    *,
    modality: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor | None]:
    """Return the recipe masks: causal language that never reads a modality key."""
    mask = torch.full(
        (language, language + modality),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    mask[:, :language] = mask[:, :language].triu(1)
    # The modality stream is unrestricted, so it reads every language key.
    return [mask, None]


def test_graft_frozen_step_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="mmdit_graft_frozen_step",
        build_module=_frozen_graft,
        build_input=_graft_batch,
        run=_run_frozen_step,
    )


def _frozen_graft() -> MMDiTGraft:
    """Build the conditioned graft with only its modality stream trainable."""
    graft = _config(conditioned=True).make()
    graft.freeze_backbone()
    return graft


def _run_frozen_step(
    module: nn.Module,
    batch: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Take one SGD step and return the loss, logits, and modality stream."""
    assert isinstance(module, MMDiTGraft)
    tokens, modality, conditioning = batch
    before = {name: parameter.clone() for name, parameter in module.named_parameters()}
    frozen = {
        name
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen
    assert len(frozen) < len(before)
    logits, streams = module(
        tokens,
        [modality],
        c=[None, conditioning],
        attn_mask=_language_only_masks(3, modality=2),
    )
    loss = logits.square().mean() + streams[0].square().mean()
    loss.backward()
    torch.optim.SGD(module.parameters(), lr=0.1).step()
    module.zero_grad(set_to_none=True)
    for name, parameter in module.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, before[name]), name
        else:
            assert not torch.equal(parameter, before[name]), name
    return torch.cat(
        [
            loss.detach().reshape(1),
            logits.detach().flatten(),
            streams[0].detach().flatten(),
        ],
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
def test_load_backbone_transfers_only_language_weights(depth: int, tie: bool) -> None:
    source = _backbone(depth=depth, tie=tie).make()
    graft = _config(depth=depth, tie=tie).make()
    randomize_parameters(source, seed=7, std=0.2)
    modality = nn.ModuleList([graft.blocks[0].attn.streams[1], graft.blocks[0].ffns[1]])
    modality_before = {
        name: value.clone()
        for name, value in DictCodec.coerce(modality.state_dict(), Tensor).items()
    }
    graft.load_backbone(source)
    _assert_transferred(source, graft=graft)
    for name, value in DictCodec.coerce(modality.state_dict(), Tensor).items():
        assert torch.equal(value, modality_before[name]), name


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("tie", [False, True])
def test_language_only_causal_mask_reproduces_standalone_logits(
    depth: int,
    tie: bool,
) -> None:
    source = _backbone(depth=depth, tie=tie).make().eval()
    graft = _config(depth=depth, tie=tie).make().eval()
    randomize_parameters(source, seed=7, std=0.2)
    graft.load_backbone(source)
    tokens = torch.tensor([[1, 2, 3]])
    other = torch.randn(1, 2, 16)
    masks = _language_only_masks(3, modality=2)
    with torch.no_grad(), host_agnostic_numerics():
        expected = source(tokens)
        logits, streams = graft(tokens, [other], attn_mask=masks)
        assert torch.equal(logits, expected)
        assert streams[0].shape == other.shape
        assert torch.equal(graft(tokens, [other + 100], attn_mask=masks)[0], expected)


@pytest.mark.parametrize("visibility", ["unmasked", "modality_visible", "noncausal"])
def test_wider_language_visibility_changes_logits(visibility: str) -> None:
    source = _backbone().make().eval()
    graft = _config().make().eval()
    randomize_parameters(source, seed=7, std=0.2)
    graft.load_backbone(source)
    tokens = torch.tensor([[1, 2, 3]])
    other = torch.randn(1, 2, 16)
    recipe = _language_only_masks(3, modality=2)[0]
    assert recipe is not None
    if visibility == "unmasked":
        mask = None
    elif visibility == "modality_visible":
        mask = recipe.clone()
        mask[:, 3:] = 0
    else:
        mask = torch.zeros_like(recipe)
        mask[:, 3:] = float("-inf")
    with torch.no_grad(), host_agnostic_numerics():
        expected = source(tokens)
        logits, _ = graft(tokens, [other], attn_mask=[mask, None])
        assert not torch.equal(logits, expected)


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
        attn_mask=_language_only_masks(3, modality=2),
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
    assert model_cost["flops", "matmul"].sum() > host["flops", "matmul"].sum()


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
        seq_len=3,
        batch_size=1,
        num_tokens=3,
        dtype=None,
        run=run_graft,
    )


def run_graft(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Run a graft on ``(tokens, stream)`` and reduce both outputs to a scalar."""
    assert isinstance(module, MMDiTGraft)
    tokens, other = inputs
    logits, streams = module(tokens, [other], attn_mask=[None, None])
    return logits.sum() + streams[0].sum()


def test_graft_finalize_skips_layers_that_are_not_prenorm_self_attention() -> None:
    config = _config(depth=2)
    block = config.backbone.block
    assert isinstance(block, TransformerBlock.Config)
    config.backbone.block = [block.copy_tree(), Linear.Config(16, 16)]
    config.backbone.num_layers = 2
    assert len(config.finalize().block) == 1


def test_graft_rejects_a_backbone_with_no_layers() -> None:
    config = _config()
    finalized = config.finalize()
    finalized.backbone.block = []
    with pytest.raises(ValueError, match="nonempty language backbone"):
        MMDiTGraft(finalized)


def test_loading_rejects_a_source_whose_blocks_are_not_native() -> None:
    source = _backbone().make()
    source.blocks[0] = nn.Identity()
    graft = _config().make()
    with pytest.raises(ValueError, match="native prenorm SelfAttention"):
        graft.load_backbone(source)


def test_reset_parameters_redraws_every_owned_module() -> None:
    graft = _config().make()
    torch.manual_seed(0)
    graft.reset_parameters()
    first = DictCodec.coerce(graft.state_dict(), Tensor)["proj_in.weight"].clone()
    torch.manual_seed(1)
    graft.reset_parameters()
    after = DictCodec.coerce(graft.state_dict(), Tensor)["proj_in.weight"]
    assert not torch.equal(first, after)


def test_forward_rejects_the_wrong_stream_count() -> None:
    graft = _config().make()
    with pytest.raises(ValueError, match="Expected 1 modality streams, got 0"):
        graft(torch.zeros(1, 4, dtype=torch.long), [], attn_mask=torch.ones(4, 4))


def test_load_backbone_state_accepts_transformer_named_weights() -> None:
    source = _backbone().make()
    graft = _config().make()
    graft.load_backbone_state(DictCodec.coerce(source.state_dict(), Tensor))
    loaded = DictCodec.coerce(graft.state_dict(), Tensor)
    expected = DictCodec.coerce(source.state_dict(), Tensor)
    assert torch.equal(loaded["proj_in.weight"], expected["proj_in.weight"])


def test_head_is_tied_reads_through_a_sequential_head() -> None:
    assert head_is_tied(_backbone(tie=True))
    assert not head_is_tied(_backbone(tie=False))
    bare = _backbone()
    bare.proj_out = TiedLinear.Config(tied="proj_in")
    assert head_is_tied(bare)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
