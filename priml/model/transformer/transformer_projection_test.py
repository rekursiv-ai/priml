"""Optional input/output projections and attention-owned causality."""

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.self_attention import SelfAttention
from priml.model.embedding import Embedding
from priml.model.generate import generate
from priml.model.linear import Linear
from priml.model.special import TiedLinear
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import randomize_parameters


def _config() -> Transformer.Config:
    config = Transformer.Config()
    config.channels_in = 8
    config.num_layers = 1
    assert isinstance(config.block, TransformerBlock.Config)
    config.block.attn = SelfAttention.Config()
    config.block.attn.num_heads = 2
    config.block.attn.attn_kernel = SdpaNaive.Config()
    return config


def test_optional_projections_accept_hidden_states() -> None:
    config = _config()
    assert config.proj_in is None
    assert config.proj_out is None
    model = config.make()
    hidden = torch.randn(2, 3, 8)
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    assert torch.equal(model(hidden), block(hidden))


def test_configurable_linear_projections() -> None:
    config = _config()
    config.channels_in = 4
    config.channels_out = 3
    config.proj_in = Linear.Config(channels_out=8)
    config.proj_out = Linear.Config()
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.proj_in, Linear.Config)
    assert isinstance(resolved.proj_out, Linear.Config)
    assert resolved.proj_in.channels_in == 4
    assert resolved.proj_in.channels_out == 8
    assert resolved.proj_out.channels_in == 8
    assert resolved.proj_out.channels_out == 3
    model = config.make()
    assert model(torch.randn(2, 3, 4)).shape == (2, 3, 3)
    assert isinstance(model.proj_out, Linear)
    assert model.proj_out.in_features == 8
    assert config.proj_in.channels_in == -1
    assert config.proj_out.channels_in == -1


def test_output_width_is_inferred_from_projection() -> None:
    config = _config()
    config.proj_out = Linear.Config(channels_out=3)
    resolved = config.copy_tree().finalize()
    assert resolved.channels_out == 3
    assert config.channels_out == -1
    assert config.make()(torch.randn(2, 3, 8)).shape == (2, 3, 3)


def test_generation_requires_token_input_projection() -> None:
    model = _config().make()
    with pytest.raises(TypeError, match=r"proj_in.*embedding"):
        generate(model, torch.tensor([[1, 2]]))


def test_tied_projection_reuses_input_weight() -> None:
    config = _config()
    config.proj_in = Embedding.Config(channels_in=17)
    config.proj_out = TiedLinear.Config(tied="proj_in")
    config.channels_out = 17
    resolved = config.copy_tree().finalize()
    assert resolved.channels_in == 8
    assert isinstance(resolved.proj_out, TiedLinear.Config)
    assert resolved.proj_out.channels_in == 8
    assert resolved.proj_out.channels_out == 17
    model = config.make()
    assert isinstance(model.proj_in, Embedding)
    tokens = torch.tensor([[1, 2, 3]])
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    hidden = block(model.proj_in(tokens))
    assert torch.equal(model(tokens), hidden @ model.proj_in.weight.T)
    assert "proj_in.weight" in model.state_dict()
    assert not any(key.startswith("proj_out.") for key in model.state_dict())
    model(tokens).square().sum().backward()
    assert model.proj_in.weight.grad is not None
    assert torch.count_nonzero(model.proj_in.weight.grad) > 0


def test_tied_projection_requires_input_weight() -> None:
    config = _config()
    config.proj_out = TiedLinear.Config(tied="proj_in")
    with pytest.raises(ValueError, match=r"tied='proj_in'"):
        config.make()


def test_layer_count_is_inferred_from_explicit_blocks() -> None:
    config = _config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block.copy_tree(), config.block.copy_tree()]
    config.num_layers = -1
    resolved = config.copy_tree().finalize()
    assert resolved.num_layers == 2
    assert config.num_layers == -1
    model = config.make()
    assert len(model.blocks) == 2
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
    depth_indices = [
        block.depth_index
        for block in model.blocks
        if isinstance(block, TransformerBlock)
    ]
    assert depth_indices == [((0, 2),), ((1, 2),)]


@pytest.mark.parametrize("causal", [False, True])
def test_attention_owns_causality(causal: bool) -> None:
    config = _config()
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.causal = causal
    model = config.make().eval()
    randomize_parameters(model, seed=7, std=0.2)
    hidden = torch.randn(1, 3, 8)
    changed = hidden.clone()
    changed[:, -1] += torch.arange(8)
    before, after = model(hidden), model(changed)
    assert torch.equal(before[:, 0], after[:, 0]) == causal


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
