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
    assert config.in_proj is None
    assert config.out_proj is None
    model = config.make()
    hidden = torch.randn(2, 3, 8)
    assert torch.equal(model(hidden), model.blocks[0](hidden))


def test_configurable_linear_projections() -> None:
    config = _config()
    config.channels_in = 4
    config.channels_out = 3
    config.in_proj = Linear.Config(channels_out=8)
    config.out_proj = Linear.Config()
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.in_proj, Linear.Config)
    assert isinstance(resolved.out_proj, Linear.Config)
    assert resolved.in_proj.channels_in == 4
    assert resolved.in_proj.channels_out == 8
    assert resolved.out_proj.channels_in == 8
    assert resolved.out_proj.channels_out == 3
    model = config.make()
    assert model(torch.randn(2, 3, 4)).shape == (2, 3, 3)
    assert isinstance(model.out_proj, Linear)
    assert model.out_proj.in_features == 8
    assert config.in_proj.channels_in == -1
    assert config.out_proj.channels_in == -1


def test_output_width_is_inferred_from_projection() -> None:
    config = _config()
    config.out_proj = Linear.Config(channels_out=3)
    resolved = config.copy_tree().finalize()
    assert resolved.channels_out == 3
    assert config.channels_out == -1
    assert config.make()(torch.randn(2, 3, 8)).shape == (2, 3, 3)


def test_generation_requires_token_input_projection() -> None:
    model = _config().make()
    with pytest.raises(TypeError, match=r"in_proj.*embedding"):
        generate(model, torch.tensor([[1, 2]]))


def test_tied_projection_reuses_input_weight() -> None:
    config = _config()
    config.in_proj = Embedding.Config(num_embeddings=17)
    config.out_proj = TiedLinear.Config(tied="in_proj")
    config.channels_out = 17
    resolved = config.copy_tree().finalize()
    assert resolved.channels_in == 8
    assert isinstance(resolved.out_proj, TiedLinear.Config)
    assert resolved.out_proj.channels_in == 8
    assert resolved.out_proj.channels_out == 17
    model = config.make()
    assert isinstance(model.in_proj, Embedding)
    tokens = torch.tensor([[1, 2, 3]])
    hidden = model.blocks[0](model.in_proj(tokens))
    assert torch.equal(model(tokens), hidden @ model.in_proj.weight.T)
    assert "in_proj.weight" in model.state_dict()
    assert not any(key.startswith("out_proj.") for key in model.state_dict())
    model(tokens).square().sum().backward()
    assert model.in_proj.weight.grad is not None
    assert torch.count_nonzero(model.in_proj.weight.grad) > 0


def test_tied_projection_requires_input_weight() -> None:
    config = _config()
    config.out_proj = TiedLinear.Config(tied="in_proj")
    with pytest.raises(ValueError, match=r"tied='in_proj'"):
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
    assert [block.depth_index for block in model.blocks] == [((0, 2),), ((1, 2),)]


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
