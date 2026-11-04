"""Exact source checks for the ARC2 reference recipe's reusable components.

The source's recorded weights and latent buffers are replay inputs; its
recorded logits remain the immutable forward oracle.
"""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi.parity.record_test import load
from priml.baselines.arcagi.parity.train_step import ArcModelConfig
from priml.baselines.arcagi.parity.train_step_test import training_config
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.embedding import Embedding
from priml.model.grid_embedding import GridEmbedding
from priml.model.grid_solver import GridSolver
from priml.model.init import corrected_fan_in_normal
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import mismatches


def reference_forward(*, perturb: bool = False) -> Tensor:
    """Replay source logits from the source's stored parameters and latent buffers."""
    expected = load("model")["model"]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        model = port_model().make()
        parameters = [
            expected[f"param/{index}"]
            for index in range(sum(key.startswith("param/") for key in expected))
        ]
        with torch.no_grad():
            for parameter, saved in zip(model.parameters(), parameters, strict=True):
                assert parameter.shape == saved.shape
                assert parameter.dtype == saved.dtype
                parameter.copy_(saved)
            model.slow_init.copy_(expected["slow_init"])
            model.fast_init.copy_(expected["fast_init"])
            if perturb:
                next(model.parameters()).add_(0.25)
        tokens = torch.arange(6).reshape(2, 3) % 2 + 2
        return model(tokens, puzzle_identifiers=torch.tensor([0, 1])).logits


def port_token_init() -> Tensor:
    """Build the port's grid embedding sized like the reference and return it."""
    candidate = GridEmbedding.Config(channels_in=4, channels_out=4, grid_shape=(3,))
    return candidate.make().embed_tokens.weight


def port_model() -> ArcModelConfig:
    """Return the port's miniature model."""
    candidate = training_config(4, torch.bfloat16).model
    assert isinstance(candidate, ArcModelConfig)
    return candidate


def test_unfilled_arc_model_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        ArcModelConfig(channels_in=4, num_layers=1).make()


def test_reference_token_initialization() -> None:
    """Keep the corrected embedding law without pinning a historical sampler."""
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        actual = port_token_init()
        torch.manual_seed(0)
        expected = Embedding.Config(4, 4).make().weight
        corrected_fan_in_normal(expected, depth_index=())
    assert torch.equal(actual, expected)


def test_reference_model_initialization_and_forward() -> None:
    """Source weights and latent buffers reproduce the immutable source logits."""
    assert torch.equal(reference_forward(), load("model")["model"]["logits"])


def test_reference_model_bites() -> None:
    """Changing an input weight changes the source forward comparison."""
    assert mismatches(
        {"logits": load("model")["model"]["logits"]},
        {"logits": reference_forward(perturb=True)},
    )


def test_full_model_cost_matches_torch() -> None:
    """Count the recurrent rotary model and sparse prefix through the common harness."""
    config = port_model()
    assert isinstance(config.embedding, GridEmbedding.Config)
    assert isinstance(config.block, RotaryBlock.Config)
    assert isinstance(config.block.attn, Attention.Config)
    assert isinstance(config.prefix, PuzzleEmbedding.Config)
    config.embedding.grid_shape = (3,)
    config.prefix.num_tokens = 2
    config.block.attn.attn_kernel = SdpaNaive.Config()

    def run(module: nn.Module, tokens: Tensor) -> Tensor:
        assert isinstance(module, GridSolver)
        output = module(tokens, puzzle_identifiers=torch.tensor([0, 1]))
        return output.logits.sum() + output.halt.sum()

    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.zeros(2, 3, dtype=torch.int32),
        batch_size=2,
        dtype=None,
        run=run,
    )


def test_rotary_block_factor_branches() -> None:
    config = RotaryBlock.Config(channels_in=4)
    assert isinstance(config.attn, Attention.Config)
    config.attn.channels_in = 4
    config.attn.num_heads = 2
    config.attn.channels_head = 2
    assert config.rope is not None
    config.rope.channels_head = 2
    block = config.make()
    factors = block.factors(3, device=torch.device("cpu"))
    assert factors is not None
    assert factors[0].shape[0] == 3
    assert block(torch.zeros(2, 3, 4)).shape == (2, 3, 4)
    no_rope = RotaryBlock.Config(channels_in=4, rope=None).make()
    assert no_rope.factors(3, device=torch.device("cpu")) is None
    with pytest.raises(ValueError, match="without a rope"):
        no_rope(torch.zeros(2, 3, 4))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
