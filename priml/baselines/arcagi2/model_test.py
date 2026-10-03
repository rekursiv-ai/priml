"""Exact source checks for the ARC2 reference recipe's reusable components.

The model this recipe was ported from was recorded once through
:func:`record_model`; the port must reproduce every recorded tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.arcagi2.model import (
    ArcModelConfig,
    PuzzleEmbedding,
    RotaryBlock,
)
from priml.baselines.arcagi2.record_test import assert_matches, load, reduce
from priml.baselines.arcagi2.train_step_test import training_config
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import SudokuNet
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import mismatches, rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable


def record_token_init(build: Callable[[], Tensor]) -> dict[str, Tensor]:
    """Record the first initialized tensor, the token embedding, from seed 0."""
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        return reduce({"embed_tokens": build().detach()})


def record_model(
    build: Callable[[], nn.Module],
    *,
    latents: Callable[[nn.Module], tuple[Tensor, Tensor]],
    forward: Callable[[nn.Module, Tensor, Tensor], Tensor],
) -> dict[str, Tensor]:
    """Record init (parameters in order, latent inits, RNG fingerprint), a forward.

    Args:
      build: Constructs the model; called seeded, under portable numerics.
      latents: Returns the model's slow and fast latent inits.
      forward: Returns the forward logits for ``(model, tokens, identifiers)``.

    Returns:
      record: Name-to-tensor record.

    """
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        model = build()
        out: dict[str, Tensor] = {"rng": rng_fingerprint()}
        for index, parameter in enumerate(model.parameters()):
            out[f"param/{index}"] = parameter.detach().clone()
        out["slow_init"], out["fast_init"] = latents(model)
        tokens = torch.arange(6).reshape(2, 3) % 2 + 2
        out["logits"] = forward(model, tokens, torch.tensor([0, 1]))
    return reduce(out)


def port_token_init() -> Tensor:
    """Build the port's grid embedding sized like the reference and return it."""
    candidate = GridEmbedding.Config(channels_in=4, channels_out=4, grid_shape=(3,))
    return candidate.make().embed_tokens.weight


def port_latents(model: nn.Module) -> tuple[Tensor, Tensor]:
    """Slow and fast latent inits of the port."""
    assert isinstance(model, SudokuNet)
    return model.slow_init, model.fast_init


def port_forward(model: nn.Module, tokens: Tensor, identifiers: Tensor) -> Tensor:
    """Forward logits of the port."""
    assert isinstance(model, SudokuNet)
    return model(tokens, puzzle_identifiers=identifiers).logits


def port_model() -> ArcModelConfig:
    """Return the port's miniature model."""
    candidate = training_config(4, torch.bfloat16).model
    assert isinstance(candidate, ArcModelConfig)
    return candidate


def test_unfilled_arc_model_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        ArcModelConfig(channels_in=4, num_layers=1).make()


def run_port_model() -> dict[str, Tensor]:
    """Record the port's miniature model."""
    return record_model(port_model().make, latents=port_latents, forward=port_forward)


def test_reference_token_initialization() -> None:
    """Match the first initialized tensor before comparing later checkpoints."""
    assert_matches("model", "token_init", record_token_init(port_token_init))


def test_reference_model_initialization_and_forward() -> None:
    """Every initialized parameter, latent init, RNG byte, and the logits match."""
    assert_matches("model", "model", run_port_model())


def test_reference_model_bites() -> None:
    """A different init seed is reported, not absorbed."""
    candidate = port_model()

    def build() -> nn.Module:
        torch.default_generator.manual_seed(1)
        return candidate.make()

    record = record_model(build, latents=port_latents, forward=port_forward)
    assert mismatches(load("model")["model"], record)


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
        assert isinstance(module, SudokuNet)
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
    from priml.lib.testing.main import test_main

    test_main(__file__)
