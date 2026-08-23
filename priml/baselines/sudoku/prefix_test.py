"""Tests for puzzle prefix tokens."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)


def _sparse(**overrides: object) -> SparsePuzzleEmbedding:
    config = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    config.hidden_size = 8
    for name, value in overrides.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    return config.make()


def test_register_tokens_are_shared_across_the_batch() -> None:
    config = RegisterTokens.Config(num_tokens=3)
    config.hidden_size = 8
    torch.manual_seed(0)
    tokens = config.make()(4)
    assert tokens.shape == (4, 3, 8)
    assert torch.equal(tokens[0], tokens[3])


def test_register_tokens_can_be_frozen() -> None:
    """A fixed random basis is a legitimate choice, so it must not train."""
    config = RegisterTokens.Config(num_tokens=2, learnable=False)
    config.hidden_size = 8
    module = config.make()
    assert not any(p.requires_grad for p in module.parameters())


def test_sparse_embedding_is_inert_at_zero_init() -> None:
    """A fresh model behaves as though the prefix were absent."""
    module = _sparse()
    module.eval()
    out = module(4, puzzle_identifiers=torch.arange(4, dtype=torch.int32))
    assert out.shape == (4, 2, 8)
    assert bool((out == 0).all())


def test_training_reads_through_the_gradient_buffer() -> None:
    """Only the batch's rows may carry gradient, or the table is undertrainable.

    A dense gradient over an 876k-row table is gigabytes per step; the local
    buffer is what keeps the cost proportional to the batch.
    """
    module = _sparse(init_std=1.0)
    module.train()
    identifiers = torch.tensor([3, 1, 3, 0], dtype=torch.int32)
    out = module(4, puzzle_identifiers=identifiers)
    out.square().mean().backward()
    assert module.local_weights.grad is not None
    assert module.local_weights.grad.shape == (4, 8)
    # The master table stays gradient-free: the optimizer scatters into it.
    assert module.weights.grad is None
    # The ids the optimizer will scatter to were recorded this step.
    assert torch.equal(module.local_ids, identifiers)


def test_eval_reads_the_master_table_directly() -> None:
    """Nothing needs a gradient at eval, so the buffer indirection is skipped."""
    module = _sparse(init_std=1.0)
    module.eval()
    identifiers = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
    out = module(4, puzzle_identifiers=identifiers)
    # The per-puzzle vector is ``channels`` wide (8) and the prefix is 2 tokens
    # of 8, so it occupies the first token and the pad fills the second.
    assert torch.allclose(out[0, 0], module.weights[2] * module.embed_scale)
    assert bool((out[0, 1] == 0).all())
    # Every row asked for the same puzzle, so every row got the same vector.
    assert torch.equal(out[0], out[3])


def test_gradient_buffer_survives_a_device_move() -> None:
    """``_apply`` replaces buffers with copies that drop ``requires_grad``.

    Without the override the table silently stops training the moment the
    model moves to a device -- the failure has no error, only a frozen table.
    """
    module = _sparse()
    module.to(torch.float64)
    assert module.local_weights.requires_grad


def test_missing_identifiers_is_rejected() -> None:
    module = _sparse()
    with pytest.raises(TypeError, match="requires puzzle_identifiers"):
        module(4)


def test_stack_concatenates_in_order() -> None:
    """The halt readout reads position 0, so order decides which part owns it."""
    puzzle = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    puzzle.hidden_size = 8
    registers = RegisterTokens.Config(num_tokens=3)
    registers.hidden_size = 8
    config = PrefixStack.Config()
    config.parts = [puzzle, registers]
    torch.manual_seed(0)
    out = config.make()(4, puzzle_identifiers=torch.arange(4, dtype=torch.int32))
    assert out.shape == (4, 5, 8)


def test_hidden_size_must_be_inherited() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RegisterTokens.Config().make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
