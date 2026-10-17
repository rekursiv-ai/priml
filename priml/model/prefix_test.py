"""Tests for puzzle prefix tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

import pytest
import torch

from priml.model.conditioning import (
    CategorySlotFiLM,
    SigmoidResidualGate,
    WeightedProjectionConfig,
)
from priml.model.prefix import (
    ComposedSparsePrefix,
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.model.special import Identity


if TYPE_CHECKING:
    from priml.model.prefix import PrefixConfig
from priml.cost import cost
from priml.model.norm import RMSNorm
from priml.testing.cost import assert_cost_matches_torch


def _sparse(**overrides: object) -> SparsePuzzleEmbedding:
    config = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    config.channels_out = 8
    for name, value in overrides.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    return config.make()


def test_register_tokens_are_shared_across_the_batch() -> None:
    config = RegisterTokens.Config(num_tokens=3)
    config.channels_out = 8
    torch.manual_seed(0)
    tokens = config.make()(4)
    assert tokens.shape == (4, 3, 8)
    assert torch.equal(tokens[0], tokens[3])


def test_register_tokens_can_be_frozen() -> None:
    """A fixed random basis is a legitimate choice, so it must not train."""
    config = RegisterTokens.Config(num_tokens=2, learnable=False)
    config.channels_out = 8
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
    # The per-puzzle vector is ``channels_out`` wide (8) and the prefix is 2 tokens
    # of 8, so it occupies the first token and the pad fills the second.
    assert torch.allclose(out[0, 0], module.weights[2] * module.embed_scale)
    assert bool((out[0, 1] == 0).all())
    # Every row asked for the same puzzle, so every row got the same vector.
    assert torch.equal(out[0], out[3])


@pytest.mark.parametrize("training", [True, False])
def test_sparse_embedding_scales_after_the_configured_upcast(training: bool) -> None:
    """Round rows to bf16, then upcast before multiplying by the scale."""
    config = SparsePuzzleEmbedding.Config(
        num_puzzles=8,
        num_tokens=2,
        batch_size=4,
        dtype=torch.bfloat16,
        dtype_scale=torch.float32,
    )
    config.channels_out = 512
    module = config.make()
    with torch.no_grad():
        module.weights[0, 0] = 0.3
    module.train(training)

    out = module(4, puzzle_identifiers=torch.zeros(4, dtype=torch.int32))

    assert out.dtype == torch.float32
    expected = torch.tensor(0.3).to(torch.bfloat16).to(torch.float32) * (512**0.5)
    assert torch.equal(out[0, 0, 0], expected)
    assert not torch.equal(out[0, 0, 0], expected.to(torch.bfloat16).float())


def test_gradient_buffer_survives_a_device_move() -> None:
    """``_apply`` replaces buffers with copies that drop ``requires_grad``.

    Without the override the table silently stops training the moment the
    model moves to a device -- the failure has no error, only a frozen table.
    """
    module = _sparse()
    module.to(torch.float64)
    assert module.local_weights.requires_grad


def test_sparse_embedding_reset_clears_grad_buffer_without_breaking_autograd() -> None:
    module = _sparse(init_std=0.0)
    module.train()
    module(4, puzzle_identifiers=torch.arange(4, dtype=torch.int32)).sum().backward()
    assert module.local_weights.grad is not None

    module.reset_parameters()

    assert bool((module.weights == 0).all())
    assert bool((module.local_weights == 0).all())
    assert bool((module.local_ids == 0).all())
    assert module.local_weights.requires_grad


def test_a_vector_narrower_than_the_model_pads_out_to_model_width_tokens() -> None:
    """``channels_out`` sizes the stored row; ``channels_in`` sizes each token.

    The row is zero-padded to ``num_tokens * channels_in`` and scaled like the
    grid tokens it precedes, and the cost prices that padded prefix.
    """
    config = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    config.channels_in = 8
    config.channels_out = 5
    config.init_std = 1.0
    module = config.make()
    module.eval()
    identifiers = torch.tensor([3, 1, 0, 2])
    out = module(4, puzzle_identifiers=identifiers)
    assert out.shape == (4, 2, 8)
    assert torch.equal(out[:, 0, :5], module.weights[identifiers] * 8**0.5)
    assert bool((out[:, 0, 5:] == 0).all())
    assert bool((out[:, 1] == 0).all())
    costed = cost(config, seq_len=1, batch_size=4, dtype=None)
    assert costed["flops", "primal", "elementwise"].sum() == 4 * 2 * 8


def test_a_vector_wider_than_its_tokens_is_rejected_at_construction() -> None:
    """The reshape into tokens cannot hold it; fail where the field is named."""
    config = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    config.channels_in = 4
    config.channels_out = 9
    with pytest.raises(ValueError, match=r"channels_out=9 exceeds num_tokens"):
        config.make()


def test_a_training_batch_must_fill_the_gradient_buffer() -> None:
    """A short batch would broadcast into, or fail to fill, the gradient buffer."""
    module = _sparse()
    module.train()
    with pytest.raises(ValueError, match=r"batch_size=4"):
        module(2, puzzle_identifiers=torch.arange(2))
    module.eval()
    assert module(2, puzzle_identifiers=torch.arange(2)).shape == (2, 2, 8)


def test_missing_identifiers_is_rejected() -> None:
    module = _sparse()
    with pytest.raises(TypeError, match="requires puzzle_identifiers"):
        module(4)


def test_stack_concatenates_in_order() -> None:
    """The halt readout reads position 0, so order decides which part owns it."""
    puzzle = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    puzzle.channels_out = 8
    registers = RegisterTokens.Config(num_tokens=3)
    registers.channels_out = 8
    config = PrefixStack.Config()
    config.parts = [puzzle, registers]
    torch.manual_seed(0)
    out = config.make()(4, puzzle_identifiers=torch.arange(4, dtype=torch.int32))
    assert out.shape == (4, 5, 8)


def test_channels_must_be_inherited() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RegisterTokens.Config().make()


def test_register_tokens_cost_is_one_scale_over_the_owned_tokens() -> None:
    """Cost one scale over every register token in the complete batch."""
    analytical = assert_cost_matches_torch(
        RegisterTokens.Config(num_tokens=2, channels_out=8),
        build_input=lambda: torch.arange(4, dtype=torch.int32),
        seq_len=1,
        batch_size=4,
        dtype=None,
        run=_run_prefix,
    )
    assert analytical.params == analytical.params_active == 2 * 8
    assert analytical["flops", "primal", "elementwise"].sum() == 4 * 2 * 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 4 * 2 * 8
    # The gradient is summed back over the four puzzles sharing the tokens.
    assert analytical["flops", "adjoint", "reduction"].sum() == 2 * 8 * (4 - 1)
    frozen = cost(
        RegisterTokens.Config(num_tokens=2, channels_out=8, learnable=False),
        seq_len=1,
        batch_size=1,
        dtype=None,
    )
    assert frozen.params == frozen.params_active == 0


def test_sparse_embedding_cost_owns_no_parameters() -> None:
    """The table is a buffer: one row gathered, padded, scaled; nothing to own.

    The scatter back into the master table is the sparse optimizer's, which
    the counting policy excludes, so the adjoint holds only the scale.
    """
    config = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    config.channels_out = 8
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.arange(4, dtype=torch.int32),
        seq_len=1,
        batch_size=4,
        dtype=None,
        run=_run_prefix,
    )
    assert analytical.params == analytical.params_active == 0
    assert analytical["bytes", "primal", "selection", torch.int64] == 4 * 3 * 8
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * 4 * (
        4 * 8 + 8 + 16
    )
    assert analytical["flops", "adjoint", "selection"].sum() == 0
    assert analytical["flops", "primal", "elementwise"].sum() == 4 * 2 * 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 4 * 2 * 8


def test_stack_cost_sums_its_parts() -> None:
    puzzle = SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4)
    puzzle.channels_out = 8
    registers = RegisterTokens.Config(num_tokens=3)
    registers.channels_out = 8
    config = PrefixStack.Config()
    config.channels_out = 8
    config.parts = [puzzle, registers]
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.arange(4, dtype=torch.int32),
        seq_len=1,
        batch_size=4,
        dtype=None,
        run=_run_prefix,
    )
    children = cost(puzzle, seq_len=1, batch_size=4, dtype=None) + cost(
        registers,
        seq_len=1,
        batch_size=4,
        dtype=None,
    )
    assert analytical["flops", "primal"] == children["flops", "primal"]
    assert analytical["flops", "adjoint"] == children["flops", "adjoint"]
    assert analytical["bytes", "adjoint"] == children["bytes", "adjoint"]
    assert analytical.params == children.params
    assert (
        analytical["bytes", "primal", "selection"].sum()
        - children["bytes", "primal", "selection"].sum()
        == 4 * 4 * 2 * 5 * 8
    )


def test_a_stack_counts_its_tokens_from_its_parts() -> None:
    """A stack is itself a prefix: its token count is the sum of its parts'.

    Nested stacks compose the same way, so the model that reads
    ``num_tokens`` never has to know a stack from a leaf.
    """
    inner = PrefixStack.Config()
    inner.parts = [RegisterTokens.Config(num_tokens=3)]
    outer = PrefixStack.Config()
    outer.parts = [
        SparsePuzzleEmbedding.Config(num_puzzles=8, num_tokens=2, batch_size=4),
        inner,
    ]
    assert inner.num_tokens == 3
    assert outer.num_tokens == 5


def test_a_stack_part_without_a_token_count_cannot_be_priced() -> None:
    """A stacked module that declares no tokens gives the concatenation no size."""
    config = PrefixStack.Config()
    config.parts = [cast("PrefixConfig", RMSNorm.Config(channels_in=8))]
    with pytest.raises(AttributeError, match="num_tokens"):
        cost(config, seq_len=1, batch_size=1, dtype=None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_sparse_prefix_traffic_counts_lookup_copy_padding_and_scale(
    dtype: torch.dtype,
) -> None:
    config = SparsePuzzleEmbedding.Config()
    config.channels_out = 8
    config.num_tokens = 2
    costed = config.cost(seq_len=1, batch_size=1, dtype=dtype)
    itemsize = dtype.itemsize
    # Three int64 bookkeeping reads; the row, pad, and prefix at ``dtype``.
    assert costed["bytes", "primal", "selection", torch.int64] == 8 * 3
    assert costed["bytes", "primal", "selection", dtype] == itemsize * (4 * 8 + 8 + 16)
    assert costed["bytes", "primal", "elementwise"].sum() == itemsize * 2 * 16
    assert costed["bytes", "adjoint", "elementwise"].sum() == itemsize * 2 * 16
    config.dtype_scale = torch.float32
    scaled = config.cost(seq_len=1, batch_size=1, dtype=torch.bfloat16)
    assert scaled["bytes", "primal", "elementwise"].sum() == 4 * 2 * 16
    assert scaled["bytes", "adjoint", "elementwise"].sum() == 4 * 2 * 16


def _run_prefix(module: nn.Module, identifiers: Tensor) -> Tensor:
    """Call a prefix module as the model does: the row count plus the batch's ids."""
    out = cast(object, module(identifiers.shape[0], puzzle_identifiers=identifiers))
    assert isinstance(out, Tensor)
    return out


def test_register_tokens_reset_replays_the_constructor_initializer() -> None:
    config = RegisterTokens.Config(num_tokens=2, channels_out=4)
    torch.manual_seed(17)
    module = config.make()
    expected = module.register_tokens.detach().clone()
    torch.manual_seed(23)
    module.reset_parameters()
    actual_rng = torch.get_rng_state()

    torch.manual_seed(23)
    expected_module = config.make()
    expected_reset_rng = torch.get_rng_state()

    assert torch.equal(module.register_tokens, expected_module.register_tokens)
    assert actual_rng.equal(expected_reset_rng)
    assert not torch.equal(module.register_tokens, expected)


def test_prefix_stack_reset_materializes_and_resets_its_parts() -> None:
    config = PrefixStack.Config(
        parts=[RegisterTokens.Config(num_tokens=2, channels_out=4)],
        channels_out=4,
    )
    module = config.make().to("meta")
    registers = module.parts[0]
    assert isinstance(registers, RegisterTokens)
    module.to_empty(device="cpu")
    with torch.no_grad():
        registers.register_tokens.fill_(float("nan"))

    module.reset_parameters()

    assert torch.isfinite(registers.register_tokens).all()


def test_prefix_stack_finalizes_child_widths_and_reports_token_count() -> None:
    config = PrefixStack.Config(
        parts=[RegisterTokens.Config(num_tokens=2)],
        channels_out=4,
    )
    final = config.copy_tree().finalize()
    first_part = final.parts[0]
    assert isinstance(first_part, RegisterTokens.Config)
    assert first_part.channels_out == 4
    assert final.num_tokens == 2


@pytest.mark.parametrize("residual_gate", [True, False])
def test_composed_sparse_prefix_builds_padded_prefix(residual_gate: bool) -> None:
    config = _composed_config(residual_gate=residual_gate)
    module = config.make()
    module.eval()
    output = module(
        5,
        category_indices=torch.tensor([0, 1, 2, 0, 1]),
        augmentation_indices=torch.tensor([1, 0, 1, 0, 1]),
        slot_permutation=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
        puzzle_identifiers=torch.tensor([1, 2, 3, 4, 5]),
    )
    assert output.shape == (5, 3, 4)
    assert output.dtype is torch.bfloat16
    assert torch.equal(output[:, 1:], torch.zeros_like(output[:, 1:]))


def test_composed_sparse_prefix_reset_reinitializes_each_owned_child() -> None:
    module = _composed_config(residual_gate=True).make()
    assert isinstance(module.composition, CategorySlotFiLM)
    assert isinstance(module.residual_gate, SigmoidResidualGate)
    with torch.no_grad():
        module.composition.category_embed.weight.fill_(1)
        module.residual.weights.fill_(1)
        assert module.residual_gate is not None
        module.residual_gate.linear.weight.fill_(1)

    module.reset_parameters()

    assert torch.equal(
        module.composition.category_embed.weight,
        torch.zeros_like(module.composition.category_embed.weight),
    )
    assert torch.equal(
        module.residual.weights,
        torch.zeros_like(module.residual.weights),
    )
    assert module.residual_gate is not None
    assert torch.equal(
        module.residual_gate.linear.weight,
        torch.zeros_like(module.residual_gate.linear.weight),
    )


def test_composed_sparse_prefix_rejects_invalid_construction() -> None:
    config = _composed_config(residual_gate=False)
    config.channels_out = -1
    with pytest.raises(ValueError, match="positive channels_out"):
        config.make()

    config = _composed_config(residual_gate=False)
    config.num_tokens = 0
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        config.make()

    config = _composed_config(residual_gate=False)
    config.residual_projection = cast(
        WeightedProjectionConfig,
        Identity.Config(channels_in=4, channels_out=4),
    )
    with pytest.raises(TypeError, match="must expose a weight tensor"):
        config.make()

    assert ComposedSparsePrefix.Config(num_tokens=5).num_prefix_tokens == 5


def test_composed_sparse_prefix_requires_tensor_task_ids() -> None:
    module = _composed_config(residual_gate=False).make()
    with pytest.raises(TypeError, match="category_indices"):
        module(5, category_indices=[0, 1, 2, 0, 1])


def _composed_config(*, residual_gate: bool) -> ComposedSparsePrefix.Config:
    config = ComposedSparsePrefix.Config(channels_out=4, num_tokens=3)
    config.composition = CategorySlotFiLM.Config(
        num_categories=3,
        num_augmentations=2,
        num_slots=2,
        channels_out=4,
    )
    config.residual.num_puzzles = 7
    config.residual.batch_size = 5
    config.residual.channels_out = 4
    if not residual_gate:
        config.residual_gate = None
    return config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
