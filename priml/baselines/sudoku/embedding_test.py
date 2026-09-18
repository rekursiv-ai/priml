"""Tests for the sudoku input-embedding channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridChannel,
    GridEmbedding,
    PredictionFeedback,
)
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from configgle import Makeable


def _embedding(*channels: Makeable[GridChannel]) -> GridEmbedding:
    config = GridEmbedding.Config(channels_in=11, channels_out=8)
    config.channels = list(channels)
    torch.manual_seed(0)
    return config.make()


def test_plain_embedding_is_token_lookup_only() -> None:
    embedding = _embedding()
    tokens = torch.randint(0, 11, (2, 81))
    assert embedding(tokens).shape == (2, 81, 8)


def test_channels_inherit_the_models_width() -> None:
    """A channel left at its sentinel is sized by the parent, not by hand."""
    config = GridEmbedding.Config(channels_in=11, channels_out=8)
    config.channels = [FactoredPositions.Config(), PredictionFeedback.Config()]
    final = config.copy_tree().finalize()
    positions, feedback = final.channels
    assert isinstance(positions, FactoredPositions.Config)
    assert isinstance(feedback, PredictionFeedback.Config)
    assert positions.channels_out == 8
    assert feedback.channels_out == 8
    # The vocabulary reaches the one channel that is sized by it.
    assert feedback.channels_in == config.channels_in


def test_factored_positions_share_a_row() -> None:
    """Cells in one row get the same row component, which is the point."""
    embedding = _embedding(FactoredPositions.Config())
    positions = embedding.channels[0]
    assert isinstance(positions, FactoredPositions)
    # Cells 0..8 are row 0; cells 9..17 are row 1.
    assert set(positions.row_index[:9].tolist()) == {0}
    assert set(positions.row_index[9:18].tolist()) == {1}
    # Column indices cycle within a row.
    assert positions.col_index[:9].tolist() == list(range(9))
    # The first three cells of the first three rows are one box.
    box = positions.box_index.reshape(9, 9)
    assert set(box[:3, :3].flatten().tolist()) == {0}


def test_box_shape_must_tile_the_grid() -> None:
    config = FactoredPositions.Config(grid_shape=(9, 9), box_shape=(4, 4))
    config.channels_out = 8
    with pytest.raises(ValueError, match="does not tile"):
        config.make()


def test_feedback_is_inert_until_stashed_and_consumed_once() -> None:
    """A zero-initialized table cannot change the forward, and the stash clears.

    Both halves matter: inertness makes an A/B against the no-feedback model
    start from identical weights, and consume-once stops a stale grid from
    leaking into a later step.
    """
    embedding = _embedding(PredictionFeedback.Config())
    feedback = embedding.channels[0]
    assert isinstance(feedback, PredictionFeedback)
    tokens = torch.randint(0, 11, (2, 81))
    base = embedding(tokens)

    feedback.set_feedback(torch.randint(0, 11, (2, 81)))
    assert torch.equal(embedding(tokens), base)  # Zero table: no contribution.
    assert feedback._feedback_ids is None

    # A trained table does contribute, and only for the step it was stashed on.
    with torch.no_grad():
        feedback.embed_feedback.fill_(0.5)
    feedback.set_feedback(torch.randint(0, 11, (2, 81)))
    assert not torch.equal(embedding(tokens), base)
    assert torch.equal(embedding(tokens), base)  # `stash` already consumed.


def test_channel_order_changes_the_result() -> None:
    """Order is a numerics contract, so the test states it rather than assumes."""
    tokens = torch.randint(0, 11, (2, 81))
    # Same channels, different draw order at init: the tables differ, so the
    # sum differs. Equality here would mean the order was not observable.
    assert not torch.equal(
        _ordered(tokens, swap=False),
        _ordered(tokens, swap=True),
    )


def test_channels_must_be_inherited_or_set() -> None:
    with pytest.raises(ValueError, match="channels_in must be positive"):
        GridEmbedding.Config().make()
    with pytest.raises(ValueError, match="channels_out must be positive"):
        GridEmbedding.Config(channels_in=11).make()


def test_factored_positions_cost_is_three_gathers_two_adds_and_a_scale() -> None:
    """Per cell: one row read from each table, summed, then scaled; no products."""
    analytical = assert_cost_matches_torch(
        FactoredPositions.Config(grid_shape=(4, 4), box_shape=(2, 2), channels_out=8),
        build_input=lambda: (
            torch.zeros(1, 16, dtype=torch.long),
            torch.zeros(1, 16, 8),
        ),
        seq_len=16,
        batch_size=1,
        num_tokens=16,
        dtype=None,
    )
    assert analytical.params == (4 + 4 + 4) * 8
    assert analytical.params_active == 3 * 8
    # Three gathers: an int64 index each, a row read and written at fp32.
    assert analytical["bytes", "primal", "selection", torch.int64] == 8 * 3
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * 3 * 2 * 8
    assert analytical["flops", "adjoint", "selection"].sum() == 3 * 8
    assert analytical["flops", "primal", "elementwise"].sum() == 3 * 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 8


def test_prediction_feedback_cost_is_a_gather_and_a_scale() -> None:
    """Per cell with a grid stashed: one table row read, scaled; the whole table owned."""
    analytical = assert_cost_matches_torch(
        PredictionFeedback.Config(channels_in=11, channels_out=8),
        build_input=lambda: (torch.randint(0, 11, (1, 16)), torch.zeros(1, 16, 8)),
        seq_len=16,
        batch_size=1,
        num_tokens=16,
        dtype=None,
        run=_run_feedback,
    )
    assert analytical.params == 11 * 8
    assert analytical.params_active == 8
    assert analytical["bytes", "primal", "selection", torch.int64] == 8
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * 2 * 8
    assert analytical["flops", "adjoint", "selection"].sum() == 8
    assert analytical["flops", "primal", "elementwise"].sum() == 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 8


def test_grid_embedding_cost_sums_the_token_table_and_every_channel() -> None:
    """The token gather and its scale, plus each channel and one add per channel."""
    config = GridEmbedding.Config(channels_in=11, channels_out=8)
    config.channels = [FactoredPositions.Config(), PredictionFeedback.Config()]
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (1, 81)),
        seq_len=81,
        batch_size=1,
        num_tokens=81,
        dtype=None,
        run=_run_with_feedback,
    )
    assert analytical.params == 11 * 8 + (9 + 9 + 9) * 8 + 11 * 8
    assert analytical.params_active == 8 + 3 * 8 + 8
    # Token row, three position rows, one feedback row.
    assert analytical["bytes", "primal", "selection", torch.int64] == 8 * 5
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * 5 * 2 * 8
    assert analytical["flops", "adjoint", "selection"].sum() == 8 + 3 * 8 + 8
    # Token scale; positions' two adds and scale; feedback scale; two channel adds.
    assert analytical["flops", "primal", "elementwise"].sum() == 8 + 3 * 8 + 8 + 2 * 8
    # The channel adds accumulate no gradient, so only the three scales pull back.
    assert analytical["flops", "adjoint", "elementwise"].sum() == 8 + 8 + 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_grid_embedding_traffic_counts_scale_and_channel_add(
    dtype: torch.dtype,
) -> None:
    config = GridEmbedding.Config()
    config.channels_in = 11
    config.channels_out = 8
    config.channels = [PredictionFeedback.Config()]
    costed = config.finalize().cost(seq_len=1, batch_size=1, dtype=dtype)
    itemsize = dtype.itemsize
    assert costed["bytes", "primal", "elementwise"].sum() == itemsize * 8 * (2 + 2 + 3)
    assert costed["bytes", "adjoint", "elementwise"].sum() == itemsize * 8 * (2 + 2)


def test_feedback_table_gradient_zeroing_amortizes_over_batch() -> None:
    config = PredictionFeedback.Config()
    config.channels_in = 11
    config.channels_out = 8
    small = config.cost(seq_len=4, batch_size=1, dtype=None)
    large = config.cost(seq_len=8, batch_size=1, dtype=None)
    assert small["bytes", "adjoint", "selection"].sum() - large[
        "bytes",
        "adjoint",
        "selection",
    ].sum() == 4 * 11 * 8 * (1 / 4 - 1 / 8)


def test_factored_position_cost_amortizes_broadcast_over_puzzles() -> None:
    config = FactoredPositions.Config()
    config.channels_out = 8
    config.grid_shape = (4, 4)
    config.box_shape = (2, 2)
    single = config.cost(seq_len=1, batch_size=1, dtype=torch.bfloat16)
    batch = config.cost(seq_len=1, batch_size=2, dtype=torch.bfloat16)
    assert batch["flops", "primal"] == single["flops", "primal"] / 2
    assert batch["bytes", "primal"] == single["bytes", "primal"] / 2
    assert batch["flops", "adjoint", "reduction"].sum() == 8 / 2
    assert batch["bytes", "adjoint", "reduction"].sum() == 2 * (8 + 8 / 2)
    assert batch.params == single.params


def _run_feedback(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Stash ``tokens`` as the fed-back grid, then run the channel."""
    assert isinstance(module, PredictionFeedback)
    tokens, embeddings = inputs
    module.set_feedback(tokens)
    out = module(tokens, embeddings)
    assert isinstance(out, Tensor)
    return out


def _run_with_feedback(module: nn.Module, tokens: Tensor) -> Tensor:
    """Stash ``tokens`` on every feedback channel, then embed them."""
    assert isinstance(module, GridEmbedding)
    for channel in module.channels:
        if isinstance(channel, PredictionFeedback):
            channel.set_feedback(tokens)
    out = module(tokens)
    assert isinstance(out, Tensor)
    return out


def _ordered(tokens: Tensor, *, swap: bool) -> Tensor:
    """Embed ``tokens`` with the two channels in one order or the other."""
    positions = FactoredPositions.Config()
    feedback = PredictionFeedback.Config(init_std=1.0)
    channels: list[Makeable[GridChannel]] = (
        [feedback, positions] if swap else [positions, feedback]
    )
    config = GridEmbedding.Config(channels_in=11, channels_out=8)
    config.channels = channels
    torch.manual_seed(0)
    embedding = config.make()
    stash = embedding.channels[0 if swap else 1]
    assert isinstance(stash, PredictionFeedback)
    stash.set_feedback(tokens)
    return embedding(tokens)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
