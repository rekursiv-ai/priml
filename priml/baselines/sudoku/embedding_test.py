"""Tests for the sudoku input-embedding channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku.embedding import (
    FactoredPositions,
    GridChannel,
    GridEmbedding,
    PredictionFeedback,
)


if TYPE_CHECKING:
    from configgle import Makeable


def _embedding(*channels: Makeable[GridChannel]) -> GridEmbedding:
    config = GridEmbedding.Config(hidden_size=8)
    config.channels = list(channels)
    torch.manual_seed(0)
    return config.make()


def test_plain_embedding_is_token_lookup_only() -> None:
    embedding = _embedding()
    tokens = torch.randint(0, 11, (2, 81))
    assert embedding(tokens).shape == (2, 81, 8)


def test_channels_inherit_the_models_width() -> None:
    """A channel left at its sentinel is sized by the parent, not by hand."""
    config = GridEmbedding.Config(hidden_size=8)
    config.channels = [FactoredPositions.Config(), PredictionFeedback.Config()]
    final = config.copy_tree().finalize()
    positions, feedback = final.channels
    assert isinstance(positions, FactoredPositions.Config)
    assert isinstance(feedback, PredictionFeedback.Config)
    assert positions.hidden_size == 8
    assert feedback.hidden_size == 8
    # The vocabulary reaches the one channel that is sized by it.
    assert feedback.vocab_size == config.vocab_size


def test_factored_positions_share_a_row() -> None:
    """Cells in one row get the same row component, which is the point."""
    embedding = _embedding(FactoredPositions.Config())
    positions = embedding.channels[0]
    assert isinstance(positions, FactoredPositions)
    # Cells 0..8 are row 0; cells 9..17 are row 1.
    assert positions.row_index[:9].unique().tolist() == [0]
    assert positions.row_index[9:18].unique().tolist() == [1]
    # Column indices cycle within a row.
    assert positions.col_index[:9].tolist() == list(range(9))
    # The first three cells of the first three rows are one box.
    box = positions.box_index.reshape(9, 9)
    assert box[:3, :3].unique().tolist() == [0]


def test_box_shape_must_tile_the_grid() -> None:
    config = FactoredPositions.Config(grid_shape=(9, 9), box_shape=(4, 4))
    config.hidden_size = 8
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
    assert torch.equal(embedding(tokens), base)  # zero table: no contribution
    assert feedback._feedback_ids is None

    # A trained table does contribute, and only for the step it was stashed on.
    with torch.no_grad():
        feedback.embed_feedback.fill_(0.5)
    feedback.set_feedback(torch.randint(0, 11, (2, 81)))
    assert not torch.equal(embedding(tokens), base)
    assert torch.equal(embedding(tokens), base)  # stash already consumed


def test_channel_order_changes_the_result() -> None:
    """Order is a numerics contract, so the test states it rather than assumes."""
    tokens = torch.randint(0, 11, (2, 81))
    # Same channels, different draw order at init: the tables differ, so the
    # sum differs. Equality here would mean the order was not observable.
    assert not torch.equal(
        _ordered(tokens, swap=False),
        _ordered(tokens, swap=True),
    )


def test_hidden_size_must_be_inherited_or_set() -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        GridEmbedding.Config().make()


def _ordered(tokens: Tensor, *, swap: bool) -> Tensor:
    """Embed ``tokens`` with the two channels in one order or the other."""
    positions = FactoredPositions.Config()
    feedback = PredictionFeedback.Config(init_std=1.0)
    channels: list[Makeable[GridChannel]] = (
        [feedback, positions] if swap else [positions, feedback]
    )
    config = GridEmbedding.Config(hidden_size=8)
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
