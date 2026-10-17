"""Tests for the sudoku input-embedding channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.model.grid_embedding import (
    FactoredPositions,
    GridChannel,
    GridEmbedding,
    PredictionFeedback,
)
from priml.model.init import scaled_normal
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from configgle import Makeable


def _embedding(*channels: Makeable[GridChannel]) -> GridEmbedding:
    config = GridEmbedding.Config(channels_in=11, channels_out=8, grid_shape=(81,))
    for channel in channels:
        if isinstance(channel, FactoredPositions.Config):
            channel.grid_shape = (9, 9)
            channel.box_shape = (3, 3)
    config.channels = list(channels)
    torch.manual_seed(0)
    return config.make()


def test_plain_embedding_is_token_lookup_only() -> None:
    embedding = _embedding()
    tokens = torch.randint(0, 11, (2, 81))
    assert embedding(tokens).shape == (2, 81, 8)


def test_scaled_normal_matches_truncated_normal_at_inverse_sqrt_width() -> None:
    expected = torch.empty(2, 3)
    actual = torch.empty_like(expected)
    torch.manual_seed(0)
    torch.nn.init.trunc_normal_(expected, std=1.0 / 3**0.5)
    torch.manual_seed(0)
    scaled_normal(actual)
    assert torch.equal(actual, expected)


def test_channels_inherit_the_models_width() -> None:
    """A channel left at its sentinel is sized by the parent, not by hand."""
    config = GridEmbedding.Config(channels_in=11, channels_out=8, grid_shape=(81,))
    config.channels = [
        FactoredPositions.Config(grid_shape=(9, 9), box_shape=(3, 3)),
        PredictionFeedback.Config(),
    ]
    final = config.copy_tree().finalize()
    positions, feedback = final.channels
    assert isinstance(positions, FactoredPositions.Config)
    assert isinstance(feedback, PredictionFeedback.Config)
    assert positions.channels_out == 8
    assert feedback.channels_out == 8
    # The vocabulary reaches the one channel that is sized by it.
    assert feedback.channels_in == config.channels_in


def test_unfilled_position_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="grid_shape and box_shape"):
        FactoredPositions.Config(channels_out=4).make()
    with pytest.raises(ValueError, match="grid_shape from the dataset"):
        GridEmbedding.Config(channels_in=6, channels_out=4).make()


def test_factored_positions_requires_a_positive_output_width() -> None:
    with pytest.raises(ValueError, match="channels_out must be positive"):
        FactoredPositions.Config(
            grid_shape=(2, 3),
            box_shape=(2, 3),
            channels_out=0,
        ).make()


def test_feedback_requires_positive_vocabulary_and_output_widths() -> None:
    with pytest.raises(
        ValueError,
        match="channels_in and channels_out must be positive",
    ):
        PredictionFeedback.Config(channels_in=4, channels_out=0).make()


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


def test_factored_positions_reset_restores_parameters_and_indices() -> None:
    config = FactoredPositions.Config(grid_shape=(2, 3), box_shape=(2, 3))
    config.channels_out = 4
    module = config.make()
    expected_indices = (
        module.row_index.clone(),
        module.col_index.clone(),
        module.box_index.clone(),
    )
    with torch.no_grad():
        module.embed_pos_row.fill_(float("nan"))
        module.embed_pos_col.fill_(float("nan"))
        module.embed_pos_box.fill_(float("nan"))
        module.row_index.fill_(-1)
        module.col_index.fill_(-1)
        module.box_index.fill_(-1)
    torch.manual_seed(17)
    module.reset_parameters()
    assert all(
        torch.isfinite(value).all()
        for value in (
            module.embed_pos_row,
            module.embed_pos_col,
            module.embed_pos_box,
        )
    )
    for actual, expected in zip(
        (module.row_index, module.col_index, module.box_index),
        expected_indices,
        strict=True,
    ):
        assert torch.equal(actual, expected)
    actual_rng = torch.get_rng_state()
    torch.manual_seed(17)
    expected_module = config.make()
    expected_rng = torch.get_rng_state()
    for actual, expected in zip(
        (module.embed_pos_row, module.embed_pos_col, module.embed_pos_box),
        (
            expected_module.embed_pos_row,
            expected_module.embed_pos_col,
            expected_module.embed_pos_box,
        ),
        strict=True,
    ):
        assert torch.equal(actual, expected)
    assert actual_rng.equal(expected_rng)


def test_factored_positions_meta_reset_rebuilds_nonpersistent_indices() -> None:
    config = FactoredPositions.Config(grid_shape=(2, 3), box_shape=(2, 3))
    config.channels_out = 4
    module = config.make().to("meta")
    module.to_empty(device="cpu")
    module.row_index.fill_(-1)
    module.col_index.fill_(-1)
    module.box_index.fill_(-1)
    module.reset_parameters()
    assert torch.equal(module.row_index, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(module.col_index, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(module.box_index, torch.zeros(6, dtype=torch.long))


def test_factored_positions_reset_preserves_bf16_storage_and_fp32_draws() -> None:
    config = FactoredPositions.Config(grid_shape=(2, 3), box_shape=(2, 3))
    config.channels_out = 4
    module = config.make().to(torch.bfloat16)
    tables = (
        module.embed_pos_row,
        module.embed_pos_col,
        module.embed_pos_box,
    )
    identities = tuple(id(table) for table in tables)

    torch.manual_seed(17)
    module.reset_parameters()
    actual_rng = torch.get_rng_state()

    torch.manual_seed(17)
    expected_module = config.make().to(torch.bfloat16)
    expected_rng = torch.get_rng_state()

    expected_tables = (
        expected_module.embed_pos_row,
        expected_module.embed_pos_col,
        expected_module.embed_pos_box,
    )
    assert all(table.dtype is torch.bfloat16 for table in tables)
    assert tuple(id(table) for table in tables) == identities
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            tables,
            expected_tables,
            strict=True,
        )
    )
    assert actual_rng.equal(expected_rng)


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


def test_prediction_feedback_reset_clears_parameters_and_stash() -> None:
    module = PredictionFeedback.Config(channels_in=11, channels_out=8).make()
    module.set_feedback(torch.ones(2, 3, dtype=torch.long))
    with torch.no_grad():
        module.embed_feedback.fill_(2)
    module.reset_parameters()
    assert torch.equal(module.embed_feedback, torch.zeros_like(module.embed_feedback))
    assert module._feedback_ids is None


def test_grid_embedding_reset_resets_owned_channels() -> None:
    embedding = _embedding(PredictionFeedback.Config())
    feedback = embedding.channels[0]
    assert isinstance(feedback, PredictionFeedback)
    feedback.set_feedback(torch.ones(2, 3, dtype=torch.long))
    with torch.no_grad():
        feedback.embed_feedback.fill_(2)

    embedding.reset_parameters()

    assert torch.equal(
        feedback.embed_feedback,
        torch.zeros_like(feedback.embed_feedback),
    )
    assert feedback._feedback_ids is None


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
    """Cost the complete grid: one row read from each table, then scaled."""
    analytical = assert_cost_matches_torch(
        FactoredPositions.Config(grid_shape=(4, 4), box_shape=(2, 2), channels_out=8),
        build_input=lambda: (
            torch.zeros(2, 16, dtype=torch.long),
            torch.zeros(2, 16, 8),
        ),
        seq_len=16,
        batch_size=1,
        dtype=None,
    )
    assert analytical.params == (4 + 4 + 4) * 8
    assert analytical.params_active == 3 * 8
    # Three gathers: an int64 index each, a row read and written at fp32.
    assert analytical["bytes", "primal", "selection", torch.int64] == 16 * 3 * 8
    assert (
        analytical["bytes", "primal", "selection", torch.float32] == 4 * 3 * 2 * 16 * 8
    )
    assert analytical["flops", "adjoint", "selection"].sum() == 3 * 16 * 8
    assert analytical["flops", "primal", "elementwise"].sum() == 3 * 16 * 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 16 * 8


def test_prediction_feedback_cost_is_a_gather_and_a_scale() -> None:
    """Cost the complete grid with a stashed feedback tensor."""
    analytical = assert_cost_matches_torch(
        PredictionFeedback.Config(channels_in=11, channels_out=8),
        build_input=lambda: (torch.randint(0, 11, (2, 16)), torch.zeros(2, 16, 8)),
        seq_len=16,
        batch_size=1,
        dtype=None,
        run=_run_feedback,
    )
    assert analytical.params == 11 * 8
    assert analytical.params_active == 8
    assert analytical["bytes", "primal", "selection", torch.int64] == 16 * 8
    assert analytical["bytes", "primal", "selection", torch.float32] == 4 * 2 * 16 * 8
    assert analytical["flops", "adjoint", "selection"].sum() == 16 * 8
    assert analytical["flops", "primal", "elementwise"].sum() == 16 * 8
    assert analytical["flops", "adjoint", "elementwise"].sum() == 16 * 8


def test_grid_embedding_cost_sums_the_token_table_and_every_channel() -> None:
    """The token gather and its scale, plus each channel and one add per channel."""
    config = GridEmbedding.Config(channels_in=11, channels_out=8, grid_shape=(81,))
    config.channels = [
        FactoredPositions.Config(grid_shape=(9, 9), box_shape=(3, 3)),
        PredictionFeedback.Config(),
    ]
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (2, 81)),
        seq_len=81,
        batch_size=1,
        dtype=None,
        run=_run_with_feedback,
    )
    assert analytical.params == 11 * 8 + (9 + 9 + 9) * 8 + 11 * 8
    assert analytical.params_active == 8 + 3 * 8 + 8
    # Token row, three position rows, one feedback row.
    assert analytical["bytes", "primal", "selection", torch.int64] == 81 * 5 * 8
    assert (
        analytical["bytes", "primal", "selection", torch.float32] == 4 * 5 * 2 * 81 * 8
    )
    assert analytical["flops", "adjoint", "selection"].sum() == 81 * (8 + 3 * 8 + 8)
    # Token scale; positions' two adds and scale; feedback scale; two channel adds.
    assert analytical["flops", "primal", "elementwise"].sum() == 81 * (
        8 + 3 * 8 + 8 + 2 * 8
    )
    # The channel adds accumulate no gradient, so only the three scales pull back.
    assert analytical["flops", "adjoint", "elementwise"].sum() == 81 * (8 + 8 + 8)


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


def test_feedback_cost_scales_with_grid_rows() -> None:
    config = PredictionFeedback.Config()
    config.channels_in = 11
    config.channels_out = 8
    single = config.cost(seq_len=4, batch_size=1, dtype=None)
    batch = config.cost(seq_len=4, batch_size=2, dtype=None)
    assert batch["bytes", "adjoint", "selection"].sum() - single[
        "bytes",
        "adjoint",
        "selection",
    ].sum() == 4 * (8 + 3 * 8 * 4)


def test_factored_position_cost_broadcasts_the_complete_batch() -> None:
    config = FactoredPositions.Config(grid_shape=(4, 4), box_shape=(2, 2))
    config.channels_out = 8
    config.grid_shape = (4, 4)
    config.box_shape = (2, 2)
    single = config.cost(seq_len=16, batch_size=1, dtype=torch.bfloat16)
    batch = config.cost(seq_len=16, batch_size=2, dtype=torch.bfloat16)
    assert batch["flops", "primal"] == single["flops", "primal"]
    assert batch["bytes", "primal"] == single["bytes", "primal"]
    assert batch["flops", "adjoint", "reduction"].sum() == 16 * 8 * (2 - 1)
    assert batch["bytes", "adjoint", "reduction"].sum() == 2 * (2 * 16 * 8 + 16 * 8)
    assert batch.params == single.params


def _run_feedback(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Stash ``tokens`` as the fed-back grid, then run the channel."""
    assert isinstance(module, PredictionFeedback)
    tokens, embeddings = inputs
    module.set_feedback(tokens)
    return module(tokens, embeddings)


def _run_with_feedback(module: nn.Module, tokens: Tensor) -> Tensor:
    """Stash ``tokens`` on every feedback channel, then embed them."""
    assert isinstance(module, GridEmbedding)
    for channel in module.channels:
        if isinstance(channel, PredictionFeedback):
            channel.set_feedback(tokens)
    return module(tokens)


def _ordered(tokens: Tensor, *, swap: bool) -> Tensor:
    """Embed ``tokens`` with the two channels in one order or the other."""
    positions = FactoredPositions.Config(grid_shape=(9, 9), box_shape=(3, 3))
    feedback = PredictionFeedback.Config(init_std=1.0)
    channels: list[Makeable[GridChannel]] = (
        [feedback, positions] if swap else [positions, feedback]
    )
    config = GridEmbedding.Config(channels_in=11, channels_out=8, grid_shape=(81,))
    config.channels = channels
    torch.manual_seed(0)
    embedding = config.make()
    stash = embedding.channels[0 if swap else 1]
    assert isinstance(stash, PredictionFeedback)
    stash.set_feedback(tokens)
    return embedding(tokens)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
