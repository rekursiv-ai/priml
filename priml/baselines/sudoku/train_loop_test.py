"""The sudoku loop lays the model, pool, and losses out on its dataset."""

from __future__ import annotations

from priml.baselines.sudoku.act import CellCorruption, FeedbackCarry, SlotScramble
from priml.baselines.sudoku.losses import CardinalityLoss
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.baselines.sudoku.train_loop import SudokuTrainLoop


def test_the_dataset_lays_out_the_model_pool_and_cardinality_groups() -> None:
    config = SudokuTrainLoop()
    spec = config.dataset.spec = SudokuSpec(
        grid_shape=(4, 4),
        box_shape=(2, 2),
        vocab_size=6,
    )
    config.dataset.batch_size = 3
    feedback = config.step.pool.feedback = FeedbackCarry.Config()
    feedback.corruption = SlotScramble.Config()
    config.step.cardinality = CardinalityLoss.Config()
    final = config.copy_tree().finalize()
    model, pool = final.step.model, final.step.pool
    assert (model.puzzle_grid_shape, model.vocab_size) == ((16,), 6)
    assert (model.pos2d_grid_shape, model.pos2d_box_shape) == ((4, 4), (2, 2))
    assert (pool.batch_size, model.puzzle_emb_batch_size) == (3, 3)
    assert (pool.grid_len, pool.seq_len) == (16, 16 + model.puzzle_emb_len)
    assert pool.channels_hidden == model.channels_in
    assert pool.feedback is not None
    assert pool.feedback.givens == (2, 5)
    assert isinstance(pool.feedback.corruption, SlotScramble.Config)
    assert pool.feedback.corruption.high == 6
    assert final.step.cardinality is not None
    assert final.step.cardinality.spec == spec


def test_disabled_position_tables_and_another_corruption_are_left_alone() -> None:
    config = SudokuTrainLoop()
    config.step.model.pos2d_grid_shape = None
    feedback = config.step.pool.feedback = FeedbackCarry.Config()
    corruption = feedback.corruption = CellCorruption.Config()
    corruption.high = 4
    final = config.copy_tree().finalize()
    assert final.step.model.pos2d_grid_shape is None
    assert final.step.pool.feedback is not None
    assert isinstance(final.step.pool.feedback.corruption, CellCorruption.Config)
    assert final.step.pool.feedback.corruption.high == 4
    assert final.step.cardinality is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
