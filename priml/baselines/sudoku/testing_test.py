"""Freeze the trained fixtures the evaluation goldens read from ``fixtures.pt``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.sudoku import (
    agreement_lock_test,
    hps_eval_test,
    puzzle_data,
    sieve_test,
    verifier_fit_test,
)
from priml.baselines.sudoku.runner import load_eval_weights
from priml.baselines.sudoku.testing import (
    FIXTURES,
    Harness,
    load_golden,
    mint_fixtures,
    replay,
    sudoku_solution,
    tiny_recursion,
    write_dataset,
)
from priml.testing.golden import assert_tensor_golden


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor


_READERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "agreement_lock": agreement_lock_test._RECORDERS["agreement_lock"],
    "hps": hps_eval_test._RECORDERS["hps"],
    "staircase": sieve_test._RECORDERS["staircase"],
    "denoising_fit": verifier_fit_test._RECORDERS["denoising_fit"],
}
"""Every golden that reads the stored fixtures instead of training them."""


def test_stored_fixtures_are_what_training_writes(tmp_path: Path) -> None:
    """The stored weights and shard equal a fresh training run's, bit for bit."""
    puzzle_data.dihedral_indices.cache_clear()
    assert_tensor_golden(FIXTURES, mint_fixtures(_READERS, tmp_path))


def test_a_known_fixture_is_written_not_retrained(tmp_path: Path) -> None:
    """Under ``train``, a fixture in ``known`` is written to its path untrained.

    A stored checkpoint holds the eval weights alone, so the loader reads them
    back unchanged whether the directory existed or not.
    """
    weights = {"head.weight": torch.tensor([1.5, -2.0])}
    known = {f"gen_a/{name}": value for name, value in weights.items()}
    harness = Harness(tmp_path, train=True, known=known)
    write_dataset(harness.data)
    path = harness.scratch / harness.generator("gen_a", seed=0).removeprefix("/")
    assert load_eval_weights(path).keys() == weights.keys()
    assert torch.equal(load_eval_weights(path)["head.weight"], weights["head.weight"])
    assert not harness.trained
    harness.generator("gen_a", seed=0)
    assert torch.equal(load_eval_weights(path)["head.weight"], weights["head.weight"])


def test_replay_narrows_and_loads_what_a_recorder_returns(tmp_path: Path) -> None:
    """``replay`` stores each record, and ``load_golden`` reads a stored golden."""
    out = replay(lambda _: {"grid": torch.tensor([3, 4])}, tmp_path)
    assert out["grid"].dtype == torch.uint8
    assert set(load_golden("staircase")) >= {"metrics/eval/n_correct"}


def test_the_tiny_recursion_is_frozen_and_the_solution_is_a_sudoku() -> None:
    model = tiny_recursion()
    assert not model.training
    assert model.config.channels_in == 4
    # sudoku_group_indices uses SudokuSpec's 9x9 row and column geometry.
    grid = sudoku_solution().reshape(9, 9) - 2
    digits = set(range(9))
    assert all(set(row.tolist()) == digits for row in grid)
    assert all(set(column.tolist()) == digits for column in grid.T)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
