"""Freeze the trained fixtures the evaluation goldens read from ``fixtures.pt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from treekle import json_loads

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.runner import file_sha256, load_eval_weights
from priml.baselines.sudoku.search import run_search
from priml.baselines.sudoku.testing import (
    FIXTURE_GROUPS,
    FIXTURES,
    Harness,
    load_golden,
    metric_tensors,
    replay,
    sudoku_solution,
    tiny_recursion,
    train_fixture,
    weight_tensors,
    write_dataset,
)
from priml.baselines.sudoku.verifier_fit import VerifierLoss
from priml.testing.golden import assert_tensor_golden_part, read_tensors


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("group", FIXTURE_GROUPS)
def test_a_stored_fixture_is_what_training_writes(group: str, tmp_path: Path) -> None:
    """Each stored fixture equals a fresh training of it, bit for bit."""
    puzzle_data.dihedral_indices.cache_clear()
    store = read_tensors(FIXTURES)
    assert {key.split("/", 1)[0] for key in store} == set(FIXTURE_GROUPS)
    assert_tensor_golden_part(
        FIXTURES,
        f"{group}/",
        train_fixture(group, tmp_path, known=store),
    )


def test_a_stored_harvest_publishes_the_manifest_a_harvest_writes(
    tmp_path: Path,
) -> None:
    """The stored shard's manifest names its source and rows, as a harvest's does."""
    harness = Harness(tmp_path, train=False)
    write_dataset(harness.data)
    source = harness.generator("gen_a", seed=0)
    out = harness.harvest(source)
    assert harness.harvest(source) == out
    text = (out / "manifest.json").read_text()
    rows = len(read_tensors(FIXTURES)["harvest/original"])
    shard = out / "shard-00000.npz"
    expected = {
        "schema_version": 1,
        "source_checkpoint": str(tmp_path / source.removeprefix("/")),
        "groups_per_shard": 8,
        "row_count": rows,
        "shard_count": 1,
        "shards": [{"file": shard.name, "rows": rows, "sha256": file_sha256(shard)}],
    }
    assert json_loads(text, dict[str, object]) == expected
    assert text == json.dumps(expected, indent=2) + "\n"


def test_each_trained_fixture_joins_those_trained_before_it(tmp_path: Path) -> None:
    store = read_tensors(FIXTURES)
    known = {k: v for k, v in store.items() if not k.startswith(("gen_a/", "gen_b/"))}
    write_dataset(tmp_path / "data")
    harness = Harness(tmp_path, train=True, known=known)
    harness.generator("gen_a", seed=0)
    harness.generator("gen_b", seed=1)
    assert {key.split("/", 1)[0] for key in harness.trained} == {"gen_a", "gen_b"}


def test_the_harness_search_runs_on_the_tiny_generator(tmp_path: Path) -> None:
    harness = Harness(tmp_path, train=False)
    batch = {
        "media": sudoku_solution()[None].repeat(2, 1),
        "valid_count": 2,
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        result = run_search(tiny_recursion(), batch, harness.search())
    assert result.final_predictions.shape == (2, 81)
    assert int(result.nodes.max()) <= harness.search().search_budget


def test_dumps_split_packed_rows_into_scores_and_grids(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "dump"
    run.mkdir(parents=True)
    np.savez(
        run / "x.npz",
        rows=np.arange(2 * (3 + 2 * 81), dtype=np.float32).reshape(2, -1),
        grid=np.ones((2, 81), dtype=np.uint8),
    )
    dumps = Harness(tmp_path, train=False).dumps("dump")
    assert dumps.keys() == {"x.npz/rows/scores", "x.npz/rows/grids", "x.npz/grid"}
    assert dumps["x.npz/rows/scores"].shape == (2, 3)
    assert dumps["x.npz/rows/grids"].shape == (2, 2 * 81)


def test_weights_are_prefixed_and_metrics_keep_only_numbers() -> None:
    assert weight_tensors("w", {"a": torch.tensor([1.0])}).keys() == {"w/a"}
    metrics = metric_tensors({"eval/n": 2, "eval/seconds": 1.5, "eval/name": "x"})
    assert metrics.keys() == {"metrics/eval/n"}
    assert metrics["metrics/eval/n"].dtype == torch.float64


def test_a_verifier_with_its_own_objective_trains_even_when_one_is_stored(
    tmp_path: Path,
) -> None:
    """A custom loss is the fit under test, so the stored member never stands in."""
    harness = Harness(tmp_path, train=False)
    write_dataset(harness.data)
    harness.harvest(harness.generator("gen_a", seed=0))
    path = harness.verifier(0, loss=VerifierLoss.Config())
    trained = load_eval_weights(path)
    stored = harness._stored("verifier_s0/")
    assert trained.keys() == stored.keys()
    assert not torch.equal(trained["cell_head.bias"], stored["cell_head.bias"])


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
