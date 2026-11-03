"""Freeze the trained fixtures the evaluation goldens read from ``fixtures.pt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from torch import Tensor
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
    FilledDenoisingVerifier,
    FilledHypotheses,
    FilledRecursion,
    FilledVerifier,
    Harness,
    metric_tensors,
    replay,
    sudoku_solution,
    tiny_recursion_config,
    train_fixture,
    write_dataset,
)
from priml.baselines.sudoku.verifier_fit import DenoisingLoss, VerifierLoss
from priml.testing import regenerate
from priml.testing.golden import (
    assert_tensor_golden_part,
    mismatches,
    read_tensors,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("group", FIXTURE_GROUPS)
def test_owned_record_ignores_constructor_sampler_changes(
    group: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = read_tensors(FIXTURES)
    expected = train_fixture(group, tmp_path / "original", known=known)

    def different_sampler(tensor: Tensor, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        with torch.no_grad():
            torch.rand(17)
            return tensor.uniform_(-0.1, 0.1)

    monkeypatch.setattr(torch.nn.init, "_no_grad_trunc_normal_", different_sampler)
    actual = train_fixture(group, tmp_path / "altered", known=known)
    assert not mismatches(expected, actual)


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


@pytest.mark.parametrize("regenerating", [False, True])
def test_a_stored_harvest_publishes_the_manifest_a_harvest_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    regenerating: bool,
) -> None:
    """The stored shard's manifest names its source and rows, as a harvest's does."""
    regenerate.override(monkeypatch, b4b=regenerating)
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
        result = run_search(
            tiny_recursion_config().make().eval(),
            batch,
            harness.search(),
        )
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


def test_metrics_keep_only_numbers() -> None:
    metrics = metric_tensors({"eval/n": 2, "eval/seconds": 1.5, "eval/name": "x"})
    assert metrics.keys() == {"metrics/eval/n"}
    assert metrics["metrics/eval/n"].dtype == torch.float64


@pytest.mark.parametrize("denoising", [False, True])
def test_custom_verifier_architectures_train_instead_of_loading_a_fixture(
    tmp_path: Path,
    denoising: bool,
) -> None:
    harness = Harness(tmp_path, train=False)
    write_dataset(harness.data)
    harness.harvest(harness.generator("gen_a", seed=0))
    model = (
        FilledDenoisingVerifier.Config().update(harness.verifier_model())
        if denoising
        else harness.verifier_model()
    )
    path = harness.verifier(
        0,
        model=model,
        loss=DenoisingLoss.Config() if denoising else None,
    )
    trained = load_eval_weights(path)
    initial = model.make().state_dict()
    assert (path.parents[1] / "metrics.json").is_file()
    assert trained.keys() == initial.keys()
    assert mismatches(trained, initial)


def test_a_verifier_with_its_own_objective_trains_even_when_one_is_stored(
    tmp_path: Path,
) -> None:
    """A custom loss is the fit under test, so the stored member never stands in."""
    harness = Harness(tmp_path, train=False)
    write_dataset(harness.data)
    harness.harvest(harness.generator("gen_a", seed=0))
    loss = VerifierLoss.Config()
    loss.cell_weight = 0.0
    path = harness.verifier(0, loss=loss)
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


def test_replay_narrows_what_a_recorder_returns(tmp_path: Path) -> None:
    out = replay(lambda _: {"grid": torch.tensor([3, 4])}, tmp_path)
    assert out["grid"].dtype == torch.uint8


@pytest.mark.parametrize("train", [False, True])
def test_replay_preserves_rng_and_passes_the_requested_harness(
    tmp_path: Path,
    train: bool,
) -> None:
    def record(harness: Harness) -> dict[str, Tensor]:
        assert harness.scratch == tmp_path
        assert harness.train is train
        return {"draws": torch.rand(3)}

    state = torch.random.get_rng_state()
    actual = replay(record, tmp_path, train=train)
    assert torch.equal(torch.random.get_rng_state(), state)
    generator = torch.Generator().manual_seed(0)
    assert torch.equal(actual["draws"], torch.rand(3, generator=generator))
    for split, rows in (("train", 9), ("test", 2)):
        path = tmp_path / "data" / split / "all__inputs.npy"
        assert np.load(path).shape == (rows, 81)


@pytest.mark.parametrize("kind", ["recursion", "hypotheses", "verifier", "denoising"])
def test_fixture_construction_ignores_sampler_bits_and_preserves_rng(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path)
    if kind == "recursion":
        config = harness.model()
        expected_type = FilledRecursion
    elif kind == "hypotheses":
        config = harness.hypothesis_model()
        expected_type = FilledHypotheses
    elif kind == "verifier":
        config = harness.verifier_model()
        expected_type = FilledVerifier
    else:
        config = FilledDenoisingVerifier.Config().update(harness.verifier_model())
        expected_type = FilledDenoisingVerifier
    state = torch.random.get_rng_state()
    original = config.make()
    assert isinstance(original, expected_type)
    assert torch.equal(state, torch.random.get_rng_state())

    def different_sampler(tensor: Tensor, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        with torch.no_grad():
            torch.rand(17)
            return tensor.fill_(0.125)

    monkeypatch.setattr(torch.nn.init, "_no_grad_trunc_normal_", different_sampler)
    altered = config.make()
    assert torch.equal(state, torch.random.get_rng_state())
    assert not mismatches(original.state_dict(), altered.state_dict())
    for name in ("slow_init", "fast_init"):
        if kind in ("recursion", "hypotheses"):
            assert torch.equal(original.get_buffer(name), altered.get_buffer(name))


def test_the_tiny_recursion_is_frozen_and_the_solution_is_a_sudoku() -> None:
    model = tiny_recursion_config().make().eval()
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
