"""Replay the ``cv-multitask-loss`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from treekle import from_plain

import pytest
import torch

from priml.baselines.mlsbench.multitask import reference
from priml.baselines.mlsbench.multitask.data import COARSE_OF_FINE
from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.testing.golden import mismatches, read_tensors


_DATA: Final = Path(__file__).resolve().parent / "testdata"


def stored(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def assert_exact(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    """Require two records to agree in every key, dtype, shape and bit."""
    assert expected
    assert mismatches(expected, actual) == []


def test_every_combination_matches_its_script_on_fixed_losses() -> None:
    assert_exact(stored("combination/"), reference.combination_record())


def test_the_superclass_map_matches_the_script() -> None:
    assert_exact(
        stored("data/"),
        {"data/coarse_of_fine": torch.tensor(COARSE_OF_FINE)},
    )


def test_the_schedule_and_best_accuracy_match_the_script() -> None:
    assert_exact(stored("schedule/"), reference.schedule_record())
    assert_exact(stored("best/"), vision.best_record())


@pytest.mark.parametrize("name", list(reference.COMBINATIONS))
def test_two_updates_with_each_combination_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("sum", states=states)
    assert len(states) == vision.STEPS
    assert not torch.equal(
        states[0]["trunk.head.weight"],
        states[1]["trunk.head.weight"],
    )


def test_the_loader_batches_carry_superclasses(tmp_path: Path) -> None:
    train, evaluation = reference.loader_batches(tmp_path)
    assert len(train) == len(evaluation) == vision.LOADER_PASSES
    for batch in train[0]:
        labels = from_plain(batch["label"].tolist(), list[int])
        expected = [COARSE_OF_FINE[c] for c in labels]
        assert batch["label_coarse"].tolist() == expected


def test_the_full_networks_match_the_script() -> None:
    assert_exact(stored("optimizer/"), reference.optimizer_record())
    assert_exact(stored("draws/"), reference.draws_record())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
