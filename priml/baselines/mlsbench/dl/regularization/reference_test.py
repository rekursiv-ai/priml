"""Replay the ``dl-regularization`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.regularization import reference
from priml.baselines.mlsbench.dl.regularization.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision
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


def test_every_regularizer_and_its_gradients_match_the_script() -> None:
    assert_exact(stored("regularizer/"), reference.regularizer_record())


def test_the_schedule_and_optimizer_match_the_setting() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/"),
        vision.schedule_record(exp000) | vision.optimizer_record(exp000),
    )


def test_the_batch_size_matches_the_setting() -> None:
    assert_exact(
        stored("setting/"),
        {"setting/batch_size": reference.port_batch_size()},
    )


def test_the_full_step_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


def test_the_best_accuracy_is_reported_as_the_script_does() -> None:
    assert_exact(stored("best/"), vision.best_record())


@pytest.mark.parametrize("name", list(reference.REGULARIZERS))
def test_two_updates_with_each_regularizer_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_golden_bites_on_one_ulp() -> None:
    key = "regularizer/orthogonal_reg/value"
    value = stored(key)[key].clone()
    value[1] = torch.nextafter(value[1], torch.tensor(0.0))
    port = {key: reference.regularizer_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
