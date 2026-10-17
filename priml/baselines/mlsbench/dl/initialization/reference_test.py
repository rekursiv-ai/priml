"""Replay the ``dl-weight-initialization`` port against values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.initialization import reference
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


def test_the_schedule_and_optimizer_match_the_script() -> None:
    expected = stored("schedule/") | stored("optimizer/")
    assert_exact(expected, reference.recipe_record())


def test_each_full_network_leaves_the_generator_where_the_script_does() -> None:
    expected = stored("draws/")
    assert set(expected) == {f"draws/{name}" for name in reference.INITIALIZATIONS}
    assert_exact(expected, reference.draws_record())


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_each_initialization_matches_the_script(name: str) -> None:
    assert_exact(stored(f"network/{name}/"), reference.network_record(name))


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_two_updates_after_each_initialization_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_golden_bites_on_one_ulp() -> None:
    key = "network/fixup/train"
    logits = stored(key)[key].clone()
    logits[0, 0] = torch.nextafter(logits[0, 0], torch.tensor(0.0))
    port = {key: reference.network_record("fixup")[key]}
    assert [line.split(":")[0] for line in mismatches({key: logits}, port)] == [key]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
