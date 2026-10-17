"""Replay the ``cv-pooling-aggregation`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.pooling import reference
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


def test_every_pool_matches_its_script_forward_and_backward() -> None:
    expected = stored("pool/")
    assert {k.split("/")[1] for k in expected} == set(reference.POOLS)
    assert "pool/gem/param_grad" in expected
    assert_exact(expected, reference.pool_record())


def test_the_schedule_optimizer_and_accuracy_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/") | stored("best/"),
        vision.schedule_record() | vision.optimizer_record() | vision.best_record(),
    )


def test_each_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


@pytest.mark.parametrize("name", list(reference.POOLS))
def test_two_updates_with_each_pool_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_golden_bites_on_one_ulp() -> None:
    expected = stored("pool/gem/value")
    value = expected["pool/gem/value"].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = {"pool/gem/value": reference.pool_record()["pool/gem/value"]}
    assert mismatches({"pool/gem/value": value}, port) == ["pool/gem/value: 1/6 differ"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
