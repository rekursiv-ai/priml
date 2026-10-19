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
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.upstream_test import StoredPart
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


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


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {
        f"draws/{name}" for name in reference.INITIALIZATIONS
    }


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_each_initialization_matches_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"network/{name}/"),
        reference.network_record(name, states=states),
    )
    assert len(states) == 1


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_two_updates_after_each_initialization_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    key = "network/fixup/train"
    logits = stored(key)[key].clone()
    logits[0, 0] = torch.nextafter(logits[0, 0], torch.tensor(0.0))
    port = {key: reference.network_record("fixup")[key]}
    assert [line.split(":")[0] for line in mismatches({key: logits}, port)] == [key]


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = stored("")
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        StoredPart(golden, "lockstep/{}/"),
    )
    monkeypatch.setattr(reference, "draws_record", StoredPart(golden, "draws/"))
    monkeypatch.setattr(reference, "network_record", StoredPart(golden, "network/{}/"))
    assert_exact(golden, reference.port_record())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
