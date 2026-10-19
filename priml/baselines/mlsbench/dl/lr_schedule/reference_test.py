"""Replay the ``dl-lr-schedule`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.lr_schedule import reference
from priml.baselines.mlsbench.dl.lr_schedule.experiments import exp000
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


def test_every_schedule_matches_its_script_for_all_200_epochs() -> None:
    expected = stored("schedule/")
    assert set(expected) == {f"schedule/{name}" for name in reference.SCHEDULES}
    assert all(v.shape == (200,) for v in expected.values())
    assert_exact(expected, reference.schedule_record())


def test_the_optimizer_matches_the_script() -> None:
    assert_exact(stored("optimizer/"), reference.optimizer_record())


def test_the_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), vision.draws_record(exp000))


def test_the_reported_accuracy_matches_the_script() -> None:
    assert_exact(stored("best/"), vision.best_record())


@pytest.mark.parametrize("name", list(reference.SCHEDULES))
def test_two_updates_with_each_schedule_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    expected = stored("schedule/cosine")
    rate = expected["schedule/cosine"].clone()
    rate[1] = torch.nextafter(rate[1], torch.tensor(0.0, dtype=torch.float64))
    port = {"schedule/cosine": reference.schedule_record()["schedule/cosine"]}
    assert [
        line.split(":")[0] for line in mismatches({"schedule/cosine": rate}, port)
    ] == [
        "schedule/cosine",
    ]


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = stored("")
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        StoredPart(golden, "lockstep/{}/"),
    )
    monkeypatch.setattr(vision, "draws_record", StoredPart(golden, "draws/"))
    assert_exact(golden, reference.port_record())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
