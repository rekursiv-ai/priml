"""Replay the ``dl-residual-connection`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.residual import reference
from priml.baselines.mlsbench.dl.vision import reference as vision
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


def test_every_block_matches_its_script_forward_backward_and_draws() -> None:
    expected = stored("block/")
    assert {k.split("/")[1] for k in expected} == set(reference.VARIANTS)
    draws = torch.cat(
        [
            v
            for k, v in expected.items()
            if k.startswith("block/stochastic_depth/") and k.endswith("/draws")
        ],
    )
    assert bool((draws < 0.5).any())
    assert bool((draws >= 0.5).any())
    assert_exact(expected, reference.block_record())


def test_the_schedule_optimizer_and_accuracy_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/") | stored("best/"),
        vision.schedule_record() | vision.optimizer_record() | vision.best_record(),
    )


def test_each_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_two_updates_with_each_block_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_port_record_holds_every_stored_key_of_its_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "VARIANTS", {"basic": reference.VARIANTS["basic"]})
    expected = {
        k
        for k in stored("")
        if k.split("/")[0] in {"schedule", "optimizer", "best"}
        or k.split("/")[1] == "basic"
    }
    assert reference.port_record().keys() == expected


def test_the_golden_bites_on_one_ulp() -> None:
    key = "block/pre_activation/projected/eval"
    value = stored(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = {key: reference.block_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
