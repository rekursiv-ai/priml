"""Replay the ``cv-classification-loss`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``, which also compares
every parameter after each update live. Equality is exact; there is no
tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.classification_loss import reference
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
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


def test_every_loss_and_its_gradient_match_the_script() -> None:
    assert_exact(stored("loss/"), reference.loss_record())


@pytest.mark.parametrize("name", list(reference.LOSSES))
def test_two_updates_with_each_loss_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_golden_holds_only_the_losses_and_their_updates() -> None:
    assert {k.split("/")[0] for k in stored("")} == {"loss", "lockstep"}


def test_the_captured_golden_fits_the_size_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= capture_reference.CEILING


def test_the_comparison_sees_a_change_of_one_ulp() -> None:
    expected = stored("loss/cross_entropy/value")
    bumped = {
        k: torch.nextafter(v, torch.tensor(torch.inf)) for k, v in expected.items()
    }
    assert mismatches(expected, bumped) == ["loss/cross_entropy/value: 1/1 differ"]


def test_a_lockstep_run_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("cross_entropy", states=states)
    assert len(states) == vision.STEPS
    assert not torch.equal(states[0]["stem.weight"], states[1]["stem.weight"])


def test_the_loss_record_holds_a_value_and_a_gradient_per_loss() -> None:
    record = reference.loss_record()
    assert set(record) == {
        f"loss/{name}/{kind}" for name in reference.LOSSES for kind in ("value", "grad")
    }
    assert record["loss/poly/grad"].shape == (3, 4)
    assert record["loss/poly/value"].shape == ()


def test_the_record_joins_the_losses_and_every_lockstep_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference,
        "loss_record",
        lambda: {"loss_record": torch.zeros(2)},
    )

    def lockstep(name: str) -> dict[str, torch.Tensor]:
        return {f"lockstep/{name}": torch.zeros(2)}

    monkeypatch.setattr(reference, "lockstep_record", lockstep)
    assert set(reference.port_record()) == {
        "loss_record",
        *(f"lockstep/{name}" for name in reference.LOSSES),
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
