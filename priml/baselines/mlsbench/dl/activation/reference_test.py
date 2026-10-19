"""Replay the ``dl-activation-function`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from configgle import InlineConfig
from torch import nn

import pytest
import torch

from priml.baselines.mlsbench.dl.activation import reference
from priml.baselines.mlsbench.dl.activation.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.upstream_test import StoredPart
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


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


def test_every_activation_and_its_gradient_match_the_script() -> None:
    expected = stored("activation/")
    assert {k.split("/")[1] for k in expected} == set(reference.ACTIVATIONS)
    assert_exact(expected, reference.activation_record())


def test_the_record_refuses_an_activation_it_would_have_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "ACTIVATIONS", {"prelu": _stateful})
    with pytest.raises(
        TypeError,
        match=r"^Expected not isinstance\(activation, Makeable\)\.$",
    ):
        _ = reference.activation_record()


def test_the_learning_rate_of_all_200_epochs_matches_the_script() -> None:
    assert_exact(stored("schedule/"), vision.schedule_record(exp000))


def test_the_optimizer_matches_the_script() -> None:
    assert_exact(stored("optimizer/"), vision.optimizer_record(exp000))


def test_the_best_accuracy_matches_the_script() -> None:
    assert_exact(stored("best/"), vision.best_record())


def test_the_resnet20_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


@pytest.mark.parametrize("name", list(reference.ACTIVATIONS))
def test_two_updates_with_each_activation_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    key = "activation/mish/grad"
    grad = stored(key)[key].clone()
    grad[0, 0] = torch.nextafter(grad[0, 0], torch.tensor(1.0))
    port = {key: reference.activation_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: grad}, port)] == [key]


def test_the_golden_fits_the_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= 32_768


def _stateful() -> VisionTrainLoop:
    """Return ``exp000`` with an activation the network has to build."""
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.activation = InlineConfig(nn.PReLU)
    return cfg


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
    assert_exact(golden, reference.port_record())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
