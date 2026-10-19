"""Replay the ``dl-regularization`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.regularization import reference
from priml.baselines.mlsbench.dl.regularization.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.train_step import LossContext
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


def test_every_regularizer_and_its_gradients_match_the_script() -> None:
    assert_exact(stored("regularizer/"), reference.regularizer_record())


def test_a_port_call_hands_on_every_input_and_states_its_context() -> None:
    seen = _Seen()
    model = nn.Identity()
    media, logits, label = torch.zeros(2, 3), torch.ones(2, 4), torch.tensor([0, 3])
    term = reference._PortCall(seen)(
        model,
        media=media,
        logits=logits,
        label=label,
        config={"num_classes": 4, "epoch": 3, "total_epochs": 5},
    )
    assert torch.equal(term, torch.tensor(8.0))
    [(got_model, got_media, got_logits, got_label, context)] = seen.calls
    assert got_model is model
    assert got_media is media
    assert got_logits is logits
    assert got_label is label
    assert context == LossContext(num_classes=4, epoch=3, total_epochs=5)


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
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    key = "regularizer/orthogonal_reg/value"
    value = stored(key)[key].clone()
    value[1] = torch.nextafter(value[1], torch.tensor(0.0))
    port = {key: reference.regularizer_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


class _Seen:
    """A regularizer that keeps what each call handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[nn.Module, Tensor, Tensor, Tensor, LossContext]] = []

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        self.calls.append((model, media, logits, label, context))
        return logits.sum()


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
