"""Random-site numbering and exact-mismatch sensitivity of EEG records."""

from __future__ import annotations

from configgle import InlineConfig
from torch import nn

import pytest
import torch

from priml.baselines.motor_imagery.testing import (
    compare_record,
    model_record,
    optimizer_record,
    random_record,
    source_preprocessing,
    source_step,
)
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.testing.golden import PinnedRandom
from priml.train.parallelism import NoParallel


def test_random_record_numbers_each_files_sites_in_encounter_order() -> None:
    pins = PinnedRandom()
    pins.calls["aten.uniform_.default/conv.py:20"] = 6
    pins.calls["aten.uniform_.default/linear.py:35"] = 3
    pins.calls["aten.bernoulli_.float/dropout.py:70"] = 8
    pins.calls["aten.uniform_.default/linear.py:30"] = 1
    pins.calls["aten.uniform_.default/linear.py:40"] = 2
    compare_record(
        {
            "random/aten.uniform_.default/conv.py:0": torch.tensor(6),
            "random/aten.uniform_.default/linear.py:0": torch.tensor(3),
            "random/aten.bernoulli_.float/dropout.py:0": torch.tensor(8),
            "random/aten.uniform_.default/linear.py:1": torch.tensor(1),
            "random/aten.uniform_.default/linear.py:2": torch.tensor(2),
        },
        random_record(pins),
    )
    moved = PinnedRandom()
    for name, count in pins.calls.items():
        moved.calls[f"{name}0"] = count
    compare_record(random_record(pins), random_record(moved))


def test_random_record_rejects_an_omitted_or_added_site() -> None:
    pins = PinnedRandom()
    pins.calls["aten.uniform_.default/linear.py:30"] = 3
    pins.calls["aten.uniform_.default/linear.py:35"] = 1
    omitted = PinnedRandom()
    omitted.calls["aten.uniform_.default/linear.py:30"] = 3
    with pytest.raises(
        ValueError,
        match=r"missing random/aten\.uniform_\.default/linear\.py:1",
    ):
        compare_record(random_record(pins), random_record(omitted))
    added = PinnedRandom()
    added.calls.update(pins.calls)
    added.calls["aten.uniform_.default/conv.py:20"] = 2
    with pytest.raises(
        ValueError,
        match=r"unexpected random/aten\.uniform_\.default/conv",
    ):
        compare_record(random_record(pins), random_record(added))


def test_compare_record_reports_a_changed_bit() -> None:
    with pytest.raises(ValueError, match="bits: 1/2 differ") as raised:
        compare_record(
            {"bits": torch.tensor([1.0, 2.0])},
            {"bits": torch.tensor([1.0, 3.0])},
        )
    assert str(raised.value) == "bits: 1/2 differ; max_ulp=4194304"


def test_compare_record_reports_every_difference_on_its_own_line() -> None:
    with pytest.raises(ValueError, match="differ") as raised:
        compare_record(
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([1.0])},
            {"a": torch.tensor([1.0, 3.0]), "b": torch.tensor([2.0])},
        )
    assert str(raised.value) == (
        "a: 1/2 differ; max_ulp=4194304\nb: 1/1 differ; max_ulp=8388608"
    )


def test_optimizer_record_rejects_multiple_optimizer_members() -> None:
    optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]),
            torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1),
        ],
    )
    with pytest.raises(
        ValueError,
        match=r"^Expected len\(optimizer.optimizers\) == 1\.$",
    ):
        optimizer_record(optimizer)


def test_model_record_keeps_detached_copies_under_the_prefix() -> None:
    model = nn.Linear(3, 2)
    record = model_record(model, prefix="final")
    assert record.keys() == {"final/weight", "final/bias"}
    assert torch.equal(record["final/weight"], model.weight)
    assert not record["final/weight"].requires_grad
    with torch.no_grad():
        model.weight.zero_()
    assert record["final/weight"].abs().sum() > 0


@pytest.mark.parametrize("composite", [False, True])
def test_optimizer_record_numbers_each_parameter_state(composite: bool) -> None:
    first, second = nn.Parameter(torch.ones(2, 3)), nn.Parameter(torch.ones(4))
    adamw = torch.optim.AdamW([first, second])
    (first.sum() + second.sum()).backward()
    adamw.step()
    record = optimizer_record(CompositeOptimizer([adamw]) if composite else adamw)
    assert record.keys() == {
        f"optimizer/{index}/{name}"
        for index in range(2)
        for name in ("step", "exp_avg", "exp_avg_sq")
    }
    assert record["optimizer/0/exp_avg"].shape == (2, 3)


def test_random_record_counts_each_pinned_primitive() -> None:
    pins = PinnedRandom()
    pins.calls["aten.uniform_.default"] = 3
    assert random_record(pins) == {"random/aten.uniform_.default:0": torch.tensor(3)}


def test_source_preprocessing_returns_a_fresh_exp000_tree() -> None:
    first = source_preprocessing()
    assert source_preprocessing() is not first


def test_source_step_trains_on_the_cpu_over_the_requested_epochs() -> None:
    model = InlineConfig(nn.Identity)
    step = source_step(model, epochs=3)
    assert step.model is model
    assert isinstance(step.parallelism, NoParallel.Config)
    assert step.parallelism.device == "cpu"
    assert step.train_budget_epochs == 3
    assert isinstance(step.lr_schedule, RecursiveCosine.Config)
    assert step.lr_schedule.T_max == 3


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
