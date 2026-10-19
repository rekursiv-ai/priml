"""Replay the ``cv-multitask-loss`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from torch import nn
from treekle import from_plain

import pytest
import torch

from priml.baselines.mlsbench.dl.multitask import reference
from priml.baselines.mlsbench.dl.multitask.data import COARSE_OF_FINE
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    assert_port_record_joins,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.dl.multitask.experiments import (
        MultiTaskTrainLoop,
    )
    from priml.baselines.mlsbench.dl.multitask.train_step import MultiTaskStep

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


def test_every_combination_matches_its_script_on_fixed_losses() -> None:
    assert_exact(stored("combination/"), reference.combination_record())


def test_the_superclass_map_matches_the_script() -> None:
    assert_exact(
        stored("data/"),
        {"data/coarse_of_fine": torch.tensor(COARSE_OF_FINE)},
    )


def test_the_schedule_and_best_accuracy_match_the_script() -> None:
    assert_exact(stored("schedule/"), reference.schedule_record())
    assert_exact(stored("best/"), vision_reference.best_record())


@pytest.mark.parametrize("name", list(reference.COMBINATIONS))
def test_two_updates_with_each_combination_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("sum", states=states)
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(
        states[0]["trunk.head.weight"],
        states[1]["trunk.head.weight"],
    )


def test_the_loader_batches_carry_superclasses(tmp_path: Path) -> None:
    train, evaluation = reference.loader_batches(tmp_path)
    assert len(train) == len(evaluation) == vision_reference.LOADER_PASSES
    for batch in train[0]:
        labels = from_plain(batch["label"].tolist(), list[int])
        expected = [COARSE_OF_FINE[c] for c in labels]
        assert batch["label_coarse"].tolist() == expected


def test_the_full_networks_optimize_what_the_script_does() -> None:
    assert_exact(stored("optimizer/"), reference.optimizer_record())


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {f"draws/{name}" for name in reference.COMBINATIONS}


@pytest.mark.parametrize("name", list(reference.COMBINATIONS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_port_record_joins(
        reference,
        monkeypatch,
        {
            reference: {"lockstep_record": "lockstep/{}/", "draws_record": "draws/"},
        },
    )


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting in reference.LOCKSTEP_SETTINGS
        for name in reference.SETTING_COMBINATIONS[setting]
    ],
)
def test_two_updates_at_each_setting_match_the_script(key: str) -> None:
    setting, name = key.split("/")
    assert_exact(
        stored_settings(f"settings/{setting}/lockstep/{name}/"),
        reference.setting_lockstep_record(key),
    )


@pytest.mark.parametrize(
    "setting",
    [
        "resnet20-cifar100mt",
        pytest.param("vgg16bn-cifar100mt", marks=pytest.mark.compute_training),
    ],
)
def test_each_settings_full_networks_and_recipe_match_the_script(
    setting: str,
) -> None:
    assert_exact(
        stored_settings(f"settings/{setting}/"),
        reference.setting_record(setting),
    )


def shallow_step(factory: Callable[[], MultiTaskTrainLoop]) -> MultiTaskStep:
    """Build ``factory``'s step at its lockstep size, in place of the full one."""
    return reference.tiny_step(factory, epochs=vision_reference.EPOCHS)


def placeholder_record(setting: str) -> dict[str, torch.Tensor]:
    """Stand in for one setting's record with a single key naming it."""
    return {setting: torch.zeros(2)}


def test_a_lockstep_setting_adds_every_combinations_two_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrinks the full step and stubs the lockstep; the lockstep is tested above."""
    monkeypatch.setattr(reference, "full_step", shallow_step)
    monkeypatch.setattr(reference, "setting_lockstep_record", placeholder_record)
    record = reference.setting_record("vgg16bn-cifar100mt")
    assert {f"vgg16bn-cifar100mt/{name}" for name in reference.COMBINATIONS} <= set(
        record,
    )
    assert not set(reference.SETTING_COMBINATIONS["resnet20-cifar100mt"]) & {
        k.rsplit("/", 1)[-1]
        for k in reference.setting_record("resnet20-cifar100mt")
        if k.startswith("resnet20-cifar100mt/")
    }


def named_record(name: str) -> Callable[..., dict[str, torch.Tensor]]:
    """Return a stand-in for a record function: one key, ``name``."""
    return lambda *_: {name: torch.zeros(2)}


def test_the_port_record_joins_every_component_and_each_lockstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = (
        "combination_record",
        "schedule_record",
        "optimizer_record",
        "draws_record",
    )
    for name in components:
        monkeypatch.setattr(reference, name, named_record(name))
    monkeypatch.setattr(vision_reference, "best_record", named_record("best_record"))
    monkeypatch.setattr(reference, "lockstep_record", placeholder_record)
    record = reference.port_record()
    assert list(record) == [
        *components,
        "best_record",
        "data/coarse_of_fine",
        *reference.COMBINATIONS,
    ]
    assert torch.equal(record["data/coarse_of_fine"], torch.tensor(COARSE_OF_FINE))


class _Sum(nn.Module):
    """A combination that adds its two losses."""

    @override
    def forward(
        self,
        loss_fine: torch.Tensor,
        loss_coarse: torch.Tensor,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        del epoch, total_epochs
        return loss_fine + loss_coarse


def test_the_spy_runs_before_first_on_the_first_fine_loss_only() -> None:
    seen: list[torch.Tensor] = []
    spy = reference.CombinationSpy(_Sum(), before_first=seen.append)
    first, second = torch.tensor(1.0), torch.tensor(2.0)
    assert spy(first, torch.tensor(10.0), 0, 5).item() == 11.0
    assert spy(second, torch.tensor(20.0), 1, 5).item() == 22.0
    assert len(seen) == 1
    assert seen[0] is first
    assert [loss.item() for loss in spy.losses] == [11.0, 22.0]
    assert spy.epochs == [[0, 5], [1, 5]]


def test_settings_record_joins_every_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "setting_record", placeholder_record)
    assert list(reference.settings_record()) == list(reference.SETTINGS)


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/vgg16bn-cifar100mt/lockstep/uncertainty/loss"
    expected = stored_settings(key)
    bumped = expected[key].clone()
    bumped[1] = torch.nextafter(bumped[1], torch.tensor(torch.inf))
    port = reference.setting_lockstep_record("vgg16bn-cifar100mt/uncertainty")
    assert mismatches({key: bumped}, {key: port[key]}) == [f"{key}: 1/2 differ"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
