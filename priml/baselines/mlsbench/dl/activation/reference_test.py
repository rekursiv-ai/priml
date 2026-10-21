"""Replay the ``dl-activation-function`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.activation import reference
from priml.baselines.mlsbench.dl.activation.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    assert_port_record_joins,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


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


def keyed(key: str) -> dict[str, torch.Tensor]:
    """Stand in for a record function: one zero tensor under ``key``."""
    return {key: torch.zeros(2)}


def test_every_activation_and_its_gradient_match_the_script() -> None:
    expected = stored("activation/")
    assert {k.split("/")[1] for k in expected} == set(reference.ACTIVATIONS)
    assert_exact(expected, reference.activation_record())


def test_the_learning_rate_of_all_200_epochs_matches_the_script() -> None:
    assert_exact(stored("schedule/"), vision_reference.schedule_record(exp000))


def test_the_optimizer_matches_the_script() -> None:
    assert_exact(stored("optimizer/"), vision_reference.optimizer_record(exp000))


def test_the_best_accuracy_matches_the_script() -> None:
    assert_exact(stored("best/"), vision_reference.best_record())


def test_the_resnet20_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


@pytest.mark.parametrize("name", list(reference.ACTIVATIONS))
def test_two_updates_with_each_activation_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision_reference.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    key = "activation/mish/grad"
    grad = stored(key)[key].clone()
    grad[0, 0] = torch.nextafter(grad[0, 0], torch.tensor(1.0))
    port = {key: reference.activation_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: grad}, port)] == [key]


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_every_golden_fits_the_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize(
    ("setting", "name"),
    [
        (setting, name)
        for setting, activations in reference.SETTING_ACTIVATIONS.items()
        for name in activations
    ],
)
def test_each_settings_full_network_with_each_activation_matches_the_script(
    setting: str,
    name: str,
) -> None:
    key = f"{setting}/{name}"
    expected = stored_settings(f"settings/{key}/")
    assert set(expected) == {
        f"settings/{key}/{part}" for part in ("draws", "train", "eval")
    }
    factory = reference.SETTING_ACTIVATIONS[setting][name]
    assert_exact(expected, vision_reference.setting_record(setting, factory, key=key))


@pytest.mark.parametrize(
    "setting",
    # The vision capture proves CIFAR-100's loader; the shared comparison skips it.
    [
        s
        for s in reference.SETTING_ACTIVATIONS
        if vision_reference.SETTINGS[s].dataset != "cifar100"
    ],
)
def test_each_settings_loader_shuffles_and_augments_like_the_script(
    setting: str,
    tmp_path: Path,
) -> None:
    assert_exact(
        stored_settings(f"settings/{setting}/loader/"),
        vision_reference.setting_loader_record(tmp_path, setting),
    )


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, activations in reference.SETTING_ACTIVATIONS.items()
        for name in activations
    ],
)
def test_two_updates_at_each_setting_with_each_activation_match_the_script(
    key: str,
) -> None:
    setting, name = key.split("/")
    assert_exact(
        stored_settings(f"settings/{setting}/lockstep/{name}/"),
        reference.setting_lockstep_record(key),
    )


def test_the_settings_record_holds_every_setting_lockstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "setting_lockstep_record", keyed)
    assert set(reference.settings_record()) == {
        f"{setting}/{name}"
        for setting, activations in reference.SETTING_ACTIVATIONS.items()
        for name in activations
    }


def test_a_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.setting_lockstep_record("mobilenetv2-fmnist/gelu", states=states)
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(
        next(iter(states[0].values())),
        next(iter(states[1].values())),
    )


def test_the_record_joins_every_component(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        def record(*args: object) -> dict[str, torch.Tensor]:
            return {f"{name}/{'/'.join(map(str, args))}": torch.zeros(2)}

        return record

    for name in ("activation_record", "draws_record", "lockstep_record"):
        monkeypatch.setattr(reference, name, fake(name))
    monkeypatch.setattr(vision_reference, "best_record", fake("best_record"))
    record = reference.port_record()
    assert {"activation_record/", "draws_record/", "best_record/"} <= set(record)
    assert {f"lockstep_record/{name}" for name in reference.ACTIVATIONS} <= set(record)
    assert {"schedule/lr", "optimizer/settings"} <= set(record)


def test_the_setting_lockstep_golden_bites_on_one_ulp() -> None:
    key = "settings/mobilenetv2-fmnist/lockstep/gelu/probe"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = reference.setting_lockstep_record("mobilenetv2-fmnist/gelu")
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/vgg16bn-cifar100/mish/eval"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = vision_reference.setting_record(
        "vgg16bn-cifar100",
        reference.SETTING_ACTIVATIONS["vgg16bn-cifar100"]["mish"],
        key="vgg16bn-cifar100/mish",
    )
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
