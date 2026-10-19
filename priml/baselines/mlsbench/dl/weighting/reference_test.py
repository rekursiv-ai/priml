"""Replay the ``cv-sample-weighting`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    assert_port_record_joins,
)
from priml.baselines.mlsbench.dl.weighting import reference
from priml.baselines.mlsbench.dl.weighting.experiments import exp000
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep

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


def test_every_weighting_and_its_loss_match_the_script() -> None:
    expected = stored("weights/") | stored("loss/")
    assert len(expected) == 4 * len(reference.WEIGHTS)
    assert_exact(expected, reference.weights_record())


def test_the_cifar100_long_tail_counts_and_weights_match_main() -> None:
    record, _ = reference.counts_record()
    assert_exact(stored("counts/") | stored("main/"), record)


def test_the_synthetic_cut_keeps_the_scripts_images_in_its_order() -> None:
    assert_exact(stored("subset/"), reference.subset_record())


def test_the_schedule_and_optimizer_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/"),
        vision_reference.schedule_record(exp000)
        | vision_reference.optimizer_record(exp000),
    )


def test_the_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


def test_the_reported_accuracy_is_the_best_the_script_reports() -> None:
    assert_exact(stored("best/"), vision_reference.best_record())


def test_the_loader_yields_the_scripts_images_in_its_order(tmp_path: Path) -> None:
    train, _ = reference.loader_batches(tmp_path)
    assert_exact(stored("loader/"), reference.decode_indices(train))


@pytest.mark.parametrize("name", list(reference.WEIGHTS))
def test_two_updates_with_each_weighting_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision_reference.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    expected = stored("weights/effective_number")
    weight = expected["weights/effective_number"].clone()
    weight[1] = torch.nextafter(weight[1], torch.tensor(0.0))
    port = {
        "weights/effective_number": reference.weights_record()[
            "weights/effective_number"
        ],
    }
    assert [
        line.split(":")[0]
        for line in mismatches({"weights/effective_number": weight}, port)
    ] == ["weights/effective_number"]


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert_port_record_joins(
        reference,
        monkeypatch,
        {
            reference: {"lockstep_record": "lockstep/{}/", "draws_record": "draws/"},
        },
        tmp_path,
    )


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize("name", list(reference.SETTINGS))
def test_each_settings_counts_and_weights_match_main(name: str) -> None:
    prefix = f"settings/{name}/"
    expected = stored_settings(f"{prefix}counts") | stored_settings(f"{prefix}weight/")
    assert len(expected) == 1 + len(reference.WEIGHTS)
    record, _ = reference.setting_counts_record(name)
    assert_exact(expected, record)


@pytest.mark.parametrize(
    "key",
    [
        f"{name}/{weighting}"
        for name, setting in reference.SETTINGS.items()
        if setting.lockstep
        for weighting in setting.experiments
    ],
)
def test_two_updates_at_each_setting_match_the_script(key: str) -> None:
    name, weighting = key.split("/")
    assert_exact(
        stored_settings(f"settings/{name}/lockstep/{weighting}/"),
        reference.setting_lockstep_record(key),
    )


def test_the_cifar10_loader_yields_the_scripts_images_in_its_order(
    tmp_path: Path,
) -> None:
    setting = reference.SETTINGS["resnet32-cifar10lt"]
    uniform = setting.experiments["uniform"]
    train, _ = reference.loader_batches(
        tmp_path,
        uniform,
        imbalance_ratio=reference.SPLIT_RATIO_10,
    )
    record = reference.decode_indices(train, uniform().dataset)
    assert_exact(
        stored_settings("settings/resnet32-cifar10lt/loader/"),
        {"settings/resnet32-cifar10lt/loader/index": record["loader/index"]},
    )


@pytest.mark.parametrize(
    "name",
    [
        "resnet32-cifar10lt",
        pytest.param("vgg16bn-cifar100lt", marks=pytest.mark.compute_training),
    ],
)
def test_each_settings_full_network_and_recipe_match_the_script(name: str) -> None:
    expected = {
        k: v
        for k, v in stored_settings(f"settings/{name}/").items()
        if "/loader/" not in k
    }
    assert_exact(expected, reference.setting_record(name))


def shallow_step(factory: Callable[[], VisionTrainLoop]) -> VisionTrainStep:
    """Build ``factory``'s step at its lockstep size, in place of the full one."""
    return vision_reference.tiny_step(
        factory,
        blocks=(1, 1, 1),
        classes=vision_reference.CLASSES,
        epochs=vision_reference.EPOCHS,
    )


def placeholder_record(name: str) -> dict[str, torch.Tensor]:
    """Stand in for one setting's record with a single key naming it."""
    return {name: torch.zeros(2)}


def test_a_lockstep_setting_adds_every_weightings_two_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrinks the VGG setting's full step to its lockstep size, to stay fast."""
    monkeypatch.setattr(vision_reference, "full_step", shallow_step)
    record = reference.setting_record("vgg16bn-cifar100lt")
    for name in reference.WEIGHTS:
        key = f"settings/vgg16bn-cifar100lt/lockstep/{name}/"
        assert_exact(
            stored_settings(key),
            {k: v for k, v in record.items() if k.startswith(key)},
        )


def test_settings_record_joins_every_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "setting_record", placeholder_record)
    assert list(reference.settings_record()) == list(reference.SETTINGS)


def test_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    record = reference.lockstep_record("uniform", states=states)
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(states[0]["head.weight"], states[1]["head.weight"])
    assert_exact(stored("lockstep/uniform/"), record)


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/vgg16bn-cifar100lt/weight/effective_number"
    weight = stored_settings(key)[key].clone()
    weight[1] = torch.nextafter(weight[1], torch.tensor(0.0))
    record, _ = reference.setting_counts_record("vgg16bn-cifar100lt")
    assert mismatches({key: weight}, {key: record[key]}) == [
        f"{key}: 1/{weight.numel()} differ",
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
