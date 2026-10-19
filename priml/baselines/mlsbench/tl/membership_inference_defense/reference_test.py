"""Replay the ``security-membership-inference-defense`` port against the driver's captured values.

``testdata/reference.pt`` holds what the pristine driver and each baseline's
edit of its defense produced, written by ``scripts/capture_reference.py``,
which also compares the image pipelines' pixels, the member split of 50,000
images, ResNet-20's initial state and every parameter after each update live.
Equality is exact; there is no tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.baselines.mlsbench.tl.membership_inference_defense import reference
from priml.baselines.mlsbench.tl.membership_inference_defense.scripts import (
    capture_reference,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def stored(prefix: str = "") -> dict[str, torch.Tensor]:
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


def test_every_defense_and_its_gradient_match_the_driver() -> None:
    expected = stored("defense/")
    assert {k.split("/")[1] for k in expected} == set(reference.DEFENSES)
    assert_exact(expected, reference.defense_record())


def test_the_attack_matches_the_driver_on_tied_scores() -> None:
    expected = stored("attack/")
    member = set(expected["attack/scores/member"].tolist())
    assert member & set(expected["attack/scores/nonmember"].tolist())
    assert_exact(expected, reference.attack_record())


def test_the_augmenting_pipeline_crops_and_mirrors_as_the_driver() -> None:
    assert_exact(
        stored("augment/"),
        reference.augment_record(
            reference.pipeline_record()["pipeline/cifar10/augment"],
        ),
    )


def test_the_rates_optimizer_and_split_match_the_driver() -> None:
    expected = stored("recipe/") | stored("split/")
    assert expected["recipe/lr"].shape == (reference.EPOCHS,)
    assert_exact(
        expected,
        reference.recipe_record() | reference.split_record(vision.LOADER_IMAGES),
    )


@pytest.mark.parametrize(
    "key",
    [
        "exp000",
        # Full-size VGG-16-BN and MobileNetV2 forwards take about a second.
        pytest.param("exp005", marks=pytest.mark.slow),
        pytest.param("exp010", marks=pytest.mark.slow),
    ],
)
def test_each_settings_network_initializes_and_runs_as_the_driver(key: str) -> None:
    setting, factory = reference.SETTINGS[key]
    assert_exact(
        stored(f"settings/{key}/"),
        vision.setting_record(setting, factory, key=key),
    )


def test_the_network_record_covers_every_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def record(setting: str, factory: object, *, key: str) -> dict[str, torch.Tensor]:
        del factory
        return {f"settings/{key}/{setting}": torch.zeros(1)}

    monkeypatch.setattr(vision, "setting_record", record)
    assert set(reference.network_record()) == {
        "settings/exp000/resnet20-cifar10",
        "settings/exp005/vgg16bn-cifar100",
        "settings/exp010/mobilenetv2-fmnist",
    }


@pytest.mark.parametrize("name", list(reference.DEFENSES))
def test_a_whole_run_with_each_defense_matches_the_driver(
    name: str,
    tmp_path: Path,
) -> None:
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, tmp_path),
    )


def test_relaxloss_ascends_flattens_and_descends_in_its_run() -> None:
    expected = stored("lockstep/relaxloss/")
    assert expected["lockstep/relaxloss/epoch"].tolist() == [0, 1]
    rates = expected["lockstep/relaxloss/lr"]
    assert torch.equal(rates, torch.tensor([0.1, 0.1 * 0.1], dtype=torch.float64))
    assert not torch.equal(
        expected["lockstep/relaxloss/loss"],
        stored("lockstep/pristine/")["lockstep/pristine/loss"],
    )


def test_a_lockstep_run_hands_back_the_state_after_each_update(tmp_path: Path) -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("pristine", tmp_path, states=states)
    assert len(states) == len(
        stored("lockstep/pristine/loss")["lockstep/pristine/loss"],
    )
    assert not torch.equal(states[0]["stem.weight"], states[-1]["stem.weight"])


def test_the_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        return lambda *args: {f"{name}{args[:1]}": torch.zeros(1)}

    parts = (
        "defense_record",
        "attack_record",
        "augment_record",
        "recipe_record",
        "split_record",
        "network_record",
        "lockstep_record",
    )
    for name in parts:
        monkeypatch.setattr(reference, name, fake(name))
    monkeypatch.setattr(
        reference,
        "pipeline_record",
        lambda: {"pipeline/cifar10/augment": torch.zeros(1)},
    )
    keys = set(reference.port_record(tmp_path))
    assert {key.split("(")[0] for key in keys} == set(parts)
    assert {k for k in keys if k.startswith("lockstep")} == {
        f"lockstep_record({name!r},)" for name in reference.DEFENSES
    }


def test_the_port_reaches_the_drivers_random_call_sites(tmp_path: Path) -> None:
    sides: dict[str, dict[str, int]] = {"reference": {}, "port": {}}
    for line in (_DATA / "reference_sites.txt").read_text().splitlines():
        side, key, count = line.split()
        sides[side][key] = int(count)
    assert capture_reference.check_sites(sides["reference"], sides["port"]) == []
    assert capture_reference.port_sites(tmp_path) == sides["port"]


def test_the_captured_golden_fits_the_size_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= vision_capture.CEILING


def test_the_golden_bites_on_one_ulp(tmp_path: Path) -> None:
    key = "lockstep/relaxloss/loss"
    want = stored(key)[key].clone()
    want[-1] = torch.nextafter(want[-1], torch.tensor(torch.inf))
    got = {key: reference.lockstep_record("relaxloss", tmp_path)[key]}
    assert mismatches({key: want}, got) == [f"{key}: 1/{want.numel()} differ"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
