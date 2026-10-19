"""Replay the ``dl-weight-initialization`` port against values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.initialization import reference
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    assert_port_record_joins,
)
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


def keyed(key: str) -> dict[str, torch.Tensor]:
    """Stand in for a record function: one zero tensor under ``key``."""
    return {key: torch.zeros(2)}


def test_the_schedule_and_optimizer_match_the_script() -> None:
    expected = stored("schedule/") | stored("optimizer/")
    assert_exact(expected, reference.recipe_record())


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {
        f"draws/{name}" for name in reference.INITIALIZATIONS
    }


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_each_initialization_matches_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"network/{name}/"),
        reference.network_record(name, states=states),
    )
    assert len(states) == 1


@pytest.mark.parametrize("name", list(reference.INITIALIZATIONS))
def test_two_updates_after_each_initialization_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision_reference.STEPS


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize(
    ("setting", "name"),
    [
        (setting, name)
        for setting, inits in reference.SETTING_INITIALIZATIONS.items()
        for name in inits
    ],
)
def test_each_settings_full_network_with_each_initialization_matches_the_script(
    setting: str,
    name: str,
) -> None:
    key = f"{setting}/{name}"
    expected = {
        k: v
        for k, v in stored_settings(f"settings/{key}/").items()
        if "/lockstep/" not in k
    }
    assert set(expected) == {
        f"settings/{key}/{part}" for part in ("draws", "train", "eval")
    }
    factory = reference.SETTING_INITIALIZATIONS[setting][name]
    assert_exact(expected, vision_reference.setting_record(setting, factory, key=key))


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, inits in reference.SETTING_INITIALIZATIONS.items()
        for name in inits
    ],
)
def test_two_updates_at_each_setting_after_each_initialization_match_the_script(
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
        for setting, inits in reference.SETTING_INITIALIZATIONS.items()
        for name in inits
    }


def test_a_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.setting_lockstep_record(f"{reference.FMNIST}/fixup", states=states)
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(
        next(iter(states[0].values())),
        next(iter(states[1].values())),
    )


def test_a_network_run_hands_back_the_state_after_initialization() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.network_record("fixup", states=states)
    (state,) = states
    assert torch.equal(
        state["stages.0.0.norm2.weight"],
        torch.zeros_like(state["stages.0.0.norm2.weight"]),
    )


def test_the_setting_lockstep_golden_bites_on_one_ulp() -> None:
    key = f"settings/{reference.FMNIST}/lockstep/fixup/probe"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = reference.setting_lockstep_record(f"{reference.FMNIST}/fixup")
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


def test_the_fmnist_loader_shuffles_and_augments_like_the_script(
    tmp_path: Path,
) -> None:
    assert_exact(
        stored_settings(f"settings/{reference.FMNIST}/loader/"),
        vision_reference.setting_loader_record(tmp_path, reference.FMNIST),
    )


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = f"settings/{reference.FMNIST}/orthogonal/eval"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = vision_reference.setting_record(
        reference.FMNIST,
        reference.SETTING_INITIALIZATIONS[reference.FMNIST]["orthogonal"],
        key=f"{reference.FMNIST}/orthogonal",
    )
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_every_golden_fits_the_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= 32_768


def test_the_golden_bites_on_one_ulp() -> None:
    key = "network/fixup/train"
    logits = stored(key)[key].clone()
    logits[0, 0] = torch.nextafter(logits[0, 0], torch.tensor(0.0))
    port = {key: reference.network_record("fixup")[key]}
    assert [line.split(":")[0] for line in mismatches({key: logits}, port)] == [key]


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_port_record_joins(
        reference,
        monkeypatch,
        {
            reference: {
                "lockstep_record": "lockstep/{}/",
                "draws_record": "draws/",
                "network_record": "network/{}/",
            },
        },
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
