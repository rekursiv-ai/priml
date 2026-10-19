"""Replay the ``dl-normalization`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.normalization import reference
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.upstream_test import StoredPart
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def keyed(key: str) -> dict[str, torch.Tensor]:
    """Stand in for a record function: one zero tensor under ``key``."""
    return {key: torch.zeros(2)}


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


def test_every_norm_matches_its_script_forward_and_backward() -> None:
    assert_exact(stored("norm/"), reference.norm_record())


def test_the_schedule_optimizer_and_accuracy_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/") | stored("best/"),
        vision.schedule_record(reference.NORMS["batch_norm"])
        | vision.optimizer_record(reference.NORMS["batch_norm"])
        | vision.best_record(),
    )


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {f"draws/{name}" for name in reference.NORMS}


@pytest.mark.parametrize("name", list(reference.NORMS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


@pytest.mark.parametrize("name", list(reference.NORMS))
def test_two_updates_with_each_norm_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize(
    ("setting", "name"),
    [
        (setting, name)
        for setting, norms in reference.SETTING_NORMS.items()
        for name in norms
    ],
)
def test_each_settings_full_network_with_each_norm_matches_the_script(
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
    factory = reference.SETTING_NORMS[setting][name]
    assert_exact(expected, vision.setting_record(setting, factory, key=key))


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, norms in reference.SETTING_NORMS.items()
        for name in norms
    ],
)
def test_two_updates_at_each_setting_with_each_norm_match_the_script(key: str) -> None:
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
        for setting, norms in reference.SETTING_NORMS.items()
        for name in norms
    }


def test_a_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    record = reference.setting_lockstep_record(
        f"{reference.FMNIST}/group_norm",
        states=states,
    )
    assert len(states) == vision.STEPS
    assert not torch.equal(
        next(iter(states[0].values())),
        next(iter(states[1].values())),
    )
    assert record.keys() == {
        f"settings/{reference.FMNIST}/lockstep/group_norm/{part}"
        for part in ("loss", "logits", "probe")
    }


def test_the_record_joins_every_component(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        return lambda *_: {name: torch.zeros(2)}

    for name in ("norm_record", "draws_record", "lockstep_record"):
        monkeypatch.setattr(reference, name, fake(name))
    for name in ("schedule_record", "optimizer_record", "best_record"):
        monkeypatch.setattr(vision, name, fake(name))
    assert set(reference.port_record()) == {
        "norm_record",
        "draws_record",
        "lockstep_record",
        "schedule_record",
        "optimizer_record",
        "best_record",
    }


def test_the_fmnist_loader_shuffles_and_augments_like_the_script(
    tmp_path: Path,
) -> None:
    assert_exact(
        stored_settings(f"settings/{reference.FMNIST}/loader/"),
        vision.setting_loader_record(tmp_path, reference.FMNIST),
    )


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = f"settings/{reference.FMNIST}/group_norm/eval"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = vision.setting_record(
        reference.FMNIST,
        reference.SETTING_NORMS[reference.FMNIST]["group_norm"],
        key=f"{reference.FMNIST}/group_norm",
    )
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


def test_the_setting_lockstep_golden_bites_on_one_ulp() -> None:
    key = f"settings/{reference.FMNIST}/lockstep/switchable_norm/probe"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = reference.setting_lockstep_record(f"{reference.FMNIST}/switchable_norm")
    assert mismatches({key: value}, {key: port[key]}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_every_golden_fits_the_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= 32_768


def test_the_golden_bites_on_one_ulp() -> None:
    key = "norm/switchable_norm/train"
    value = stored(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = {key: reference.norm_record()[key]}
    assert mismatches({key: value}, port) == [f"{key}: 1/120 differ"]


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
