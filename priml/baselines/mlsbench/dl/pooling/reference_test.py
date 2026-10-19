"""Replay the ``cv-pooling-aggregation`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.pooling import reference
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
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


def test_every_pool_matches_its_script_forward_and_backward() -> None:
    expected = stored("pool/")
    assert {k.split("/")[1] for k in expected} == set(reference.POOLS)
    assert "pool/gem/param_grad" in expected
    assert_exact(expected, reference.pool_record())


def test_the_schedule_optimizer_and_accuracy_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/") | stored("best/"),
        vision_reference.schedule_record()
        | vision_reference.optimizer_record()
        | vision_reference.best_record(),
    )


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {f"draws/{name}" for name in reference.POOLS}


@pytest.mark.parametrize("name", list(reference.POOLS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


@pytest.mark.parametrize("name", list(reference.POOLS))
def test_two_updates_with_each_pool_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision_reference.STEPS


def stored_settings() -> dict[str, torch.Tensor]:
    """Return every captured value of the other settings."""
    return read_tensors(_DATA / "reference_settings.pt")


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_the_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_settings_golden_holds_every_pools_fork() -> None:
    keys = set(stored_settings())
    assert {
        f"settings/{setting}/{name}/{kind}"
        for setting, forks in reference.SETTING_POOLS.items()
        for name in forks
        for kind in ("draws", "train", "eval")
    } <= keys
    assert any(k.startswith("settings/mobilenetv2-fmnist/loader/") for k in keys)


@pytest.mark.compute_large_fixture
def test_every_setting_fork_builds_and_runs_like_the_script(tmp_path: Path) -> None:
    assert_exact(
        {k: v for k, v in stored_settings().items() if "/lockstep/" not in k},
        vision_reference.task_settings_record(reference.SETTING_POOLS, tmp_path),
    )


@pytest.mark.parametrize(
    "name",
    [
        f"{setting}/{pool}"
        for setting, forks in reference.SETTING_POOLS.items()
        for pool in forks
    ],
)
def test_two_updates_of_each_setting_fork_match_the_script(name: str) -> None:
    setting, pool = name.split("/")
    prefix = f"settings/{setting}/lockstep/{pool}/"
    expected = {k: v for k, v in stored_settings().items() if k.startswith(prefix)}
    assert set(expected) == {f"{prefix}{kind}" for kind in ("loss", "logits", "probe")}
    assert_exact(expected, reference.setting_lockstep_record(name))


def test_the_settings_lockstep_record_joins_every_setting_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = {k: v for k, v in stored_settings().items() if "/lockstep/" in k}
    monkeypatch.setattr(
        reference,
        "setting_lockstep_record",
        StoredPart(golden, "settings/{}/lockstep/{}/"),
    )
    assert_exact(golden, reference.settings_lockstep_record())


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/mobilenetv2-fmnist/gem/eval"
    expected = {key: stored_settings()[key]}
    bumped = expected[key].clone()
    bumped.view(-1)[3] = torch.nextafter(bumped.view(-1)[3], torch.tensor(torch.inf))
    assert mismatches(expected, {key: bumped}) == [f"{key}: 1/{bumped.numel()} differ"]


def test_the_golden_bites_on_one_ulp() -> None:
    expected = stored("pool/gem/value")
    value = expected["pool/gem/value"].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = {"pool/gem/value": reference.pool_record()["pool/gem/value"]}
    assert mismatches({"pool/gem/value": value}, port) == ["pool/gem/value: 1/6 differ"]


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
