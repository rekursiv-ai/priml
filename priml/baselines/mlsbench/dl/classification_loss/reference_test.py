"""Replay the ``cv-classification-loss`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``, which also compares
every parameter after each update live. Equality is exact; there is no
tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.classification_loss import reference
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
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


def test_every_loss_and_its_gradient_match_the_script() -> None:
    assert_exact(stored("loss/"), reference.loss_record())


@pytest.mark.parametrize("name", list(reference.LOSSES))
def test_two_updates_with_each_loss_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_golden_holds_only_the_losses_and_their_updates() -> None:
    assert {k.split("/")[0] for k in stored("")} == {"loss", "lockstep"}


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_the_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def test_every_loss_matches_the_script_at_each_settings_class_count() -> None:
    expected = stored_settings("settings/")
    expected = {k: v for k, v in expected.items() if "/loss/" in k}
    assert expected["settings/vgg16bn-cifar100/loss/poly/grad"].shape == (3, 100)
    assert expected["settings/mobilenetv2-fmnist/loss/focal/grad"].shape == (3, 10)
    assert_exact(expected, reference.setting_loss_record())


@pytest.mark.compute_large_fixture
def test_every_setting_fork_builds_and_runs_like_the_script(tmp_path: Path) -> None:
    expected = {
        k: v
        for k, v in stored_settings("settings/").items()
        if "/loss/" not in k and "/lockstep/" not in k
    }
    assert_exact(
        expected,
        vision_reference.task_settings_record(reference.SETTING_LOSSES, tmp_path),
    )


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, forks in reference.SETTING_LOSSES.items()
        for name in forks
    ],
)
def test_two_updates_of_each_setting_fork_match_the_script(key: str) -> None:
    setting, name = key.split("/")
    assert_exact(
        stored_settings(f"settings/{setting}/lockstep/{name}/"),
        reference.setting_lockstep_record(key),
    )


def test_a_setting_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    record = reference.setting_lockstep_record(
        "mobilenetv2-fmnist/focal",
        states=states,
    )
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(states[0]["head.weight"], states[1]["head.weight"])
    assert record["settings/mobilenetv2-fmnist/lockstep/focal/context"].tolist() == [
        [vision_reference.CLASSES, 0, vision_reference.EPOCHS],
        [vision_reference.CLASSES, 1, vision_reference.EPOCHS],
    ]


def test_the_settings_golden_holds_every_fork() -> None:
    keys = set(stored_settings(""))
    for setting, forks in reference.SETTING_LOSSES.items():
        for name in forks:
            assert {
                f"settings/{setting}/{name}/{kind}"
                for kind in ("draws", "train", "eval")
            } | {
                f"settings/{setting}/loss/{name}/{kind}" for kind in ("value", "grad")
            } | {
                f"settings/{setting}/lockstep/{name}/{kind}"
                for kind in ("loss", "logits", "probe", "context")
            } <= keys
    assert any(k.startswith("settings/mobilenetv2-fmnist/loader/") for k in keys)


def test_the_settings_comparison_sees_a_change_of_one_ulp() -> None:
    key = "settings/vgg16bn-cifar100/loss/focal/grad"
    port = {key: reference.setting_loss_record()[key]}
    bumped = stored_settings(key)[key].clone()
    bumped.view(-1)[7] = torch.nextafter(bumped.view(-1)[7], torch.tensor(torch.inf))
    assert mismatches({key: bumped}, port) == [f"{key}: 1/300 differ"]


def test_setting_logits_span_the_class_count_and_reach_the_last_class() -> None:
    logits, label = reference.setting_logits(10)
    assert logits.shape == (3, 10)
    assert label.tolist() == [0, 9, 5]
    assert len(set(logits[0].tolist())) == 10


def test_the_comparison_sees_a_change_of_one_ulp() -> None:
    expected = stored("loss/cross_entropy/value")
    bumped = {
        k: torch.nextafter(v, torch.tensor(torch.inf)) for k, v in expected.items()
    }
    assert mismatches(expected, bumped) == ["loss/cross_entropy/value: 1/1 differ"]


def test_a_lockstep_run_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("cross_entropy", states=states)
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(states[0]["stem.weight"], states[1]["stem.weight"])


def test_the_settings_record_joins_the_losses_and_every_forks_lockstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def losses() -> dict[str, torch.Tensor]:
        return {"setting_loss_record": torch.zeros(2)}

    def lockstep(name: str) -> dict[str, torch.Tensor]:
        return {f"lockstep/{name}": torch.zeros(2)}

    monkeypatch.setattr(reference, "setting_loss_record", losses)
    monkeypatch.setattr(reference, "setting_lockstep_record", lockstep)
    assert set(reference.settings_port_record()) == {
        "setting_loss_record",
        *(
            f"lockstep/{setting}/{name}"
            for setting, forks in reference.SETTING_LOSSES.items()
            for name in forks
        ),
    }


def test_the_loss_record_holds_a_value_and_a_gradient_per_loss() -> None:
    record = reference.loss_record()
    assert set(record) == {
        f"loss/{name}/{kind}" for name in reference.LOSSES for kind in ("value", "grad")
    }
    assert record["loss/poly/grad"].shape == (3, 4)
    assert record["loss/poly/value"].shape == ()


def test_the_record_joins_the_losses_and_every_lockstep_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference,
        "loss_record",
        lambda: {"loss_record": torch.zeros(2)},
    )

    def lockstep(name: str) -> dict[str, torch.Tensor]:
        return {f"lockstep/{name}": torch.zeros(2)}

    monkeypatch.setattr(reference, "lockstep_record", lockstep)
    assert set(reference.port_record()) == {
        "loss_record",
        *(f"lockstep/{name}" for name in reference.LOSSES),
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
