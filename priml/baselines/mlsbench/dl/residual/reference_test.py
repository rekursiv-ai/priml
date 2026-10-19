"""Replay the ``dl-residual-connection`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.residual import reference
from priml.baselines.mlsbench.dl.residual.blocks import StochasticDepthBlock
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.model import ResNet, sized_blocks
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


def test_every_block_matches_its_script_forward_backward_and_draws() -> None:
    expected = stored("block/")
    assert {k.split("/")[1] for k in expected} == set(reference.VARIANTS)
    draws = torch.cat(
        [
            v
            for k, v in expected.items()
            if k.startswith("block/stochastic_depth/") and k.endswith("/draws")
        ],
    )
    assert bool((draws < 0.5).any())
    assert bool((draws >= 0.5).any())
    assert_exact(expected, reference.block_record())


def test_the_tiny_stochastic_depth_network_resets_its_count_as_the_full_one() -> None:
    model = reference.VARIANTS["stochastic_depth"]().step.model
    vision.tiny_model(model, blocks=(2, 1, 1))
    assert isinstance(model, ResNet.Config)
    block = model.block
    assert isinstance(block, StochasticDepthBlock.Config)
    assert block.reset_width == vision.TINY_WIDTHS[16]
    copies = [b for stage in sized_blocks(model) for b, _ in stage]
    # Upstream restarts on every stride-1 block of the first stage, as the
    # full network's do: both of that stage's blocks stand at position 1.
    assert [b.depth for b in copies if isinstance(b, StochasticDepthBlock.Config)] == [
        1,
        1,
        2,
        3,
    ]


def test_the_schedule_optimizer_and_accuracy_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/") | stored("best/"),
        vision.schedule_record() | vision.optimizer_record() | vision.best_record(),
    )


def test_the_golden_holds_the_draws_of_every_full_network() -> None:
    assert set(stored("draws/")) == {f"draws/{name}" for name in reference.VARIANTS}


@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_each_full_network_leaves_the_generator_where_the_script_does(
    name: str,
) -> None:
    key = f"draws/{name}"
    assert_exact({key: stored(key)[key]}, reference.draws_record([name]))


@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_two_updates_with_each_block_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision.STEPS


def test_the_port_record_holds_every_stored_key_of_its_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "VARIANTS", {"basic": reference.VARIANTS["basic"]})
    expected = {
        k
        for k in stored("")
        if k.split("/")[0] in {"schedule", "optimizer", "best"}
        or k.split("/")[1] == "basic"
    }
    assert reference.port_record().keys() == expected


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_each_settings_network_builds_and_runs_like_the_script(
    setting: str,
    name: str,
) -> None:
    expected = stored_settings(f"settings/{setting}/{name}/")
    expected.pop(f"settings/{setting}/{name}/positions", None)
    assert set(expected) == {
        f"settings/{setting}/{name}/{key}" for key in ("draws", "train", "eval")
    }
    assert_exact(
        expected,
        vision.setting_record(
            setting,
            reference.SETTINGS[setting][name],
            key=f"{setting}/{name}",
        ),
    )


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_stochastic_depth_numbers_its_blocks_as_the_script_at_each_setting(
    setting: str,
) -> None:
    key = f"settings/{setting}/stochastic_depth/positions"
    assert_exact(stored_settings(key), reference.positions_record(setting))


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_two_updates_with_each_block_match_the_script_at_each_setting(
    setting: str,
    name: str,
) -> None:
    states: list[dict[str, torch.Tensor]] = []
    expected = stored_settings(f"{reference.lockstep_key(name, setting=setting)}/")
    assert len(expected) == 3
    assert_exact(
        expected,
        reference.lockstep_record(name, states=states, setting=setting),
    )
    assert len(states) == vision.STEPS


def test_the_resnet20_cifar10_loader_matches_the_script(tmp_path: Path) -> None:
    assert_exact(
        stored_settings("settings/resnet20-cifar10/loader/"),
        vision.setting_loader_record(tmp_path, "resnet20-cifar10"),
    )


def labeled(label: str) -> Callable[..., dict[str, torch.Tensor]]:
    """Return a stand-in record function: one tensor named by ``label`` and its call."""

    def record(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"/".join([label, *map(str, [*args, *kwargs.values()])]): torch.zeros(2)}

    return record


def test_the_settings_record_joins_every_settings_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("positions_record", "lockstep_record"):
        monkeypatch.setattr(reference, name, labeled(name))
    monkeypatch.setattr(vision, "task_settings_record", labeled("networks"))
    record = reference.settings_record(tmp_path)
    assert set(record) == {
        f"networks/{reference.SETTINGS}/{tmp_path}",
        *(f"positions_record/{setting}" for setting in reference.SETTINGS),
        *(
            f"lockstep_record/{name}/{setting}"
            for setting, variants in reference.SETTINGS.items()
            for name in variants
        ),
    }


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/resnet110-cifar100/stochastic_depth/train"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = vision.setting_record(
        "resnet110-cifar100",
        reference.SETTINGS["resnet110-cifar100"]["stochastic_depth"],
        key="resnet110-cifar100/stochastic_depth",
    )
    port = {key: port[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_each_golden_fits_the_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= 32_768


def test_the_golden_bites_on_one_ulp() -> None:
    key = "block/pre_activation/projected/eval"
    value = stored(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    port = {key: reference.block_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
