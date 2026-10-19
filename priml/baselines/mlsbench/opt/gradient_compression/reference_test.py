"""Replay the ``optimization-gradient-compression`` port against the script's captured values.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.scripts.capture_reference import CEILING
from priml.baselines.mlsbench.opt.gradient_compression import reference
from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    ErrorFeedback,
)
from priml.baselines.mlsbench.opt.gradient_compression.scripts import (
    capture_reference,
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


@pytest.mark.parametrize("name", list(reference.COMPRESSORS))
def test_each_compressor_matches_the_script_round_by_round(name: str) -> None:
    assert_exact(stored(f"compressor/{name}/"), reference.compressor_record(name))


def test_error_feedback_residuals_are_part_of_the_record() -> None:
    assert {k.rsplit("/", 1)[-1] for k in stored("compressor/topk_ef/residual/")} == {
        "weight",
        "bias",
        "zero",
    }
    assert not stored("compressor/qsgd/residual/")


@pytest.mark.parametrize("name", list(reference.SETTINGS))
def test_each_setting_schedule_and_optimizer_match_the_script(name: str) -> None:
    assert_exact(
        stored(f"schedule/{name}") | stored(f"optimizer/{name}"),
        reference.schedule_record(name),
    )


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("name", list(reference.SETTINGS))
def test_each_setting_network_matches_the_script(name: str) -> None:
    assert_exact(
        stored(f"draws/{name}") | stored(f"forward/{name}/"),
        reference.network_record(name),
    )


def test_the_evaluated_epochs_and_reported_accuracy_match_the_script() -> None:
    assert_exact(
        stored("eval/") | stored("best/"),
        reference.eval_epochs_record() | reference.best_record(),
    )


def test_the_test_metrics_match_the_scripts_evaluate() -> None:
    assert_exact(stored("metric/"), reference.metric_record())


def test_the_test_pipelines_match_the_script() -> None:
    assert_exact(stored("transform/"), reference.transform_record())


def test_the_loader_shuffles_and_augments_like_the_script(tmp_path: Path) -> None:
    assert_exact(stored("loader/"), reference.loader_record(tmp_path))


@pytest.mark.parametrize("name", list(reference.COMPRESSORS))
def test_three_updates_with_each_compressor_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_port_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    sides: dict[str, dict[str, int]] = {"reference": {}, "port": {}}
    for line in (_DATA / "reference_sites.txt").read_text().splitlines():
        side, key, count = line.split()
        sides[side][key] = int(count)
    assert capture_reference.check_sites(sides["reference"], sides["port"]) == []
    assert reference.draw_sites(tmp_path) == sides["port"]


def test_the_golden_bites_on_one_ulp_of_a_residual() -> None:
    key = "compressor/topk_ef/residual/weight"
    feedback = reference.COMPRESSORS["topk_ef"]().step.compressor.make()
    assert isinstance(feedback, ErrorFeedback)
    record = reference.compressor_record("topk_ef", feedback)
    bumped = feedback.residuals["weight"].clone()
    bumped.view(-1)[0] = torch.nextafter(bumped.view(-1)[0], torch.tensor(1.0))
    assert mismatches(stored(key), {key: record[key]}) == []
    assert mismatches(stored(key), {key: bumped}) == [f"{key}: 1/200 differ"]


def test_the_golden_bites_on_one_ulp_of_a_qsgd_draw() -> None:
    key = "compressor/qsgd/weight"
    value = reference.compressor_record("qsgd")[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(1.0))
    assert mismatches(stored(key), {key: value}) == [f"{key}: 1/600 differ"]


def test_the_golden_fits_the_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= CEILING


def test_lockstep_hands_back_the_state_after_every_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    record = reference.lockstep_record("signsgd", states=states)
    assert len(states) == reference.LOCKSTEP_EPOCHS
    assert states[0].keys() == states[-1].keys()
    assert not torch.equal(states[0]["head.weight"], states[1]["head.weight"])
    assert record["lockstep/signsgd/loss"].shape == (reference.LOCKSTEP_EPOCHS,)


def test_the_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        return lambda *args: {f"{name}/{'/'.join(map(str, args))}": torch.zeros(2)}

    components = (
        "eval_epochs_record",
        "best_record",
        "metric_record",
        "transform_record",
        "loader_record",
        "schedule_record",
        "network_record",
        "compressor_record",
        "lockstep_record",
    )
    for name in components:
        monkeypatch.setattr(reference, name, fake(name))
    keys = set(reference.port_record(tmp_path))
    assert {key.split("/")[0] for key in keys} == set(components)
    assert {f"schedule_record/{name}" for name in reference.SETTINGS} <= keys
    assert {f"lockstep_record/{name}" for name in reference.COMPRESSORS} <= keys


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
