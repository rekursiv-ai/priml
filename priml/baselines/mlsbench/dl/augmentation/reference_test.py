"""Replay the ``cv-data-augmentation`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
The capture itself compares every pixel; the golden pins the augmentations
through the logits a fixed network gives them.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import reference
from priml.baselines.mlsbench.dl.augmentation.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.upstream_test import StoredPart
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

_DATA: Final = _CWD / "testdata"


def stored(prefix: str) -> dict[str, Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def assert_exact(expected: dict[str, Tensor], actual: dict[str, Tensor]) -> None:
    """Require two records to agree in every key, dtype, shape and bit."""
    assert expected
    assert mismatches(expected, actual) == []


@pytest.mark.parametrize("name", list(reference.AUGMENTS))
def test_each_training_pipeline_matches_its_script(name: str) -> None:
    assert_exact(stored(f"augment/{name}/"), reference.pipeline_record(name))


def test_the_test_pipeline_only_normalizes_as_the_script_does() -> None:
    assert_exact(stored("augment/test/"), reference.test_record())


def test_the_schedule_and_optimizer_match_the_script() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/"),
        vision.schedule_record(exp000) | vision.optimizer_record(exp000),
    )


def test_the_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), vision.draws_record())


@pytest.mark.parametrize("name", list(reference.AUGMENTS))
def test_two_updates_on_each_augmentation_match_the_script(name: str) -> None:
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.stored(reference.lockstep_record(name)),
    )


def test_the_golden_bites_on_one_wrong_mirror() -> None:
    images = reference.pipeline_images(reference.GOLDEN_IMAGES)
    media = reference.augmented("trivialaugment", images, seed=vision.SEED)
    media[0] = media[0].flip(-1)
    key = "augment/trivialaugment/logits"
    assert mismatches(stored(key), {key: reference.classified(media)}) == [
        f"{key}: {vision.CLASSES}/{vision.CLASSES * reference.GOLDEN_IMAGES} differ",
    ]


def test_the_stored_lockstep_omits_only_the_batches_and_can_keep_states() -> None:
    states: list[dict[str, Tensor]] = []
    record = reference.lockstep_record("cutout", states=states)
    assert set(record) - set(reference.stored(record)) == {"lockstep/cutout/media"}
    assert len(states) == vision.STEPS
    assert not torch.equal(states[0]["head.weight"], states[1]["head.weight"])


def test_the_whole_port_record_holds_every_part_the_golden_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = stored("")
    lockstep = golden | {
        f"lockstep/{n}/media": torch.zeros(1) for n in reference.AUGMENTS
    }
    monkeypatch.setattr(reference, "pipeline_record", StoredPart(golden, "augment/{}/"))
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        StoredPart(lockstep, "lockstep/{}/"),
    )
    monkeypatch.setattr(reference, "test_record", StoredPart(golden, "augment/test/"))
    monkeypatch.setattr(vision, "draws_record", StoredPart(golden, "draws/"))
    assert_exact(golden, reference.port_record())


def test_the_port_record_joins_every_part(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("pipeline_record", "lockstep_record", "test_record"):
        monkeypatch.setattr(reference, name, labeled(name))
    for name in ("schedule_record", "optimizer_record", "draws_record"):
        monkeypatch.setattr(vision, name, labeled(name))
    assert set(reference.port_record()) == {
        "schedule_record/exp000",
        "optimizer_record/exp000",
        "draws_record",
        "test_record",
        *(
            f"{part}/{name}"
            for name in reference.AUGMENTS
            for part in ("pipeline_record", "lockstep_record")
        ),
    }


def test_loaders_run_each_augmentation_over_the_synthetic_split(
    tmp_path: Path,
) -> None:
    train, evaluation = reference.loader_batches("cutout", tmp_path)
    assert len(train) == len(evaluation) == vision.LOADER_PASSES
    assert train[0][0]["media"].dtype == torch.float32


def stored_settings(prefix: str) -> dict[str, Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
@pytest.mark.parametrize("name", list(reference.AUGMENTS))
def test_each_settings_network_builds_and_runs_like_the_script(
    setting: str,
    name: str,
) -> None:
    prefix = f"settings/{setting}/{name}/"
    expected = {
        k: v for k, v in stored_settings(prefix).items() if not k.endswith("/augment")
    }
    assert set(expected) == {f"{prefix}{key}" for key in ("draws", "train", "eval")}
    assert_exact(
        expected,
        vision.setting_record(
            setting,
            reference.SETTINGS[setting][name],
            key=f"{setting}/{name}",
        ),
    )


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
@pytest.mark.parametrize("name", list(reference.AUGMENTS))
def test_each_settings_training_pipeline_matches_its_script(
    setting: str,
    name: str,
) -> None:
    key = f"settings/{setting}/{name}/augment"
    assert_exact(stored_settings(key), reference.setting_pipeline_record(setting, name))


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_each_settings_pristine_loader_matches_the_script(
    setting: str,
    tmp_path: Path,
) -> None:
    assert_exact(
        stored_settings(f"settings/{setting}/loader/"),
        vision.setting_loader_record(tmp_path, setting),
    )


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
@pytest.mark.parametrize("name", list(reference.AUGMENTS))
def test_two_updates_on_each_augmentation_match_the_script_at_each_setting(
    setting: str,
    name: str,
) -> None:
    states: list[dict[str, Tensor]] = []
    expected = stored_settings(f"{reference.lockstep_key(name, setting=setting)}/")
    assert len(expected) == 3
    assert_exact(
        expected,
        reference.stored(
            reference.lockstep_record(name, states=states, setting=setting),
        ),
    )
    assert len(states) == vision.STEPS


def labeled(label: str) -> Callable[..., dict[str, Tensor]]:
    """Return a stand-in record function: one tensor named by ``label`` and its call."""

    def record(*args: object, **kwargs: object) -> dict[str, Tensor]:
        names = [a.__name__ if isinstance(a, FunctionType) else str(a) for a in args]
        return {"/".join([label, *names, *map(str, kwargs.values())]): torch.zeros(2)}

    return record


def test_the_settings_record_joins_every_settings_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("setting_pipeline_record", "lockstep_record"):
        monkeypatch.setattr(reference, name, labeled(name))
    monkeypatch.setattr(vision, "task_settings_record", labeled("networks"))
    record = reference.settings_record(tmp_path)
    assert set(record) == {
        f"networks/{reference.SETTINGS}/{tmp_path}",
        *(
            key
            for setting, variants in reference.SETTINGS.items()
            for name in variants
            for key in (
                f"setting_pipeline_record/{setting}/{name}",
                f"lockstep_record/{name}/{setting}",
            )
        ),
    }


def test_the_fmnist_golden_bites_on_one_wrong_grayscale_mirror() -> None:
    setting = "mobilenetv2-fmnist"
    images = reference.prepared(
        setting,
        reference.source_images(setting, reference.GOLDEN_IMAGES),
    )
    assert images.shape[1] == 1
    media = reference.augmented(
        "randaugment",
        images,
        seed=vision.SEED,
        factory=reference.SETTINGS[setting]["randaugment"],
    )
    media[0] = media[0].flip(-1)
    key = f"settings/{setting}/randaugment/augment"
    assert mismatches(stored_settings(key), {key: reference.classified(media)}) == [
        f"{key}: {vision.CLASSES}/{vision.CLASSES * reference.GOLDEN_IMAGES} differ",
    ]


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "settings/mobilenetv2-fmnist/trivialaugment/eval"
    value = stored_settings(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(0.0))
    assert mismatches(stored_settings(key), {key: value}) == [
        f"{key}: 1/{value.numel()} differ",
    ]


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_each_golden_fits_the_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= 32_768


def test_the_census_counts_the_ports_operations_by_sign() -> None:
    with reference.op_census() as counts:
        _ = reference.augmented(
            "trivialaugment",
            reference.pipeline_images(40),
            seed=vision.SEED,
        )
    assert sum(counts.values()) == 40
    assert any(key.endswith("-") for key in counts)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
