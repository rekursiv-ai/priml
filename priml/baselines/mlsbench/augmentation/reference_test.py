"""Replay the ``cv-data-augmentation`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
The capture itself compares every pixel; the golden pins the augmentations
through the logits a fixed network gives them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.augmentation import reference
from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from torch import Tensor

_DATA: Final = Path(__file__).resolve().parent / "testdata"


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
        reference.schedule_record() | reference.optimizer_record(),
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


def test_the_stored_lockstep_omits_only_the_batches() -> None:
    record = reference.lockstep_record("cutout")
    assert set(record) - set(reference.stored(record)) == {"lockstep/cutout/media"}


def test_loaders_run_each_augmentation_over_the_synthetic_split(
    tmp_path: Path,
) -> None:
    train, evaluation = reference.loader_batches("cutout", tmp_path)
    assert len(train) == len(evaluation) == vision.LOADER_PASSES
    assert train[0][0]["media"].dtype == torch.float32


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
