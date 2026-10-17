"""Replay the ``cv-sample-weighting`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.weighting import reference
from priml.baselines.mlsbench.weighting.experiments import exp000
from priml.testing.golden import mismatches, read_tensors


_DATA: Final = Path(__file__).resolve().parent / "testdata"


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
        vision.schedule_record(exp000) | vision.optimizer_record(exp000),
    )


def test_the_full_network_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


def test_the_reported_accuracy_is_the_best_the_script_reports() -> None:
    assert_exact(stored("best/"), vision.best_record())


def test_the_loader_yields_the_scripts_images_in_its_order(tmp_path: Path) -> None:
    train, _ = reference.loader_batches(tmp_path)
    assert_exact(stored("loader/"), reference.decode_indices(train))


@pytest.mark.parametrize("name", list(reference.WEIGHTS))
def test_two_updates_with_each_weighting_match_the_script(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
