"""Tests for the pretrain, unlearn and evaluate job.

``reference_test.py`` replays a whole run of each variant against the harness;
these pin the job's wiring and its refusals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging

import pytest
import torch

from priml.baselines.mlsbench.tl.machine_unlearning import reference
from priml.baselines.mlsbench.tl.machine_unlearning.experiments import (
    exp000,
    exp004,
)
from priml.baselines.mlsbench.tl.machine_unlearning.unlearning import (
    seed_like_the_reference,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_the_dataset_resolves_beneath_the_jobs_base_dir() -> None:
    cfg = exp000()
    cfg.base_dir = "/root"
    assert (
        str(cfg.copy_tree().finalize().dataset.working_dir) == "/root/datasets/cifar10"
    )


def test_negative_unlearning_epochs_are_refused() -> None:
    cfg = exp000()
    cfg.unlearn_epochs = -1
    with pytest.raises(ValueError, match="unlearn_epochs"):
        cfg.make()


def test_seeding_is_torchs_own_unsalted_seed() -> None:
    seed_like_the_reference(42)
    drawn = torch.rand(3)
    torch.manual_seed(42)
    assert torch.equal(drawn, torch.rand(3))


def test_a_tiny_run_reports_the_reference_metrics_line(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference.write_synthetic(tmp_path)
    job = reference.tiny_config(exp004, tmp_path).make()
    with caplog.at_level(logging.INFO):
        job.run()
    (line,) = [
        r.getMessage() for r in caplog.records if "TEST_METRICS" in r.getMessage()
    ]
    assert line.startswith("TEST_METRICS retain_acc=")
    assert "unlearn_score=" in line


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
