"""Tests for the ETTh1 training step."""

from pathlib import Path
from typing import Final, cast

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.testing import tiny_batches, tiny_config, training_record
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors


_GOLDEN: Final = Path(__file__).parent / "testdata" / "dlinear_training.pt"


@pytest.mark.compute_training
def test_three_updates_match_source_golden() -> None:
    with host_agnostic_numerics():
        actual = training_record(tiny_config().make())
    assert not mismatches(read_tensors(_GOLDEN), actual)


@pytest.mark.compute_training
def test_training_golden_detects_changed_recipe() -> None:
    cfg = tiny_config()
    cfg.optimizer = PartialConfig(torch.optim.Adam, lr=2e-4)
    with host_agnostic_numerics():
        actual = training_record(cfg.make())
    assert any(
        "parameter/" in diff for diff in mismatches(read_tensors(_GOLDEN), actual)
    )


def test_eval_does_not_change_weights_and_uses_mse() -> None:
    step = tiny_config().make()
    batch = next(tiny_batches())
    before = {key: value.clone() for key, value in step.model.state_dict().items()}
    actual = step.eval_loss(**batch)
    expected = torch.nn.functional.mse_loss(
        cast(Tensor, step.model(batch["media"])),
        batch["label"],
    )
    assert torch.equal(actual["loss"], expected.reshape(1))
    assert not mismatches(before, step.model.state_dict())
    assert step.global_step == 0
    assert not step.model.training
    assert torch.equal(step.train_loss(**batch)["loss"], actual["loss"])
    assert step.model.training
