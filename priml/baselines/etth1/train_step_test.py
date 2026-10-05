"""Tests for the ETTh1 training step."""

from pathlib import Path
from typing import Final, cast

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.testing import (
    golden_record,
    tiny_batches,
    tiny_config,
    training_record,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors
from priml.train.ema import EMA


_CWD: Final = Path(__file__).resolve().parent

_GOLDEN: Final = _CWD / "testdata" / "dlinear_training.pt"


@pytest.mark.compute_training
def test_three_updates_match_source_golden() -> None:
    with host_agnostic_numerics():
        actual = training_record(tiny_config().make())
    assert not mismatches(read_tensors(_GOLDEN), golden_record(actual, training=True))


@pytest.mark.compute_training
def test_training_golden_detects_changed_recipe() -> None:
    cfg = tiny_config()
    cfg.optimizer = PartialConfig(torch.optim.Adam, lr=2e-4)
    with host_agnostic_numerics():
        actual = training_record(cfg.make())
    assert any(
        "parameter/" in diff
        for diff in mismatches(
            read_tensors(_GOLDEN),
            golden_record(actual, training=True),
        )
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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("accumulate_grad_batches", 3),
        ("gradient_clip_norm", 1e-12),
        ("compile", PartialConfig(torch.compile)),
        ("dtype_autocast", torch.bfloat16),
        ("autocast_cache_enabled", True),
        ("ema", EMA.Config()),
        ("skip_step_on_nonfinite_grad", True),
    ],
)
def test_unsupported_controls_fail_before_training(name: str, value: object) -> None:
    cfg = tiny_config()
    setattr(cfg, name, value)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        cfg.make()
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.compute_training
def test_forward_and_eval_timers_count_calls() -> None:
    step = tiny_config().make()
    batch = next(tiny_batches())
    step.train_step(**batch)
    step.train_loss(**batch)
    step.eval_loss(**batch)
    assert step.timer_forward.global_count == 2
    assert step.timer_eval.global_count == 1
    assert step.timer_step.global_count == 1


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
