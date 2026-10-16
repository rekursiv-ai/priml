"""Tests for the ETTh1 training step."""

from pathlib import Path
from typing import Final, cast

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.checkpointer import Etth1Checkpointer
from priml.baselines.etth1.data_test import fixture_config
from priml.baselines.etth1.experiments import exp000
from priml.baselines.etth1.testing import (
    golden_record,
    tiny_batches,
    tiny_config,
    training_record,
)
from priml.baselines.etth1.train_step import Etth1TrainLoop
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors
from priml.train.ema import EMA


_CWD: Final = Path(__file__).resolve().parent

_GOLDEN: Final = _CWD / "testdata" / "dlinear_training.pt"


def test_three_updates_match_source_golden() -> None:
    with host_agnostic_numerics():
        actual = training_record(tiny_config().make())
    assert not mismatches(read_tensors(_GOLDEN), golden_record(actual, training=True))


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


def test_forward_and_eval_timers_count_calls() -> None:
    step = tiny_config().make()
    batch = next(tiny_batches())
    step.train_step(**batch)
    step.train_loss(**batch)
    step.eval_loss(**batch)
    assert step.timer_forward.global_count == 2
    assert step.timer_eval.global_count == 1
    assert step.timer_step.global_count == 1


def _loop_config(directory: Path) -> Etth1TrainLoop.Config:
    cfg = exp000()
    cfg.base_dir = directory
    cfg.dataset = fixture_config(directory / "data")
    cfg.dataset.base_dir = "/"
    cfg.max_epochs = 2
    cfg.max_steps = float("inf")
    cfg.checkpointer = None
    return cfg


def test_nonpositive_patience_is_rejected(tmp_path: Path) -> None:
    cfg = _loop_config(tmp_path)
    cfg.patience = 0
    with pytest.raises(ValueError, match="patience must be positive"):
        cfg.make()


def test_train_validates_each_epoch_then_closes(tmp_path: Path) -> None:
    loop = _loop_config(tmp_path).make()
    loop.train()
    assert loop.current_epoch == 2
    assert len(loop.validation_losses) == 2
    assert loop.step.global_step == 10
    with pytest.raises(RuntimeError, match="closed TrainLoop"):
        loop.train()


def test_worsening_validation_stops_after_patience(tmp_path: Path) -> None:
    cfg = _loop_config(tmp_path)
    cfg.patience = 2
    loop = cfg.make()
    try:
        for value in (0.5, 0.6):
            loop._publish_eval_metrics(
                {"total_loss": value},
                eval_time=0,
                step=0,
                is_final=False,
            )
        assert loop.bad_validation_epochs == 1
        assert not loop._should_stop_early()
        loop._publish_eval_metrics(
            {"total_loss": 0.7},
            eval_time=0,
            step=0,
            is_final=False,
        )
        assert loop.bad_validation_epochs == 2
        assert loop._should_stop_early()
        loop._training = True
        with pytest.raises(StopIteration):
            loop._get_next_batch()
    finally:
        loop._training = False
        loop.close()


def test_failed_epoch_validation_saves_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _loop_config(tmp_path)
    cfg.checkpointer = Etth1Checkpointer.Config()
    cfg.checkpointer.base_dir = "/"
    cfg.checkpointer.working_dir = tmp_path / "checkpoints"
    cfg.checkpointer.save_every = 5
    loop = cfg.make()
    try:
        for _ in range(5):
            loop._do_train_step(loop._get_next_batch())

        def fail() -> dict[str, object]:
            raise RuntimeError("validation failed")

        monkeypatch.setattr(loop, "eval", fail)
        with pytest.raises(RuntimeError, match="validation failed"):
            loop._complete_pending_epoch()
        assert (tmp_path / "checkpoints" / "step_00000005.pt").is_file()
        assert loop.validation_losses == []
    finally:
        loop.close()


def test_legacy_checkpoint_without_resume_state_is_rejected(tmp_path: Path) -> None:
    loop = _loop_config(tmp_path).make()
    try:
        state = dict(loop.state_dict())
        del state["best_validation_loss"]
        with pytest.raises(ValueError, match="legacy ETTh1 checkpoint"):
            loop.load_state_dict(state)
    finally:
        loop.close()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
