"""Tests for evaluating after chosen epochs and stopping early on a metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch
import torch.distributed as dist

from priml.data.dummy import DummyDataset
from priml.runtime import SingleProcess
from priml.train.epoch_loop import (
    EarlyStopping,
    EpochLoopConfig,
    EpochTrainLoop,
    first_periodic_and_last,
)
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput


class _Model(nn.Module):
    class Config(Fig["_Model"], make_with_kwargs=True):
        in_features: int = 2
        out_features: int = 3

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    @override
    def forward(self, media: Tensor, **_kwargs: object) -> Tensor:
        return self.linear(media)


def _cross_entropy(output: Tensor, *, label: Tensor, **_kwargs: object) -> LossOutput:
    return {"loss": nn.functional.cross_entropy(output, label, reduction="none")}


def _loop_config(
    *,
    epochs: int,
) -> EpochLoopConfig[TrainStep.Config, DummyDataset.Config]:
    """Return a frozen-weights loop, so every evaluation sees the same loss."""
    step = TrainStep.Config()
    step.model = _Model.Config()
    step.optimizer = PartialConfig(torch.optim.SGD, lr=0.0)
    step.loss = PartialConfig(_cross_entropy)
    step.parallelism = NoParallel.Config(device="cpu")
    step.compile = None
    config = EpochLoopConfig[TrainStep.Config, DummyDataset.Config]()
    config.step = step
    config.dataset = DummyDataset.Config(
        input_shape=(2,),
        num_classes=3,
        num_samples=4,
        batch_size=2,
        device="cpu",
    )
    config.runtime = SingleProcess.Config(device="cpu")
    config.checkpointer = None
    config.max_epochs = epochs
    config.num_steps_eval = 0
    return config


def test_the_reference_policy_picks_the_first_every_tenth_and_the_last() -> None:
    picked = [
        e for e in range(1, 26) if first_periodic_and_last(e, total_epochs=25, every=10)
    ]
    assert picked == [1, 10, 20, 25]


def test_the_loop_evaluates_only_after_the_picked_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loop_config(epochs=4)
    config.eval_epochs = PartialConfig(first_periodic_and_last, every=3)
    loop = config.make()
    assert isinstance(loop, EpochTrainLoop)
    epochs: list[int] = []
    evaluate = loop.eval

    def spy() -> dict[str, object]:
        epochs.append(loop.current_epoch)
        return evaluate()

    monkeypatch.setattr(loop, "eval", spy)
    loop.train()
    assert epochs == [1, 3, 4]


def test_early_stopping_counts_epochs_since_a_strict_improvement() -> None:
    stopping = EarlyStopping.Config(patience=2).make()
    verdicts = [
        stopping.update({"total_loss": loss}, epoch=epoch)
        for epoch, loss in enumerate([5.0, 4.0, 4.0, 4.0, 3.0], start=1)
    ]
    assert verdicts == [False, False, False, True, False]
    assert (stopping.best, stopping.best_epoch) == (3.0, 5)


def test_early_stopping_waits_for_min_epochs_and_honors_the_mode() -> None:
    stopping = EarlyStopping.Config(mode="max", patience=1, min_epochs=3).make()
    verdicts = [
        stopping.update({"total_loss": loss}, epoch=epoch)
        for epoch, loss in enumerate([1.0, 0.0, 0.0, 0.0], start=1)
    ]
    assert verdicts == [False, False, True, True]


def test_early_stopping_state_resumes_and_bad_settings_are_refused() -> None:
    stopping = EarlyStopping.Config(patience=1).make()
    stopping.update({"total_loss": 1.0}, epoch=1)
    stopping.update({"total_loss": 1.0}, epoch=2)
    resumed = EarlyStopping.Config(patience=1).make()
    resumed.load_state_dict(stopping.state_dict())
    assert resumed.state_dict() == {"best": 1.0, "best_epoch": 1, "stop": True}
    with pytest.raises(KeyError, match="not an eval metric"):
        resumed.update({"other": 1.0}, epoch=3)
    with pytest.raises(ValueError, match="patience must be positive"):
        EarlyStopping.Config(patience=0).make()


def test_the_loop_stops_when_the_metric_stalls_and_keeps_the_record() -> None:
    config = _loop_config(epochs=10)
    config.early_stopping = EarlyStopping.Config(patience=1)
    loop = config.make()
    assert isinstance(loop, EpochTrainLoop)
    loop.train()
    assert loop.step.global_step == 5
    state = loop.state_dict()
    assert "early_stopping" in state
    assert state["early_stopping"]["stop"]
    resumed = config.make()
    assert isinstance(resumed, EpochTrainLoop)
    resumed.load_state_dict(state)
    assert resumed.early_stopping is not None
    assert resumed.early_stopping.stop
    resumed.close()


def test_the_loop_without_early_stopping_runs_every_epoch() -> None:
    loop = _loop_config(epochs=3).make()
    assert isinstance(loop, EpochTrainLoop)
    loop.train()
    assert loop.step.global_step == 6
    assert "early_stopping" not in loop.state_dict()


def test_the_loop_runs_with_unbounded_epochs() -> None:
    config = _loop_config(epochs=1)
    config.max_epochs = float("inf")
    config.max_steps = 6
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 6


def test_ranks_follow_rank_zeros_stop_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loop_config(epochs=3)
    config.early_stopping = EarlyStopping.Config()
    loop = config.make()
    assert isinstance(loop, EpochTrainLoop)

    def rank_zero_says_stop(shared: list[bool], src: int) -> None:
        assert src == 0
        shared[0] = True

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", rank_zero_says_stop)
    assert not (loop.early_stopping and loop.early_stopping.stop)
    assert loop._should_stop_early()
    loop.close()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
