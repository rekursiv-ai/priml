"""Tests for ETTh1 checkpoint selection and resume behavior."""

from pathlib import Path
from typing import cast

import pytest
import torch

from priml.baselines.etth1.checkpointer import Etth1Checkpointer
from priml.baselines.etth1.data_test import fixture_config
from priml.baselines.etth1.experiments import exp_smoke
from priml.baselines.etth1.train_step import Etth1TrainStep
from priml.lib.codec import from_plain, loads
from priml.testing.golden import mismatches
from priml.train.checkpointer import AsyncLocalStateDictStorer


@pytest.mark.parametrize(
    "asynchronous",
    [False, pytest.param(True, marks=pytest.mark.compute_distributed)],
)
def test_tied_score_selects_latest_checkpoint(
    tmp_path: Path,
    asynchronous: bool,
) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    loop = cfg.make()
    checker_cfg = Etth1Checkpointer.Config()
    checker_cfg.working_dir = tmp_path / "checkpoints"
    checker_cfg.best_metric = "total_loss"
    if asynchronous:
        checker_cfg.storer = AsyncLocalStateDictStorer.Config()
    checker = checker_cfg.make()
    try:
        if asynchronous:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=(tmp_path / "pg-init").resolve().as_uri(),
                rank=0,
                world_size=1,
            )
        assert checker.on_eval(loop, step=1, metrics={"total_loss": 0.7})
        assert checker.on_eval(loop, step=2, metrics={"total_loss": 0.7})
        checker.close()
        record = from_plain(
            loads((tmp_path / "checkpoints/best.json").read_text()),
            dict[str, object],
        )
        assert record["step"] == 2
        assert record["value"] == 0.7
    finally:
        try:
            checker.close()
        finally:
            loop.close()
            if asynchronous and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


@pytest.mark.compute_training
def test_epoch_checkpoint_resumes_after_validation(tmp_path: Path) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    cfg.max_steps = 10
    cfg.max_epochs = 2
    cfg.eval_every_epoch = True
    cfg.num_steps_eval = float("inf")
    cfg.checkpointer = Etth1Checkpointer.Config()
    cfg.checkpointer.best_metric = "total_loss"
    cfg.checkpointer.save_every = cfg.max_steps
    loop = cfg.make()
    try:
        loop._training = True
        for _ in range(5):
            loop._do_train_step(loop._get_next_batch())
        next_batch = loop._get_next_batch()
        assert loop.current_epoch == 1
        assert len(loop.validation_losses) == 1
        checkpoint = tmp_path / "runs/etth1/exp_smoke/checkpoints/step_00000005.pt"
        state = from_plain(
            cast(object, torch.load(checkpoint, weights_only=True)),
            dict[str, object],
        )
        dataset = from_plain(state["dataset"], dict[str, object])
        assert (
            from_plain(dataset["timer_epoch"], dict[str, object])["global_count"] == 1
        )
        assert len(cast(list[float], state["validation_losses"])) == 1
        loop._do_train_step(next_batch)
        assert isinstance(loop.step, Etth1TrainStep)
        expected = {
            key: value.clone() for key, value in loop.step.model.state_dict().items()
        }
        rng = torch.get_rng_state()
    finally:
        loop._training = False
        loop.close()
    resumed = cfg.make()
    try:
        assert resumed.current_epoch == 1
        assert len(resumed.validation_losses) == 1
        actual_batch = resumed._get_next_batch()
        assert not mismatches(
            from_plain(next_batch, dict[str, torch.Tensor]),
            from_plain(actual_batch, dict[str, torch.Tensor]),
        )
        resumed._do_train_step(actual_batch)
        assert isinstance(resumed.step, Etth1TrainStep)
        assert not mismatches(expected, resumed.step.model.state_dict())
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        resumed.close()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
