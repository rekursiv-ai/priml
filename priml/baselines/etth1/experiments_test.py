"""Tests for the ETTh1 experiment setup and training loop."""

from pathlib import Path
from typing import cast

from configgle.testing import assert_pprint_golden
from torch import Tensor

import pytest
import torch

from priml.baselines.etth1.data_test import fixture_config
from priml.baselines.etth1.experiments import dlinear_type1, exp000, exp_smoke
from priml.baselines.etth1.train_step import Etth1TrainStep
from priml.lib.custom_json import DictCodec
from priml.testing.golden import mismatches


def test_canonical_config_golden() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_geometry_and_seed_propagation_without_data(tmp_path: Path) -> None:
    cfg = exp000()
    cfg.base_dir = tmp_path
    cfg.dataset.seq_len = 5
    cfg.dataset.pred_len = 3
    cfg.dataset.channels = 4
    cfg.seed = 37
    resolved = cfg.copy_tree().finalize()
    assert resolved.dataset.working_dir == tmp_path / "datasets/etth1"
    assert resolved.step.model.seq_len == 5
    assert resolved.step.model.pred_len == 3
    assert resolved.step.model.channels == 4
    assert resolved.step.seed == 37
    assert cfg.step.seed is None
    assert not (tmp_path / "datasets").exists()
    assert cfg.max_steps == 10260


def test_schedule_matches_source_epoch_boundaries() -> None:
    assert [dlinear_type1(epoch / 10) for epoch in range(10)] == [
        1,
        1,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
    ]
    assert dlinear_type1(1 / 10 - 1e-6) == 1
    assert dlinear_type1(2 / 10 - 1e-6) == 1


@pytest.mark.compute_training
def test_smoke_runs_end_to_end(tmp_path: Path) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.dataset = fixture_config(tmp_path / "datasets/etth1")
    # An absolute path should stay unchanged.
    cfg.dataset.base_dir = "/"
    loop = cfg.make()
    loop.train()
    assert loop.step.global_step == 3
    assert len(loop.validation_losses) == 1
    assert torch.isfinite(torch.tensor(loop.validation_losses)).all()


def test_early_stopping_ties_and_state_restore(tmp_path: Path) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    loop = cfg.make()
    try:
        for value in (0.8, 0.7, 0.9, 0.7):
            loop._publish_eval_metrics(
                {"total_loss": value},
                eval_time=0,
                step=0,
                is_final=False,
            )
        assert loop.bad_validation_epochs == 0
        for value in (0.8, 0.9):
            loop._publish_eval_metrics(
                {"total_loss": value},
                eval_time=0,
                step=0,
                is_final=False,
            )
        state = loop.state_dict()
        loop.bad_validation_epochs = 0
        loop.load_state_dict(state)
        assert loop.bad_validation_epochs == 2
        loop._publish_eval_metrics(
            {"total_loss": 0.8},
            eval_time=0,
            step=0,
            is_final=False,
        )
        assert loop._should_stop_early()
        assert loop.best_validation_loss == 0.7
    finally:
        loop.close()


@pytest.mark.compute_training
def test_mid_epoch_resume_replays_exact_next_update(tmp_path: Path) -> None:
    cfg = exp_smoke()
    cfg.base_dir = tmp_path
    cfg.dataset = fixture_config(tmp_path / "data")
    cfg.dataset.base_dir = "/"
    loop = cfg.make()
    try:
        first = loop._get_next_batch()
        loop._do_train_step(first)
        # Copy the state because state_dict values are live views.
        checkpoint = tmp_path / "resume.pt"
        torch.save(loop.state_dict(), checkpoint)
        batch = loop._get_next_batch()
        loop._do_train_step(batch)
        assert isinstance(loop.step, Etth1TrainStep)
        expected = {
            name: value.clone() for name, value in loop.step.model.state_dict().items()
        }
        expected_rng = torch.get_rng_state()
    finally:
        loop.close()
    resumed = cfg.make()
    try:
        resumed.load_state_dict(
            DictCodec.coerce(cast(object, torch.load(checkpoint, weights_only=True))),
        )
        actual_batch = resumed._get_next_batch()
        assert not mismatches(
            DictCodec.coerce(batch, Tensor),
            DictCodec.coerce(actual_batch, Tensor),
        )
        resumed._do_train_step(actual_batch)
        assert isinstance(resumed.step, Etth1TrainStep)
        assert not mismatches(expected, resumed.step.model.state_dict())
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert resumed.step.global_step == 2
    finally:
        resumed.close()
