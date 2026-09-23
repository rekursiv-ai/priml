"""Tests for the ImageNet experiment configs.

``exp000`` is pinned whole by a pprint golden: it is the control, so any
change to it must be deliberate. Its budget-derived fields are also checked by
value against the ffcv-imagenet 16-epoch config.
"""

from __future__ import annotations

from configgle import PartialConfig
from configgle.pprinting import pformat
from configgle.testing import assert_pprint_golden

from priml.baselines.imagenet.data import NUM_TRAIN_SAMPLES
from priml.baselines.imagenet.experiments import exp000, exp_smoke
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.math.schedules import cyclic


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_matches_the_ffcv_16_epoch_budget() -> None:
    cfg = exp000()
    steps_per_epoch = NUM_TRAIN_SAMPLES // 512
    assert cfg.num_steps_eval == steps_per_epoch
    assert cfg.max_steps == cfg.step.train_budget_steps == 16 * steps_per_epoch
    assert cfg.step.schedule == PartialConfig(cyclic, peak=2 / 16)
    assert (cfg.step.resize_start, cfg.step.resize_end) == (11 / 16, 13 / 16)
    assert cfg.step.learning_rate == 0.5
    assert cfg.step.label_smoothing == 0.1


def test_exp_smoke_shrinks_only_the_budget_and_batch() -> None:
    smoke, base = exp_smoke(), exp000()
    assert smoke.max_steps == smoke.step.train_budget_steps == 4
    smoke.step.train_budget_steps = base.step.train_budget_steps
    assert pformat(smoke.step) == pformat(base.step)
    train = smoke.dataset.train_data_pipeline
    assert isinstance(train, DataPipeline.Config)
    (batcher,) = (p for p in train.processors if isinstance(p, Batcher.Config))
    assert batcher.size == 16


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
