"""Initial controls for learning from an imperfect simulator."""

from configgle import PartialConfig

import torch

from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection


def exp000() -> ForecastStudy.Config:
    """DLinear trained only on the fixed reference sample.

    Hypothesis:
      An established forecaster gives a useful control before adding an agent.

    References:
      https://arxiv.org/abs/2205.13504
      priml.baselines.etth1.experiments.exp000 for the DLinear implementation.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = ForecastStudy.Config()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.step.optimizer = PartialConfig(torch.optim.Adam, lr=1e-3)
    return cfg


def exp001() -> ForecastStudy.Config:
    """exp000 with half its update budget spent on cheap simulator data.

    Hypothesis:
      Simulator pretraining may help when its dynamics match the reference,
      but may hurt when its missing force matters.

    References:
      exp000.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.pretrain_updates = cfg.method.updates // 2
    return cfg


def exp002() -> ForecastStudy.Config:
    """exp000 with ridge regression correcting the cheap simulator's forecasts.

    Hypothesis:
      A simple residual correction may solve this task without an agent.

    References:
      exp000 for the reference-data budget and held-out splits.
      https://doi.org/10.1080/00401706.1970.10488634

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp_smoke() -> ForecastStudy.Config:
    """Check the complete evaluation path with a small exp000 run.

    Hypothesis:
      Small shapes and two updates exercise the baseline wiring.

    References:
      exp000.

    Results:
      Integration only; not a forecasting result.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.data.seq_len = 5
    cfg.data.pred_len = 3
    cfg.data.train.count = 4
    cfg.data.validation.count = 4
    cfg.data.synthetic.count = 6
    cfg.data.test.count = 4
    cfg.data.shifted.count = 4
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.updates = 2
    cfg.method.batch_size = 4
    return cfg
