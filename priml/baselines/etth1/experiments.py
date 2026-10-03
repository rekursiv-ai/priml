"""ETTh1 long-horizon forecasting experiments."""

from __future__ import annotations

from configgle import PartialConfig

from priml.baselines.etth1.checkpointer import Etth1Checkpointer
from priml.baselines.etth1.metrics import ForecastMSE
from priml.baselines.etth1.train_step import Etth1TrainLoop
from priml.runtime import SingleProcess


def dlinear_type1(progress: float, *, epochs: float = 10) -> float:
    """Match the reference DLinear type1 learning-rate schedule."""
    epoch = int(progress * epochs + 1e-9)
    drops = max(0, epoch - 1)
    return 0.5**drops


def exp000() -> Etth1TrainLoop.Config:
    """DLinear on ETTh1 with a 336-step history and 96-step forecast.

    This is the canonical forecasting baseline.

    Hypothesis:
      A decomposition-linear model provides a compact canonical baseline for
      long-horizon multivariate forecasting and a fast target for automated
      experiment hillclimbing.

    Returns:
      cfg: Canonical ETTh1 DLinear training configuration.

    References:
      https://arxiv.org/abs/2205.13504
      Zeng et al. 2023. Are Transformers Effective for Time Series Forecasting?

    Results:
      Reference-hardware benchmark: TBD. See README.md for the separately
      recorded local CPU reproduction and source-parity evidence.

    """
    cfg = Etth1TrainLoop.Config()

    cfg.study_name = "etth1"
    cfg.experiment_name = "exp000"
    cfg.seed = 2021

    steps_per_epoch = (
        cfg.dataset.train_rows - cfg.dataset.seq_len - cfg.dataset.pred_len + 1
    ) // cfg.dataset.batch_size

    cfg.max_epochs = 10
    cfg.max_steps = 10 * steps_per_epoch

    cfg.step.train_budget_epochs = cfg.max_epochs
    cfg.step.lr_schedule = PartialConfig(dlinear_type1, epochs=cfg.max_epochs)

    cfg.num_steps_eval = float("inf")
    cfg.eval_every_epoch = True

    cfg.num_steps_log = 100
    cfg.early_train_log_steps = 0

    cfg.metrics_eval[""] = ForecastMSE.Config()

    cfg.checkpointer = Etth1Checkpointer.Config()
    # The epoch hook saves after validation.
    cfg.checkpointer.save_every = cfg.max_steps
    cfg.checkpointer.keep_last_n = 3
    cfg.checkpointer.best_metric = "total_loss"
    cfg.checkpointer.best_mode = "min"

    cfg.runtime = SingleProcess.Config(device="cpu")

    return cfg


def exp_smoke() -> Etth1TrainLoop.Config:
    """Tiny ETTh1 DLinear run for end-to-end integration checking.

    Returns:
      cfg: Minimal forecasting run that exercises model, data, loss,
      optimization, and evaluation.

    Hypothesis:
      A short, narrow run exercises the canonical data and training wiring.

    References:
      exp000.

    Results:
      Integration only; not a forecasting quality benchmark.

    """
    cfg = exp000()

    cfg.experiment_name = "exp_smoke"

    cfg.max_epochs = 1
    cfg.max_steps = 3
    cfg.dataset.seq_len = 5
    cfg.dataset.pred_len = 3
    cfg.dataset.batch_size = 2
    cfg.dataset.eval_batch_size = 2

    cfg.num_steps_eval = cfg.max_steps
    cfg.eval_every_epoch = False

    cfg.checkpointer = None

    return cfg
