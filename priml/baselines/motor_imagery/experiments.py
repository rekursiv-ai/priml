"""Cross-subject EEG: public recipe, five-minute candidate, and wiring smoke."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import math


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")

from configgle import PartialConfig

from priml.baselines.motor_imagery.campaign import MotorImageryCampaign
from priml.baselines.motor_imagery.data import NUM_CLASSES
from priml.baselines.motor_imagery.folds import FixedCohort, GroupedFolds
from priml.baselines.motor_imagery.metric import EEGAccuracy
from priml.baselines.motor_imagery.model import (
    AttentionEEGNet,
    MultiScaleEEGNet,
    ShallowConvNet,
)
from priml.baselines.motor_imagery.train_step import cross_entropy_off_target
from priml.loss.simple_loss import SimpleLoss
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import CudnnSingleProcess


def exp000() -> MotorImageryCampaign.Config:
    """Reproduce the strongest publicly released competition notebook.

    Hypothesis:
      Avik Das's complete five-fold validation and three-model ensemble provide
      the reproducible public baseline for subsequent campaign forks. The
      notebook seeds once and draws every initialization, dropout and shuffle
      from that one stream, so the campaign's `seed` is the single seed and
      restores its stream after each loop's own seeding. The ensemble loop is
      the validation loop's recipe over the whole cohort, for 16 epochs.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444
      Apache-2.0; GitHub e2d892f158c68afa92ecda8229b9ad255c7d0aaf.

    Results:
      Source reports public 0.56 and 291.8 seconds on T4 x2.
      At 90ac943b1a88323d667b06fe517d7f960a2b18b3, one H200/Torch 2.11cu128:
      237.53 s outer walltime; full preprocessing, five folds, ensemble, 360 rows.
      Local OOF trial accuracy 1208/1795=0.6729805014 with source epoch selection.
      This is not a Kaggle score or complete source/native trajectory parity.

    """
    cfg = MotorImageryCampaign.Config()
    cfg.study_name = "motor_imagery"
    cfg.experiment_name = "exp000"
    cfg.seed = 42
    cfg.diagnostic_batch_size = 4
    cfg.expected_submission_rows = 360
    assert isinstance(cfg.folds, GroupedFolds.Config)
    cfg.folds.count = 5
    cfg.ensemble = [
        MultiScaleEEGNet.Config(),
        ShallowConvNet.Config(),
        AttentionEEGNet.Config(),
    ]

    validation = cfg.validation
    validation.max_epochs = 18
    validation.num_steps_eval = math.inf
    validation.checkpointer = None
    runtime = validation.runtime = CudnnSingleProcess.Config()
    runtime.cudnn_deterministic = True
    runtime.cudnn_benchmark = False
    validation.dataset.batch_size = 64
    optimizer = validation.step.optimizer = CompositeOptimizer.Config()
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-3
    adamw.weight_decay = 1e-3
    optimizer.optimizers = [adamw]
    objective = validation.step.loss = SimpleLoss.Config()
    objective.loss_fn = partial(cross_entropy_off_target, smoothing=0.05)
    objective.channels_out = NUM_CLASSES
    schedule = validation.step.lr_schedule = RecursiveCosine.Config()
    schedule.T_max = 18
    schedule.eta_min = 1e-5
    validation.step.train_budget_epochs = validation.max_epochs
    validation.step.gradient_clip_norm = 1.0
    validation.step.compile = None

    # Copied before the validation-only slots below, so the ensemble loop
    # declares only what it builds: it never evaluates or reads raw data.
    ensemble = cfg.ensemble_loop = validation.copy_tree()
    ensemble.max_epochs = 16
    ensemble.eval_every_epoch = False
    ensemble.step.train_budget_epochs = ensemble.max_epochs
    ensemble_schedule = ensemble.step.lr_schedule
    assert isinstance(ensemble_schedule, RecursiveCosine.Config)
    ensemble_schedule.T_max = 16

    validation.step.model = MultiScaleEEGNet.Config()
    validation.metrics_eval[""] = EEGAccuracy.Config()
    preprocessing = validation.dataset.preprocessing
    preprocessing.channels = ("Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz")
    preprocessing.sample_rate = 250.0
    preprocessing.notch_hz = 50.0
    preprocessing.notch_quality = 30.0
    preprocessing.band_hz = (1.0, 45.0)
    preprocessing.filter_order = 4
    preprocessing.eps = 1e-8
    preprocessing.window_size = 500
    preprocessing.window_step = 125
    return cfg


def exp_smoke() -> MotorImageryCampaign.Config:
    """exp000 with smaller dimensions and passes. Not a result.

    Hypothesis:
      All three architectures and the subject-fold workflow wire correctly.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not a result; installation and wiring only.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    assert isinstance(cfg.folds, GroupedFolds.Config)
    cfg.folds.count = 2
    for loop in (cfg.validation, cfg.ensemble_loop):
        loop.max_epochs = 3
        loop.step.train_budget_epochs = loop.max_epochs
        schedule = loop.step.lr_schedule
        assert isinstance(schedule, RecursiveCosine.Config)
        schedule.T_max = int(loop.max_epochs)
    cfg.validation.dataset.preprocessing.window_size = 100
    cfg.validation.dataset.preprocessing.window_step = 25
    for model in [cfg.validation.step.model, *cfg.ensemble]:
        assert isinstance(
            model,
            (MultiScaleEEGNet.Config, ShallowConvNet.Config, AttentionEEGNet.Config),
        )
        model.channels_hidden = 2
        if isinstance(model, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
            model.channels_projection = 2
    return cfg


def exp001() -> MotorImageryCampaign.Config:
    """exp000 with one fixed, subject-disjoint validation cohort.

    Hypothesis:
      Holding out S002, S006, and S019 defines one repeatable local protocol
      in place of the five-fold GroupKFold traversal.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not measured independently.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cohort = cfg.folds = FixedCohort.Config()
    cohort.subjects = ["S002", "S006", "S019"]
    return cfg


def exp002() -> MotorImageryCampaign.Config:
    """exp001 with three complete passes per validation and ensemble fit.

    Hypothesis:
      Reducing fit duration preserves the source epoch-clock protocol by moving
      its loop stop, budget, and cosine horizon together.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not measured independently.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    for loop in (cfg.validation, cfg.ensemble_loop):
        loop.max_epochs = 3
        loop.step.train_budget_epochs = loop.max_epochs
        schedule = loop.step.lr_schedule
        assert isinstance(schedule, RecursiveCosine.Config)
        schedule.T_max = int(loop.max_epochs)
    return cfg


def exp003() -> MotorImageryCampaign.Config:
    """exp002 with temporal window stride 250.

    Hypothesis:
      Fewer overlapping windows reduce training work while retaining the cue data.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not measured independently.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.validation.dataset.preprocessing.window_step = 250
    return cfg


def exp004() -> MotorImageryCampaign.Config:
    """exp003 with only the MultiScaleEEGNet full-cohort ensemble member.

    Hypothesis:
      One architecture reduces retraining and inference work.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not measured independently.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    cfg.ensemble = [cfg.ensemble[0]]
    return cfg


def exp005() -> MotorImageryCampaign.Config:
    """exp004 with half-width temporal and projection channels.

    Hypothesis:
      A coupled half-width capacity point lowers both MultiScaleEEGNet widths
      while retaining their source ratio in validation and retraining.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      Not measured independently.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    for model in [cfg.validation.step.model, *cfg.ensemble]:
        assert isinstance(model, MultiScaleEEGNet.Config)
        model.channels_hidden = 4
        model.channels_projection = 8
    return cfg


def exp006() -> MotorImageryCampaign.Config:
    """exp005 with an enforced 300-second outer wall allowance.

    Hypothesis:
      The candidate fits a five-minute budget including cold preprocessing,
      evaluation, export, and durable receipt publication.

    References:
      https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444

    Results:
      This combined candidate was named exp001 at
      90ac943b1a88323d667b06fe517d7f960a2b18b3: one H200/Torch 2.11cu128,
      47.16 s cold outer walltime; 164/305=0.5377049180 local trial accuracy;
      all 360 rows exported. Acquisition excluded; full trajectory parity open.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    cfg.budget.seconds = 300
    return cfg
