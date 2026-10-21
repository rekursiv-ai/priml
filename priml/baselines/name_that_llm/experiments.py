"""Full public reference, a five-minute candidate, and an installation smoke run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.name_that_llm.benchmark import Benchmark
from priml.baselines.name_that_llm.model import (
    CenteredHeads,
    QwenHeads,
    SvmBranch,
)


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


def exp000() -> Benchmark.Config:
    """Reproduce the complete 0.475 public Qwen8B/SVM/centred-head recipe.

    Hypothesis:
      The strongest fully public notebook supplies the starting point for
      hill climbing without dropping any ensemble component.

    References:
      https://www.kaggle.com/code/qasminumber/name-that-llm-v5?scriptVersionId=358565580

    Returns:
      config: Full five-fold prompt-exclusive evaluation and submission recipe.

    Results:
      Public source: 0.475 leaderboard, 0.513400447427293 grouped CV.
      Rewrite runtime and full-data score: TBD. Tiny CPU goldens minted in-tree
      pin the three branches and the encoder. No training-time cap applies.

    """
    config = Benchmark.Config()
    config.experiment_name = "exp000"
    config.dataset.directory = "/datasets/name-that-llm/prepared"

    config.encoder.layers = [4, 6, 8, 22]
    config.encoder.batch_size = 8
    config.encoder.dtype = torch.float16
    config.encoder.device = "cuda"

    svm = config.branches["svm"] = SvmBranch.Config()
    svm.features.style_weight = 0.3
    svm.inner_folds = 3

    plain = config.branches["plain"] = QwenHeads.Config()
    plain.classifier.inverse_regularization = 0.002

    centred = config.branches["centred"] = CenteredHeads.Config()
    centred.heads.classifier.inverse_regularization = 0.002
    centred.heads.classifier.optimizer.max_iter = 3000
    centred.chunk_sizes = [2, 3, 4, 6]
    centred.seed = 42

    config.blend.weights = {"svm": 0.4, "plain": 0.3, "centred": 0.3}
    config.validation.chunk_size = 3
    config.validation.seed = 43
    config.outer_folds = 5
    return config


def exp_smoke() -> Benchmark.Config:
    """exp000 with six real prompts, short extraction batches and two folds.

    Hypothesis:
      Small row, token and fold dimensions exercise the complete recipe after
      setup.

    References:
      https://www.kaggle.com/code/qasminumber/name-that-llm-v5?scriptVersionId=358565580

    Returns:
      config: Smoke configuration retaining all branches and pretrained weights.

    Results:
      Installation check only; no leaderboard or campaign quality claim.

    """
    config = exp000()
    config.experiment_name = "exp_smoke"
    config.dataset.max_prompts = 6
    config.encoder.batch_size = 2
    config.encoder.max_length = 64
    config.outer_folds = 2
    # The pretrained snapshot fixes width and depth; shrinking them changes weights.
    return config


def exp001() -> Benchmark.Config:
    """exp000 with a finite five-minute workload budget.

    Hypothesis:
      A fixed wall-clock budget bounds repeatable local hill-climbing trials
      while retaining every model, feature branch and validation fold.

    References:
      https://www.kaggle.com/code/qasminumber/name-that-llm-v5?scriptVersionId=358565580

    Returns:
      config: The separately budgeted candidate, preserving all ensemble members.

    Results:
      Rewrite runtime and score: TBD. No five-minute result is admitted.

    """
    config = exp000()
    config.experiment_name = "exp001"
    config.budget.seconds = 300
    return config


def exp002() -> Benchmark.Config:
    """exp001 with two prompt-exclusive validation folds.

    Hypothesis:
      Two folds reduce trial cost while retaining every classifier branch.

    References:
      https://www.kaggle.com/code/qasminumber/name-that-llm-v5?scriptVersionId=358565580

    Returns:
      config: Two-fold candidate using exp001's five-minute budget.

    Results:
      Runtime and score: TBD. Compare candidates under the same fold protocol.

    """
    config = exp001()
    config.experiment_name = "exp002"
    config.outer_folds = 2
    return config
