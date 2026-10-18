r"""MLS-Bench ``dl-normalization``: ResNet-56 on CIFAR-100, the norm layer as a slot.

``exp000`` is the task's pristine recipe, batch normalization. Every later
experiment forks it and changes ``step.model.norm``, the one class the task
lets an edit touch; the model builds the stem's, every block's and every
projection's normalization from it.

    exp000  batch norm
      +-- exp001  group norm
      +-- exp002  batch-instance norm
      +-- exp003  switchable norm

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.normalization.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.normalization.norms import (
    BatchInstanceNorm,
    GroupNorm,
    SwitchableNorm,
)
from priml.baselines.mlsbench.dl.vision import experiments
from priml.baselines.mlsbench.dl.vision.model import ResNet


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, batch norm, cross-entropy, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with every ``nn.BatchNorm2d``
    behind a ``CustomNorm`` that wraps one, so this is that task's ``exp000``.
    The wrapper only renames the state's keys, and the script's
    ``initialize_weights`` sets each norm to the scale and shift it is built
    with. Frozen: improvements belong in a fork.

    Hypothesis:
      Batch norm is the bar a normalization layer must clear.

    References:
      https://arxiv.org/abs/1502.03167
        Ioffe and Szegedy 2015, "Batch Normalization: Accelerating Deep Network
        Training by Reducing Internal Covariate Shift."
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-normalization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "dl_normalization"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + group norm, up to 32 groups.

    Hypothesis:
      Statistics over channel groups of one image do not depend on the batch,
      so training and evaluation normalize alike, which group norm reports as
      matching batch norm on classification.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``group_norm`` of ``dl-normalization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = GroupNorm
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + batch-instance norm.

    Hypothesis:
      A learned per-channel gate between batch and instance statistics lets
      each channel drop the per-image contrast it does not need.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``batch_instance_norm`` of ``dl-normalization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = BatchInstanceNorm
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + switchable norm.

    Hypothesis:
      Learning a mixture of instance, layer and batch statistics per layer
      picks the normalization each depth suits, which switchable norm reports
      as beating each of the three alone.

    References:
      https://arxiv.org/abs/1806.10779
        Luo et al. 2019, "Differentiable Learning-to-Normalize via Switchable
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``switchable_norm`` of ``dl-normalization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = SwitchableNorm
    return cfg
