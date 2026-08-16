r"""CIFAR-10 experiments.

``exp000`` is the baseline: the strongest recipe that uses nothing exotic --
a residual network, AdamW, cosine decay, random crops and flips. Every later
experiment forks a named parent and applies ONE change, stating its hypothesis
and source, so the chain reads as an argument rather than a pile of settings.

A variant is a different VALUE or a different INJECTED PIECE. ``exp002`` does
not select an optimizer from an enumeration; it supplies one, which is why a
reader can add Lion here without touching the train step.

    exp000  ResNet + AdamW, 30 epochs
      +-- exp001  SpeedNet architecture (PCA whitening), 8 epochs
            +-- exp002  Muon on the convolutions
                  +-- exp003  test-time augmentation
                        +-- exp004  identity initialization

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.cifar10.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000

``--override PATH=VALUE`` adapts a run to the machine it lands on -- where the
data lives, where output goes::

    uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000 --override dataset.working_dir=/datasets/my-cifar10

Path fields are logical: ``dataset.working_dir`` resolves beneath the run's
``base_dir`` (``/opt/scratch``), so pass ``/datasets/...``, not a full on-disk
path.

A hyperparameter belongs in an experiment, not on the command line. Overriding
one produces a result whose config exists nowhere in the code, so it cannot be
rerun or compared later -- write a fork instead; that is what they are for.
"""

from __future__ import annotations

from dataclasses import field
from typing import Final

from configgle import Makes, PartialConfig

import torch

from priml.baselines.cifar10.data import Cifar10Data
from priml.baselines.cifar10.model import ResNet, SpeedNet
from priml.baselines.cifar10.train_step import Cifar10TrainStep
from priml.math.schedules import polynomial
from priml.metrics.topk import TopK
from priml.model.init import dirac
from priml.optimizers import (
    CompositeOptimizer,
    Muon,
    complement,
    excluding,
)
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


NUM_TRAIN_SAMPLES: Final = 50_000
"""Images in the CIFAR-10 training split, fixed by the dataset itself."""


class Cifar10TrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    """A training loop with the CIFAR-10 step and dataset already in place.

    Narrowing the two slots here rather than at each call site is what lets a
    factory read ``cfg.step.model`` directly, with no ``isinstance`` narrow to
    reach a field it is about to set.
    """

    step: Cifar10TrainStep.Config = field(default_factory=Cifar10TrainStep.Config)
    """Model, optimization, and augmentation."""

    dataset: Cifar10Data.Config = field(default_factory=Cifar10Data.Config)
    """Prepared CIFAR-10 tensors, served from device memory."""


def exp000() -> Cifar10TrainLoop:
    """Pre-activation ResNet trained with AdamW and cosine decay.

    The baseline every other experiment forks, and the only one that states a
    recipe rather than a change. Frozen: improvements belong in a fork, never
    in an edit here, so a result measured against it stays comparable.

    Hypothesis:
      A residual network with AdamW, cosine decay, and pad-crop-flip
      augmentation is the strongest recipe that uses nothing exotic -- the
      bar any additional mechanism must clear to earn its complexity.

    References:
      https://arxiv.org/abs/1603.05027
      He et al. 2016. Identity mappings in deep residual networks.

    Results:
      TBD.

    """
    cfg = Cifar10TrainLoop()
    cfg.study_name = "cifar10"
    cfg.experiment_name = "exp000"

    cfg.step.model = ResNet.Config()
    cfg.dataset.batch_size = 512

    # Floors: a short final batch is not a whole step.
    steps_per_epoch = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
    cfg.num_steps_eval = steps_per_epoch
    # Equal, or the schedule anneals past the end of training or short of it.
    cfg.max_steps = cfg.step.total_train_steps = 30 * steps_per_epoch

    topk = cfg.metrics["accuracy"] = TopK.Config()
    topk.k_values = [1]

    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> Cifar10TrainLoop:
    """exp000 + the SpeedNet architecture: PCA whitening, wide and shallow.

    Replaces the 13-convolution residual stack with 8 wider convolutions
    behind a frozen PCA-whitening layer, at 8 epochs rather than 30.
    Optimization is untouched (AdamW, cosine); ``exp002`` changes that.

    Hypothesis:
      Whitening the first layer's patches removes the strong local
      correlation of natural images, so a much shallower network reaches
      comparable accuracy in far fewer steps. The budget moves with the
      architecture because reaching accuracy quickly is the claim.

    References:
      https://github.com/KellerJordan/cifar10-airbench
      Jordan 2024. 94% on CIFAR-10 in 3.29 seconds on a single A100.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.model = SpeedNet.Config()
    steps_per_epoch = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
    cfg.max_steps = cfg.step.total_train_steps = 8 * steps_per_epoch
    return cfg


def exp002() -> Cifar10TrainLoop:
    """exp001 + Muon on the convolution weights, SGD on the rest.

    Hypothesis:
      Muon orthogonalizes each update via a Newton-Schulz iteration, making
      the step scale-invariant with respect to the weight matrix, which
      should suit convolution kernels. Orthogonalizing a per-channel vector
      is meaningless, so the head and norms stay on SGD with Nesterov
      momentum. The schedule anneals polynomially, as Muon's recipe
      prescribes, so the two travel together as one change.

    References:
      https://kellerjordan.github.io/posts/muon/
      Jordan et al. 2024. Muon: an optimizer for hidden layers in neural
      networks.

    Results:
      TBD.

    """
    cfg = exp001()
    cfg.experiment_name = "exp002"

    # Muon orthogonalizes matrix updates; the head is left out because
    # orthogonalizing class logits measurably hurts. Everything Muon does not
    # claim -- norms, biases, the head -- goes to SGD, so the pair is total.
    on_muon = excluding(Muon.eligible_tensor, "head")

    split = cfg.step.optimizer = CompositeOptimizer.Config()
    split.select = [complement(on_muon), on_muon]

    sgd = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.67
    sgd.momentum = 0.85
    sgd.nesterov = True

    muon = Muon.Config()
    muon.lr = 0.24
    muon.momentum = 0.6
    muon.nesterov = True
    muon.ns_steps = 3
    muon.weight_decay = 0.0125
    muon.adjust_lr_fn = "conv_heuristic"

    split.optimizers = [sgd, muon]

    cfg.step.schedule = PartialConfig(polynomial)
    cfg.step.schedule.power = 1.2
    return cfg


def exp003() -> Cifar10TrainLoop:
    """exp002 + test-time augmentation over mirrored, shifted crops.

    Hypothesis:
      Averaging logits over six views -- the image and two one-pixel shifts,
      each with its mirror -- cancels the error a single view happens to
      make. Training is untouched, so any gain is bought with evaluation
      time alone.

    References:
      https://arxiv.org/abs/1409.4842
      Szegedy et al. 2014. Going deeper with convolutions, section 7.

    Results:
      TBD.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.step.use_tta = True
    return cfg


def exp004() -> Cifar10TrainLoop:
    """exp003 + identity (Dirac) initialization of the convolutions.

    Hypothesis:
      An identity kernel makes an untrained block pass its input through
      unchanged, so the network starts shallow and deepens as training
      proceeds. That should matter most at a short budget, where there is no
      time to recover from a poor start -- hence testing it on the 8-epoch
      chain rather than on exp000.

    References:
      https://arxiv.org/abs/1511.06856
      Zagoruyko & Komodakis 2015. Diracnets: training very deep neural
      networks without skip-connections.

    Results:
      TBD.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    assert isinstance(cfg.step.model, SpeedNet.Config)
    cfg.step.model.init_conv = dirac
    return cfg


def exp_smoke() -> Cifar10TrainLoop:
    """exp000 at minimum size, for verifying an installation end to end.

    Not a result. It answers one question -- is the data prepared and does the
    loop run -- so it is cut on every axis that costs time without bearing on
    that answer: one epoch, and a network narrow and shallow enough to finish
    in seconds. Accuracy will be poor, which is expected.
    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.max_steps = cfg.step.total_train_steps = (
        NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
    )
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.channels_hidden = (4, 8)
    cfg.step.model.blocks_per_stage = 1
    return cfg
