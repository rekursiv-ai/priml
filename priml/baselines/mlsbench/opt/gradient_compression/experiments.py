r"""MLS-Bench ``optimization-gradient-compression``: the compressor as a slot.

``exp000`` is the task's pristine recipe, no compression, at its
``resnet20-cifar10`` setting. ``exp001`` to ``exp003`` fork it and change only
``step.compressor``, the one class the task lets an edit touch. ``exp004`` and
``exp005`` fork it at the task's other two settings, keeping no compression.

    exp000  ResNet-20, CIFAR-10, no compression
      +-- exp001  top-1% with error feedback
      +-- exp002  QSGD, 256 levels, clipped to norm 1
      +-- exp003  scaled sign with error feedback
      +-- exp004  VGG-11, CIFAR-100, rate 0.05   (vgg11-cifar100)
      +-- exp005  ResNet-56, CIFAR-10            (resnet56-cifar10, hidden)

Prepare both datasets once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.opt.gradient_compression.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.multitask.model import keep_default
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    ErrorFeedback,
    qsgd,
    scaled_sign,
    top_k,
)
from priml.baselines.mlsbench.opt.gradient_compression.metrics import (
    MeanCrossEntropy,
)
from priml.baselines.mlsbench.opt.gradient_compression.model import VGG
from priml.baselines.mlsbench.opt.gradient_compression.train_loop import (
    CompressionTrainLoop,
)
from priml.baselines.mlsbench.opt.gradient_compression.train_step import (
    warmup_cosine,
)


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


def exp000() -> CompressionTrainLoop.Config:
    """ResNet-20 on CIFAR-10, SGD with warmup and cosine decay, no compression.

    The pristine recipe of the task's ``resnet20-cifar10`` setting. Its script
    shares the vision tasks' ResNet but keeps torch's default initialization,
    warms up linearly for five epochs before a cosine to a hundredth of the
    rate, loads with two workers and a test batch of 256, and evaluates after
    the first epoch, every tenth and the last. Frozen: improvements belong in
    a fork.

    Hypothesis:
      Uncompressed gradients are the bar a compressor's accuracy is measured
      against.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``optimization-gradient-compression`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = CompressionTrainLoop.Config()
    cfg.study_name = "optimization_gradient_compression"
    cfg.experiment_name = "exp000"
    # The reference's ``--seed``; the loop salts it, so the stream still differs.
    cfg.seed = 42

    model = cfg.step.model = ResNet.Config()
    model.channels_in = 3
    model.channels_out = 10
    model.init_conv = model.init_head = model.init_bias = keep_default

    sgd = cfg.step.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    sgd.momentum = 0.9
    sgd.weight_decay = 5e-4
    cfg.step.loss = PartialConfig(cross_entropy)
    schedule = cfg.step.schedule = PartialConfig(warmup_cosine)
    schedule.warmup_epochs = 5
    schedule.min_lr_ratio = 0.01

    cfg.dataset.working_dir = "/datasets/cifar10"
    cfg.dataset.mean = (0.4914, 0.4822, 0.4465)
    cfg.dataset.std = (0.2023, 0.1994, 0.2010)
    cfg.dataset.num_streams = 2
    cfg.dataset.eval_batch_size = 256

    # Equal, or the schedule anneals past the end of training or short of it.
    cfg.max_epochs = cfg.step.train_budget_epochs = 200
    cfg.num_steps_eval = -1
    cfg.metrics_eval["test_acc"] = BestAccuracy.Config()
    cfg.metrics_eval["test_loss"] = MeanCrossEntropy.Config()
    return cfg


def exp001() -> CompressionTrainLoop.Config:
    """exp000 + top-1% sparsification with error feedback.

    Hypothesis:
      Sending only the largest hundredth of each gradient, and carrying what
      was dropped into the next step, converges near the uncompressed run at
      a hundredth of the traffic.

    References:
      https://arxiv.org/abs/1809.07599
        Stich, Cordonnier and Jaggi 2018, "Sparsified SGD with Memory."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``topk_ef`` of ``optimization-gradient-compression`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    feedback = cfg.step.compressor = ErrorFeedback.Config()
    sparsify = feedback.compressor = PartialConfig(top_k)
    sparsify.ratio = 0.01
    return cfg


def exp002() -> CompressionTrainLoop.Config:
    """exp000 + QSGD with 256 levels, each gradient clipped to norm 1.

    Hypothesis:
      Stochastic rounding to 256 levels is unbiased, so plain SGD's guarantees
      hold with added variance; clipping keeps that variance from compounding
      through momentum.

    References:
      https://arxiv.org/abs/1610.02132
        Alistarh et al. 2017, "QSGD: Communication-Efficient SGD via Gradient
        Quantization and Encoding."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``qsgd`` of ``optimization-gradient-compression`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    quantize = cfg.step.compressor = PartialConfig(qsgd)
    quantize.levels = 256
    quantize.clip_norm = 1.0
    return cfg


def exp003() -> CompressionTrainLoop.Config:
    """exp000 + the sign of each entry scaled by the mean magnitude, with error feedback.

    Hypothesis:
      One bit per entry keeps the descent direction; scaling by the mean
      magnitude and feeding back the error restores the step size the sign
      throws away.

    References:
      https://arxiv.org/abs/1802.04434
        Bernstein et al. 2018, "signSGD: Compressed Optimisation for
        Non-Convex Problems."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``signsgd`` of ``optimization-gradient-compression`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    feedback = cfg.step.compressor = ErrorFeedback.Config()
    feedback.compressor = PartialConfig(scaled_sign)
    return cfg


def exp004() -> CompressionTrainLoop.Config:
    """exp000 at the ``vgg11-cifar100`` setting: VGG-11 on CIFAR-100, rate 0.05.

    Hypothesis:
      A wide network without skip connections on a hundred classes tests a
      compressor where gradients are larger and less redundant. The setting's
      script halves the rate with the network, so the two travel together.

    References:
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg11-cifar100`` of ``optimization-gradient-compression``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    model = cfg.step.model = VGG.Config()
    model.channels_in = 3
    model.channels_out = 100
    sgd = cfg.step.optimizer
    assert isinstance(sgd, PartialConfig)
    sgd.lr = 0.05
    cfg.dataset.working_dir = "/datasets/cifar100"
    cfg.dataset.mean = (0.5071, 0.4867, 0.4408)
    cfg.dataset.std = (0.2675, 0.2565, 0.2761)
    return cfg


def exp005() -> CompressionTrainLoop.Config:
    """exp000 at the hidden ``resnet56-cifar10`` setting: ResNet-56 on CIFAR-10.

    Hypothesis:
      Three times the depth tests whether a compressor's error accumulates
      through more layers.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet56-cifar10`` of ``optimization-gradient-compression``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp005"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.blocks_per_stage = (9, 9, 9)
    return cfg
