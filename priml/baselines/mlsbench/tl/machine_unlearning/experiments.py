r"""MLS-Bench ``security-machine-unlearning``: forget one class, the update as a slot.

``exp000`` is the task's pristine recipe at its ``resnet20-cifar10-class0``
setting: ResNet-20 pretrained on all of CIFAR-10 for 80 epochs, then 20 epochs
of retain-only finetuning with Adam. Every baseline forks it and changes
``method``, the one function the task lets an edit touch; the settings fork it
and change the network and the dataset.

    exp000  retain-only finetuning, ResNet-20 on CIFAR-10      (pristine)
      +-- exp001  retain-only finetuning                       (retain_finetune)
      +-- exp002  negative gradient                            (negative_gradient)
      +-- exp003  bad teacher                                  (bad_teacher)
      +-- exp004  SCRUB                                        (scrub)
      +-- exp005  VGG-16-BN on CIFAR-100                       (vgg16bn-cifar100-class0)
      +-- exp006  MobileNetV2 on Fashion-MNIST                 (mobilenetv2-fmnist-class0)

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.machine_unlearning.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.machine_unlearning.methods import (
    BadTeacher,
    NegativeGradient,
    Scrub,
)
from priml.baselines.mlsbench.tl.machine_unlearning.unlearning import (
    MachineUnlearning,
)
from priml.model.swiglu import relu


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


def exp000() -> MachineUnlearning.Config:
    """ResNet-20 on CIFAR-10, forget class 0 by retain-only finetuning.

    The pristine recipe of the ``resnet20-cifar10-class0`` setting: 80 epochs
    of SGD with cosine annealing on every class, then 20 epochs of Adam on the
    retained classes alone. Its method carries an unused ``forget_weight`` of
    zero and otherwise computes what ``retain_finetune`` does. Frozen:
    improvements belong in a fork.

    Hypothesis:
      Finetuning on the retained data alone is the bar an unlearning update
      must clear: it forgets only by drift, with no signal from the forget set.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``security-machine-unlearning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = MachineUnlearning.Config()
    cfg.study_name = "security_machine_unlearning"
    cfg.experiment_name = "exp000"

    assert isinstance(cfg.pretrain.model, ResNet.Config)
    cfg.pretrain.model.channels_in = 3
    cfg.pretrain.model.channels_out = 10
    sgd = cfg.pretrain.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    sgd.momentum = 0.9
    sgd.weight_decay = 5e-4
    cfg.pretrain.loss = PartialConfig(cross_entropy)
    cfg.pretrain.train_budget_epochs = 80

    cfg.dataset.working_dir = "/datasets/cifar10"
    cfg.dataset.mean = (0.4914, 0.4822, 0.4465)
    cfg.dataset.std = (0.2023, 0.1994, 0.2010)
    return cfg


def exp001() -> MachineUnlearning.Config:
    """exp000 + the ``retain_finetune`` baseline's method, which computes the same.

    Hypothesis:
      The published baseline equals the pristine method, so this run measures
      the run-to-run spread of the recipe.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``retain_finetune`` of ``security-machine-unlearning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    return cfg


def exp002() -> MachineUnlearning.Config:
    """exp000 + negative gradient: ascend half the forget loss.

    Hypothesis:
      Ascending the forget class's cross-entropy drives its accuracy to zero
      directly, at the price of retained accuracy the descent must recover.

    References:
      https://arxiv.org/abs/1911.04933
        Golatkar, Achille and Soatto 2020, "Eternal Sunshine of the Spotless
        Net: Selective Forgetting in Deep Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``negative_gradient`` of ``security-machine-unlearning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.method = NegativeGradient.Config()
    return cfg


def exp003() -> MachineUnlearning.Config:
    """exp000 + bad teacher: distill forget images from a re-initialized network.

    Hypothesis:
      Matching a random network on the forget class and the original on the
      rest removes the forget class's signal without disturbing the others.

    References:
      https://arxiv.org/abs/2205.08096
        Chundawat et al. 2023, "Can Bad Teaching Induce Forgetting?
        Unlearning in Deep Networks using an Incompetent Teacher."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``bad_teacher`` of ``security-machine-unlearning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    cfg.method = BadTeacher.Config()
    return cfg


def exp004() -> MachineUnlearning.Config:
    """exp000 + SCRUB: two epochs of max steps from the teacher, then min steps.

    Hypothesis:
      Pushing the student away from the frozen original on forget images,
      while distilling it back on retained ones, forgets without the collapse
      plain gradient ascent risks.

    References:
      https://arxiv.org/abs/2302.09880
        Kurmanji et al. 2023, "Towards Unbounded Machine Unlearning."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``scrub`` of ``security-machine-unlearning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    cfg.method = Scrub.Config()
    return cfg


def exp005() -> MachineUnlearning.Config:
    """exp000 at the ``vgg16bn-cifar100-class0`` setting: VGG-16-BN on CIFAR-100.

    Changes the network and the dataset only; the method stays pristine.

    Hypothesis:
      A wide network without skip connections, forgetting one class of a
      hundred, sets the bar on a harder retain set with a smaller forget set.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100-class0`` of ``security-machine-unlearning``
        at commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp005"
    model = cfg.pretrain.model = VGG16BN.Config()
    model.channels_in = 3
    model.channels_out = 100
    cfg.dataset.working_dir = "/datasets/cifar100"
    cfg.dataset.mean = (0.5071, 0.4867, 0.4408)
    cfg.dataset.std = (0.2675, 0.2565, 0.2761)
    return cfg


def exp006() -> MachineUnlearning.Config:
    """exp000 at the ``mobilenetv2-fmnist-class0`` setting: MobileNetV2 on Fashion-MNIST.

    Changes the network and the dataset only; the method stays pristine. This
    harness's MobileNetV2 activates with ReLU, not the paper's ReLU6.

    Hypothesis:
      Grayscale clothing with ten classes forgets one class from a network
      of depthwise blocks, the hidden setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist-class0`` of ``security-machine-unlearning``
        at commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp006"
    model = cfg.pretrain.model = MobileNetV2.Config()
    model.channels_in = 3
    model.channels_out = 10
    model.activation = relu
    cfg.dataset.working_dir = "/datasets/fmnist"
    cfg.dataset.mean = (0.2860, 0.2860, 0.2860)
    cfg.dataset.std = (0.3530, 0.3530, 0.3530)
    return cfg
