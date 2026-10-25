r"""The recipe every MLS-Bench DL task forks: ResNet-56 on CIFAR-100.

``exp000`` is the recipe of the pristine ``cv-classification-loss`` script at
its ``resnet56-cifar100`` setting: cross-entropy, SGD with cosine annealing.
Every DL task's script is a variant of that script, so each task's ``exp000``
forks this one under the task's own study name, and the task's baselines fork
that.

Prepare the data once, then launch a task's ``exp000``::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data

The recipe is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, override

from configgle import Makes, PartialConfig

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import (
    VisionTrainStep,
    cross_entropy,
)
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class VisionTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config],
):
    """A training loop with the vision step and CIFAR-100 already in place.

    Narrowing the two slots here rather than at each call site is what lets a
    factory read ``cfg.step.loss`` directly, with no ``isinstance`` narrow to
    reach a field it is about to set.
    """

    step: VisionTrainStep.Config = field(default_factory=VisionTrainStep.Config)
    """Model, optimizer, loss and schedule."""

    dataset: Cifar100Data.Config = field(default_factory=Cifar100Data.Config)
    """Prepared CIFAR-100 tensors, shuffled and augmented per pass."""

    @override
    def finalize(self) -> Self:
        check_reference_loop(self, budget_epochs=self.step.train_budget_epochs)
        return super().finalize()


def check_reference_loop(config: TrainLoop.Config, *, budget_epochs: float) -> None:
    """Refuse loop settings that would take the reference's run off its draws.

    Every evaluation pass takes one draw from the global generator, as the
    reference's loader does, so an evaluation the reference never runs moves
    every shuffle after it. The cosine recurrence divides by zero one epoch past
    its horizon, so the run may not outlast it.

    Args:
      config: The loop.
      budget_epochs: The step's schedule horizon.

    Raises:
      ValueError: A setting adds an evaluation pass or outruns the horizon.

    """
    if config.max_epochs > budget_epochs:
        raise ValueError(
            f"max_epochs ({config.max_epochs}) exceeds the schedule's "
            f"train_budget_epochs ({budget_epochs}).",
        )
    extra = {
        "num_steps_eval": config.num_steps_eval != -1,
        "eval_warmup_batches": config.eval_warmup_batches != 0,
        "eval_only": config.eval_only,
    }
    for name in (name for name, added in extra.items() if added):
        raise ValueError(
            f"{name} adds an evaluation pass the reference never runs, and "
            "its draw would move every later shuffle.",
        )


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, cross-entropy, SGD with cosine annealing.

    The reference recipe of the ``cv-classification-loss`` script's
    ``resnet56-cifar100`` setting, which every DL task's ``exp000`` forks under
    its own study name. Frozen: improvements belong in a task's fork, never in
    an edit here.

    Hypothesis:
      The unedited recipe is the bar any task's edit must clear to earn its
      complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-classification-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = VisionTrainLoop()
    cfg.study_name = "mlsbench_dl_vision"
    cfg.experiment_name = "exp000"
    # The reference's ``--seed``; the loop salts it, so the stream still differs.
    cfg.seed = 42

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.channels_in = 3
    cfg.step.model.channels_out = 100
    cfg.step.model.blocks_per_stage = (9, 9, 9)

    sgd = cfg.step.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    sgd.momentum = 0.9
    sgd.weight_decay = 5e-4
    cfg.step.loss = PartialConfig(cross_entropy)

    # Equal, or the schedule anneals past the end of training or short of it.
    cfg.max_epochs = cfg.step.train_budget_epochs = 200

    # Evaluate after every epoch and not in between: an evaluation advances the
    # generator, so one inside an epoch would move every shuffle after it.
    cfg.num_steps_eval = -1
    cfg.metrics_eval["test_acc"] = BestAccuracy.Config()
    return cfg
