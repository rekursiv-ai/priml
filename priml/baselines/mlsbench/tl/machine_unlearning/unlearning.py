"""Pretrain a classifier, unlearn one class, and score how well it was forgotten.

The reference harness, as one job. It seeds torch with the seed itself,
unsalted, so a run draws what the reference's ``main`` draws:

1. Pretrain the network on every training image: the vision recipe's step
   (SGD, cosine annealing per epoch, cross-entropy) for its
   ``train_budget_epochs``.
2. Unlearn for ``unlearn_epochs``. Each epoch takes as many steps as the
   shorter of the retain and forget loaders has batches; each step hands the
   ``method`` slot the next batch of each loader, both cycling pass after pass,
   and one Adam optimizer built here.
3. Evaluate: accuracy on the retained and the forget test images, and the
   membership AUC of the forget class's training images against its test
   images.

The method is the task's editable function; every other value is fixed by the
reference and is a field here only so it prints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import logging

from configgle import Fig, Makeable, PartialConfig

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.tl.machine_unlearning.data import (
    UnlearningData,
    cycle,
)
from priml.baselines.mlsbench.tl.machine_unlearning.methods import (
    RetainFinetune,
    UnlearningMethod,
)
from priml.baselines.mlsbench.tl.machine_unlearning.metrics import (
    Report,
    auc_from_scores,
    evaluate_accuracy,
)
from priml.runtime import best_device
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor, nn


logger = logging.getLogger(__name__)


class MachineUnlearning:
    """The reference's pretrain, unlearn and evaluate pipeline for one setting."""

    class Config(Fig["MachineUnlearning"]):
        """Every value of one run: data, pretraining, the method and its optimizer."""

        study_name: str = ""
        """Run family; the launcher fills it when empty."""

        experiment_name: str = ""
        """Run name; the launcher fills it with the factory's name when empty."""

        doc: str = ""
        """The factory's docstring; the launcher fills it when empty."""

        base_dir: Path | str | None = "/opt/scratch"
        """Resource root the dataset's ``working_dir`` is resolved beneath."""

        seed: int = 42
        """Seeds Python, NumPy and torch directly, as the reference's ``--seed``."""

        device: torch.device | str | None = None
        """Where the network runs; ``None`` is the best accelerator, else the CPU."""

        dataset: UnlearningData.Config = field(default_factory=UnlearningData.Config)
        """The prepared splits, their pipelines, and the class to forget."""

        pretrain: VisionTrainStep.Config = field(
            default_factory=VisionTrainStep.Config,
        )
        """Network and recipe trained on every class; one epoch per budgeted epoch."""

        unlearn_epochs: int = 20
        """Passes of the unlearning loop."""

        unlearn_optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(torch.optim.Adam, lr=0.001),
        )
        """Built on the pretrained network's parameters and handed to every step."""

        method: Makeable[UnlearningMethod] = field(
            default_factory=RetainFinetune.Config,
        )
        """The unlearning update: the task's editable slot."""

        @override
        def finalize(self) -> Self:
            if self.dataset.base_dir is None:
                self.dataset.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.unlearn_epochs < 0:
            raise ValueError(
                f"unlearn_epochs must not be negative; got {config.unlearn_epochs}.",
            )
        self.config = config
        self.device = (
            best_device() if config.device is None else torch.device(config.device)
        )

    def run(self, *args: str) -> None:
        """Run the pipeline and log the reference's metrics line.

        Args:
          *args: Unused; the job takes no flags of its own.

        """
        del args
        logger.info("%s", self.unlearn().metrics_line())

    def unlearn(self) -> Report:
        """Seed, pretrain, unlearn with the ``method`` slot, and evaluate.

        Returns:
          report: The metrics the reference prints and the attack's scores.

        """
        seed_like_the_reference(self.config.seed)
        data = self.config.dataset.make()
        step = self.build()
        self.pretrain(step, data.full_loader())
        self.forget(step.model, data, method=self.config.method.make())
        return self.evaluate(step.model, data)

    def build(self) -> VisionTrainStep:
        """Build and initialize the network and its pretraining optimizer.

        Returns:
          step: The pretraining step, on :attr:`device`.

        """
        config = self.config.pretrain.copy_tree()
        if isinstance(config.parallelism, NoParallel.Config):
            config.parallelism.device = config.parallelism.device or self.device
        step = config.make()
        assert isinstance(step, VisionTrainStep)
        return step

    def pretrain(
        self,
        step: VisionTrainStep,
        loader: Iterable[tuple[Tensor, Tensor]],
    ) -> None:
        """Train ``step``'s network, one pass over ``loader`` per budgeted epoch.

        Args:
          step: From :meth:`build`.
          loader: Re-iterable training batches; every image of every class.

        """
        for epoch in range(int(step.config.train_budget_epochs)):
            step.timer_epoch.global_count = epoch
            for media, label in loader:
                _ = step.train_step(
                    media=media.to(self.device),
                    label=label.to(self.device),
                )
            logger.info("pretrain epoch %d done", epoch + 1)

    def forget(
        self,
        model: nn.Module,
        data: UnlearningData,
        *,
        method: UnlearningMethod,
    ) -> None:
        """Run the unlearning loop on ``model`` in place.

        Args:
          model: The pretrained network.
          data: The splits.
          method: The built ``method`` slot, fresh for this run.

        """
        optimizer = self.config.unlearn_optimizer.make()(model.parameters())
        retain_loader, forget_loader = data.retain_loader(), data.forget_loader()
        retain, forget = cycle(retain_loader), cycle(forget_loader)
        steps_per_epoch = max(1, min(len(retain_loader), len(forget_loader)))
        for epoch in range(self.config.unlearn_epochs):
            model.train()
            losses: list[float] = []
            for index in range(steps_per_epoch):
                retain_x, retain_y = next(retain)
                forget_x, forget_y = next(forget)
                loss = method.unlearn_step(
                    model,
                    (retain_x.to(self.device), retain_y.to(self.device)),
                    (forget_x.to(self.device), forget_y.to(self.device)),
                    optimizer,
                    step=index,
                    epoch=epoch,
                )
                losses.append(float(loss))
            logger.info("unlearn epoch %d loss=%.4f", epoch + 1, np.mean(losses))

    def evaluate(self, model: nn.Module, data: UnlearningData) -> Report:
        """Score ``model`` as the reference does, in the reference's loader order.

        Args:
          model: The unlearned network.
          data: The splits.

        Returns:
          report: Retained and forget accuracy and the membership AUC.

        """
        retain_acc, _ = evaluate_accuracy(
            model,
            data.retain_test_loader(),
            device=self.device,
        )
        forget_acc, forget_test = evaluate_accuracy(
            model,
            data.forget_test_loader(),
            device=self.device,
        )
        _, forget_train = evaluate_accuracy(
            model,
            data.forget_train_eval_loader(),
            device=self.device,
        )
        return Report(
            retain_acc=retain_acc,
            forget_acc=forget_acc,
            forget_mia_auc=auc_from_scores(forget_train, forget_test),
            forget_train_confidence=forget_train,
            forget_test_confidence=forget_test,
        )


def seed_like_the_reference(seed: int) -> None:
    """Seed torch's generator with ``seed`` itself, unsalted, as the reference does.

    The reference also seeds Python's and NumPy's generators, but nothing it
    runs in the main process draws from either: its loader workers reseed both
    themselves, and the capture checks that both are untouched after its
    ``main``. Only torch's stream decides the run.

    Args:
      seed: The reference's ``--seed``.

    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
