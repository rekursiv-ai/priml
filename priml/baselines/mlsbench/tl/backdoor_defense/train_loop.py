"""The driver's whole run: train a victim, filter its training split, retrain, measure.

``train`` runs the vision loop on the poisoned split, scores every training
image with the ``defense`` slot from the victim's penultimate features, removes
the most suspicious, then trains a fresh network on the rest with the same
recipe and reports the retrained model's clean accuracy and attack success
rate. Both phases evaluate the test split after the epochs the driver does.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, override

import logging

from configgle import Makeable, Makes

import torch

from priml.baselines.mlsbench.dl.vision.train_loop import VisionLoop
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.tl.backdoor_defense.defenses import (
    BackdoorDefense,
    MaxConfidence,
)
from priml.baselines.mlsbench.tl.backdoor_defense.harness import (
    Report,
    attack_success_rate,
    clean_accuracy,
    collect_features,
    filter_poison,
)
from priml.baselines.mlsbench.tl.backdoor_defense.poison import PoisonedData
from priml.train.epoch_loop import EpochTrainLoop


if TYPE_CHECKING:
    from priml.baselines.mlsbench.tl.backdoor_defense.harness import Filtered


logger = logging.getLogger(__name__)


class BackdoorTrainLoop(EpochTrainLoop):
    """Trains on poisoned data, filters it with a defense, and retrains on the rest."""

    class Config(
        Makes["BackdoorTrainLoop"],
        VisionLoop[VisionTrainStep.Config, PoisonedData.Config],
    ):
        """The vision recipe, the poisoned data and the defense that filters it."""

        dataset: PoisonedData.Config = field(default_factory=PoisonedData.Config)
        """Prepared tensors with a backdoor planted in part of the training split."""

        defense: Makeable[BackdoorDefense] = field(
            default_factory=MaxConfidence.Config,
        )
        """Scores every training image from the victim's features and logits."""

        filter_multiplier: float = 1.5
        """Images removed over poisoned images; above one leaves the scorer margin."""

    def __init__(self, config: Config) -> None:
        # Both networks train in one run directory, so a checkpointer would
        # resume the retraining from the victim's last step.
        if config.checkpointer is not None:
            raise ValueError("BackdoorTrainLoop needs checkpointer=None.")
        self.config = config
        self.filtered: Filtered | None = None
        """What the defense removed, once the victim is trained."""
        self.report: Report | None = None
        """The retrained model's numbers, once ``train`` returns."""
        super().__init__(config)

    @override
    def train(self) -> None:
        super().train()
        config = self.config
        victim, step = self.dataset, self.step
        assert isinstance(victim, PoisonedData)
        assert isinstance(step, VisionTrainStep)
        features, logits, labels = collect_features(
            step.model,
            victim.analysis_dataloader(),
            device=step.device,
        )
        filtered = self.filtered = filter_poison(
            config.defense.make(),
            features=features,
            logits=logits,
            labels=labels,
            is_poison=victim.is_poison.numpy(),
            poison_fraction=config.dataset.poison_fraction,
            multiplier=config.filter_multiplier,
        )
        logger.info(
            "Defense removed %d of %d flagged images (poison recall %.4f).",
            int(victim.is_poison[filtered.removed].sum()),
            len(filtered.removed),
            filtered.poison_recall,
        )
        # A fresh loop reseeds from ``config.seed``, so the retrained network
        # starts from the victim's initialization; the driver's continues its
        # generator instead, which the loop's salted seed never matched anyway.
        retrain = EpochTrainLoop(config)
        data, step = retrain.dataset, retrain.step
        assert isinstance(data, PoisonedData)
        assert isinstance(step, VisionTrainStep)
        data.keep(torch.from_numpy(filtered.kept))
        retrain.train()
        model, device = step.model, step.device
        report = self.report = Report(
            clean_acc=clean_accuracy(model, data.eval_dataloader(), device=device),
            asr=attack_success_rate(
                model,
                data.eval_media,
                data.eval_label,
                trigger=config.dataset.trigger,
                target_class=config.dataset.target_class,
                mean=torch.tensor(config.dataset.mean),
                std=torch.tensor(config.dataset.std),
                batch_size=config.dataset.batch_size,
                device=device,
            ),
            poison_recall=filtered.poison_recall,
        )
        logger.info(
            "TEST_METRICS clean_acc=%.4f asr=%.4f poison_recall=%.4f "
            "defense_score=%.4f",
            report.clean_acc,
            report.asr,
            report.poison_recall,
            report.defense_score,
        )
