r"""MLS-Bench ``security-backdoor-defense``: filter a poisoned split, retrain, measure.

``exp000`` is the task's pristine recipe at its ``resnet20-cifar10-badnets``
setting: train ResNet-20 for 100 epochs on CIFAR-10 with a BadNets square in 5%
of the training images, score every image by its top softmax probability,
remove the highest 7.5%, retrain from scratch on the rest and report the
retrained model's clean accuracy and attack success rate. Every baseline forks
it and changes ``defense``, the one class the task lets an edit touch; the
other two settings fork it and change the network, the data and the trigger.

    exp000  max softmax confidence           resnet20-cifar10-badnets
      +-- exp001  confidence z-score within the predicted class
      +-- exp002  spectral signature
      +-- exp003  activation clustering
      +-- exp004  per-class feature z-score
      +-- exp005  vgg16bn-cifar100-blend, 1% poisoned
      +-- exp006  mobilenetv2-fmnist-badnets, 8% poisoned (hidden upstream)

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.backdoor_defense.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.tl.backdoor_defense.defenses import (
    ActivationClustering,
    ConfidenceZScore,
    FeatureZScore,
    SpectralSignature,
)
from priml.baselines.mlsbench.tl.backdoor_defense.poison import PoisonedData
from priml.baselines.mlsbench.tl.backdoor_defense.train_loop import (
    BackdoorTrainLoop,
)
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import blend


def exp000() -> BackdoorTrainLoop.Config:
    """ResNet-20 on CIFAR-10 with BadNets in 5% of it, filtered by max softmax confidence.

    The pristine recipe of the task's ``resnet20-cifar10-badnets`` setting.
    Its driver trains with the ``cv-classification-loss`` recipe -- SGD at
    0.1 with momentum 0.9 and weight decay 5e-4, cosine annealing, the same
    augmentation -- for 100 epochs instead of 200, and evaluates the test
    split after epoch 1, every tenth and the last rather than after every
    epoch; those evaluations are diagnostic and change no reported number.
    The poisoned positions are drawn from Python's generator seeded 42, the
    run's seed. The victim's features are taken over the training split in
    order through the training transform, 1.5 times the poisoned fraction is
    removed, and a fresh network is retrained on the rest. Frozen:
    improvements belong in a fork.

    Hypothesis:
      A poisoned image is learned until the model is sure of it, so the top
      softmax probability is the bar a defense must clear; as the clean
      images are as confidently fit, it should barely beat chance.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``security-backdoor-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1708.06733
        Gu, Dolan-Gavitt and Garg 2017, "BadNets: Identifying Vulnerabilities
        in the Machine Learning Model Supply Chain."

    Results:
      TBD.

    """
    parent = experiments.exp000()
    cfg = BackdoorTrainLoop.Config().update(parent, skip_missing=True)
    cfg.study_name = "security_backdoor_defense"
    cfg.experiment_name = "exp000"
    cfg.dataset = PoisonedData.Config().update(parent.dataset)
    settings.resnet20_cifar10(cfg)
    cfg.max_epochs = cfg.step.train_budget_epochs = 100
    # The victim and the retrained network share the run directory.
    cfg.checkpointer = None
    return cfg


def exp001() -> BackdoorTrainLoop.Config:
    """exp000 + score each image's confidence as a z-score within its predicted class.

    Hypothesis:
      A trigger forces a prediction more confidently than the target class's
      own images earn theirs, so standardizing confidence per predicted class
      should lift the poisoned images above the clean ones.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_filter`` of ``security-backdoor-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.defense = ConfidenceZScore.Config()
    return cfg


def exp002() -> BackdoorTrainLoop.Config:
    """exp000 + spectral signatures: squared projection on each class's top singular vector.

    Hypothesis:
      Poisoned images share a direction in feature space that dominates the
      target class's covariance, so their projections on its top singular
      vector stand out.

    References:
      https://arxiv.org/abs/1811.00636
        Tran, Li and Madry 2018, "Spectral Signatures in Backdoor Attacks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``spectral_signature`` of ``security-backdoor-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.defense = SpectralSignature.Config()
    return cfg


def exp003() -> BackdoorTrainLoop.Config:
    """exp000 + activation clustering: two-means per class, the smaller cluster scored high.

    Hypothesis:
      Poisoned images form their own cluster inside the target class, the
      smaller one, so membership in each class's smaller cluster flags them.

    References:
      https://arxiv.org/abs/1811.03728
        Chen et al. 2018, "Detecting Backdoor Attacks on Deep Neural Networks
        by Activation Clustering."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``activation_clustering`` of ``security-backdoor-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    cfg.defense = ActivationClustering.Config()
    return cfg


def exp004() -> BackdoorTrainLoop.Config:
    """exp000 + mean squared per-feature z-score within each image's class.

    Hypothesis:
      A trigger drives features that the target class's clean images leave
      dormant, so a diagonal Mahalanobis distance to the class flags them.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``zscore_outlier`` of ``security-backdoor-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    cfg.defense = FeatureZScore.Config()
    return cfg


def exp005() -> BackdoorTrainLoop.Config:
    """exp000 at the ``vgg16bn-cifar100-blend`` setting: VGG-16-BN, a blended checkerboard, 1%.

    Hypothesis:
      A blended trigger covers the whole image rather than a corner, and at
      1% poisoning the target class is a third poisoned, the regime Tran et
      al. assume; the pristine defense is the bar there too.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100-blend`` of ``security-backdoor-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1712.05526
        Chen et al. 2017, "Targeted Backdoor Attacks on Deep Learning Systems
        Using Data Poisoning."

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp005"
    cfg.dataset.trigger = blend
    cfg.dataset.poison_fraction = 0.01
    return cfg


def exp006() -> BackdoorTrainLoop.Config:
    """exp000 at the ``mobilenetv2-fmnist-badnets`` setting: MobileNetV2, Fashion-MNIST, 8%.

    Upstream hides this setting from the task's agents and scores on it.

    Hypothesis:
      Grayscale clothing has less texture for a corner square to hide in, so
      the backdoor should be learned at least as reliably as on CIFAR-10.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist-badnets`` of ``security-backdoor-defense``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp006"
    cfg.dataset.poison_fraction = 0.08
    return cfg
