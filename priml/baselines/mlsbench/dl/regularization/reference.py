"""Port-side runs the ``dl-regularization`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's edit
and these runs on the same inputs under the same pins, compares them bit for
bit, and stores the script's values under ``testdata/``; ``reference_test.py``
replays these runs against them. The network, data and loss are the
``cv-classification-loss`` port's, already proven there; this task adds the
regularizer and the step that adds it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast, override

import copy

from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.regularization.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
)
from priml.baselines.mlsbench.dl.regularization.train_step import (
    RegularizedTrainStep,
    Regularizer,
)
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.train_step import LossContext
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, rng_fingerprint


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


REGULARIZERS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "zero": exp000,
    "dropblock": exp001,
    "confidence_penalty": exp002,
    "orthogonal_reg": exp003,
}
"""The experiment that carries each regularizer, keyed as the script's variants."""

SETTING_REGULARIZERS: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "vgg16bn-cifar100": {
        "zero": exp004,
        "dropblock": exp005,
        "confidence_penalty": exp006,
        "orthogonal_reg": exp007,
    },
    "mobilenetv2-fmnist": {
        "zero": exp008,
        "dropblock": exp009,
        "confidence_penalty": exp010,
        "orthogonal_reg": exp011,
    },
}
"""Per other setting, keyed as the vision ``SETTINGS``, each regularizer's experiment."""

PROBE_EPOCHS: Final = (0, 3, 4)
"""Epochs of a five-epoch run the probe evaluates at: before, during and at the end of a ramp."""

type ProbeCall = Callable[..., Tensor]
"""A regularizer called as ``(model, *, media, logits, label, config)``."""


class ProbeNet(nn.Module):
    """A parameter container whose names and kernels reach every selector branch.

    ``conv1`` and ``conv2`` are 3x3 kernels; ``shortcut`` is a 3x1 kernel, too
    narrow for a 3x3 block and named as ResNet's projection is; ``norm`` and
    ``fc`` are not convolutions. Every value is a multiple of one eighth.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, bias=False)
        self.conv2 = nn.Conv2d(4, 5, 3)
        self.shortcut = nn.Conv2d(3, 5, (3, 1), bias=False)
        self.norm = nn.BatchNorm2d(5)
        self.fc = nn.Linear(5, 2)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                count = parameter.numel()
                values = (torch.arange(count) * (5 + 2 * index) % 13 - 6) / 8
                parameter.copy_(values.reshape(parameter.shape))

    @override
    def forward(self, media: Tensor) -> Tensor:
        raise NotImplementedError("The probe holds parameters; it is never run.")


def regularizer_run(
    name: str,
    function: ProbeCall,
    *,
    network: Callable[[], nn.Module] = ProbeNet,
    prefix: str = "",
) -> dict[str, Tensor]:
    """Evaluate a regularizer on a fresh network at each of :data:`PROBE_EPOCHS`.

    Args:
      name: Which regularizer, for the record's keys.
      function: The regularizer, in the :data:`ProbeCall` convention.
      network: Builds the network the regularizer reads, once per epoch.
      prefix: Put before every key, naming a setting.

    Returns:
      record: Per epoch the value, its gradient with respect to the logits, and
        its gradient with respect to every parameter of the probe, joined in
        registration order with a missing gradient as zeros.

    """
    logits, label = reference.fixed_logits()
    values: list[Tensor] = []
    logit_grads: list[Tensor] = []
    parameter_grads: list[Tensor] = []
    with host_agnostic_numerics():
        for epoch in PROBE_EPOCHS:
            model = network()
            leaf = logits.clone().requires_grad_()
            value = function(
                model,
                media=reference.fixed_media(0),
                logits=leaf,
                label=label,
                config={"num_classes": 4, "epoch": epoch, "total_epochs": 5},
            )
            if value.requires_grad:
                value.backward()
            values.append(value.detach())
            logit_grads.append(_grad(leaf))
            parameter_grads.append(joined(_grad(p) for p in model.parameters()))
    return {
        f"{prefix}regularizer/{name}/value": torch.stack(values),
        f"{prefix}regularizer/{name}/logits_grad": torch.stack(logit_grads),
        f"{prefix}regularizer/{name}/parameters_grad": torch.stack(parameter_grads),
    }


def regularizer_record() -> dict[str, Tensor]:
    """Run each baseline's regularizer, as its experiment builds it, on the probe.

    Returns:
      record: :func:`regularizer_run` of every regularizer but ``zero``.

    """
    record: dict[str, Tensor] = {}
    for name, factory in REGULARIZERS.items():
        if name != "zero":
            step = factory().step
            assert isinstance(step, RegularizedTrainStep.Config)
            record |= regularizer_run(name, _PortCall(step.regularization.make()))
    return record


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building the full ``exp000`` step.

    Returns:
      record: The next eight draws.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        _ = reference.full_step(exp000)
        return {"draws/exp000": rng_fingerprint()}


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest ResNet for two updates, one per epoch.

    Args:
      name: A key of :data:`REGULARIZERS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``lockstep/<name>/``.

    """
    return lockstep_run(REGULARIZERS[name], prefix=f"lockstep/{name}/", states=states)


def setting_lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train a setting's shallowest network for two updates, one per epoch.

    Args:
      name: ``<setting>/<regularizer>``, keys of :data:`SETTING_REGULARIZERS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``settings/<setting>/lockstep/<regularizer>/``.

    """
    setting, regularizer = name.split("/")
    return lockstep_run(
        SETTING_REGULARIZERS[setting][regularizer],
        prefix=f"settings/{setting}/lockstep/{regularizer}/",
        states=states,
    )


def lockstep_run(
    factory: Callable[[], VisionTrainLoop],
    *,
    prefix: str,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train an experiment's shallowest network for two updates, one per epoch.

    Args:
      factory: The experiment; its network is cut as ``tiny_step`` cuts it.
      prefix: Put before every key.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss minimized, the regularization term, the
        logits, a fixed probe's logits in evaluation mode afterwards, and what
        the term was told about the run.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            factory,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        )
        assert isinstance(step, RegularizedTrainStep)
        spy = step.regularization = _RegularizationSpy(step.regularization)
        for index in range(reference.STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=reference.fixed_media(index),
                label=reference.fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            # Outside inference mode, as the script's own probe is: an
            # inference tensor escapes the float64 upcast, so a 512-wide layer
            # would round by host.
            step.model.eval()
            with torch.no_grad():
                probes.append(
                    reference.model_logits(
                        step.model,
                        reference.fixed_media(reference.STEPS),
                    ),
                )
            step.model.train()
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        f"{prefix}loss": torch.stack(losses),
        f"{prefix}regularization": torch.stack(spy.values),
        f"{prefix}logits": torch.stack(logits),
        f"{prefix}probe": torch.stack(probes),
        f"{prefix}context": torch.tensor(
            [[c.num_classes, c.epoch, c.total_epochs] for c in spy.contexts],
        ),
    }


def tiny_network(setting: str) -> nn.Module:
    """Build a setting's shallowest network as its unregularized experiment initializes it.

    Cut as the vision ``tiny_step`` cuts it: the script's own layer names and
    kinds, at ``reference.CLASSES`` logits, so every selector and kernel test
    of a regularizer meets what it meets on the full network.

    Args:
      setting: A key of :data:`SETTING_REGULARIZERS`.

    Returns:
      model: Built under the task's seed, in training mode.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        return reference.tiny_step(
            SETTING_REGULARIZERS[setting]["zero"],
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        ).model


def setting_regularizer_record(setting: str, model: nn.Module) -> dict[str, Tensor]:
    """Run each baseline's regularizer, as the setting's fork builds it, on a network.

    Args:
      setting: A key of :data:`SETTING_REGULARIZERS`.
      model: The setting's :func:`tiny_network`, or the script's; each run
        reads a copy of it.

    Returns:
      record: :func:`regularizer_run` of every regularizer but ``zero``, under
        ``settings/<setting>/``.

    """
    record: dict[str, Tensor] = {}
    for name, factory in SETTING_REGULARIZERS[setting].items():
        if name != "zero":
            step = factory().step
            assert isinstance(step, RegularizedTrainStep.Config)
            record |= regularizer_run(
                name,
                _PortCall(step.regularization.make()),
                network=Copies(model),
                prefix=f"settings/{setting}/",
            )
    return record


def setting_recipe_record(setting: str) -> dict[str, Tensor]:
    """Return a setting's 200 learning rates, optimizer and batch sizes.

    Args:
      setting: A key of :data:`SETTING_REGULARIZERS`.

    Returns:
      record: ``schedule/lr``, ``optimizer/settings`` and ``setting/batch_size``,
        under ``settings/<setting>/``.

    """
    factory = SETTING_REGULARIZERS[setting]["zero"]
    dataset = factory().dataset
    record = (
        reference.schedule_record(factory)
        | first_epoch_optimizer_record(factory)
        | {
            "setting/batch_size": torch.tensor(
                [dataset.batch_size, dataset.eval_batch_size],
            ),
        }
    )
    return {f"settings/{setting}/{key}": value for key, value in record.items()}


def first_epoch_optimizer_record(
    factory: Callable[[], VisionTrainLoop],
) -> dict[str, Tensor]:
    """Return an experiment's optimizer settings as they stand during epoch 0.

    The script's trace reads them inside its first ``train_epoch``, after its
    scheduler has set epoch 0's rate: for VGG's ``LambdaLR`` that is the first
    warmup rate, not the base one. The port's step sets its rate before each
    update, so this applies epoch 0's schedule before reading.

    Args:
      factory: The experiment.

    Returns:
      record: The entries of ``reference.HYPERPARAMETERS``, as float64.

    """
    step = reference.tiny_step(
        factory,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=200,
    )
    for group in step.optimizer.param_groups:
        group["lr"] = step.schedule(
            0,
            base_lr=from_plain(cast(object, group["initial_lr"]), float),
            total_epochs=200,
        )
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    values = [float(cast(float, group[name])) for name in reference.HYPERPARAMETERS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def setting_batch_size(argv: list[str]) -> Tensor:
    """Return the ``--batch-size`` the setting passes, once per loader.

    ``main`` hands it to ``get_dataloaders``, which builds both loaders with it.

    Args:
      argv: The setting's command line.

    Returns:
      sizes: ``(2,)`` the training then evaluation batch size.

    """
    size = int(argv[argv.index("--batch-size") + 1])
    return torch.tensor([size, size])


def port_batch_size() -> Tensor:
    """Return the training and evaluation batch sizes of ``exp000``."""
    dataset = exp000().dataset
    return torch.tensor([dataset.batch_size, dataset.eval_batch_size])


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = (
        regularizer_record()
        | reference.schedule_record(exp000)
        | reference.optimizer_record(exp000)
        | draws_record()
        | reference.best_record()
        | {"setting/batch_size": port_batch_size()}
    )
    for name in REGULARIZERS:
        record |= lockstep_record(name)
    return record


def settings_record() -> dict[str, Tensor]:
    """Run the task's own port-side comparisons at its other settings.

    The full networks and the Fashion-MNIST loader are the vision
    ``task_settings_record``'s.

    Returns:
      record: Per setting the recipe, each regularizer on the shallowest
        network, and two updates per regularizer.

    """
    record: dict[str, Tensor] = {}
    for setting, variants in SETTING_REGULARIZERS.items():
        record |= setting_recipe_record(setting)
        record |= setting_regularizer_record(setting, tiny_network(setting))
        for name in variants:
            record |= setting_lockstep_record(f"{setting}/{name}")
    return record


class Copies:
    """Builds a fresh copy of one network per call, so each run starts from it."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __call__(self) -> nn.Module:
        """Return a deep copy of the network."""
        return copy.deepcopy(self.model)


class _RegularizationSpy:
    """Forwards to a regularizer and keeps what it returned and was told."""

    def __init__(self, regularizer: Regularizer) -> None:
        self.regularizer = regularizer
        self.values: list[Tensor] = []
        self.contexts: list[LossContext] = []

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Record and return the wrapped regularizer's term."""
        value = self.regularizer(
            model,
            media=media,
            logits=logits,
            label=label,
            context=context,
        )
        self.values.append(value.detach())
        self.contexts.append(context)
        return value


def _grad(tensor: Tensor) -> Tensor:
    """Return a leaf's gradient, or zeros where none reached it."""
    return torch.zeros_like(tensor) if tensor.grad is None else tensor.grad


class _PortCall:
    """Calls a port regularizer in the :data:`ProbeCall` convention."""

    def __init__(self, regularizer: Regularizer) -> None:
        self.regularizer = regularizer

    def __call__(
        self,
        model: nn.Module,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the term, telling it ``config`` as a :class:`LossContext`."""
        return self.regularizer(
            model,
            media=media,
            logits=logits,
            label=label,
            context=LossContext(**config),
        )
