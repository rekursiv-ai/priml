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

from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.pytorch_vision import reference
from priml.baselines.mlsbench.pytorch_vision.train_step import LossContext
from priml.baselines.mlsbench.regularization.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.regularization.train_step import (
    RegularizedTrainStep,
    Regularizer,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


REGULARIZERS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "zero": exp000,
    "dropblock": exp001,
    "confidence_penalty": exp002,
    "orthogonal_reg": exp003,
}
"""The experiment that carries each regularizer, keyed as the script's variants."""

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


def regularizer_run(name: str, function: ProbeCall) -> dict[str, Tensor]:
    """Evaluate a regularizer on the probe at each of :data:`PROBE_EPOCHS`.

    Args:
      name: Which regularizer, for the record's keys.
      function: The regularizer, in the :data:`ProbeCall` convention.

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
            model = ProbeNet()
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
        f"regularizer/{name}/value": torch.stack(values),
        f"regularizer/{name}/logits_grad": torch.stack(logit_grads),
        f"regularizer/{name}/parameters_grad": torch.stack(parameter_grads),
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


def schedule_record() -> dict[str, Tensor]:
    """Return the learning rate of each of the 200 epochs of ``exp000``.

    Returns:
      record: The 200 rates, as float64.

    """
    step = reference.tiny_step(exp000, blocks=(1, 1, 1), classes=3, epochs=200)
    base_lr = from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    return {"schedule/lr": torch.tensor(rates, dtype=torch.float64)}


def optimizer_record() -> dict[str, Tensor]:
    """Return the settings of the optimizer ``exp000`` builds.

    Returns:
      record: The entries of ``HYPERPARAMETERS``, as float64.

    """
    step = reference.tiny_step(exp000, blocks=(1, 1, 1), classes=3, epochs=200)
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    values = [float(cast(float, group[name])) for name in reference.HYPERPARAMETERS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building the full ``exp000`` step.

    Returns:
      record: The next eight draws.

    """
    config = exp000().step
    config.parallelism = NoParallel.Config(device="cpu")
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        _ = config.make()
        return {"draws/exp000": rng_fingerprint()}


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`REGULARIZERS`.
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
            REGULARIZERS[name],
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
            probe = step.call_eval(media=reference.fixed_media(reference.STEPS))
            assert isinstance(probe, Tensor)
            probes.append(probe)
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        f"lockstep/{name}/loss": torch.stack(losses),
        f"lockstep/{name}/regularization": torch.stack(spy.values),
        f"lockstep/{name}/logits": torch.stack(logits),
        f"lockstep/{name}/probe": torch.stack(probes),
        f"lockstep/{name}/context": torch.tensor(
            [[c.num_classes, c.epoch, c.total_epochs] for c in spy.contexts],
        ),
    }


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = (
        regularizer_record()
        | schedule_record()
        | optimizer_record()
        | draws_record()
        | reference.best_record()
    )
    for name in REGULARIZERS:
        record |= lockstep_record(name)
    return record


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
