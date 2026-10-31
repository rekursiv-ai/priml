"""Runs of the pristine ``dl-regularization`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's, but with the arguments of the setting's own launch script, so the
recipe is the one the benchmark runs rather than the script's defaults. The
lockstep run repeats ``main``'s epoch loop on the shallowest network. Nothing
runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable
from unittest import mock

import re
import shlex
import sys

import torch

from priml.baselines.mlsbench.dl.regularization import (
    reference,
)
from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream,
)
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__dl-regularization/tests/meta/pristine"
    "/pytorch-vision/custom_reg.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-regularization/edits")
"""Directory of the task's baseline edits, relative to the clone."""

SETTING: Final = Path("tasks/dl-regularization/scripts/resnet56_cifar100.sh")
"""Launch script of the ``resnet56-cifar100`` setting, relative to the clone."""

SETTING_SCRIPTS: Final = {
    "vgg16bn-cifar100": Path("tasks/dl-regularization/scripts/vgg16bn_cifar100.sh"),
    "mobilenetv2-fmnist": Path(
        "tasks/dl-regularization/scripts/mobilenetv2_fmnist.sh",
    ),
}
"""Launch script of each other setting, keyed as the vision ``SETTINGS``."""

EDITED: Final = ("dropblock", "confidence_penalty", "orthogonal_reg")
"""The baselines, each named as its edit file and as its port's variant."""

type Regularization = Callable[
    [nn.Module, Tensor, Tensor, Tensor, dict[str, int]],
    Tensor,
]
"""The script's ``compute_regularization``: model, inputs, outputs, targets, config."""


@runtime_checkable
class Script(Protocol):
    """The part of the ``dl-regularization`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    compute_regularization: Regularization

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    train_epoch: Callable[
        [
            nn.Module,
            Iterable[tuple[Tensor, Tensor]],
            Callable[[Tensor, Tensor], Tensor],
            torch.optim.Optimizer,
            torch.device,
            dict[str, int],
        ],
        tuple[float, float],
    ]
    """Train for one pass: model, loader, criterion, optimizer, device, config."""


def variants(root: Path, scratch: Path) -> dict[str, Upstream]:
    """Import the pristine script, made to add zero, and each baseline's edit of it.

    The pristine body raises; its docstring says to return zero for no
    regularization, which is what the ``zero`` variant's function does.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``zero`` and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    upstream.narrowed(pristine, Script).compute_regularization = no_regularization
    return {"zero": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def no_regularization(
    model: nn.Module,
    inputs: Tensor,
    outputs: Tensor,
    targets: Tensor,
    config: dict[str, int],
) -> Tensor:
    """Return the zero the pristine script's docstring asks for, on the logits' device."""
    del model, inputs, targets, config
    return torch.tensor(0.0, device=outputs.device)


def pristine_raises(root: Path) -> bool:
    """Whether the unedited script's ``compute_regularization`` raises as published.

    Args:
      root: Clone of the benchmark.

    Returns:
      raises: True when the body raises ``NotImplementedError``.

    """
    script = upstream.narrowed(upstream.load_script(root / PRISTINE), Script)
    logits, label = vision_reference.fixed_logits()
    try:
        script.compute_regularization(
            reference.ProbeNet(),
            vision_reference.fixed_media(0),
            logits,
            label,
            {"num_classes": 4, "epoch": 0, "total_epochs": 1},
        )
    except NotImplementedError:
        return True
    return False


def setting_argv(root: Path, launch: Path = SETTING) -> list[str]:
    """Return the arguments a setting's launch script passes to the training script.

    Shell defaults (``${SEED:-42}``) resolve to their default.

    Args:
      root: Clone of the benchmark.
      launch: The launch script, relative to the clone; ``resnet56-cifar100``'s
        by default.

    Returns:
      argv: ``sys.argv`` as the script would see it.

    """
    text = (root / launch).read_text().replace("\\\n", " ")
    script = PRISTINE.parent.name + "/" + PRISTINE.name
    (command,) = (line for line in text.splitlines() if script in line)
    tokens = shlex.split(re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", command))
    return ["script.py", *tokens[tokens.index(script) + 1 :]]


def trace_setting(
    script: Upstream,
    argv: list[str],
    *,
    classes: int = 100,
) -> MainTrace:
    """Trace ``main`` as the setting launches it.

    Args:
      script: A ``dl-regularization`` script, pristine or edited.
      argv: The setting's :func:`setting_argv`.
      classes: What the replaced loaders report for the setting's dataset.

    Returns:
      trace: What the replaced calls saw.

    """
    with mock.patch.object(script, "main", _WithArgv(script.main, argv)):
        return upstream.trace_main(script, classes=classes)


def with_epochs(argv: list[str], epochs: int) -> list[str]:
    """Return ``argv`` with its ``--epochs`` value replaced.

    Args:
      argv: A setting's :func:`setting_argv`.
      epochs: The new horizon.

    Returns:
      argv: A copy, every other argument as it was.

    """
    index = argv.index("--epochs") + 1
    return [*argv[:index], str(epochs), *argv[index + 1 :]]


def upstream_regularizers(scripts: dict[str, Upstream]) -> dict[str, Tensor]:
    """Run each baseline's regularizer on the probe network and fixed logits.

    Args:
      scripts: The variants of :func:`variants`; ``zero`` has no gradient and
        is skipped.

    Returns:
      record: Keyed as :func:`reference.regularizer_record`.

    """
    record: dict[str, Tensor] = {}
    for name in EDITED:
        task = upstream.narrowed(scripts[name], Script)
        record |= reference.regularizer_run(
            name,
            _ScriptCall(task.compute_regularization),
        )
    return record


def upstream_lockstep(
    script: Upstream,
    name: str,
    trace: MainTrace,
    *,
    arch: str = "resnet56",
    prefix: str | None = None,
) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the regularizer under test.
      name: Which regularizer, for the record's keys.
      trace: ``main`` traced over ``reference.EPOCHS`` epochs: the source of
        the optimizer settings and of each epoch's rate, as the script's own
        scheduler set it; for a VGG that is a warmup no fixed scheduler here
        could stand in for.
      arch: The setting's ``--arch``; its network is cut as ``tiny_model`` cuts it.
      prefix: Put before every key; ``lockstep/<name>/`` by default.

    Returns:
      lockstep: Per update the loss, the regularization term, the logits, a
        probe's logits afterwards and the configuration the term was given;
        and the model's whole state after each update.

    """
    task = upstream.narrowed(script, Script)
    spy = _RegularizationSpy(task.compute_regularization)
    criterion = upstream.CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with (
        vision_reference.seeded(vision_reference.SEED),
        host_agnostic_numerics(),
        mock.patch.object(task, "compute_regularization", spy),
    ):
        model = upstream.tiny_model(script, arch, vision_reference.CLASSES)
        task.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            dampening=trace.settings["dampening"],
            weight_decay=trace.settings["weight_decay"],
            nesterov=bool(trace.settings["nesterov"]),
        )
        config = {
            "num_classes": vision_reference.CLASSES,
            "epoch": 0,
            "total_epochs": vision_reference.EPOCHS,
        }
        for index in range(vision_reference.STEPS):
            config["epoch"] = index
            for group in optimizer.param_groups:
                group["lr"] = trace.rates[index]
            batch = [
                (
                    vision_reference.fixed_media(index),
                    vision_reference.fixed_label(index),
                ),
            ]
            task.train_epoch(
                model,
                batch,
                criterion,
                optimizer,
                torch.device("cpu"),
                config,
            )
            model.eval()
            with torch.no_grad():
                probes.append(
                    vision_reference.model_logits(
                        model,
                        vision_reference.fixed_media(vision_reference.STEPS),
                    ),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        # The script adds the two terms and keeps no sum; this repeats its one
        # addition under the same pins.
        losses = [
            ce + reg for ce, reg in zip(criterion.losses, spy.values, strict=True)
        ]
    prefix = prefix or f"lockstep/{name}/"
    record = {
        f"{prefix}loss": torch.stack(losses),
        f"{prefix}regularization": torch.stack(spy.values),
        f"{prefix}logits": torch.stack(criterion.logits),
        f"{prefix}probe": torch.stack(probes),
        f"{prefix}context": torch.tensor(
            [[c["num_classes"], c["epoch"], c["total_epochs"]] for c in spy.configs],
        ),
    }
    return record, states


def upstream_setting_regularizers(
    scripts: dict[str, Upstream],
    setting: str,
    model: nn.Module,
) -> dict[str, Tensor]:
    """Run each baseline's regularizer on a setting's full network.

    Args:
      scripts: The variants of :func:`variants`; ``zero`` is skipped.
      setting: A key of :data:`SETTING_SCRIPTS`, naming the record.
      model: The script's shallowest network at the setting, as
        :func:`tiny_network` builds it; each run reads a copy.

    Returns:
      record: Keyed as :func:`reference.setting_regularizer_record`.

    """
    record: dict[str, Tensor] = {}
    for name in EDITED:
        task = upstream.narrowed(scripts[name], Script)
        record |= reference.regularizer_run(
            name,
            _ScriptCall(task.compute_regularization),
            network=reference.Copies(model),
            prefix=f"settings/{setting}/",
        )
    return record


def tiny_network(script: Upstream, arch: str) -> nn.Module:
    """Build and initialize the script's shallowest network of an ``--arch``.

    Args:
      script: The pristine script.
      arch: The setting's ``--arch``.

    Returns:
      model: Built under the task's seed, as :func:`reference.tiny_network` is.

    """
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = upstream.tiny_model(script, arch, vision_reference.CLASSES)
        upstream.narrowed(script, Script).initialize_weights(model)
    return model


def upstream_setting_recipe(
    trace: MainTrace,
    setting: str,
    argv: list[str],
) -> dict[str, Tensor]:
    """Return a setting's rates, optimizer and batch sizes as its launch traced them.

    Args:
      trace: The setting's ``main``, traced with its launch arguments.
      setting: A key of :data:`SETTING_SCRIPTS`, naming the record.
      argv: The setting's :func:`setting_argv`.

    Returns:
      record: Keyed as :func:`reference.setting_recipe_record`.

    """
    record = {
        "schedule/lr": torch.tensor(trace.rates, dtype=torch.float64),
        "optimizer/settings": torch.tensor(
            [trace.settings[name] for name in vision_reference.HYPERPARAMETERS],
            dtype=torch.float64,
        ),
        "setting/batch_size": reference.setting_batch_size(argv),
    }
    return {f"settings/{setting}/{key}": value for key, value in record.items()}


def upstream_record(
    trace: MainTrace,
    regularizers: dict[str, Tensor],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the reference's value of every quantity the default test replays.

    Args:
      trace: A run of the setting's ``main``.
      regularizers: :func:`upstream_regularizers`.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`reference.port_record`.

    """
    record = dict(regularizers)
    record["schedule/lr"] = torch.tensor(trace.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [trace.settings[name] for name in vision_reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["draws/exp000"] = trace.draws
    record["best/accuracy"] = torch.tensor(trace.best, dtype=torch.float64)
    for values, _ in lockstep.values():
        record |= values
    return record


class _WithArgv:
    """Runs a ``main`` with ``sys.argv`` set, inside whatever the caller patched."""

    def __init__(self, main: Callable[[], None], argv: list[str]) -> None:
        self.main = main
        self.argv = argv

    def __call__(self) -> None:
        """Run ``main`` seeing :attr:`argv`."""
        with mock.patch.object(sys, "argv", self.argv):
            self.main()


class _ScriptCall:
    """Calls the script's regularizer in the port's keyword convention."""

    def __init__(self, function: Regularization) -> None:
        self.function = function

    def __call__(
        self,
        model: nn.Module,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the script's term for these inputs."""
        return self.function(model, media, logits, label, config)


class _RegularizationSpy:
    """Forwards to the script's regularizer and keeps what it saw."""

    def __init__(self, function: Regularization) -> None:
        self.function = function
        self.values: list[Tensor] = []
        self.configs: list[dict[str, int]] = []

    def __call__(
        self,
        model: nn.Module,
        inputs: Tensor,
        outputs: Tensor,
        targets: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the script's term for the batch."""
        value = self.function(model, inputs, outputs, targets, config)
        self.values.append(value.detach())
        self.configs.append(dict(config))
        return value
