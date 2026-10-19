"""Runs of the pristine ``optimization-gradient-compression`` script, under the capture's pins.

The script and each baseline's edit are imported from a clone of the benchmark,
not vendored, and driven through their own functions. ``train`` runs as
written with the network shrunk, the loaders replaced by fixed batches and an
optimizer that records every update, so its seeding, schedule, compression,
evaluation cadence and best-accuracy bookkeeping are observed rather than
re-implemented. The data pipeline runs through the script's own
``get_dataloaders`` over a synthetic split. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable
from unittest import mock

import contextlib
import io
import sys

from PIL import Image
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream,
)
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.opt.gradient_compression import reference
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from torch.utils.data import DataLoader

    from priml.baselines.mlsbench.dl.vision.upstream import Batches, Lockstep

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__optimization-gradient-compression/tests/meta"
    "/pristine/pytorch-vision/custom_compressor.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/optimization-gradient-compression/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("topk_ef", "qsgd", "signsgd")
"""The published baselines, each named as its edit file and as its port's variant."""

SETTINGS: Final = {
    "exp000": ("resnet20", "cifar10", 0.1),
    "exp004": ("vgg11", "cifar100", 0.05),
    "exp005": ("resnet56", "cifar10", 0.1),
}
"""Per port experiment, the setting's ``--model``, ``--dataset`` and ``--lr``."""

CLASSES: Final = {"cifar10": 10, "cifar100": 100}
"""Classes of each ``--dataset``."""

DATASETS: Final = {"cifar10": "CIFAR10", "cifar100": "CIFAR100"}
"""The torchvision class each ``--dataset`` reads."""

COMPRESS_RATIO: Final = 0.01
"""Every setting's ``--compress-ratio``."""


class Compressing(Protocol):
    """An instance of the script's editable ``Compressor``."""

    def compress(self, tensor: Tensor, name: str) -> tuple[list[Tensor], object]:
        """Return what would be communicated and the local context."""
        raise NotImplementedError

    def decompress(self, compressed_tensors: list[Tensor], ctx: object) -> Tensor:
        """Rebuild the gradient."""
        raise NotImplementedError


@runtime_checkable
class Script(Protocol):
    """The part of the ``optimization-gradient-compression`` script the capture uses."""

    __file__: str
    BasicBlock: type[nn.Module]
    datasets: ModuleType
    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]
    Compressor: Callable[[float], Compressing]

    def build_model(
        self,
        model_name: str,
        num_classes: int,
        device: torch.device,
    ) -> nn.Module:
        """Build a setting's network."""
        raise NotImplementedError

    def get_dataloaders(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = ...,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor]],
        int,
    ]:
        """Build the loaders."""
        raise NotImplementedError

    def evaluate(
        self,
        model: nn.Module,
        test_loader: object,
        device: torch.device,
    ) -> tuple[float, float]:
        """Return accuracy in percent and mean cross-entropy."""
        raise NotImplementedError

    def train(self, args: Namespace) -> None:
        """Run the whole training of one setting."""

    def main(self) -> None:
        """Parse the command line and train."""


def variants(root: Path, scratch: Path) -> dict[str, Script]:
    """Import the pristine script and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``none`` (the pristine script) and each of :data:`EDITED`.

    """
    scripts = {"none": upstream.load_script(root / PRISTINE)} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }
    return {name: upstream.narrowed(script, Script) for name, script in scripts.items()}


def upstream_compressor(script: Script, name: str) -> dict[str, Tensor]:
    """Run one script's ``Compressor`` on every round of fixed gradients.

    Args:
      script: The script carrying the compressor under test.
      name: Which compressor, for the record's keys.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.opt.gradient_compression.reference.compressor_record`.

    """
    compressor = script.Compressor(COMPRESS_RATIO)
    outputs: dict[str, list[Tensor]] = {key: [] for key in reference.GRADIENT_SHAPES}
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        for step in range(reference.COMPRESSION_STEPS):
            for key, grad in reference.fixed_gradients(step).items():
                compressed, ctx = compressor.compress(grad, key)
                outputs[key].append(compressor.decompress(compressed, ctx))
    record = {
        f"compressor/{name}/{key}": torch.stack(values)
        for key, values in outputs.items()
    }
    residuals = cast(dict[str, dict[str, Tensor]], vars(compressor)).get(
        "residuals",
        {},
    )
    for key, residual in residuals.items():
        record[f"compressor/{name}/residual/{key}"] = residual
    return record


def upstream_network(
    script: Script,
    name: str,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build one setting's network from the seed, as ``train`` does, and run it.

    Args:
      script: The pristine script.
      name: A key of :data:`SETTINGS`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.opt.gradient_compression.reference.network_record`.
      state: The network's whole state after the build, for a live comparison.

    """
    model_name, dataset, _ = SETTINGS[name]
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = script.build_model(model_name, CLASSES[dataset], torch.device("cpu"))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        draws = rng_fingerprint()
        model.train()
        train = vision_reference.logits_of(model, reference.fixed_images32())
        model.eval()
        evaluated = vision_reference.logits_of(model, reference.fixed_images32())
    record = {
        f"draws/{name}": draws,
        f"forward/{name}/train": train.detach(),
        f"forward/{name}/eval": evaluated.detach(),
    }
    return record, state


class Run:
    """What one call of the script's ``train`` did, as its replaced pieces saw it."""

    def __init__(self, script: Script, *, scripted: bool) -> None:
        self.script = script
        self.scripted = scripted
        self.model: nn.Module | None = None
        self.rates: list[float] = []
        self.settings: dict[str, float] = {}
        self.states: list[dict[str, Tensor]] = []
        self.probes: list[Tensor] = []
        self.losses: list[Tensor] = []
        self.logits: list[Tensor] = []
        self.eval_epochs: list[int] = []
        self.accuracies: list[float] = []
        self.printed = ""
        # Kept before ``run_train`` patches the attributes with this run's own.
        self._cross_entropy = functional.cross_entropy
        self._evaluate = script.evaluate
        self._sgd = torch.optim.SGD

    def build_model(
        self,
        model_name: str,
        num_classes: int,
        device: object,
    ) -> nn.Module:
        """Build the shallowest ResNet with the lockstep's classes, whatever was asked.

        Args:
          model_name: The setting's ``--model``; ignored.
          num_classes: The dataset's classes; ignored.
          device: Where the script places the network; ignored.

        Returns:
          model: The script's ``ResNet`` with one block per stage.

        """
        del model_name, num_classes, device
        with upstream.narrowed_widths(self.script):
            self.model = self.script.ResNet(
                self.script.BasicBlock,
                [1, 1, 1],
                vision_reference.CLASSES,
            )
        return self.model

    def get_dataloaders(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 0,
    ) -> tuple[_EpochBatches, list[tuple[Tensor, Tensor]], int]:
        """Return one fixed batch per epoch, one fixed test batch and the lockstep's classes.

        Args:
          dataset_name: Unused; the batches are fixed.
          batch_size: Unused; each batch holds the fixed images.
          num_workers: Unused; nothing is loaded.

        Returns:
          train: A fixed batch in each epoch.
          test: The fixed probe batch.
          classes: The lockstep's class count.

        """
        del dataset_name, batch_size, num_workers
        probe = (
            vision_reference.fixed_media(vision_reference.STEPS),
            vision_reference.fixed_label(vision_reference.STEPS),
        )
        return _EpochBatches(), [probe], vision_reference.CLASSES

    def evaluate(
        self,
        model: nn.Module,
        test_loader: object,
        device: torch.device,
    ) -> tuple[float, float]:
        """Note the epoch, then evaluate with the script's own or report a scripted accuracy.

        The script evaluates the last epoch's model twice, in the loop and after
        it; the second report repeats the first, as the unchanged model would.

        Args:
          model: The network under training.
          test_loader: What the script's ``get_dataloaders`` returned for testing.
          device: Where the script runs.

        Returns:
          accuracy: Test accuracy in percent.
          loss: Mean test cross-entropy; zero when scripted.

        """
        epoch = len(self.rates)
        repeat = bool(self.eval_epochs) and self.eval_epochs[-1] == epoch
        self.eval_epochs.append(epoch)
        if not self.scripted:
            return self._evaluate(model, test_loader, device)
        if not repeat:
            correct = vision_reference.SCRIPTED_CORRECT[len(self.accuracies)]
            self.accuracies.append(100.0 * correct / vision_reference.SCRIPTED_TOTAL)
        return self.accuracies[-1], 0.0

    def optimizer(
        self,
        params: object,
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> torch.optim.SGD:
        """Build torch's SGD as the script asks, recording every update it makes.

        Args:
          params: The network's parameters.
          lr: Initial learning rate.
          momentum: Momentum factor.
          weight_decay: L2 penalty.

        Returns:
          optimizer: Torch's SGD with a hook that records each step.

        """
        optimizer = self._sgd(
            cast("list[Tensor]", params),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        group = cast(dict[str, object], optimizer.param_groups[0])
        self.settings = {
            key: float(cast(float, group[key])) for key in reference.SGD_SETTINGS
        }
        _ = optimizer.register_step_post_hook(self.after_step)
        return optimizer

    def after_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: object,
        kwargs: object,
    ) -> None:
        """Record the update's rate, the state after it and an evaluation-mode probe.

        Args:
          optimizer: The optimizer that just stepped.
          args: Its ``step`` arguments; unused.
          kwargs: Its ``step`` keyword arguments; unused.

        """
        del args, kwargs
        model = self.model
        assert isinstance(model, nn.Module)
        self.rates.append(
            from_plain(cast(object, optimizer.param_groups[0]["lr"]), float),
        )
        training = model.training
        model.eval()
        with torch.no_grad():
            self.probes.append(
                vision_reference.logits_of(
                    model,
                    vision_reference.fixed_media(vision_reference.STEPS),
                ),
            )
        model.train(training)
        self.states.append({k: v.clone() for k, v in model.state_dict().items()})

    def cross_entropy(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute the script's loss; keep the training ones, which take the mean."""
        loss = self._cross_entropy(logits, labels, reduction=reduction)
        if reduction == "mean":
            self.losses.append(loss.detach())
            self.logits.append(logits.detach())
        return loss


def script_arguments(script: Script, argv: list[str]) -> Namespace:
    """Parse ``argv`` with the script's own ``main``, without training.

    Args:
      script: A variant of the script.
      argv: Command-line arguments after the program name.

    Returns:
      args: What ``main`` hands ``train``, every flag not in ``argv`` at its
        default.

    """
    parsed: list[Namespace] = []
    with (
        mock.patch.object(sys, "argv", ["script", *argv]),
        mock.patch.object(script, "train", parsed.append),
    ):
        script.main()
    (args,) = parsed
    return args


def run_train(
    script: Script,
    name: str = "exp000",
    *,
    epochs: int,
    scripted: bool,
    warmup: int | None = None,
) -> Run:
    """Call the script's ``train`` for ``epochs`` at one setting's arguments.

    Args:
      script: A variant of the script.
      name: A key of :data:`SETTINGS`, whose ``--lr`` and names are passed.
      epochs: The ``--epochs``.
      scripted: Report scripted accuracies instead of evaluating.
      warmup: The ``--warmup-epochs``; None leaves the script's default.

    Returns:
      run: What the replaced pieces saw.

    """
    model_name, dataset, lr = SETTINGS[name]
    argv = [
        f"--model={model_name}",
        f"--dataset={dataset}",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--compress-ratio={COMPRESS_RATIO}",
        f"--seed={vision_reference.SEED}",
    ]
    if warmup is not None:
        argv.append(f"--warmup-epochs={warmup}")
    args = script_arguments(script, argv)
    run = Run(script, scripted=scripted)
    printed = io.StringIO()
    with (
        vision_reference.seeded(vision_reference.SEED),
        host_agnostic_numerics(),
        mock.patch.object(script, "build_model", run.build_model),
        mock.patch.object(script, "get_dataloaders", run.get_dataloaders),
        mock.patch.object(script, "evaluate", run.evaluate),
        mock.patch.object(torch.optim, "SGD", run.optimizer),
        mock.patch.object(functional, "cross_entropy", run.cross_entropy),
        contextlib.redirect_stdout(printed),
    ):
        script.train(args)
    run.printed = printed.getvalue()
    return run


def upstream_lockstep(script: Script, name: str) -> Lockstep:
    """Run ``train`` on the shallowest network, one update per epoch.

    Args:
      script: The script carrying the compressor under test.
      name: Which compressor, for the record's keys.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    run = run_train(
        script,
        epochs=reference.LOCKSTEP_EPOCHS,
        scripted=False,
        warmup=reference.LOCKSTEP_WARMUP,
    )
    record = {
        f"lockstep/{name}/loss": torch.stack(run.losses),
        f"lockstep/{name}/logits": torch.stack(run.logits),
        f"lockstep/{name}/probe": torch.stack(run.probes),
    }
    return record, run.states


def upstream_trace(
    script: Script,
    *,
    epochs: int = reference.EPOCHS,
) -> dict[str, Tensor]:
    """Run ``train`` for every setting with scripted evaluations.

    Args:
      script: The pristine script.
      epochs: The ``--epochs``; the settings' 200 by default.

    Returns:
      record: Per setting the rates and the optimizer's settings, the epochs
        ``exp000`` evaluates after, and the best accuracy it printed.

    """
    record: dict[str, Tensor] = {}
    for name in SETTINGS:
        run = run_train(script, name, epochs=epochs, scripted=True)
        record[f"schedule/{name}"] = torch.tensor(run.rates, dtype=torch.float64)
        record[f"optimizer/{name}"] = torch.tensor(
            [run.settings[key] for key in reference.SGD_SETTINGS],
            dtype=torch.float64,
        )
        if name == "exp000":
            # The trailing call re-evaluates the last epoch's unchanged model, so
            # it names no epoch of its own.
            record["eval/epochs"] = torch.tensor(run.eval_epochs[:-1])
            best = float(run.printed.rsplit("best_acc=", 1)[1].split()[0])
            record["best/accuracy"] = torch.tensor(best, dtype=torch.float64)
    return record


def upstream_metrics(script: Script) -> dict[str, Tensor]:
    """Evaluate the shallowest network on the metric batches with the script's ``evaluate``.

    Args:
      script: The pristine script.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.opt.gradient_compression.reference.metric_record`.

    """
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        with upstream.narrowed_widths(script):
            model = script.ResNet(
                script.BasicBlock,
                [1, 1, 1],
                vision_reference.CLASSES,
            )
        accuracy, loss = script.evaluate(
            model,
            reference.metric_batches(),
            torch.device("cpu"),
        )
    return {
        "metric/accuracy": torch.tensor(accuracy, dtype=torch.float64),
        "metric/loss": torch.tensor(loss, dtype=torch.float64),
    }


def upstream_batches(
    script: Script,
    dataset: str,
    *,
    num_workers: int | None = None,
) -> tuple[Batches, Batches]:
    """Run the script's loaders over the synthetic split, alternating the splits.

    Args:
      script: The pristine script.
      dataset: The setting's ``--dataset``.
      num_workers: Loader workers; None takes the script's own default.

    Returns:
      passes: The training passes, then the test passes.

    """
    train: Batches = []
    evaluation: Batches = []
    with (
        vision_reference.seeded(vision_reference.SEED),
        vision_reference.script_workers(),
    ):
        workers = (
            upstream.default_workers(script.get_dataloaders)
            if num_workers is None
            else num_workers
        )
        train_loader, test_loader = loaders(script, dataset, num_workers=workers)
        for _ in range(vision_reference.LOADER_PASSES):
            train.append([{"media": x, "label": y} for x, y in train_loader])
            evaluation.append([{"media": x, "label": y} for x, y in test_loader])
    return train, evaluation


def upstream_transforms(script: Script) -> dict[str, Tensor]:
    """Run each dataset's test pipeline on fixed uint8 images.

    Args:
      script: The pristine script.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.opt.gradient_compression.reference.transform_record`.

    """
    pixels = [
        image.permute(1, 2, 0).contiguous().numpy()
        for image in reference.fixed_test_images()
    ]
    record: dict[str, Tensor] = {}
    for name in ("exp000", "exp004"):
        _, test_loader = loaders(script, SETTINGS[name][1], num_workers=0)
        transform = cast(_FakeCifar, test_loader.dataset).transform
        record[f"transform/{name}"] = torch.stack(
            [transform(Image.fromarray(p)) for p in pixels],
        )
    return record


def upstream_sites(script: Script, qsgd: Script) -> dict[str, int]:
    """Count the random draws of a network build, the loaders in-process and QSGD.

    Args:
      script: The pristine script.
      qsgd: The script with the ``qsgd`` edit.

    Returns:
      counts: As :func:`~priml.baselines.mlsbench.opt.gradient_compression.reference.draw_sites`.

    """
    with recorded_draws() as counts:
        _ = upstream_network(script, "exp000")
        _ = upstream_batches(script, "cifar10", num_workers=0)
        _ = upstream_compressor(qsgd, "qsgd")
    # The edited script lives in a scratch directory whose name changes every run.
    scratch = Path(qsgd.__file__).parent.name
    return {key.replace(f"@{scratch}/", "@edits/"): n for key, n in counts.items()}


def loaders(
    script: Script,
    dataset: str,
    *,
    num_workers: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build the script's loaders over the synthetic split.

    Args:
      script: The pristine script.
      dataset: The setting's ``--dataset``.
      num_workers: Loader workers.

    Returns:
      loaders: The training loader, then the test loader.

    """
    with mock.patch.object(script.datasets, DATASETS[dataset], _FakeCifar):
        train, test, _ = script.get_dataloaders(
            dataset,
            vision_reference.LOADER_BATCH,
            num_workers=num_workers,
        )
    return train, test


class _EpochBatches:
    """One fixed batch per pass, a new one each pass."""

    def __init__(self) -> None:
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        batch = (
            vision_reference.fixed_media(self.epoch),
            vision_reference.fixed_label(self.epoch),
        )
        self.epoch += 1
        yield batch

    def __len__(self) -> int:
        return 1


class _FakeCifar:
    """The synthetic split, in the shape the script expects of a dataset."""

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        del root, train, download
        self.media, self.label = vision_reference.synthetic_split()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        pixels = self.media[index].permute(1, 2, 0).contiguous().numpy()
        return self.transform(Image.fromarray(pixels)), int(self.label[index])
