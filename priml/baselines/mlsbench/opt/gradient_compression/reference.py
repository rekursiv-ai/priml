"""Port-side runs the ``optimization-gradient-compression`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's
edit, and these runs on the same inputs under the same pins, compares them bit
for bit, and stores the script's values under ``testdata/``;
``reference_test.py`` replays these runs against them. The loader and the
ResNet are the vision port's, reused where this script's are the same; the
inputs and pins are the vision capture's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.multitask.model import keep_default
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    Compressor,
    ErrorFeedback,
)
from priml.baselines.mlsbench.opt.gradient_compression.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
)
from priml.baselines.mlsbench.opt.gradient_compression.metrics import (
    MeanCrossEntropy,
)
from priml.baselines.mlsbench.opt.gradient_compression.train_step import (
    CompressedTrainStep,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.upstream import Batches
    from priml.baselines.mlsbench.opt.gradient_compression.train_loop import (
        CompressionTrainLoop,
    )


COMPRESSORS: Final[dict[str, Callable[[], CompressionTrainLoop.Config]]] = {
    "none": exp000,
    "topk_ef": exp001,
    "qsgd": exp002,
    "signsgd": exp003,
}
"""The experiment that carries each compressor, keyed as the script's variants."""

SETTINGS: Final[dict[str, Callable[[], CompressionTrainLoop.Config]]] = {
    "exp000": exp000,
    "exp004": exp004,
    "exp005": exp005,
}
"""The experiment of each of the task's settings, by experiment name."""

SGD_SETTINGS: Final = tuple(
    name for name in reference.HYPERPARAMETERS if name != "initial_lr"
)
"""Parameter-group entries compared; the script sets no ``initial_lr``."""

EPOCHS: Final = 200
"""Epochs of every setting."""

LOCKSTEP_EPOCHS: Final = 3
"""Epochs of a lockstep run, one update each: one of warmup, two of cosine."""

LOCKSTEP_WARMUP: Final = 1
"""Warmup epochs of a lockstep run; :func:`schedule_record` pins the recipe's five."""

COMPRESSION_STEPS: Final = 3
"""Rounds of fixed gradients each compressor is run on."""

GRADIENT_SHAPES: Final = {"weight": (8, 25), "bias": (7,), "zero": (2, 3)}
"""Each fixed parameter's gradient shape; ``weight`` is large enough that top-1% keeps two."""

IMAGE_SHAPE: Final = (2, 3, 32, 32)
"""Batch, channels, height and width of the full networks' inputs."""

TEST_IMAGE_SHAPE: Final = (2, 3, 4, 5)
"""Uint8 images the test pipeline normalizes."""

METRIC_BATCHES: Final = (2, 2, 1)
"""Sizes of the test batches the evaluation metrics see; the last is short."""


def fixed_gradients(step: int) -> dict[str, Tensor]:
    """Return round ``step``'s gradient of each parameter in :data:`GRADIENT_SHAPES`.

    ``weight`` has a norm above one, so QSGD clips it; ``bias`` below one, so it
    does not; ``zero`` is zero, which QSGD returns before drawing.

    Args:
      step: Round, which shifts the patterns.

    Returns:
      grads: Float32 gradients of multiples of a power of two.

    """
    weight = (torch.arange(200) * 7 + 3 * step) % 61 - 30
    bias = (torch.arange(7) * 3 + step) % 11 - 5
    return {
        "weight": (weight / 16).reshape(GRADIENT_SHAPES["weight"]),
        "bias": bias / 64,
        "zero": torch.zeros(GRADIENT_SHAPES["zero"]),
    }


def fixed_images32() -> Tensor:
    """Return float32 images of shape :data:`IMAGE_SHAPE`: multiples of one eighth."""
    count = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2] * IMAGE_SHAPE[3]
    return ((torch.arange(count) * 5 % 17 - 8) / 8).reshape(IMAGE_SHAPE)


def fixed_test_images() -> Tensor:
    """Return uint8 images the test pipeline normalizes.

    Returns:
      images: ``TEST_IMAGE_SHAPE`` uint8 pixels in a fixed pattern.

    """
    count = (
        TEST_IMAGE_SHAPE[0]
        * TEST_IMAGE_SHAPE[1]
        * TEST_IMAGE_SHAPE[2]
        * TEST_IMAGE_SHAPE[3]
    )
    return (torch.arange(count) * 37 % 251).reshape(TEST_IMAGE_SHAPE).to(torch.uint8)


def compressor_record(
    name: str,
    compressor: Compressor | None = None,
) -> dict[str, Tensor]:
    """Run one experiment's compressor on every round of fixed gradients.

    Args:
      name: A key of :data:`COMPRESSORS`.
      compressor: Runs instead of the experiment's; for a test's perturbation.

    Returns:
      record: Per parameter its reconstruction in each round, and for an
        error-feedback compressor the residual it ends with.

    """
    made = compressor or COMPRESSORS[name]().step.compressor.make()
    outputs: dict[str, list[Tensor]] = {key: [] for key in GRADIENT_SHAPES}
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        for step in range(COMPRESSION_STEPS):
            for key, grad in fixed_gradients(step).items():
                outputs[key].append(made(grad, name=key))
    record = {
        f"compressor/{name}/{key}": torch.stack(values)
        for key, values in outputs.items()
    }
    if isinstance(made, ErrorFeedback):
        for key, residual in made.residuals.items():
            record[f"compressor/{name}/residual/{key}"] = residual
    return record


def tiny_step(
    factory: Callable[[], CompressionTrainLoop.Config],
    *,
    epochs: int,
    warmup: int | None = None,
) -> CompressedTrainStep:
    """Build the step of ``factory`` with one block per stage and three classes.

    Args:
      factory: The experiment whose recipe is kept.
      epochs: Schedule horizon.
      warmup: The schedule's warmup epochs; None keeps the recipe's.

    Returns:
      step: A CPU step with the experiment's optimizer, schedule and compressor.

    """
    config = factory().step
    # The schedule and optimizer do not depend on the network, so every setting
    # shares the shallowest ResNet here, VGG-11's included.
    model = config.model = ResNet.Config()
    model.channels_in = 3
    model.channels_out = reference.CLASSES
    reference.tiny_model(model)
    model.init_conv = model.init_head = model.init_bias = keep_default
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = epochs
    if warmup is not None:
        config.schedule = config.schedule.copy_tree().update(warmup_epochs=warmup)
    step = config.make()
    assert isinstance(step, CompressedTrainStep)
    return step


def full_step(
    factory: Callable[[], CompressionTrainLoop.Config],
) -> CompressedTrainStep:
    """Build an experiment's step at full size on the CPU."""
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    step = config.make()
    assert isinstance(step, CompressedTrainStep)
    return step


def network_record(name: str) -> dict[str, Tensor]:
    """Build one setting's full network from the seed and run it on fixed images.

    Args:
      name: A key of :data:`SETTINGS`.

    Returns:
      record: Where the generator stands after the build, then the logits of
        a training-mode forward (whose dropout draws) and an evaluation-mode one.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = full_step(SETTINGS[name]).model
        draws = rng_fingerprint()
        model.train()
        train = reference.model_logits(model, fixed_images32())
        model.eval()
        evaluated = reference.model_logits(model, fixed_images32())
    return {
        f"draws/{name}": draws,
        f"forward/{name}/train": train.detach(),
        f"forward/{name}/eval": evaluated.detach(),
    }


def schedule_record(name: str) -> dict[str, Tensor]:
    """Return the learning rate of each epoch of one setting and its optimizer's settings.

    Args:
      name: A key of :data:`SETTINGS`.

    Returns:
      record: The rates and :data:`SGD_SETTINGS`, as float64.

    """
    factory = SETTINGS[name]
    step = tiny_step(factory, epochs=EPOCHS)
    group = cast("Mapping[str, object]", step.optimizer.param_groups[0])
    base_lr = from_plain(group["lr"], float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=EPOCHS)
        for epoch in range(EPOCHS)
    ]
    settings = [float(cast(float, group[key])) for key in SGD_SETTINGS]
    return {
        f"schedule/{name}": torch.tensor(rates, dtype=torch.float64),
        f"optimizer/{name}": torch.tensor(settings, dtype=torch.float64),
    }


def eval_epochs_record() -> dict[str, Tensor]:
    """Return the completed epochs after which ``exp000`` evaluates the test split."""
    policy = exp000().eval_epochs.make()
    epochs = [e for e in range(1, EPOCHS + 1) if policy(e, total_epochs=EPOCHS)]
    return {"eval/epochs": torch.tensor(epochs)}


def best_record() -> dict[str, Tensor]:
    """Report the scripted evaluations of ``exp000``'s evaluation epochs to the metric.

    Evaluation ``i`` of the run reports ``SCRIPTED_CORRECT[i]`` hits out of
    ``SCRIPTED_TOTAL``, as the capture's scripted ``evaluate`` does.

    Returns:
      record: The best accuracy, rounded to the two places the script prints.

    """
    metric = BestAccuracy.Config().make()
    total = reference.SCRIPTED_TOTAL
    label = torch.zeros(total, dtype=torch.int64)
    hit, miss = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    evaluations = len(eval_epochs_record()["eval/epochs"])
    for correct in reference.SCRIPTED_CORRECT[:evaluations]:
        metric.reset()
        metric.update(
            torch.cat([hit.expand(correct, 2), miss.expand(total - correct, 2)]),
            label=label,
        )
        _ = metric.compute()
    best = float(f"{metric.compute()['best']:.2f}")
    return {"best/accuracy": torch.tensor(best, dtype=torch.float64)}


def metric_batches() -> list[tuple[Tensor, Tensor]]:
    """Return the test batches the evaluation metrics see.

    Returns:
      batches: Per size in :data:`METRIC_BATCHES`, fixed media and labels.

    """
    batches: list[tuple[Tensor, Tensor]] = []
    for index, size in enumerate(METRIC_BATCHES):
        batches.append(
            (reference.fixed_media(index)[:size], reference.fixed_label(index)[:size]),
        )
    return batches


def metric_record() -> dict[str, Tensor]:
    """Evaluate the shallowest network on :func:`metric_batches` with both test metrics.

    Returns:
      record: The accuracy in percent and the mean cross-entropy, as float64.

    """
    accuracy = BestAccuracy.Config().make()
    loss = MeanCrossEntropy.Config().make()
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = tiny_step(exp000, epochs=EPOCHS)
        for media, label in metric_batches():
            evaluated = step.call_eval(media=media)
            assert isinstance(evaluated, Tensor)
            # ``host_agnostic_numerics`` does not upcast ops on an inference
            # tensor, so the loss would round natively while the script's,
            # computed under ``no_grad``, is upcast; a plain copy is upcast too.
            logits = evaluated.clone()
            accuracy.update(logits, label=label)
            loss.update(logits, label=label)
    return {
        "metric/accuracy": torch.tensor(
            accuracy.compute()["final"],
            dtype=torch.float64,
        ),
        "metric/loss": torch.tensor(loss.compute()["mean"], dtype=torch.float64),
    }


def transform_record() -> dict[str, Tensor]:
    """Normalize fixed uint8 images with each dataset's statistics.

    Returns:
      record: ``transform/<experiment>`` for CIFAR-10 and CIFAR-100, as the
        test pipeline normalizes them.

    """
    record: dict[str, Tensor] = {}
    for name in ("exp000", "exp004"):
        config = SETTINGS[name]().dataset
        record[f"transform/{name}"] = normalize(
            fixed_test_images(),
            mean=torch.tensor(config.mean),
            std=torch.tensor(config.std),
        )
    return record


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network one update per epoch for :data:`LOCKSTEP_EPOCHS`.

    Args:
      name: A key of :data:`COMPRESSORS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = tiny_step(
            COMPRESSORS[name],
            epochs=LOCKSTEP_EPOCHS,
            warmup=LOCKSTEP_WARMUP,
        )
        for index in range(LOCKSTEP_EPOCHS):
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
        f"lockstep/{name}/logits": torch.stack(logits),
        f"lockstep/{name}/probe": torch.stack(probes),
    }


def loader_batches(name: str, directory: Path) -> tuple[Batches, Batches]:
    """Run one setting's loaders over the synthetic split, alternating the splits.

    Args:
      name: A key of :data:`SETTINGS`.
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order.

    """
    media, label = reference.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = SETTINGS[name]().dataset
    config.working_dir = directory
    config.batch_size = reference.LOADER_BATCH
    train: Batches = []
    evaluation: Batches = []
    with reference.seeded(reference.SEED):
        data = config.make()
        for _ in range(reference.LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def loader_record(directory: Path) -> dict[str, Tensor]:
    """Return what the CIFAR-100 setting's training loader drew over the synthetic split.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      record: Per training pass the label, crop offsets and mirror of every image.

    """
    train, _ = loader_batches("exp004", directory)
    return reference.decode_batches(
        train,
        config=exp004().dataset,
        source=reference.synthetic_split()[0],
    )


def draw_sites(directory: Path) -> dict[str, int]:
    """Count the random draws of a network build, the loaders and QSGD.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      counts: Maps ``"<api>@<file>:<function>"`` to the number of calls.

    """
    with recorded_draws() as counts:
        _ = network_record("exp000")
        _ = loader_batches("exp000", directory)
        _ = compressor_record("qsgd")
    return counts


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison the golden holds.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = eval_epochs_record() | best_record() | metric_record()
    record |= transform_record()
    record |= loader_record(directory)
    for name in SETTINGS:
        record |= schedule_record(name) | network_record(name)
    for name in COMPRESSORS:
        record |= compressor_record(name) | lockstep_record(name)
    return record
