"""Tests for the ffcv-imagenet training step."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, override

import copy

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.imagenet.model import BlurPoolConv2d, TorchvisionResNet
from priml.baselines.imagenet.model_test import strided_net
from priml.baselines.imagenet.train_step import ImageNetTrainStep
from priml.math.schedules import constant
from priml.optimizers import CompositeOptimizer, learning_rate
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent


def tiny_step() -> ImageNetTrainStep.Config:
    """Return the recipe at minimum size: CPU, eager, float32, tiny net."""
    config = ImageNetTrainStep.Config()
    model = config.model = TorchvisionResNet.Config()
    model.arch = PartialConfig(strided_net)
    model.channels_out = 10
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    # CPU has no float16 autocast worth testing; the scaler is then disabled.
    config.dtype_autocast = None
    config.train_budget_steps = 8
    config.resolution_min = 32
    config.resolution_max = 64
    return config


def tiny_batch(*, size: int = 4, image: int = 48, seed: int = 0) -> dict[str, Tensor]:
    """Return a batch shaped as the pipeline emits it, after ``preprocess_batch``."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "image": torch.randn(size, 3, image, image, generator=generator),
        "label": torch.randint(0, 10, (size,), generator=generator),
    }


def test_train_step_returns_per_example_loss_and_logits() -> None:
    out = tiny_step().make().train_step(**tiny_batch())
    assert out["loss"].shape == (4,)
    assert out["model"].shape == (4, 10)


def test_batchnorm_parameters_escape_weight_decay() -> None:
    """ffcv-imagenet decays every parameter whose name lacks ``bn``."""
    config = tiny_step()
    model = config.model = TorchvisionResNet.Config()
    model.channels_out = 10
    with torch.device("meta"):
        step_model = model.make()
    optimizer = config.optimizer.make()(step_model)
    assert isinstance(optimizer, CompositeOptimizer)
    bn, rest = optimizer.optimizers
    assert bn.param_groups[0]["weight_decay"] == 0.0
    assert rest.param_groups[0]["weight_decay"] == 1e-4
    names = {id(p): n for n, p in step_model.named_parameters()}
    assert all("bn" in names[id(p)] for p in _group_parameters(bn.param_groups[0]))
    assert not any(
        "bn" in names[id(p)] for p in _group_parameters(rest.param_groups[0])
    )


def test_every_group_peaks_at_the_step_learning_rate() -> None:
    step = tiny_step().make()
    assert {g["initial_lr"] for g in step.optimizer.param_groups} == {0.5}


def test_rate_follows_the_cyclic_schedule() -> None:
    step = tiny_step().make()
    rates: list[float] = []
    for _ in range(8):
        _ = step.train_step(**tiny_batch())
        rates.append(learning_rate(step.optimizer))
    # peak=2/16 of 8 steps: the first update is at progress 0 -> 1e-4 * 0.5.
    assert rates[0] == pytest.approx(0.5e-4)
    assert rates[-1] < rates[1]


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 32), (11, 32), (11.9, 32), (12, 64), (13, 64), (16, 64)],
)
def test_resolution_matches_ffcv_get_resolution(epoch: float, expected: int) -> None:
    """Ffcv's ramp over epochs 11..13, 32->64, rounded to a multiple of 32."""
    config = tiny_step()
    config.train_budget_steps = 160
    step = config.make()
    step.timer_step.global_count = round(epoch * 10)
    assert step.resolution == expected


def test_train_step_resizes_to_the_scheduled_resolution() -> None:
    step = tiny_step().make()
    seen: list[int] = []
    model = step.model
    assert isinstance(model, TorchvisionResNet)
    step._model = _Recorder(model, seen)
    _ = step.train_step(**tiny_batch(image=48))
    assert seen == [32]


def test_tta_sums_the_image_and_its_mirror() -> None:
    step = tiny_step().make()
    media = tiny_batch()["image"]
    model = step.model
    assert isinstance(model, TorchvisionResNet)
    model.eval()
    with torch.no_grad():
        plain = model.forward(media)
        mirrored = model.forward(media.flip(-1))
    out = step.call_eval(image=media)
    assert isinstance(out, Tensor)
    assert torch.allclose(out, plain + mirrored)


def test_preprocess_drops_the_frame_axis_and_goes_channels_last() -> None:
    step = tiny_step().make()
    batch: dict[str, object] = {
        "image": torch.randn(2, 3, 1, 8, 8),
        "label": torch.zeros(2),
    }
    image = step.preprocess_batch(batch)["image"]
    assert isinstance(image, Tensor)
    assert image.shape == (2, 3, 8, 8)
    assert image.is_contiguous(memory_format=torch.channels_last)


def test_state_dict_round_trip_restores_progress_and_scaler() -> None:
    step = tiny_step().make()
    for _ in range(2):
        _ = step.train_step(**tiny_batch())
    restored = tiny_step().make()
    restored.load_state_dict(step.state_dict())
    assert restored.global_step == 2
    assert "scaler" in step.state_dict()


def test_three_train_steps_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="three_steps",
        build_module=lambda: _TrainStepModule(tiny_step()),
        build_input=tiny_batch,
        seed=42,
    )


class _TrainStepModule(nn.Module):
    """Adapts a train step to the module interface the golden harness drives."""

    def __init__(self, config: ImageNetTrainStep.Config) -> None:
        super().__init__()
        self.step = config.make()
        self.inner = self.step.model

    @override
    def forward(self, image: Tensor, label: Tensor) -> Tensor:
        return torch.stack(
            [self.step.train_step(image=image, label=label)["loss"] for _ in range(3)],
        )


class _Recorder(nn.Module):
    """Records the side length of every image batch it forwards."""

    def __init__(self, model: TorchvisionResNet, seen: list[int]) -> None:
        super().__init__()
        self.model = model
        self.seen = seen

    @override
    def forward(self, media: Tensor) -> Tensor:
        self.seen.append(media.shape[-1])
        return self.model.forward(media)


def _group_parameters(group: dict[str, object]) -> list[nn.Parameter]:
    params = group["params"]
    assert isinstance(params, list)
    return [p for p in cast(list[object], params) if isinstance(p, nn.Parameter)]


def test_gradient_is_ffcvs_fused_label_smoothed_mean() -> None:
    """The backward differentiates ``CrossEntropyLoss``, as ffcv-imagenet does.

    The mean of per-example smoothed losses differs from the fused mean in the
    last bits; that alone moved 267 of ResNet-50's 326 tensors after 5 steps.
    """
    config = tiny_step()
    # A shape where the two means' gradients differ on CPU float32.
    assert isinstance(config.model, TorchvisionResNet.Config)
    config.model.channels_out = 13
    config.optimizer = PartialConfig(torch.optim.SGD, lr=1.0)
    config.schedule = PartialConfig(constant)
    step = config.make()
    reference = copy.deepcopy(step.model)
    batch = tiny_batch(size=7, image=32, seed=1)
    _ = step.train_step(**batch)

    reference.train()
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
        reference(batch["image"]),
        batch["label"],
    )
    loss.backward()
    with torch.no_grad():
        for param in reference.parameters():
            if param.grad is not None:
                _ = param.sub_(0.5 * param.grad)
    for (name, ours), theirs in zip(
        step.model.state_dict().items(),
        reference.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(ours, theirs), name


def test_blurpool_is_on_by_default() -> None:
    model = tiny_step().make().model
    assert isinstance(model, TorchvisionResNet)
    assert isinstance(model.net.get_submodule("conv2"), BlurPoolConv2d)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
