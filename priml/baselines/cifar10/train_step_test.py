"""Tests for the CIFAR-10 training step."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast, override

import math

from configgle import Makeable, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.cifar10.model import ConvBlock, ResNet, SpeedNet
from priml.baselines.cifar10.train_step import Cifar10TrainStep
from priml.math.schedules import cosine, polynomial
from priml.optimizers import (
    CompositeOptimizer,
    Muon,
    complement,
    excluding,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.timer import CheckpointableStepTimer
from priml.train.parallelism import NoParallel


_GOLDEN_DIR = Path(__file__).parent.resolve() / "goldens"


def tiny_step(
    *,
    model_kind: Literal["resnet", "speednet"] = "resnet",
) -> Cifar10TrainStep.Config:
    """Return the smallest train-step config that still runs every branch."""
    config = Cifar10TrainStep.Config()
    if model_kind == "resnet":
        model = ResNet.Config()
        model.channels_hidden = (8, 16)
        model.blocks_per_stage = 1
        config.model = model
    else:
        speed = SpeedNet.Config()
        speed.channels_hidden = (8, 16, 24)
        block = speed.block = ConvBlock.Config()
        block.num_convs = 1
        config.model = speed
    config.total_train_steps = 4
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.translate_pad = 1
    config.whiten_num_images = 4
    return config


def _constant_half(progress: float) -> float:
    """A schedule holding the learning rate at half its peak."""
    del progress
    return 0.5


def muon_optimizer() -> CompositeOptimizer.Config:
    """Return the split Muon/SGD optimizer recipe ``exp002`` injects."""
    on_muon = excluding(Muon.eligible_tensor, "head")
    split = CompositeOptimizer.Config()
    split.select = [complement(on_muon), on_muon]
    sgd = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.67
    muon = Muon.Config()
    muon.lr = 0.24
    muon.ns_steps = 3
    split.optimizers = [sgd, muon]
    return split


def tiny_batch(*, size: int = 4, image: int = 8, seed: int = 0) -> dict[str, Tensor]:
    """Return a deterministic batch of noise images and labels."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "media": torch.randn(size, 3, image, image, generator=generator),
        "label": torch.randint(0, 10, (size,), generator=generator),
    }


def test_train_step_returns_per_example_loss_and_logits() -> None:
    step = tiny_step().make()
    out = step.train_step(**tiny_batch())
    assert out["loss"].shape == (4,)
    assert out["model"].shape == (4, 10)


def test_train_step_advances_the_step_counters() -> None:
    step = tiny_step().make()
    for _ in range(3):
        _ = step.train_step(**tiny_batch())
    assert step.global_step == 3
    assert step.local_step == 3


def test_train_step_updates_the_weights() -> None:
    step = tiny_step().make()
    before = step.model.head.weight.detach().clone()
    _ = step.train_step(**tiny_batch())
    assert not torch.equal(before, step.model.head.weight)


def test_loss_decreases_on_a_memorizable_batch() -> None:
    torch.manual_seed(0)
    config = tiny_step()
    config.total_train_steps = 4
    config.translate_pad = 0
    config.label_smoothing = 0.0
    step = config.make()
    batch = tiny_batch()
    losses = [float(step.train_step(**batch)["loss"].mean()) for _ in range(4)]
    assert losses[-1] < losses[0]


def test_muon_stack_splits_convolutions_from_the_rest() -> None:
    config = tiny_step()
    config.optimizer = muon_optimizer()
    step = config.make()
    assert isinstance(step.optimizer, CompositeOptimizer)
    sgd, muon = step.optimizer.optimizers
    assert isinstance(sgd, torch.optim.SGD)
    assert isinstance(muon, Muon)
    # Muon orthogonalizes a matrix-shaped update, which is meaningless for the
    # 1-D norm parameters; those and the head stay on SGD.
    assert {p.ndim for p in muon.param_groups[0]["params"]} == {4}
    named = dict(step.model.named_parameters())
    on_sgd = {id(p) for p in sgd.param_groups[0]["params"]}
    assert id(named["head.weight"]) in on_sgd
    assert id(named["head.bias"]) in on_sgd


def test_every_parameter_lands_in_exactly_one_optimizer_group() -> None:
    config = tiny_step()
    config.optimizer = muon_optimizer()
    step = config.make()
    owned = [id(p) for group in step.optimizer.param_groups for p in group["params"]]
    assert sorted(owned) == sorted(id(p) for p in step.model.parameters())


def test_a_split_recipe_still_presents_one_optimizer() -> None:
    """The step holds one optimizer whatever the recipe does.

    That is the composite's purpose: a step written for a single optimizer
    drives a split recipe with no branch, and nothing downstream iterates.
    """
    config = tiny_step()
    config.optimizer = muon_optimizer()
    assert isinstance(config.make().optimizer, torch.optim.Optimizer)


def test_adamw_stack_is_a_single_optimizer() -> None:
    optimizer = tiny_step().make().optimizer
    assert isinstance(optimizer, CompositeOptimizer)
    assert [type(o) for o in optimizer.optimizers] == [torch.optim.AdamW]


def test_schedule_warms_up_then_decays() -> None:
    step = object.__new__(Cifar10TrainStep)
    step.config = Cifar10TrainStep.Config(
        total_train_steps=100,
        warmup_fraction=0.1,
    )
    step.schedule = cosine
    step.timer_step = CheckpointableStepTimer()
    step.optimizer = cast(
        torch.optim.Optimizer,
        SimpleNamespace(param_groups=[{"initial_lr": 1.0, "lr": 1.0}]),
    )
    peak = 1.0

    rates: list[float] = []
    for global_step in (0, 9, 50, 99):
        step.timer_step.global_count = global_step
        step._apply_schedule()
        rates.append(step.optimizer.param_groups[0]["lr"])

    assert rates[0] < peak  # warming up
    assert rates[1] == pytest.approx(peak, rel=0.05)
    assert rates[-1] < rates[2] < rates[1]


@pytest.mark.compute_training
def test_the_injected_schedule_drives_the_learning_rate() -> None:
    def trace(schedule: PartialConfig[float]) -> list[float]:
        config = tiny_step()
        config.total_train_steps = 20
        config.warmup_fraction = 0.0
        config.schedule = schedule
        step = config.make()
        out: list[float] = []
        for _ in range(20):
            _ = step.train_step(**tiny_batch())
            out.append(step.optimizer.param_groups[0]["lr"])
        return out

    assert trace(PartialConfig(cosine)) != trace(PartialConfig(polynomial, power=1.2))


def test_an_arbitrary_callable_serves_as_a_schedule() -> None:
    """A schedule is injected, so it need not come from the library at all."""
    config = tiny_step()
    config.warmup_fraction = 0.0
    config.schedule = PartialConfig(_constant_half)
    step = config.make()
    _ = step.train_step(**tiny_batch())
    group = step.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.5 * group["initial_lr"])


def test_whitening_is_fitted_from_the_first_batch_only() -> None:
    step = tiny_step(model_kind="speednet").make()
    assert not step._whitened
    _ = step.train_step(**tiny_batch(image=32))
    assert step._whitened
    fitted = step.model.whiten.weight.detach().clone()
    _ = step.train_step(**tiny_batch(image=32, seed=1))
    assert torch.equal(fitted, step.model.whiten.weight)


def test_whitening_weights_survive_a_cache_round_trip(tmp_path: Path) -> None:
    config = tiny_step(model_kind="speednet")
    config.whiten_cache_path = tmp_path / "whiten.pt"
    first = config.make()
    _ = first.train_step(**tiny_batch(image=32))

    second = config.make()
    _ = second.train_step(**tiny_batch(image=32, seed=7))
    assert torch.equal(first.model.whiten.weight, second.model.whiten.weight)


def test_augmentation_preserves_the_image_shape() -> None:
    step = tiny_step().make()
    batch = tiny_batch()
    assert step._augment(batch["media"]).shape == batch["media"].shape


def test_derandomized_flip_alternates_with_the_step() -> None:
    config = tiny_step()
    config.derandomized_flip = True
    config.translate_pad = 0
    step = config.make()
    media = tiny_batch()["media"]

    # Driven through the step timer, which is what ``global_step`` reads: the
    # counter is derived, so a test that could assign it would be pinning the
    # flip against a position training cannot actually reach.
    assert step.global_step == 0
    assert torch.equal(step._augment(media), media)
    step.timer_step.global_count = 1
    assert torch.equal(step._augment(media), media.flip(-1))


def test_cutout_zeroes_part_of_the_image() -> None:
    torch.manual_seed(0)
    config = tiny_step()
    config.cutout_size = 4
    config.translate_pad = 0
    step = config.make()
    media = tiny_batch()["media"]
    assert (step._augment(media) == 0).any()


def test_tta_averages_six_views() -> None:
    counter = _CountingModel()
    config = tiny_step()
    config.use_tta = True
    step = config.make()
    # The backing attribute, because ``model`` is a read-only property: it is
    # declared that way so a subclass can narrow its return type, which a
    # settable attribute cannot express (``train_step.py:349``).
    step._model = counter
    _ = step.call_eval(media=tiny_batch()["media"])
    # Three crops, each with its mirror.
    assert counter.calls == 6


def test_tta_matches_the_plain_forward_for_a_shift_invariant_model() -> None:
    config = tiny_step()
    config.use_tta = True
    step = config.make()
    step._model = _ConstantModel()  # Read-only property; see the note above.
    media = tiny_batch()["media"]
    assert torch.allclose(step.call_eval(media=media), step.model(media))


def test_gradient_clipping_shrinks_the_gradients() -> None:
    """Clipping caps the gradient norm the optimizer sees.

    Asserted on the gradients, not on the weight delta: AdamW normalizes by
    the second moment, so its first update has magnitude ~lr whatever the
    gradient scale, and a weight-space check would pass with clipping removed.
    """

    def grad_norm(clip: float) -> float:
        torch.manual_seed(0)
        config = tiny_step()
        config.gradient_clip_norm = clip
        step = config.make()
        batch = tiny_batch()
        # Reproduce the step's forward and clip, stopping before the optimizer
        # consumes (and zeroes) the gradients.
        media = step._augment(batch["media"])
        loss = step._loss(step.model(media), batch["label"])
        loss.sum().backward()
        if math.isfinite(clip):
            _ = nn.utils.clip_grad_norm_(step.model.parameters(), clip)
        return float(
            torch.cat(
                [
                    p.grad.flatten()
                    for p in step.model.parameters()
                    if p.grad is not None
                ],
            ).norm(),
        )

    assert grad_norm(1e-4) == pytest.approx(1e-4, rel=1e-3)
    assert grad_norm(math.inf) > 1e-3


def test_train_loss_scores_without_backward() -> None:
    step = tiny_step().make()
    before = step.model.head.weight.detach().clone()
    out = step.train_loss(**tiny_batch())
    assert out["loss"].shape == (4,)
    assert torch.equal(before, step.model.head.weight)
    assert step.global_step == 0


def test_autocast_runs_the_forward_in_the_configured_dtype() -> None:
    config = tiny_step()
    config.dtype_autocast = torch.bfloat16
    step = config.make()
    # Evaluated, not trained: this CPU's oneDNN has no bf16 backward, and the
    # autocast context is what is under test, not the backward kernel.
    assert step.call_eval(**tiny_batch()).dtype == torch.bfloat16


def test_full_precision_is_the_default() -> None:
    step = tiny_step().make()
    assert step.call_eval(**tiny_batch()).dtype == torch.float32


def test_gradient_clipping_is_skipped_when_disabled() -> None:
    # The default is infinite, so the clip call must not run at all; a finite
    # default would silently rescale every baseline's updates.
    assert tiny_step().gradient_clip_norm == math.inf


def test_eval_loss_does_not_train() -> None:
    step = tiny_step().make()
    before = step.model.head.weight.detach().clone()
    _ = step.eval_loss(**tiny_batch())
    assert torch.equal(before, step.model.head.weight)
    assert step.global_step == 0


def test_state_dict_round_trip_restores_weights_and_progress() -> None:
    step = tiny_step().make()
    for _ in range(2):
        _ = step.train_step(**tiny_batch())
    saved = step.state_dict()

    restored = tiny_step().make()
    restored.load_state_dict(saved)
    assert restored.global_step == 2
    for a, b in zip(
        step.model.state_dict().values(),
        restored.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_resume_continues_the_same_trajectory() -> None:
    """A resumed run matches an uninterrupted one, given the loop's RNG state.

    Augmentation draws from the global generator, so reproducing a trajectory
    needs both the step's state dict and that generator. The step deliberately
    checkpoints only its own state: ``TrainLoop`` owns the RNG and restores it
    alongside, and duplicating it here would let the two disagree.
    """
    batches = [tiny_batch(seed=i) for i in range(4)]

    torch.manual_seed(0)
    reference = tiny_step().make()
    for batch in batches:
        _ = reference.train_step(**batch)

    torch.manual_seed(0)
    interrupted = tiny_step().make()
    for batch in batches[:2]:
        _ = interrupted.train_step(**batch)
    rng_state = torch.get_rng_state()

    resumed = tiny_step().make()
    resumed.load_state_dict(interrupted.state_dict())
    torch.set_rng_state(rng_state)
    for batch in batches[2:]:
        _ = resumed.train_step(**batch)

    assert torch.equal(reference.model.head.weight, resumed.model.head.weight)


def test_rejects_nonpositive_horizon() -> None:
    config = tiny_step()
    config.total_train_steps = 0
    with pytest.raises(ValueError, match="total_train_steps must be positive"):
        _ = config.make()


def test_rejects_out_of_range_warmup() -> None:
    config = tiny_step()
    config.warmup_fraction = 1.0
    with pytest.raises(ValueError, match="warmup_fraction"):
        _ = config.make()


def test_adamw_train_steps_bfb() -> None:
    _assert_train_bfb(golden_name="adamw_three_steps_min_cpu", optimizer=None)


def test_muon_train_steps_bfb() -> None:
    _assert_train_bfb(
        golden_name="muon_three_steps_min_cpu",
        optimizer=muon_optimizer(),
    )


def _assert_train_bfb(
    *,
    golden_name: str,
    optimizer: Makeable[Callable[..., torch.optim.Optimizer]] | None,
) -> None:
    """Pin three optimizer steps: loss, logits, and every resulting weight.

    Wrapping a train step rather than a forward is what makes the golden cover
    the recipe: the loss, the augmentation draws, the schedule, and the
    optimizer arithmetic all reach the post-run state the harness compares.
    """

    def build() -> nn.Module:
        config = tiny_step()
        if optimizer is not None:
            config.optimizer = optimizer
        return _TrainStepModule(config)

    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name=golden_name,
        build_module=build,
        build_input=tiny_batch,
        seed=42,
    )


class _TrainStepModule(nn.Module):
    """Adapts a train step to the module interface the golden harness drives.

    The harness randomizes parameters and compares ``state_dict`` before and
    after, so the step's model must BE this module's parameters -- hence the
    delegating ``state_dict``.
    """

    def __init__(self, config: Cifar10TrainStep.Config) -> None:
        super().__init__()
        self.step = config.make()
        self.inner = self.step.model

    @override
    def forward(self, media: Tensor, label: Tensor) -> Tensor:
        losses = [
            self.step.train_step(media=media, label=label)["loss"] for _ in range(3)
        ]
        return torch.stack(losses)


class _CountingModel(nn.Module):
    """Records how many forward passes an evaluation issues."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @override
    def forward(self, media: Tensor) -> Tensor:
        self.calls += 1
        return torch.zeros(len(media), 10)

    @override
    def eval(self) -> Any:
        return self


class _ConstantModel(nn.Module):
    """Returns the same logits for any input, so averaging views is a no-op."""

    @override
    def forward(self, media: Tensor) -> Tensor:
        return torch.ones(len(media), 10)

    @override
    def eval(self) -> Any:
        return self


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
