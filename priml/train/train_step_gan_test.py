"""Tests for GANTrainStep."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, Final, overload, override

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.loss.gan import AdversarialLoss
from priml.timer import CheckpointableStepTimer
from priml.train.parallelism import NoParallel
from priml.train.profiler import PhaseTimer
from priml.train.train_step import TrainStep
from priml.train.train_step_gan import GANTrainStep, NoiseToMedia


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.loss.custom_types import LossOutput


LATENT: Final = 5
HEIGHT: Final = 3
WIDTH: Final = 4
BATCH: Final = 2
DEFAULT_MEDIA_SHAPE: Final = NoiseToMedia.Config().media_shape


class StrictGenerator(nn.Module):
    """Maps noise to images; its forward takes ``noise`` and nothing else."""

    class Config(Fig["StrictGenerator"], make_with_kwargs=True):
        latent_dim: int = LATENT

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 3 * HEIGHT * WIDTH)

    @override
    def forward(self, noise: Tensor) -> Tensor:
        return self.fc(noise).view(-1, 3, HEIGHT, WIDTH)


class StrictDiscriminator(nn.Module):
    """Scores images; ``media`` is keyword-only, and BatchNorm tracks statistics."""

    class Config(Fig["StrictDiscriminator"], make_with_kwargs=True):
        patch: bool = False

    def __init__(self, patch: bool) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 1, kernel_size=2) if patch else None
        self.fc = nn.Linear(3 * HEIGHT * WIDTH, 1)

    @override
    def forward(self, *, media: Tensor) -> Tensor:
        x = self.norm(media)
        if self.conv is not None:
            return self.conv(x)
        return self.fc(x.flatten(1))


class ModeDependentGenerator(nn.Module):
    """Adds a mode-dependent offset, so train and eval outputs differ."""

    class Config(Fig["ModeDependentGenerator"]): ...

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.inner = StrictGenerator(LATENT)

    @override
    def forward(self, noise: Tensor) -> Tensor:
        return self.inner(noise) + (1.0 if self.training else -1.0)


def _step(model: Makeable[nn.Module]) -> TrainStep.Config:
    return TrainStep.Config(
        model=model,
        optimizer=PartialConfig(torch.optim.Adam, lr=1e-2),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )


def _gan_config(
    n_discriminator_steps: int = 1,
    *,
    generator: Makeable[nn.Module] | None = None,
    patch: bool = False,
) -> tuple[GANTrainStep.Config, TrainStep.Config, TrainStep.Config]:
    """Build a small GAN config, returning its two sub-step configs for edits."""
    generator_step = _step(generator or StrictGenerator.Config())
    generator_step.loss = AdversarialLoss.Config()
    discriminator_step = _step(StrictDiscriminator.Config(patch=patch))
    config = GANTrainStep.Config(
        generator=generator_step,
        discriminator=discriminator_step,
        n_discriminator_steps=n_discriminator_steps,
    )
    return config, generator_step, discriminator_step


def _make_gan(
    n_discriminator_steps: int = 1,
    *,
    generator: Makeable[nn.Module] | None = None,
    patch: bool = False,
) -> GANTrainStep:
    config, _, _ = _gan_config(n_discriminator_steps, generator=generator, patch=patch)
    return config.make()


def _record_output_dtype(
    seen: list[torch.dtype],
    module: nn.Module,
    args: tuple[object, ...],
    output: Tensor,
) -> None:
    del module, args
    seen.append(output.dtype)


def _record_media(
    seen: list[Tensor],
    module: nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    del module, args
    media = kwargs["media"]
    assert isinstance(media, Tensor)
    seen.append(media.detach())


def _snapshot(snapshots: list[Tensor], tensor: Tensor, *args: object) -> None:
    del args
    snapshots.append(tensor.clone())


def _count_call(calls: list[None], *args: object) -> None:
    del args
    calls.append(None)


def _batch() -> dict[str, Tensor]:
    return {
        "noise": torch.randn(BATCH, LATENT),
        "media": torch.randn(BATCH, 3, HEIGHT, WIDTH),
    }


def test_train_step_shapes_and_keys() -> None:
    torch.manual_seed(0)
    result = _make_gan().train_step(**_batch())
    assert result["loss"].shape == (BATCH,)
    assert result["model"].shape == (BATCH, 3, HEIGHT, WIDTH)
    assert set(result.get("metrics", {})) == {"d_loss"}


def test_strict_generator_forward_receives_only_its_inputs() -> None:
    """Loss-only keys (fake_logits, fake_media, real_media) never reach forward."""
    torch.manual_seed(0)
    _make_gan().train_step(**_batch())  # StrictGenerator rejects unknown kwargs.


@pytest.mark.parametrize("n_discriminator_steps", [1, 3])
def test_generator_runs_once_per_train_step(n_discriminator_steps: int) -> None:
    torch.manual_seed(0)
    gan = _make_gan(n_discriminator_steps)
    calls: list[None] = []
    gan.generator.model.register_forward_hook(partial(_count_call, calls))
    gan.train_step(**_batch())
    assert len(calls) == 1


def test_discriminator_media_is_keyword_everywhere() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    batch = _batch()
    gan.train_step(**batch)
    gan.train_loss(**batch)
    gan.eval_loss(**batch)  # StrictDiscriminator's media is keyword-only.


def test_generator_phase_freezes_discriminator_batchnorm() -> None:
    """The G phase scores fakes in eval mode, so D's BN statistics hold still."""
    torch.manual_seed(0)
    gan = _make_gan()
    norm = gan.discriminator.model.get_submodule("norm")
    assert isinstance(norm, nn.BatchNorm2d)
    running_mean = norm.running_mean
    assert running_mean is not None
    # Snapshot after every D forward; the last is the G phase's scoring pass.
    snapshots: list[Tensor] = []
    norm.register_forward_hook(partial(_snapshot, snapshots, running_mean))
    generator_before = [p.detach().clone() for p in gan.generator.model.parameters()]

    gan.train_step(**_batch())

    after_d_phase, after_g_phase = snapshots
    assert not torch.equal(after_d_phase, torch.zeros_like(after_d_phase))
    torch.testing.assert_close(after_g_phase, after_d_phase, rtol=0, atol=0)
    assert norm.training
    assert all(
        not torch.equal(before, after)
        for before, after in zip(
            generator_before,
            gan.generator.model.parameters(),
            strict=True,
        )
    ), "generator did not learn through the frozen discriminator"


def test_generator_phase_leaves_no_gradient_on_discriminator() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    gan.train_step(**_batch())
    for name, param in gan.discriminator.model.named_parameters():
        assert param.grad is None, f"{name} carries a generator-phase gradient"
        assert param.requires_grad, f"{name} left frozen"


def test_generator_phase_keeps_a_user_frozen_discriminator_param_frozen() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    frozen = gan.discriminator.model.get_parameter("fc.bias")
    frozen.requires_grad_(False)
    gan.train_step(**_batch())
    assert frozen.requires_grad is False
    assert gan.discriminator.model.get_parameter("fc.weight").requires_grad


def test_forwards_run_under_the_sub_steps_autocast() -> None:
    torch.manual_seed(0)
    config, generator_step, discriminator_step = _gan_config()
    generator_step.dtype_autocast = torch.bfloat16
    discriminator_step.dtype_autocast = torch.bfloat16
    gan = config.make()
    seen: list[torch.dtype] = []
    gan.discriminator.model.get_submodule("fc").register_forward_hook(
        partial(_record_output_dtype, seen),
    )
    gan.train_step(**_batch())
    assert seen == [torch.bfloat16, torch.bfloat16]


def test_train_loss_scores_the_train_mode_generator_output() -> None:
    torch.manual_seed(0)
    gan = _make_gan(generator=ModeDependentGenerator.Config())
    batch = _batch()
    model = gan.generator.model
    assert isinstance(model, ModeDependentGenerator)
    model.train()
    expected = model(noise=batch["noise"]).detach()
    seen: list[Tensor] = []
    gan.discriminator.model.register_forward_pre_hook(
        partial(_record_media, seen),
        with_kwargs=True,
    )
    result = gan.train_loss(**batch)
    torch.testing.assert_close(result["model"], expected)
    torch.testing.assert_close(seen[0][BATCH:], expected)  # D phase's fake half.
    torch.testing.assert_close(seen[1], expected)  # G phase's scored fake.


def _patch_adversarial_loss(
    prediction: object,
    *,
    fake_logits: Tensor,
    **batch: object,
) -> LossOutput:
    """Non-saturating generator loss averaged over every patch of each sample."""
    del prediction, batch
    loss = nn.functional.binary_cross_entropy_with_logits(
        fake_logits,
        torch.ones_like(fake_logits),
        reduction="none",
    )
    return {"loss": loss.flatten(1).mean(dim=1)}


def test_patch_discriminator_trains() -> None:
    """Targets follow D's output shape, so a [B, 1, H', W'] patch D trains."""
    torch.manual_seed(0)
    gan = _make_gan(patch=True)
    gan.generator.loss = _patch_adversarial_loss
    labels: list[Tensor] = []
    d_loss = gan.discriminator.loss

    def record(prediction: object, **batch: object) -> LossOutput:
        label = batch["label"]
        assert isinstance(label, Tensor)
        labels.append(label)
        return d_loss(prediction, **batch)

    gan.discriminator.loss = record
    result = gan.train_step(**_batch())
    assert result["loss"].shape == (BATCH,)
    expected = torch.cat(
        [
            torch.ones(BATCH, 1, HEIGHT - 1, WIDTH - 1),
            torch.zeros(BATCH, 1, HEIGHT - 1, WIDTH - 1),
        ],
    )
    torch.testing.assert_close(labels[0], expected)


def _accumulating_gan() -> GANTrainStep:
    config, generator_step, discriminator_step = _gan_config()
    generator_step.accumulate_grad_batches = 2
    discriminator_step.accumulate_grad_batches = 2
    return config.make()


def test_global_step_counts_generator_updates() -> None:
    torch.manual_seed(0)
    gan = _accumulating_gan()
    gan.train_step(**_batch())
    assert gan.global_step == 0
    gan.train_step(**_batch())
    assert gan.global_step == 1
    gan.state_dict()  # Both sub-steps are on an update boundary.


def test_checkpoint_refuses_pending_accumulation() -> None:
    torch.manual_seed(0)
    gan = _accumulating_gan()
    gan.train_step(**_batch())
    with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
        gan.state_dict()


def test_misaligned_accumulation_is_rejected_at_construction() -> None:
    config, _, discriminator_step = _gan_config(2)
    discriminator_step.accumulate_grad_batches = 3
    with pytest.raises(ValueError, match="must divide"):
        config.make()


def test_discriminator_metrics_are_carried() -> None:
    torch.manual_seed(0)
    config, _, discriminator_step = _gan_config()
    discriminator_step.skip_step_on_nonfinite_grad = True
    gan = config.make()
    metrics = gan.train_step(**_batch()).get("metrics", {})
    assert metrics["d_skipped_steps"] == 0
    assert isinstance(metrics["d_loss"], Tensor)


def test_loss_is_the_generator_objective_alone() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    batch = _batch()
    result = gan.eval_loss(**batch)
    fake = gan.generator.call_eval(noise=batch["noise"])
    assert isinstance(fake, Tensor)
    expected = gan.generator.loss(
        fake,
        fake_logits=gan.discriminator.call_eval(media=fake),
        fake_media=fake,
        real_media=batch["media"],
    )["loss"]
    torch.testing.assert_close(result["loss"], expected)


def test_n_discriminator_steps_zero_rejected() -> None:
    with pytest.raises(ValueError, match="n_discriminator_steps"):
        _make_gan(0)


def _default_gan() -> GANTrainStep:
    """Build the default recipe eager: compiling costs ~17s and no numerics."""
    config = GANTrainStep.Config()
    for sub_step in (config.generator, config.discriminator):
        assert isinstance(sub_step, TrainStep.Config)
        sub_step.compile = None
    return config.make()


def test_default_config_trains() -> None:
    torch.manual_seed(0)
    gan = _default_gan()
    result = gan.train_step(
        media=torch.randn(BATCH, *DEFAULT_MEDIA_SHAPE),
        noise=torch.randn(BATCH, LATENT),
    )
    assert result["model"].shape == (BATCH, *DEFAULT_MEDIA_SHAPE)
    assert gan.global_step == 1


def test_default_generator_shape_mismatch_raises() -> None:
    gan = _default_gan()
    with pytest.raises(ValueError, match="media_shape"):
        gan.train_step(
            media=torch.randn(BATCH, 3, HEIGHT, WIDTH),
            noise=torch.randn(BATCH, LATENT),
        )


def test_sub_steps_on_different_devices_are_rejected() -> None:
    config, _, discriminator_step = _gan_config()
    discriminator_step.parallelism = NoParallel.Config(device="meta")
    with pytest.raises(ValueError, match="share one device"):
        config.make()


def test_loop_wiring_reaches_both_sub_steps() -> None:
    gan = _make_gan()
    timer = PhaseTimer.Config().make()
    gan.timer = timer
    assert gan.generator.timer is timer
    assert gan.discriminator.timer is timer
    assert gan.timer is timer
    epoch = CheckpointableStepTimer()
    gan.bind_epoch_timer(epoch)
    assert gan.generator.timer_epoch is epoch
    assert gan.discriminator.timer_epoch is epoch


def test_preprocess_batch_moves_tensors_to_the_shared_device() -> None:
    gan = _make_gan()
    out = gan.preprocess_batch({"media": torch.randn(BATCH, 3, HEIGHT, WIDTH), "k": 1})
    media = out["media"]
    assert isinstance(media, Tensor)
    assert media.device == gan.generator.device
    assert out["k"] == 1


def test_call_eval_returns_the_generators_eval_output() -> None:
    gan = _make_gan(generator=ModeDependentGenerator.Config())
    noise = torch.randn(BATCH, LATENT)
    model = gan.generator.model
    assert isinstance(model, ModeDependentGenerator)
    model.eval()
    expected = model(noise)
    model.train()
    torch.testing.assert_close(gan.call_eval(noise=noise), expected)


def test_epoch_end_discards_both_partial_accumulations() -> None:
    torch.manual_seed(0)
    gan = _accumulating_gan()
    gan.train_step(**_batch())
    gan.on_epoch_end()
    assert gan.generator.accumulation_steps == 0
    assert gan.discriminator.accumulation_steps == 0
    gan.state_dict()


class _ClosureSGD(torch.optim.SGD):
    """SGD that asks for, runs, and records the loss-recomputing closure."""

    requires_closure = True
    closure_losses: ClassVar[list[float]] = []

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        assert closure is not None
        loss = closure()
        assert isinstance(loss, Tensor)
        assert loss.requires_grad
        type(self).closure_losses.append(float(loss.detach()))
        super().step()
        return None


def test_closures_recompute_both_objectives() -> None:
    """A closure-based optimizer re-runs each sub-step's own GAN objective."""
    torch.manual_seed(0)
    config, generator_step, discriminator_step = _gan_config()
    generator_step.optimizer = PartialConfig(_ClosureSGD, lr=1e-2)
    discriminator_step.optimizer = PartialConfig(_ClosureSGD, lr=1e-2)
    _ClosureSGD.closure_losses = []
    config.make().train_step(**_batch())
    assert len(_ClosureSGD.closure_losses) == 2
    assert all(math.isfinite(loss) for loss in _ClosureSGD.closure_losses)


def test_a_non_tensor_generator_output_is_refused() -> None:
    gan = _make_gan()
    with pytest.raises(TypeError, match="batch 'media'"):
        gan.train_step(noise=torch.randn(BATCH, LATENT))


def test_checkpoint_round_trip() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    batch = _batch()
    gan.train_step(**batch)
    restored = _make_gan()
    restored.load_state_dict(gan.state_dict())
    assert restored.global_step == gan.global_step == 1
    torch.testing.assert_close(
        restored.eval_loss(**batch)["loss"],
        gan.eval_loss(**batch)["loss"],
    )


def test_load_state_dict_forwards_strict_and_load_optimizer() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    gan.train_step(**_batch())
    state = gan.state_dict()
    del state["discriminator"]["model"]["fc.bias"]
    restored = _make_gan()
    optimizer_before = restored.generator.optimizer.state_dict()
    restored.load_state_dict(state, strict=False, load_optimizer=False)
    assert restored.generator.optimizer.state_dict() == optimizer_before
    with pytest.raises(RuntimeError, match=r"fc\.bias"):
        _make_gan().load_state_dict(state)


def test_generator_aux_loss_keys_survive() -> None:
    torch.manual_seed(0)
    gan = _make_gan()
    adversarial = gan.generator.loss

    def with_aux(prediction: object, **batch: object) -> LossOutput:
        loss = adversarial(prediction, **batch)["loss"]
        return {"loss": loss, "adversarial": loss.detach() * 2.0}

    gan.generator.loss = with_aux
    assert "adversarial" in gan.train_step(**_batch())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
