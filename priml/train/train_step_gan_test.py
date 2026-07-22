"""Tests for GANTrainStep."""

from __future__ import annotations

from typing import Any, cast, override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.loss.custom_types import LossOutput
from priml.loss.gan import AdversarialLoss
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep
from priml.train.train_step_gan import GANTrainStep


class SimpleGenerator(nn.Module):
    """Simple generator: maps noise to images."""

    class Config(Fig["SimpleGenerator"], make_with_kwargs=True):
        latent_dim: int = -1
        image_size: int = -1

    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, image_size * image_size * 3)
        self.image_size = image_size

    @override
    def forward(self, noise: Tensor, **_kwargs: Any) -> Tensor:
        """Generate images from noise."""
        x = self.fc(noise)
        return x.view(-1, 3, self.image_size, self.image_size)


class SimpleDiscriminator(nn.Module):
    """Simple discriminator: classifies real vs fake images."""

    class Config(Fig["SimpleDiscriminator"], make_with_kwargs=True):
        image_size: int = -1

    def __init__(self, image_size: int):
        super().__init__()
        self.fc = nn.Linear(image_size * image_size * 3, 1)
        self.image_size = image_size

    @override
    def forward(self, media: Tensor, **_kwargs: Any) -> Tensor:
        """Classify media as real (1) or fake (0)."""
        x = media.view(media.shape[0], -1)
        return self.fc(x)


def test_gan_train_step_basic():
    """Test GANTrainStep basic functionality."""
    torch.manual_seed(42)

    # Create GAN config
    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(adversarial_weight=1.0, content_weight=10.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.n_discriminator_steps = 1

    gan = config.make()

    # Create batch
    batch = {
        "noise": torch.randn(4, 10),
        "media": torch.randn(4, 3, 8, 8),
    }

    # Run train step
    result = gan.train_step(**batch)

    assert "loss" in result
    assert "model" in result
    assert result["loss"].shape == (4,)  # Per-sample loss
    assert result["model"].shape == (4, 3, 8, 8)  # Generated images


def test_gan_train_loss():
    """Test GANTrainStep.train_loss."""
    torch.manual_seed(42)

    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )

    gan = config.make()

    batch = {
        "noise": torch.randn(4, 10),
        "media": torch.randn(4, 3, 8, 8),
    }

    result = gan.train_loss(**batch)

    assert "loss" in result
    assert "model" in result
    assert result["loss"].shape == (4,)  # Per-sample loss
    assert result["loss"].mean().item() > 0


def test_gan_eval_loss():
    """Test GANTrainStep.eval_loss."""
    torch.manual_seed(42)

    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )

    gan = config.make()

    batch = {
        "noise": torch.randn(4, 10),
        "media": torch.randn(4, 3, 8, 8),
    }

    result = gan.eval_loss(**batch)

    assert "loss" in result
    assert "model" in result
    assert result["loss"].shape == (4,)  # Per-sample loss
    assert result["loss"].mean().item() > 0


def test_gan_checkpointing():
    """Test GANTrainStep state_dict and load_state_dict."""
    torch.manual_seed(42)

    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )

    gan1 = config.make()

    batch = {
        "noise": torch.randn(4, 10),
        "media": torch.randn(4, 3, 8, 8),
    }

    # Train for a few steps
    for _ in range(5):
        gan1.train_step(**batch)

    # Save state
    state = gan1.state_dict()

    # Create new GAN and load state
    gan2 = config.make()
    gan2.load_state_dict(state)

    # Check that outputs match
    result1 = gan1.eval_loss(**batch)
    result2 = gan2.eval_loss(**batch)

    torch.testing.assert_close(result1["loss"], result2["loss"])
    torch.testing.assert_close(result1["model"], result2["model"])


class _SyncCountingTensor(Tensor):
    """Tensor subclass that records each ``.item()`` call into a shared counter."""

    _item_calls: list[int] = []  # noqa: RUF012  -- shared test counter, set per-use

    @override
    def item(self) -> Any:
        type(self)._item_calls[0] += 1
        return super().item()


class _FakeDiscriminator:
    """Minimal discriminator whose train_step returns a sync-counting loss."""

    def __init__(self, item_calls: list[int]) -> None:
        self.device = torch.device("cpu")
        self._item_calls = item_calls
        self.model = SimpleDiscriminator(image_size=8)

    def _loss(self, batch_size: int) -> Tensor:
        base = torch.ones(2 * batch_size)
        counting = base.as_subclass(_SyncCountingTensor)
        type(counting)._item_calls = self._item_calls
        return counting

    def train_step(self, *, media: Tensor, label: Tensor) -> dict[str, Tensor]:
        del label
        return {"loss": self._loss(media.shape[0] // 2), "model": media}


def test_gan_train_step_disc_loss_no_per_step_item() -> None:
    """T-022: the GAN disc loop must not call ``.item()`` per discriminator step."""
    item_calls = [0]

    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.n_discriminator_steps = 5
    gan = config.make()
    # Swap in a fake discriminator whose per-step loss tensor counts .item().
    gan.discriminator = _FakeDiscriminator(item_calls)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment] -- deliberately swaps a test double (not a TrainStep) into a typed attribute

    batch = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}
    gan.train_step(**batch)

    # 5 disc steps: a correct impl syncs the accumulated tensor exactly once.
    assert item_calls[0] <= 1, (
        f".item() called {item_calls[0]} times on per-step disc losses (expected <=1)"
    )


def test_gan_preprocess_batch_forwards_to_generator() -> None:
    """T-024: GANTrainStep must expose preprocess_batch (TrainLoop calls it)."""
    torch.manual_seed(42)

    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )

    gan = config.make()

    raw = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}
    out = gan.preprocess_batch(raw)

    # Same keys, tensors on the generator's device.
    assert set(out.keys()) == set(raw.keys())
    assert out["media"].device == gan.generator.device


def _make_gan(n_discriminator_steps: int = 1) -> GANTrainStep:
    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.001),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.n_discriminator_steps = n_discriminator_steps
    return config.make()


def test_gan_generator_phase_does_not_pollute_discriminator_grads() -> None:
    """#337(1): the generator phase must not leave gradients on D parameters.

    The G phase runs ``D(G(z))`` so the generator can be scored, and its
    backward flows THROUGH the (frozen) discriminator activations. If D's
    parameters are not frozen, that backward accumulates ``.grad`` on D
    AFTER D's own ``train_step`` already zeroed them -- polluting the next
    discriminator update. D parameters must carry no gradient post-step.
    """
    torch.manual_seed(42)
    gan = _make_gan()
    batch = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}

    gan.train_step(**batch)

    for name, param in gan.discriminator.model.named_parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0, (
            f"discriminator parameter {name!r} carries G-phase gradient "
            "(grad pollution)"
        )
    # Discriminator must remain trainable after the frozen G phase restores it.
    assert all(p.requires_grad for p in gan.discriminator.model.parameters()), (
        "discriminator left frozen after generator phase"
    )


def test_gan_generator_phase_preserves_user_frozen_discriminator_params() -> None:
    """#337 regression: the G phase must not unfreeze a user-frozen D param.

    The generator phase freezes the discriminator to avoid grad pollution, but
    a blanket ``requires_grad_(True)`` restore clobbers any param the user
    intentionally froze. Snapshot per-param ``requires_grad`` and restore it
    exactly: a param frozen before ``train_step`` must remain frozen after.
    """
    torch.manual_seed(0)
    gan = _make_gan()

    # User intentionally freezes one discriminator parameter.
    frozen_name, frozen_param = next(iter(gan.discriminator.model.named_parameters()))
    frozen_param.requires_grad_(False)

    batch = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}
    gan.train_step(**batch)

    restored = dict(gan.discriminator.model.named_parameters())
    assert restored[frozen_name].requires_grad is False, (
        f"discriminator param {frozen_name!r} was unfrozen by the generator phase"
    )
    # Params the user left trainable must remain trainable.
    for name, param in restored.items():
        if name != frozen_name:
            assert param.requires_grad is True, f"param {name!r} wrongly frozen"


def test_gan_discriminator_receives_media_under_consistent_key() -> None:
    """#337(2): the discriminator model must receive media under ``media=``.

    The trainer's vocabulary is ``media`` everywhere (dataset, generator,
    eval). The discriminator sub-step used a one-off ``images=`` kwarg, an
    inconsistency that diverges from the rest of the contract and breaks any
    discriminator whose forward names its input ``media``. Record the kwargs
    the discriminator model actually receives and assert ``media`` is the key.
    """
    torch.manual_seed(0)
    gan = _make_gan()

    seen_keys: list[str] = []
    real_forward = gan.discriminator.model.forward

    def recording_forward(*args: Any, **kwargs: Any) -> Tensor:
        seen_keys.extend(kwargs.keys())
        return cast(Tensor, real_forward(*args, **kwargs))

    gan.discriminator.model.forward = recording_forward  # ty: ignore[invalid-assignment]  -- test patches a typed nn.Module attribute

    batch = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}
    gan.train_step(**batch)

    assert "media" in seen_keys, (
        f"discriminator model received keys {seen_keys}; expected 'media'"
    )
    assert "images" not in seen_keys, "trainer still uses the one-off 'images' key"


def test_gan_n_discriminator_steps_zero_rejected() -> None:
    """#337(3): n_discriminator_steps=0 would divide by zero; reject at init."""
    config = GANTrainStep.Config()
    config.generator = TrainStep.Config(
        model=SimpleGenerator.Config(latent_dim=10, image_size=8),
        loss=AdversarialLoss.Config(),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.discriminator = TrainStep.Config(
        model=SimpleDiscriminator.Config(image_size=8),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
    )
    config.n_discriminator_steps = 0
    with pytest.raises(ValueError, match="n_discriminator_steps"):
        config.make()


def test_gan_train_step_preserves_generator_aux_losses() -> None:
    """#337(4): generator auxiliary loss keys must survive into the result.

    ``AdversarialLoss`` returns only ``loss`` here, so attach a custom loss
    that emits an auxiliary key and confirm ``train_step`` does not drop it.
    """
    torch.manual_seed(0)
    gan = _make_gan()

    real_loss = gan.generator.loss

    def loss_with_aux(prediction: Any, **batch: Any) -> LossOutput:
        loss = cast("dict[str, Tensor]", {**real_loss(prediction, **batch)})["loss"]
        # ``adversarial`` is an auxiliary key the trainer must preserve;
        # LossOutput is total=False so extra keys are carried at runtime.
        return cast("LossOutput", {"loss": loss, "adversarial": loss.detach() * 2.0})

    gan.generator.loss = loss_with_aux

    batch = {"noise": torch.randn(4, 10), "media": torch.randn(4, 3, 8, 8)}
    result = gan.train_step(**batch)

    assert "adversarial" in result, "generator auxiliary loss key dropped"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
