"""One optimizer update through priml's training infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.speedrundit.experiments import exp_smoke
from priml.baselines.speedrundit.model_test import tiny_model
from priml.baselines.speedrundit.train_step import SpeedrunTrainStep
from priml.model.vision_ae.latent_norm import ScaleLatents
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent


class FakeTeacher(nn.Module):
    """Small deterministic stand-in for the downloaded DINOv2 teacher."""

    class Config(Fig["FakeTeacher"]):
        channels: int = 8

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels = config.channels

    @override
    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return the three reference alignment depths and CLS feature."""
        tokens = 4 if image.shape[-1] == 4 else image.shape[-1] // 16
        channels = self.channels if tokens == 4 else 768
        feature = torch.ones(
            image.shape[0],
            tokens * tokens + 1,
            channels,
            device=image.device,
        )
        return feature, feature, feature


def _small_step_config() -> SpeedrunTrainStep.Config:
    config = SpeedrunTrainStep.Config()
    config.model = tiny_model().config
    config.teacher = FakeTeacher.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.ema = NoEMA.Config()
    config.dtype_autocast = None
    config.compile = None
    config.train_budget_steps = 2
    config.latent_norm = ScaleLatents.Config(scale=0.3099)
    return config


def _small_batch(step: SpeedrunTrainStep) -> dict[str, object]:
    return step.preprocess_batch(
        {
            # SpeedrunDiT.forward enforces cfg.in_channels and cfg.input_size.
            "image": torch.zeros(5, 3, 4, 4, dtype=torch.uint8),
            # SpeedrunDiT.forward enforces cfg.in_channels and cfg.input_size.
            "latent": torch.randn(5, 2, 4, 4),
            "label": torch.tensor([0, 1, 2, 3, 0]),
        },
    )


def test_train_step_updates_model_and_advances_budget() -> None:
    step = _small_step_config().make()
    result = step.train_step(**_small_batch(step))
    assert step.global_step == 1
    assert result["loss"].shape == (5,)
    assert torch.isfinite(result["loss"]).all()


def test_accumulation_averages_micro_batch_gradients_before_stepping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_step_config()
    config.accumulate_grad_batches = 2
    step = config.make()
    batch = _small_batch(step)
    torch.manual_seed(0)
    step.train_step(**batch)
    assert step.global_step == 0
    first = {
        name: parameter.grad.clone()
        for name, parameter in step.model.named_parameters()
        if parameter.grad is not None
    }
    seen: dict[str, Tensor] = {}

    def record_grads() -> None:
        for name, parameter in step.model.named_parameters():
            if parameter.grad is not None:
                seen[name] = parameter.grad.clone()

    # The optimizer step is replaced so the averaged gradients stay readable.
    monkeypatch.setattr(step, "step", record_grads)
    torch.manual_seed(0)
    step.train_step(**batch)
    # Identical seeded micro-batches give identical gradients, so their summed
    # accumulation divided by two recovers the first micro-batch's gradient.
    assert first.keys() == seen.keys()
    assert all(torch.allclose(seen[name], grad) for name, grad in first.items())
    assert (step.accumulation_steps, step.accumulated_samples) == (0, 0)


def test_train_and_eval_losses_report_every_term_without_updating() -> None:
    step = _small_step_config().make()
    batch = _small_batch(step)
    before = [p.detach().clone() for p in step.model.parameters()]
    for result in (step.train_loss(**batch), step.eval_loss(**batch)):
        assert result["loss"].shape == (5,)
        assert set(result.get("metrics", {})) == {
            "velocity_loss",
            "cls_loss",
            "projection_loss",
            "cfm_loss",
        }
    assert step.global_step == 0
    assert all(
        torch.equal(a, b) for a, b in zip(before, step.model.parameters(), strict=True)
    )


class _SmokeSteps(nn.Module):
    """Expose three updates with the smoke recipe at small test dimensions."""

    def __init__(self) -> None:
        super().__init__()
        smoke = exp_smoke()
        config = smoke.step
        # What SpeedrunTrainLoop.finalize fills; the step is built without the loop.
        config.latent_norm = smoke.dataset.source.autoencoder.latent_norm
        config.model.input_size = 4
        config.model.in_channels = 2
        # Three blocks provide a dense encoder, routed sparse middle, and dense
        # decoder. RoPE requires two channels per axis for the two-head layout.
        config.model.hidden_size = 8
        config.model.timestep_frequencies = 4
        config.model.num_heads = 2
        config.model.cls_channels = 4
        config.model.projector_hidden = 4
        config.model.num_classes = 4
        config.model.depth = 3
        config.model.encoder_blocks = 1
        config.model.decoder_blocks = 1
        config.model.projection_depths = (1, 2, 3)
        # Flatten the feed-forward expansion ramp to 1x: the per-block MLP is
        # otherwise 2x-6x the width. The arithmetic (attention, alignment,
        # adaLN) is unchanged; only the hidden width shrinks.
        config.model.mlp_ratio_min = 1.0
        config.model.mlp_ratio_max = 1.0
        config.teacher = FakeTeacher.Config(channels=4)
        config.parallelism = NoParallel.Config(device="cpu")
        config.dtype_autocast = None
        self.step = config.make()
        self.model = self.step.model

    @override
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Train for three updates on fixed inputs; retain each scalar loss."""
        prepared = self.step.preprocess_batch(dict[str, object](batch))
        losses: list[Tensor] = []
        for _ in range(3):
            result = self.step.train_step(**prepared)
            losses.append(result["loss"].mean())
        return torch.stack(losses)


@pytest.mark.compute_training
def test_exp_smoke_three_steps_bfb() -> None:
    """Freeze initialization, three forward/backward passes, and weight updates."""

    def build_input() -> dict[str, Tensor]:
        return {
            # SpeedrunDiT.forward enforces cfg.in_channels and cfg.input_size.
            "image": torch.zeros(5, 3, 4, 4, dtype=torch.uint8),
            # SpeedrunDiT.forward enforces cfg.in_channels and cfg.input_size.
            "latent": torch.arange(5 * 2 * 4 * 4, dtype=torch.float32)
            .reshape(5, 2, 4, 4)
            .remainder(97)
            .div(97),
            "label": torch.tensor([0, 1, 2, 3, 0]),
        }

    def run(module: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        assert isinstance(module, _SmokeSteps)
        return module(batch)

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="exp_smoke",
        build_module=_SmokeSteps,
        build_input=build_input,
        seed=0,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
