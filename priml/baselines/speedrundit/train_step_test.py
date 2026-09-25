"""One optimizer update through priml's training infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import override

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.speedrundit.experiments import exp_smoke
from priml.baselines.speedrundit.model_test import tiny_model
from priml.baselines.speedrundit.train_step import SpeedrunTrainStep
from priml.testing.bfb import assert_bfb_against_golden
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


class FakeTeacher(nn.Module):
    """Small deterministic stand-in for the downloaded DINOv2 teacher."""

    class Config(Fig["FakeTeacher"]):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()
        del config

    @override
    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return the three reference alignment depths and CLS feature."""
        tokens = 4 if image.shape[-1] == 4 else image.shape[-1] // 16
        channels = 8 if tokens == 4 else 768
        feature = torch.ones(
            image.shape[0], tokens * tokens + 1, channels, device=image.device
        )
        return feature, feature, feature


def test_train_step_updates_model_and_advances_budget() -> None:
    config = SpeedrunTrainStep.Config()
    config.model = tiny_model().config
    config.teacher = FakeTeacher.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.ema = NoEMA.Config()
    config.dtype_autocast = None
    config.compile = None
    config.train_budget_steps = 2
    step = config.make()
    batch = step.preprocess_batch(
        {
            "image": torch.zeros(2, 3, 4, 4, dtype=torch.uint8),
            "latent": torch.randn(2, 2, 4, 4),
            "label": torch.tensor([1, 2]),
        }
    )
    result = step.train_step(**batch)
    assert step.global_step == 1
    assert result["loss"].shape == (2,)
    assert torch.isfinite(result["loss"]).all()


class _SmokeSteps(nn.Module):
    """Expose five updates with the smoke recipe at small test dimensions."""

    def __init__(self) -> None:
        super().__init__()
        config = exp_smoke().step
        config.model.input_size = 4
        config.model.in_channels = 2
        config.model.hidden_size = 16
        config.model.num_heads = 4
        config.model.cls_channels = 8
        config.model.projector_hidden = 16
        config.model.num_classes = 4
        config.teacher = FakeTeacher.Config()
        config.parallelism = NoParallel.Config(device="cpu")
        config.dtype_autocast = None
        self.step = config.make()
        self.model = self.step.model

    @override
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Train for five updates on fixed inputs; retain each scalar loss."""
        prepared = self.step.preprocess_batch(dict[str, object](batch))
        losses: list[Tensor] = []
        for _ in range(5):
            result = self.step.train_step(**prepared)
            losses.append(result["loss"].mean())
        return torch.stack(losses)


@pytest.mark.compute_training
def test_exp_smoke_five_steps_bfb() -> None:
    """Freeze initialization, five forward/backward passes, and weight updates."""

    def build_input() -> dict[str, Tensor]:
        return {
            "image": torch.zeros(2, 3, 4, 4, dtype=torch.uint8),
            "latent": torch.arange(2 * 2 * 4 * 4, dtype=torch.float32)
            .reshape(2, 2, 4, 4)
            .remainder(97)
            .div(97),
            "label": torch.tensor([1, 2]),
        }

    def run(module: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        assert isinstance(module, _SmokeSteps)
        return module(batch)

    assert_bfb_against_golden(
        golden_dir=Path(__file__).parent / "testdata",
        golden_name="exp_smoke",
        build_module=_SmokeSteps,
        build_input=build_input,
        seed=0,
        run=run,
    )
