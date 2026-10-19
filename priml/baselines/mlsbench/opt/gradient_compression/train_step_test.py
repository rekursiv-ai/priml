"""Tests for the step that compresses gradients, and its warmup cosine schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import itertools
import math

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.train_step_test import (
    batch,
    sized,
    tiny_config,
)
from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    ErrorFeedback,
    top_k,
)
from priml.baselines.mlsbench.opt.gradient_compression.train_step import (
    CompressedTrainStep,
    warmup_cosine,
)


if TYPE_CHECKING:
    from torch import Tensor


def config() -> CompressedTrainStep.Config:
    """Return the vision test step's config, on the compressing step."""
    return sized(CompressedTrainStep.Config())


class Halving:
    """A compressor that halves every gradient and notes the names it saw."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def __call__(self, grad: Tensor, /, *, name: str) -> Tensor:
        """Return half of ``grad``."""
        self.names.append(name)
        return grad / 2


def test_the_default_compressor_leaves_the_update_unchanged() -> None:
    plain = tiny_config().make()
    compressed = config().make()
    compressed.model.load_state_dict(plain.model.state_dict())
    _ = plain.train_step(**batch()), compressed.train_step(**batch())
    for a, b in zip(
        plain.model.parameters(),
        compressed.model.parameters(),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_every_gradient_is_compressed_before_the_step() -> None:
    plain, halved = config().make(), config().make()
    halved.model.load_state_dict(plain.model.state_dict())
    halving = halved.compressor = Halving()
    _ = plain.train_step(**batch()), halved.train_step(**batch())
    assert halving.names == [name for name, _ in halved.model.named_parameters()]
    for p, h in zip(plain.model.parameters(), halved.model.parameters(), strict=True):
        assert p.grad is not None
        assert h.grad is not None
        assert torch.equal(h.grad, p.grad / 2)


def test_the_residuals_survive_a_checkpoint() -> None:
    cfg = config()
    feedback = cfg.compressor = ErrorFeedback.Config()
    feedback.compressor = PartialConfig(top_k, ratio=0.01)
    step = cfg.make()
    _ = step.train_step(**batch())
    state = step.state_dict()
    restored = cfg.make()
    restored.load_state_dict(state)
    assert isinstance(step.compressor, ErrorFeedback)
    assert isinstance(restored.compressor, ErrorFeedback)
    assert step.compressor.residuals.keys() == restored.compressor.residuals.keys()
    for name, residual in step.compressor.residuals.items():
        assert torch.equal(restored.compressor.residuals[name], residual)
    fresh = cfg.make()
    fresh.load_state_dict(state, load_optimizer=False)
    assert isinstance(fresh.compressor, ErrorFeedback)
    assert not fresh.compressor.residuals


def test_a_stateless_compressor_checkpoints_nothing() -> None:
    step = config().make()
    assert step.state_dict()["compressor"] == {}
    step.load_state_dict(step.state_dict())


def test_settings_the_bare_update_bypasses_are_refused() -> None:
    cfg = config()
    cfg.gradient_clip_norm = 1.0
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        cfg.make()


def test_warmup_rises_linearly_without_reaching_the_peak() -> None:
    rates = [
        warmup_cosine(
            e,
            base_lr=0.6,
            total_epochs=10,
            warmup_epochs=2,
            min_lr_ratio=0.1,
        )
        for e in range(10)
    ]
    assert rates[:2] == [0.6 * 1 / 3, 0.6 * 2 / 3]
    floor = 0.6 * 0.1
    assert rates[2] == floor + 0.5 * (0.6 - floor) * 2
    assert rates[-1] == floor + 0.5 * (0.6 - floor) * (1 + math.cos(math.pi * 7 / 8))
    assert all(a > b for a, b in itertools.pairwise(rates[2:]))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
