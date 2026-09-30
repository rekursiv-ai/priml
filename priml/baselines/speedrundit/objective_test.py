"""Probability paths, REG alignment, and objective input validation."""

from __future__ import annotations

from typing import Literal

import math

from torch import Tensor
from torch.nn import functional

import pytest
import torch

from priml.baselines.speedrundit.model import ModelOutput, Projection
from priml.baselines.speedrundit.objective import (
    SpeedrunObjective,
    interpolant,
    projection_loss,
)
from priml.math.diffusion.time_shift import time_shift


def test_cosine_path_is_a_quarter_turn_with_its_derivatives() -> None:
    t = torch.tensor([0.0, 0.25, 1.0])
    alpha, sigma, d_alpha, d_sigma = interpolant(t, "cosine")
    angle = math.pi / 2 * t
    assert torch.allclose(alpha, angle.cos())
    assert torch.allclose(sigma, angle.sin())
    assert torch.allclose(d_alpha, -math.pi / 2 * angle.sin())
    assert torch.allclose(d_sigma, math.pi / 2 * angle.cos())
    assert torch.allclose(alpha.square() + sigma.square(), torch.ones(3))


def test_projection_loss_gathers_the_tokens_sparse_routing_kept() -> None:
    teacher = torch.randn(2, 5, 3)
    kept = torch.tensor([[0, 3, 1, 4], [4, 1, 2, 0]])
    student = torch.randn(2, 4, 3)
    loss = projection_loss((Projection(student, kept),), (teacher,))
    selected = torch.stack([teacher[0, [0, 3, 1, 4]], teacher[1, [4, 1, 2, 0]]])
    expected = -functional.cosine_similarity(student, selected, dim=-1).mean(dim=1)
    assert torch.allclose(loss, expected)


def test_projection_loss_rejects_mismatched_depth_counts() -> None:
    student = Projection(torch.randn(2, 5, 3), None)
    with pytest.raises(ValueError, match="projection depths must match"):
        projection_loss((student,), (torch.randn(2, 5, 3), torch.randn(2, 5, 3)))


def test_projection_loss_rejects_mismatched_token_shapes() -> None:
    student = Projection(torch.randn(2, 4, 3), None)
    with pytest.raises(ValueError, match="teacher tokens do not match"):
        projection_loss((student,), (torch.randn(2, 5, 3),))


@pytest.mark.parametrize("path", ["linear", "cosine"])
def test_lognormal_time_maps_a_lognormal_sigma_per_path(
    path: Literal["linear", "cosine"],
) -> None:
    latents = torch.zeros(3, 2, 4, 5)
    torch.manual_seed(0)
    sigma = torch.randn(3).exp()
    expected = sigma / (1 + sigma) if path == "linear" else 2 * sigma.atan() / math.pi
    objective = SpeedrunObjective(path=path, weighting="lognormal", shift_time=False)
    torch.manual_seed(0)
    assert torch.allclose(objective.sample_time(latents), expected)


def test_shifted_time_follows_the_latent_dimension() -> None:
    latents = torch.zeros(3, 2, 4, 5)
    torch.manual_seed(0)
    uniform = torch.rand(3)
    torch.manual_seed(0)
    shifted = SpeedrunObjective(shift_base=8).sample_time(latents)
    assert torch.allclose(shifted, time_shift(uniform, 2 * 4 * 5, 8))


def _unused_model(*args: Tensor) -> ModelOutput:
    raise AssertionError(f"validation must precede the model; got {len(args)} args")


def test_objective_rejects_mismatched_label_count() -> None:
    with pytest.raises(ValueError, match="different batch sizes"):
        SpeedrunObjective()(
            _unused_model,
            torch.zeros(2, 3, 4, 5),
            torch.zeros(3, dtype=torch.int64),
            (torch.zeros(2, 6, 7),),
        )


def test_objective_requires_a_teacher_feature_map() -> None:
    with pytest.raises(ValueError, match="at least one teacher feature map"):
        SpeedrunObjective()(
            _unused_model,
            torch.zeros(2, 3, 4, 5),
            torch.zeros(2, dtype=torch.int64),
            (),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
