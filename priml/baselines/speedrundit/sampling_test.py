"""Exact score conversion and latent sampling behavior."""

from __future__ import annotations

from typing import override

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.speedrundit.model import ModelOutput
from priml.baselines.speedrundit.sampling import (
    sample_latents,
    score_from_velocity,
)


class _ZeroModel(nn.Module):
    """Return zero velocities while recording the sampling call contract."""

    class _Config:
        num_classes = 3

    config = _Config()

    def __init__(self) -> None:
        super().__init__()
        self.seen_labels: list[Tensor] = []
        self.seen_options: list[tuple[bool, bool]] = []
        self.seen_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    @override
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        cls_token: Tensor,
        *,
        drop_sparse_path: bool,
        route_tokens: bool,
    ) -> ModelOutput:
        self.seen_labels.append(y.clone())
        self.seen_options.append((drop_sparse_path, route_tokens))
        self.seen_dtypes.append((x.dtype, t.dtype, cls_token.dtype))
        return ModelOutput(torch.zeros_like(x), torch.zeros_like(cls_token), ())


def test_score_from_linear_velocity_is_exact() -> None:
    velocity = torch.tensor([[2.0, -1.0], [4.0, 3.0]], dtype=torch.float64)
    noisy = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float64)
    time = torch.tensor([0.25, 0.5], dtype=torch.float64)
    expected = torch.tensor([[-10.0, -5.0], [0.0, -5.0]], dtype=torch.float64)
    assert torch.equal(score_from_velocity(velocity, noisy, time, "linear"), expected)


def test_score_from_cosine_velocity_matches_closed_form() -> None:
    velocity = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    noisy = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    time = torch.tensor([1.0, 1.0], dtype=torch.float64)
    expected = torch.tensor([[-2.0, -1.0], [1.0, -2.0]], dtype=torch.float64)
    assert torch.allclose(
        score_from_velocity(velocity, noisy, time, "cosine"),
        expected,
        rtol=0,
        atol=1e-14,
    )


def test_sampler_runs_unconditional_then_cfg_branches_and_restores_mode() -> None:
    model = _ZeroModel().train()
    latents = torch.zeros(2, 3, 4)
    cls = torch.zeros(2, 5)
    labels = torch.tensor([0, 2])
    sampled, sampled_cls = sample_latents(
        model,
        latents,
        cls,
        labels,
        num_steps=2,
        cfg_scale=2.0,
        path_drop_guidance=True,
        shift_time=False,
    )
    assert sampled.shape == latents.shape
    assert sampled_cls.shape == cls.shape
    assert model.training
    assert len(model.seen_labels) == 4
    assert all(torch.equal(seen, labels) for seen in model.seen_labels[::2])
    assert all(
        torch.equal(seen, torch.full_like(labels, 3))
        for seen in model.seen_labels[1::2]
    )
    assert model.seen_options == [(False, False), (True, False)] * 2
    assert model.seen_dtypes == [(torch.float32, torch.float32, torch.float32)] * 4


def test_sampler_rejects_one_step_before_changing_model_mode() -> None:
    model = _ZeroModel().train()
    with pytest.raises(ValueError, match="num_steps must be at least two"):
        sample_latents(
            model,
            torch.zeros(2, 3, 4),
            torch.zeros(2, 5),
            torch.tensor([0, 1]),
            num_steps=1,
        )
    assert model.training


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
