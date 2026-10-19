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


def test_score_from_linear_velocity_is_exact() -> None:
    velocity = torch.tensor([[2.0, -1.0], [4.0, 3.0]], dtype=torch.float64)
    noisy = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float64)
    time = torch.tensor([0.25, 0.5], dtype=torch.float64)
    expected = torch.tensor([[-10.0, -5.0], [0.0, -5.0]], dtype=torch.float64)
    assert torch.equal(
        score_from_velocity(velocity, noisy=noisy, t=time, path="linear"),
        expected,
    )


def test_score_from_cosine_velocity_matches_closed_form() -> None:
    velocity = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    noisy = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    time = torch.tensor([1.0, 1.0], dtype=torch.float64)
    expected = torch.tensor([[-2.0, -1.0], [1.0, -2.0]], dtype=torch.float64)
    assert torch.allclose(
        score_from_velocity(velocity, noisy=noisy, t=time, path="cosine"),
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
        latents=latents,
        cls_latents=cls,
        labels=labels,
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


def test_sampler_shifts_times_by_the_latent_dimension_ratio() -> None:
    plain, shifted = _TimedModel(), _TimedModel()
    for model, shift_time in ((plain, False), (shifted, True)):
        sample_latents(
            model,
            latents=torch.zeros(2, 3, 4),
            cls_latents=torch.zeros(2, 5),
            labels=torch.tensor([0, 2]),
            num_steps=2,
            shift_time=shift_time,
            shift_base=3,
        )
    unshifted = torch.stack(plain.seen_times)
    assert unshifted.shape == (2,)
    # Shift = sqrt(12 / 3) = 2, so a grid time t becomes 2t / (1 + t).
    torch.testing.assert_close(
        torch.stack(shifted.seen_times),
        2 * unshifted / (1 + unshifted),
    )


def test_sampler_rejects_one_step_before_changing_model_mode() -> None:
    model = _ZeroModel().train()
    with pytest.raises(ValueError, match="num_steps must be at least two"):
        sample_latents(
            model,
            latents=torch.zeros(2, 3, 4),
            cls_latents=torch.zeros(2, 5),
            labels=torch.tensor([0, 1]),
            num_steps=1,
        )
    assert model.training


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
        return ModelOutput(
            velocity=torch.zeros_like(x),
            cls_velocity=torch.zeros_like(cls_token),
            projections=(),
        )


class _TimedModel(_ZeroModel):
    """Record the flow time of each call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_times: list[Tensor] = []

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
        self.seen_times.append(t[0].clone())
        return super().forward(
            x,
            t,
            y,
            cls_token,
            drop_sparse_path=drop_sparse_path,
            route_tokens=route_tokens,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
