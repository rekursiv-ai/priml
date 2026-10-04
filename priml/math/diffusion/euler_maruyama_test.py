"""Tests for reverse-SDE integration math."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from priml.math.diffusion.euler_maruyama import (
    euler_maruyama_grid,
    guide_drift,
    integrate_two_streams,
    repa_diffusion,
    velocity_to_drift,
    velocity_to_score,
)


if TYPE_CHECKING:
    import pytest


def test_scalar_drift_helpers() -> None:
    t = torch.tensor([0.25, 0.5])
    assert torch.equal(repa_diffusion(t), 2 * t)
    assert torch.equal(
        guide_drift(torch.tensor([3.0, 5.0]), torch.tensor([1.0, 2.0]), 2),
        torch.tensor([5.0, 8.0]),
    )
    assert torch.equal(
        velocity_to_drift(
            torch.tensor([3.0, 5.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([2.0, 4.0]),
        ),
        torch.tensor([2.0, 1.0]),
    )


def test_velocity_to_score_recovers_the_closed_form() -> None:
    result = velocity_to_score(
        torch.tensor([2.0, 4.0]),
        torch.tensor([1.0, 3.0]),
        alpha=torch.tensor([2.0, 2.0]),
        sigma=torch.tensor([3.0, 4.0]),
        d_alpha=torch.tensor([1.0, 1.0]),
        d_sigma=torch.tensor([1.0, 1.0]),
    )
    expected = (2 * torch.tensor([2.0, 4.0]) - torch.tensor([1.0, 3.0])) / (
        torch.tensor([3.0, 4.0]) ** 2 - 2 * torch.tensor([3.0, 4.0])
    )
    torch.testing.assert_close(result, expected)


def test_grid_ends_with_deterministic_zero() -> None:
    grid = euler_maruyama_grid(3, last_time=0.2)
    assert grid.dtype == torch.float64
    torch.testing.assert_close(
        grid,
        torch.tensor([1.0, 0.6, 0.2, 0.0], dtype=torch.float64),
    )


def test_grid_uses_requested_device_and_default_final_time() -> None:
    grid = euler_maruyama_grid(2, device=torch.device("cpu"))
    assert grid.device == torch.device("cpu")
    torch.testing.assert_close(
        grid,
        torch.tensor([1.0, 0.04, 0.0], dtype=torch.float64),
    )


def test_grid_passes_requested_device_to_linspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_linspace = torch.linspace
    kwargs: dict[str, object] = {}

    def linspace(
        start: float,
        end: float,
        steps: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        kwargs.update(dtype=dtype, device=device)
        return original_linspace(
            start,
            end,
            steps=steps,
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(torch, "linspace", linspace)
    grid = euler_maruyama_grid(2, device=torch.device("cpu"))

    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["dtype"] == torch.float64
    assert grid.device == torch.device("cpu")


def test_velocity_to_score_uses_each_path_derivative() -> None:
    velocity = torch.tensor([2.0, -3.0])
    state = torch.tensor([1.0, 4.0])
    alpha = torch.tensor([2.0, 3.0])
    sigma = torch.tensor([4.0, 5.0])
    d_alpha = torch.tensor([2.0, 1.5])
    d_sigma = torch.tensor([1.0, 2.0])
    ratio = alpha / d_alpha
    variance = sigma.square() - ratio * d_sigma * sigma
    expected = (ratio * velocity - state) / variance

    torch.testing.assert_close(
        velocity_to_score(
            velocity,
            state,
            alpha=alpha,
            sigma=sigma,
            d_alpha=d_alpha,
            d_sigma=d_sigma,
        ),
        expected,
        rtol=0,
        atol=0,
    )


def test_zero_diffusion_integration_is_deterministic() -> None:
    media = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cls = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def drift(
        media: torch.Tensor,
        cls: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        return 2 * torch.ones_like(media), 3 * torch.ones_like(cls)

    def zero_diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    actual_media, actual_cls = integrate_two_streams(
        media,
        cls,
        torch.tensor([1.0, 0.5, 0.0]),
        drift,
        diffusion=zero_diffusion,
    )
    torch.testing.assert_close(actual_media, (media - 2).double())
    torch.testing.assert_close(actual_cls, (cls - 3).double())


def test_integration_uses_two_stream_noise_then_a_deterministic_final_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = torch.ones(2, 3)
    cls = torch.full((2, 4), 2.0)
    draws = iter((1.0, 3.0))
    draw_shapes: list[torch.Size] = []
    times: list[float] = []

    def randn_like(tensor: torch.Tensor) -> torch.Tensor:
        draw_shapes.append(tensor.shape)
        return torch.full_like(tensor, next(draws))

    def drift(
        latent: torch.Tensor,
        cls_token: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times.append(float(time))
        return torch.ones_like(latent), torch.full_like(cls_token, 2.0)

    monkeypatch.setattr(torch, "randn_like", randn_like)
    actual_media, actual_cls = integrate_two_streams(
        media,
        cls,
        torch.tensor([1.0, 0.5, 0.0]),
        drift,
        diffusion=lambda time: 4 * time,
    )

    torch.testing.assert_close(
        actual_media,
        torch.full(
            (2, 3),
            float(torch.sqrt(torch.tensor(2.0, dtype=torch.float64))),
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        actual_cls,
        torch.full(
            (2, 4),
            3 * float(torch.sqrt(torch.tensor(2.0, dtype=torch.float64))),
            dtype=torch.float64,
        ),
    )
    assert draw_shapes == [torch.Size((2, 3)), torch.Size((2, 4))]
    assert times == [1.0, 0.5]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
