"""Tests for reverse-SDE integration math."""

from __future__ import annotations

import torch

from priml.math.diffusion.euler_maruyama import (
    euler_maruyama_grid,
    guide_drift,
    integrate_two_streams,
    repa_diffusion,
    velocity_to_drift,
    velocity_to_score,
)


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


def test_zero_diffusion_integration_is_deterministic() -> None:
    media = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cls = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def drift(
        media: torch.Tensor,
        cls: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        return torch.ones_like(media), torch.ones_like(cls)

    def zero_diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    actual_media, actual_cls = integrate_two_streams(
        media,
        cls,
        torch.tensor([1.0, 0.5, 0.0]),
        drift,
        diffusion=zero_diffusion,
    )
    torch.testing.assert_close(actual_media, (media - 1).double())
    torch.testing.assert_close(actual_cls, (cls - 1).double())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
