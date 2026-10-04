"""Tests for resolution-aware timestep shifting."""

from __future__ import annotations

import pytest
import torch

from priml.math.diffusion.time_shift import time_shift


def test_time_shift_preserves_endpoints_and_matches_formula() -> None:
    times = torch.tensor([-0.5, 0.25, 0.75, 1.5])
    shifted = time_shift(times, latent_dimensions=16, reference_dimensions=9)
    assert torch.equal(shifted, torch.tensor([0.0, 0.30769232, 0.8, 1.0]))


def test_time_shift_uses_the_reference_dimension_default() -> None:
    times = torch.tensor([0.25, 0.5, 0.75])
    assert torch.equal(time_shift(times, latent_dimensions=4_096), times)


def test_time_shift_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match=r"^latent and reference dimensions must be positive$",
    ):
        time_shift(torch.tensor([0.5, 0.75]), latent_dimensions=0)
    with pytest.raises(
        ValueError,
        match=r"^latent and reference dimensions must be positive$",
    ):
        time_shift(
            torch.tensor([0.5, 0.75]),
            latent_dimensions=2,
            reference_dimensions=0,
        )

    assert torch.equal(
        time_shift(
            torch.tensor([0.25, 0.75]),
            latent_dimensions=1,
            reference_dimensions=1,
        ),
        torch.tensor([0.25, 0.75]),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
