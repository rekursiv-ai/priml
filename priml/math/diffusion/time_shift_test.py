"""Tests for resolution-aware timestep shifting."""

from __future__ import annotations

import pytest
import torch

from priml.math.diffusion.time_shift import time_shift


def test_time_shift_preserves_endpoints_and_is_monotone() -> None:
    times = torch.tensor([0.0, 0.25, 1.0])
    shifted = time_shift(times, latent_dimensions=16, reference_dimensions=4)
    assert shifted[0] == 0
    assert shifted[-1] == 1
    assert bool(torch.all(shifted[1:] >= shifted[:-1]))
    assert shifted[1] > times[1]


def test_time_shift_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        time_shift(torch.tensor([0.5, 0.75]), latent_dimensions=0)
    with pytest.raises(ValueError, match="must be positive"):
        time_shift(
            torch.tensor([0.5, 0.75]),
            latent_dimensions=2,
            reference_dimensions=-1,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
