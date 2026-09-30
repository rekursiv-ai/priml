"""Tests for diffusion conditioning functions."""

from __future__ import annotations

import torch

from priml.math.diffusion.conditioning import modulate, timestep_embedding


def test_modulate_broadcasts_shift_and_scale_per_example() -> None:
    x = torch.tensor([[[2.0, 3.0], [5.0, 7.0]], [[11.0, 13.0], [17.0, 19.0]]])
    shift = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    expected = x * (1 + scale[:, None]) + shift[:, None]
    assert torch.equal(modulate(x, shift, scale), expected)


def test_timestep_embedding_pads_odd_width() -> None:
    times = torch.tensor([0.0, 0.5])
    embedding = timestep_embedding(times, width=5, period=10.0)
    assert embedding.shape == (2, 5)
    assert torch.equal(embedding[:, -1], torch.zeros(2))
    torch.testing.assert_close(embedding[0, :2], torch.ones(2))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
