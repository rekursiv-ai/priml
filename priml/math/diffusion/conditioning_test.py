"""Tests for diffusion conditioning functions."""

from __future__ import annotations

import math

import torch

from priml.math.diffusion.conditioning import modulate, timestep_embedding


def test_modulate_broadcasts_shift_and_scale_per_example() -> None:
    x = torch.tensor(
        [
            [[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]],
            [[17.0, 19.0], [23.0, 29.0], [31.0, 37.0]],
        ],
    )
    shift = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    expected = torch.tensor(
        [
            [[7.0, 14.0], [16.0, 30.0], [34.0, 54.0]],
            [[88.0, 118.0], [118.0, 178.0], [158.0, 226.0]],
        ],
    )
    assert torch.equal(modulate(x, shift, scale), expected)


def test_timestep_embedding_pads_odd_width() -> None:
    times = torch.tensor([0.0, 0.5])
    embedding = timestep_embedding(times, width=5, period=10.0)
    assert embedding.shape == (2, 5)
    assert torch.equal(embedding[:, -1], torch.zeros(2))
    torch.testing.assert_close(embedding[0, :2], torch.ones(2))


def test_timestep_embedding_even_width_orders_cosines_then_sines() -> None:
    embedding = timestep_embedding(torch.tensor([0.0, 1.5]), width=4, period=9.0)
    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [
                torch.cos(torch.tensor(1.5)),
                torch.cos(torch.tensor(0.5)),
                torch.sin(torch.tensor(1.5)),
                torch.sin(torch.tensor(0.5)),
            ],
        ],
    )
    torch.testing.assert_close(embedding, expected)
    assert embedding.dtype == torch.float32


def test_timestep_embedding_stays_on_the_input_device() -> None:
    times = torch.tensor([0.0, 1.5])
    embedding = timestep_embedding(times, width=4)
    assert embedding.device == times.device
    assert embedding.shape == (2, 4)
    assert timestep_embedding(times).shape == (2, 256)

    meta_embedding = timestep_embedding(torch.empty(2, device="meta"), width=4)
    assert meta_embedding.device.type == "meta"


def test_timestep_embedding_uses_the_default_period() -> None:
    times = torch.tensor([0.0, 1.5])
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(2, dtype=torch.float32) / 2,
    )
    angles = times[:, None] * frequencies[None]
    expected = torch.cat((angles.cos(), angles.sin()), dim=-1)
    assert torch.equal(timestep_embedding(times, width=4), expected)


def test_timestep_embedding_frequency_dtype_is_explicit() -> None:
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        embedding = timestep_embedding(torch.tensor([0.0, 1.5]), width=4, period=9.0)
    finally:
        torch.set_default_dtype(default_dtype)
    assert embedding.dtype == torch.float32


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
