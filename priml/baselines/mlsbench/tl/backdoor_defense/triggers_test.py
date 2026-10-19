"""Tests for the backdoor triggers."""

from __future__ import annotations

from typing import Final

import torch

from priml.baselines.mlsbench.tl.backdoor_defense.triggers import badnets, blend


MEAN: Final = torch.tensor([0.25, 0.5, 0.75])
STD: Final = torch.tensor([0.5, 0.25, 0.125])


def pixels(media: torch.Tensor) -> torch.Tensor:
    """Undo the normalization of ``MEAN`` and ``STD``."""
    # As badnets and blend do, broadcast the (C,) statistics over (H, W).
    return media * STD.view(-1, 1, 1) + MEAN.view(-1, 1, 1)


def test_badnets_whitens_the_bottom_right_square_and_nothing_else() -> None:
    media = torch.zeros(2, 3, 5, 6)
    out = pixels(badnets(media, mean=MEAN, std=STD))
    assert (out[..., -3:, -3:] == 1).all()
    corner = torch.zeros(5, 6, dtype=torch.bool)
    corner[-3:, -3:] = True
    assert torch.equal(out[..., ~corner], pixels(media)[..., ~corner])


def test_badnets_clips_pixels_into_the_unit_range() -> None:
    media = torch.full((2, 3, 5, 6), 8.0)
    out = pixels(badnets(media, mean=MEAN, std=STD))
    assert torch.equal(out, torch.ones_like(out))


def test_blend_mixes_a_fifth_of_a_checkerboard_into_every_pixel() -> None:
    # As badnets and blend do, broadcast the (C,) statistics over (H, W).
    media = (torch.zeros(2, 3, 5, 6) - MEAN.view(-1, 1, 1)) / STD.view(-1, 1, 1)
    out = pixels(blend(media, mean=MEAN, std=STD))
    rows, cols = torch.meshgrid(torch.arange(5), torch.arange(6), indexing="ij")
    on = (rows + cols) % 2 == 0
    torch.testing.assert_close(out[..., on], torch.full_like(out[..., on], 0.2))
    torch.testing.assert_close(out[..., ~on], torch.zeros_like(out[..., ~on]))


def test_triggers_leave_their_input_untouched() -> None:
    media = torch.zeros(2, 3, 5, 6)
    _ = badnets(media, mean=MEAN, std=STD)
    _ = blend(media, mean=MEAN, std=STD)
    assert torch.equal(media, torch.zeros(2, 3, 5, 6))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
