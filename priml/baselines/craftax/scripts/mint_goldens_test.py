"""Tests that the references are glibc's and PufferLib's, not a copy of the port's."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import platform

import numpy as np
import pytest

from priml.baselines.craftax.scripts import mint_goldens


if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_the_transcription_draws_pufferlibs_first_draws_from_73() -> None:
    expected = mint_goldens.SEED_73_DRAWS
    draws = mint_goldens.glibc_rand_r(np.array([73], np.uint32), len(expected))
    assert draws.tolist() == [list(expected)]


def test_the_transcription_is_the_live_glibc() -> None:
    if platform.libc_ver()[0] != "glibc":
        pytest.skip("rand_r differs between libcs; the reference is glibc's")
    seeds = (0, 1, 73, 511, 2_047)
    expected = mint_goldens.glibc_rand_r(np.array(seeds, np.uint32), 64)
    for row, seed in enumerate(seeds):
        draws = cast("NDArray[np.int32]", expected[row])
        assert np.array_equal(mint_goldens.libc_rand_r(seed, 64), draws), seed


def test_the_libm_light_level_is_pufferlibs_at_the_anchors() -> None:
    anchors = mint_goldens.LIGHT_ANCHORS
    levels = cast(
        "list[int]",
        mint_goldens.light_levels_libm(anchors).view(np.uint32).tolist(),
    )
    assert dict(zip(anchors, levels, strict=True)) == anchors


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
