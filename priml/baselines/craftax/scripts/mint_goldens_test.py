"""Tests that the references are glibc's and PufferLib's, not a copy of the port's."""

from __future__ import annotations

from typing import cast

import numpy as np

from priml.baselines.craftax.scripts import mint_goldens


def test_the_transcription_draws_pufferlibs_first_draws_from_73() -> None:
    expected = mint_goldens.SEED_73_DRAWS
    draws = mint_goldens.glibc_rand_r(np.array([73], np.uint32), len(expected))
    assert draws.tolist() == [list(expected)]


def test_the_live_libc_draws_count_values_from_its_seed() -> None:
    # Which libc's stream it is, the parity capture checks on glibc.
    draws = mint_goldens.libc_rand_r(73, 4)
    assert (draws.dtype, draws.shape) == (np.int32, (4,))
    assert np.array_equal(draws, mint_goldens.libc_rand_r(73, 4))


def test_the_libm_light_level_is_pufferlibs_at_the_anchors() -> None:
    anchors = mint_goldens.LIGHT_ANCHORS
    levels = cast(
        "list[int]",
        mint_goldens.light_levels_libm(anchors).view(np.uint32).tolist(),
    )
    assert dict(zip(anchors, levels, strict=True)) == anchors


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
