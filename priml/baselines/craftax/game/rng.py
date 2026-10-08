"""glibc's ``rand_r`` and the draws the game builds on it.

Every environment owns one 32-bit ``rand_r`` state, seeded with its index, and
every random decision in the game -- terrain, loot, mob moves, spawns -- is a
draw from that stream in a fixed order. Matching the C bit for bit therefore
starts here: the generator must produce glibc's integers, ``rng_f32_numba`` must
round them to float32 exactly as C does, and every consumer must draw in C's
order. Draws are scalar and many are conditional, which is why the game draws
per environment rather than in batches (the package docstring says why).

C passes the state by pointer. A kernel here passes it as a one-element
``uint32`` array, so ``rng_f32_numba(rng)`` advances the caller's stream just as
``rng_f32(&env->rng)`` does; ``rng_f32_at_numba`` is the one pure draw, from a state
derived from a seed and a cell index.

Two edges are load-bearing. ``rng_f32_numba`` returns exactly ``1.0`` when the draw
is within 64 of 2^31, because the int-to-float conversion rounds up, and then
``rng_int(rng, lo, hi)`` returns ``hi``. Callers that index with the result
handle that case explicitly rather than reading past an array.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from priml.baselines.craftax.game.jit import jit


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import Array1


MASK32: Final = np.uint64(0xFFFF_FFFF)
"""Numba widens uint32 arithmetic to uint64, so every LCG step is masked back."""

F32_PER_DRAW: Final = np.float32(1.0 / 2_147_483_648.0)
"""``1.0f / ((float)RAND_MAX + 1.0f)``: 2^-31, exact in float32."""


@jit
def rand_r_step_numba(seed: np.uint64) -> tuple[np.int32, np.uint64]:
    """Return glibc ``rand_r``'s draw and next state for one 32-bit state.

    Three steps of the ``1103515245 * x + 12345`` LCG, taking 11, 10 and 10
    bits from the top halves. The state is carried as a masked uint64.

    Args:
      seed: The current state, below 2^32.

    Returns:
      draw: The 31-bit draw.
      next_seed: The state after the draw.

    """
    next_seed = (seed * np.uint64(1_103_515_245) + np.uint64(12_345)) & MASK32
    result = (next_seed >> np.uint64(16)) & np.uint64(0x7FF)
    next_seed = (next_seed * np.uint64(1_103_515_245) + np.uint64(12_345)) & MASK32
    result = (result << np.uint64(10)) ^ (
        (next_seed >> np.uint64(16)) & np.uint64(0x3FF)
    )
    next_seed = (next_seed * np.uint64(1_103_515_245) + np.uint64(12_345)) & MASK32
    result = (result << np.uint64(10)) ^ (
        (next_seed >> np.uint64(16)) & np.uint64(0x3FF)
    )
    return np.int32(result), next_seed


@jit
def rand_r_numba(rng: Array1[np.uint32]) -> np.int32:
    """Advance the stream in ``rng[0]`` and return glibc's draw in ``[0, 2^31)``."""
    draw, next_seed = rand_r_step_numba(np.uint64(rng[0]))
    rng[0] = np.uint32(next_seed)
    return draw


@jit
def rng_f32_numba(rng: Array1[np.uint32]) -> np.float32:
    """Draw a float32 in ``[0, 1]``: ``rand_r(rng) * (1.0f / 2^31)``.

        The draw converts to float32 with round-to-nearest, so a draw within 64
        of 2^31 rounds up and the result is exactly ``1.0``.

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    Returns:
      value: A float32 in ``[0, 1]``.

    """
    return np.float32(rand_r_numba(rng)) * F32_PER_DRAW


@jit
def rng_int_numba(rng: Array1[np.uint32], lo: int, hi: int) -> int:
    """Draw an int in ``[lo, hi)``; ``hi`` itself when ``rng_f32_numba`` returns 1.0.

        A span of one or less returns ``lo`` without consuming a draw, as the C
        does, so a caller's draw count depends on the span.

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      lo: Inclusive lower bound.
      hi: Exclusive upper bound, except on the 1.0 edge.

    Returns:
      value: The drawn int.

    """
    span = hi - lo
    if span <= 1:
        return lo
    return lo + int(rng_f32_numba(rng) * np.float32(span))


@jit
def rng_f32_at_numba(seed: np.uint64, index: np.uint64) -> np.float32:
    """Draw a float32 from a state mixed from ``seed`` and a cell ``index``.

        One ``rand_r`` step from ``seed ^ (index * 747796405) + 2891336453``, in
        uint32 arithmetic. Terrain, ore and tree fields use it so every cell gets
        an independent draw instead of the LCG's raster-order lattice.

    Args:
      seed: The lattice seed, below 2^32.
      index: The cell index.

    Returns:
      value: A float32 in ``[0, 1]``.

    """
    local = (seed ^ ((index * np.uint64(747_796_405)) & MASK32)) & MASK32
    local = (local + np.uint64(2_891_336_453)) & MASK32
    draw, _ = rand_r_step_numba(local)
    return np.float32(draw) * F32_PER_DRAW


@jit
def choice_valid_numba(rng: Array1[np.uint32], valid: Array1[int], count: int) -> int:
    """Pick a uniformly random index among the first ``count`` set entries of ``valid``.

        Returns 0 without a draw when none is set. When ``rng_int_numba`` returns the
        count itself (the 1.0 edge) the scan runs off the end and the last set
        index is returned, as the C's does.

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      valid: A boolean or uint8 array.
      count: Entries to consider.

    Returns:
      index: The chosen index.

    """
    n = 0
    last = 0
    for i in range(count):
        if valid[i]:
            n += 1
            last = i
    if n == 0:
        return 0
    k = rng_int_numba(rng, 0, n)
    for i in range(count):
        if valid[i]:
            if k == 0:
                return i
            k -= 1
    return last


@jit
def choose_weighted_numba(
    rng: Array1[np.uint32],
    weights: Array1[np.float32],
    count: int,
) -> int:
    """Pick an index with probability proportional to its float32 weight.

        The total and the running sum accumulate in float32 in index order; the
        draw is ``total * rng_f32``, and the last index catches a draw the sums
        never exceed (all-zero weights, or the 1.0 edge).

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      weights: float32 weights.
      count: Entries to consider.

    Returns:
      index: The chosen index.

    """
    total = np.float32(0.0)
    for i in range(count):
        total += weights[i]
    draw = total * rng_f32_numba(rng)
    cumulative = np.float32(0.0)
    for i in range(count):
        cumulative += weights[i]
        if cumulative > draw:
            return i
    return count - 1


@jit
def choose_weighted_pair_numba(
    rng: Array1[np.uint32],
    first: np.float32,
    second: np.float32,
) -> int:
    """Return :func:`choose_weighted_numba` over the two weights, without an array.

        The same float32 sums in the same order, so the draw and the choice are
        the ones ``choose_weighted(rng, [first, second], 2)`` makes. The step's
        creatures choose an axis five times per tick, and an ``np.empty`` for
        two weights each time was an NRT allocation per call (measured: 7
        allocations per tick, 5 of them here).

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      first: The weight of index 0.
      second: The weight of index 1.

    Returns:
      index: 0 or 1.

    """
    total = np.float32(0.0)
    total += first
    total += second
    draw = total * rng_f32_numba(rng)
    cumulative = np.float32(0.0)
    cumulative += first
    if cumulative > draw:
        return 0
    return 1
