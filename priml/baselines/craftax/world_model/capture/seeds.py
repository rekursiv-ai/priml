"""World seeds of captured episodes, their training/validation split, and sampler seeds.

Episode ``n`` that a capture worker starts (counting from 0 across both splits)
is a validation episode when ``n % 20 == 19`` and a training episode otherwise.
Its world seed is ``split_base + 400,000,000 * generation + 40,000,000 * arm +
10,000,000 * worker + i``, where ``split_base`` is 100,000,000 for training
and 300,000,000 for validation, and ``i`` counts the worker's earlier episodes
in the same split. Generation 0 is the first dataset; every later generation
shifts the whole 100M-460M block by 400M, so the ranges of generations 0-9
are disjoint from each other and all lie below 2**32, the game's seed width,
and above the worlds an RL run trains and evaluates on (seeds below a few
million).

A dataset version takes one generation per kind of worker: fresh episodes in
one, branches of earlier episodes, whose worlds are their parents', in
another, which then only separates their sampling seeds.

Every episode's receipt records its own ``sampling_seed``, which also seeds
its epsilon override (``env.py``). A policy's action streams start from the
worker's ``rollout_seed``, so no two workers, arms, or generations repeat one.
"""

from typing import Final


TRAIN: Final = 0
VALIDATION: Final = 1

_MASK64: Final = (1 << 64) - 1


def episode_seed(
    ordinal: int,
    *,
    arm: int,
    worker: int,
    generation: int,
) -> tuple[int, int]:
    """Return the split and world seed of a worker's ``ordinal``-th started episode.

    Args:
      ordinal: Episodes the worker started before this one, over both splits.
      arm: Behaviour-mixture arm, 0-3.
      worker: Capture worker, 0-3.
      generation: Seed generation, 0-9.

    Returns:
      split: ``TRAIN`` or ``VALIDATION``.
      world_seed: The seed the episode's world is generated from.

    Raises:
      ValueError: An argument is out of range, or the split's index would leave
        the worker's 10,000,000-seed range.

    """
    validation_count = (ordinal + 1) // 20
    split = VALIDATION if ordinal % 20 == 19 else TRAIN
    index = validation_count - 1 if split == VALIDATION else ordinal - validation_count
    if min(ordinal, arm, worker, generation) < 0 or max(arm, worker) > 3:
        raise ValueError(f"Episode {ordinal}, arm {arm}, worker {worker} out of range.")
    if generation > 9:
        raise ValueError(f"Generation {generation} out of range 0-9.")
    if index >= 10_000_000:
        raise ValueError(f"Episode {ordinal} leaves the worker's seed range.")
    base = 300_000_000 if split == VALIDATION else 100_000_000
    return split, (
        base + 400_000_000 * generation + 40_000_000 * arm + 10_000_000 * worker + index
    )


def sampling_seed(
    ordinal: int,
    *,
    arm: int,
    worker: int,
    environment: int,
    generation: int,
) -> int:
    """Return the sampling seed recorded in one captured episode's receipt.

    Args:
      ordinal: Episodes the worker started before this one, over both splits.
      arm: Behaviour-mixture arm, 0-3.
      worker: Capture worker, 0-3.
      environment: Index of the environment within the worker's rollout.
      generation: Seed generation, 0-9.

    Returns:
      seed: Arm, worker, environment, generation and ordinal packed into 2, 2,
        20, 4 and 36 bits of a uint64, so every episode of every capture has
        its own.

    Raises:
      ValueError: A field does not fit its bits.

    """
    fields = ((ordinal, 36), (arm, 2), (worker, 2), (environment, 20))
    if (
        generation < 0
        or generation > 9
        or any(value < 0 or value >= 1 << bits for value, bits in fields)
    ):
        raise ValueError(
            f"Episode {ordinal}, arm {arm}, worker {worker}, environment "
            f"{environment}, generation {generation} out of range.",
        )
    return arm << 62 | worker << 60 | environment << 40 | generation << 36 | ordinal


def rollout_seed(*, arm: int, worker: int, base: int, generation: int) -> int:
    """Return the action-stream seed of one capture worker.

    A sampler seeds buffer ``b``'s streams with ``seed + b``. Spacing workers
    1,000,000 apart, and generations 16 workers apart, keeps every worker's
    streams distinct from every other's.

    Args:
      arm: Behaviour-mixture arm, 0-3.
      worker: Capture worker, 0-3.
      base: The arm policy's own sampler seed.
      generation: Seed generation, 0-9.

    Returns:
      seed: The worker's sampler seed.

    """
    return base + 1_000_000 * (16 * generation + 4 * arm + worker)


def splitmix64(state: int) -> tuple[int, int]:
    """Return SplitMix64's advanced state and its next output.

    The generator a sampling seed drives: the epsilon override's draws and the
    uniform legal actions of random play.

    Args:
      state: The generator's state, below 2**64.

    Returns:
      state: The advanced state.
      draw: The next 64-bit output.

    """
    state = (state + 0x9E37_79B9_7F4A_7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
    return state, z ^ (z >> 31)
