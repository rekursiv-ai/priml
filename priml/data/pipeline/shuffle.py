"""Shuffling processor for data pipeline.

Processor for shuffling samples in data pipelines.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import random

from configgle import Fig


__all__ = [
    "ShuffleBuffer",
]


class ShuffleBuffer:
    """Shuffle samples using a fixed-size buffer.

    Inspired by TensorFlow Datasets (TFDS) shuffle mechanism:
    - Maintains a buffer of fixed size (in number of samples)
    - Fills buffer initially, then randomly samples from it
    - Each sample removed is replaced with a new one (if available)
    - Provides approximate shuffling without loading entire dataset

    Reference: tensorflow/python/data/ops/dataset_ops.py (shuffle operation)

    Buffer sizing:
    - size is in number of samples, not bytes
    - For perfect shuffling: size >= dataset size
    - For memory efficiency: use smaller buffer (e.g., 1_000-10_000)
    - Memory usage depends on sample size (metadata vs images vs tensors)
    - Place early in pipeline (before image loading) for smaller memory footprint

    Shuffling is approximate: a sample can move at most ``size`` positions
    forward. With ``size <= 1`` the stream passes through unchanged. The buffer
    is fully drained (shuffled) at end of stream, so all inputs are emitted.

    Reproducibility: pass ``seed`` to draw from a dedicated ``random.Random``
    rather than the shared global RNG, isolating shuffle order from other
    consumers of ``random.*``.
    """

    class Config(Fig["ShuffleBuffer"]):
        size: int = 1_000
        """Buffer size in number of samples."""

        seed: int | None = None
        """Seed for a dedicated RNG; None uses the shared global RNG."""

    # Transparent pass-through: the shuffler never reads or mutates sample
    # fields, so a sample is any read-only mapping (matching the pipeline's
    # `Processor` sample bound, `Mapping[str, object]`).
    Input = Mapping[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.size = config.size
        # Dedicated RNG when seeded; otherwise defer to the global random module.
        self._rng = (
            random.Random(config.seed)  # noqa: S311 -- This RNG only determines sample order and never protects secrets.
            if config.seed is not None
            else random
        )

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Shuffle samples using a fixed-size buffer.

        Requires:
          - (any fields) - passes through all sample fields

        Adds:
          - (none) - transparent pass-through with shuffling

        """
        buffer: list[ShuffleBuffer.Input] = []

        # Fill buffer initially.
        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= self.size:
                break

        # Yield random samples and refill.
        for sample in samples:
            # Pick random index from buffer.
            idx = self._rng.randrange(len(buffer))
            yield buffer[idx]
            # Replace with new sample.
            buffer[idx] = sample

        # Shuffle and drain remaining buffer.
        self._rng.shuffle(buffer)
        yield from buffer
