"""Tests for shuffle processors."""

from __future__ import annotations

import random

from priml.data.pipeline.shuffle import ShuffleBuffer
from priml.lib.codec import from_plain


def test_shuffle_buffer():
    """Test ShuffleBuffer shuffles samples."""
    config = ShuffleBuffer.Config()
    config.size = 10
    shuffler = config.make()

    samples = [{"key": f"test_{i}", "index": i} for i in range(20)]

    results = list(shuffler(iter(samples)))

    assert len(results) == 20
    # Check that not all samples are in original order.
    indices = [r["index"] for r in results]
    assert indices != list(range(20))


def test_shuffle_buffer_small():
    """Test ShuffleBuffer with fewer samples than buffer size."""
    config = ShuffleBuffer.Config()
    config.size = 100
    shuffler = config.make()

    samples = [{"key": f"test_{i}", "index": i} for i in range(10)]

    results = list(shuffler(iter(samples)))
    assert len(results) == 10


def test_shuffle_buffer_seed_is_reproducible():
    """A fixed seed yields a deterministic, reproducible shuffle order (M13/M14)."""
    samples = [{"key": f"test_{i}", "index": i} for i in range(50)]

    a = [
        from_plain(r.get("index"), int, default=0)
        for r in ShuffleBuffer.Config(size=10, seed=123).make()(iter(samples))
    ]
    b = [
        from_plain(r.get("index"), int, default=0)
        for r in ShuffleBuffer.Config(size=10, seed=123).make()(iter(samples))
    ]

    assert a == b
    assert a != list(range(50))  # Actually shuffled.
    assert sorted(a) == list(range(50))  # No loss/dup.


def test_shuffle_buffer_different_seeds_differ():
    """Different seeds produce different shuffle orders (M14)."""
    samples = [{"key": f"test_{i}", "index": i} for i in range(50)]

    a = [
        r["index"] for r in ShuffleBuffer.Config(size=10, seed=1).make()(iter(samples))
    ]
    b = [
        r["index"] for r in ShuffleBuffer.Config(size=10, seed=2).make()(iter(samples))
    ]

    assert a != b


def test_shuffle_buffer_fills_exactly_its_capacity() -> None:
    """The first replacement is drawn only from the initial capacity."""
    samples = [{"index": index} for index in range(4)]
    results = list(ShuffleBuffer.Config(size=3, seed=0).make()(iter(samples)))

    assert results[0]["index"] == 1
    indices: list[int] = []
    for sample in results:
        index = sample["index"]
        assert isinstance(index, int)
        indices.append(index)
    assert sorted(indices) == [0, 1, 2, 3]


def test_shuffle_buffer_without_seed_uses_global_random_state() -> None:
    """Unseeded shuffling consumes the shared RNG stream."""
    random.seed(42)
    before = random.getstate()

    list(ShuffleBuffer.Config(size=2).make()(iter([{"index": 0}, {"index": 1}])))

    assert random.getstate() != before


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
