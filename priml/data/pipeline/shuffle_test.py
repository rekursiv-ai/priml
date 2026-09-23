"""Tests for shuffle processors."""

from __future__ import annotations

from priml.data.pipeline.shuffle import ShuffleBuffer
from priml.lib.custom_json import IntCodec


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
        IntCodec.coerce(r["index"], 0)
        for r in ShuffleBuffer.Config(size=10, seed=123).make()(iter(samples))
    ]
    b = [
        IntCodec.coerce(r["index"], 0)
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
