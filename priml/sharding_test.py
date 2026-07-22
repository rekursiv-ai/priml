"""Tests for sharding utilities."""

from __future__ import annotations

from priml.sharding import parse_shard_spec
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_parse_shard_spec_individual_shards() -> None:
    """Test parsing individual shard IDs."""
    assert parse_shard_spec("1,3,5", 10) == {1, 3, 5}
    assert parse_shard_spec("0", 10) == {0}
    assert parse_shard_spec("9", 10) == {9}


def test_parse_shard_spec_ranges() -> None:
    """Test parsing ranges with start:end syntax."""
    assert parse_shard_spec("1:5", 10) == {1, 2, 3, 4}
    assert parse_shard_spec("0:3", 10) == {0, 1, 2}
    assert parse_shard_spec("5:10", 10) == {5, 6, 7, 8, 9}


def test_parse_shard_spec_open_ranges() -> None:
    """Test parsing open ranges."""
    assert parse_shard_spec(":5", 10) == {0, 1, 2, 3, 4}
    assert parse_shard_spec("5:", 10) == {5, 6, 7, 8, 9}
    assert parse_shard_spec(":", 10) == set(range(10))


def test_parse_shard_spec_step() -> None:
    """Test parsing ranges with step."""
    assert parse_shard_spec("::2", 10) == {0, 2, 4, 6, 8}
    assert parse_shard_spec("1:10:2", 20) == {1, 3, 5, 7, 9}
    assert parse_shard_spec("0:10:3", 20) == {0, 3, 6, 9}


def test_parse_shard_spec_negative_indices() -> None:
    """Test parsing negative indices."""
    assert parse_shard_spec("-1", 10) == {9}
    assert parse_shard_spec("-5:", 10) == {5, 6, 7, 8, 9}
    assert parse_shard_spec(":-5", 10) == {0, 1, 2, 3, 4}
    assert parse_shard_spec("-8:-2", 10) == {2, 3, 4, 5, 6, 7}


def test_parse_shard_spec_negative_step() -> None:
    """Negative step honors full Python slice semantics (issue CORE-003)."""
    assert parse_shard_spec("::-1", 5) == {0, 1, 2, 3, 4}
    assert parse_shard_spec("::-2", 5) == {0, 2, 4}
    assert parse_shard_spec("4:0:-1", 5) == {1, 2, 3, 4}


def test_parse_shard_spec_mixed() -> None:
    """Test parsing mixed specifications."""
    assert parse_shard_spec("1,4:9,3", 20) == {1, 3, 4, 5, 6, 7, 8}
    assert parse_shard_spec("0,5:10,15", 20) == {0, 5, 6, 7, 8, 9, 15}


def test_parse_shard_spec_deduplication() -> None:
    """Test that duplicates are automatically removed."""
    assert parse_shard_spec("1,1,1", 10) == {1}
    assert parse_shard_spec("1:5,3:7", 10) == {1, 2, 3, 4, 5, 6}


def test_parse_shard_spec_clamping() -> None:
    """Test that indices are clamped to valid range."""
    # Out of bounds positive indices
    assert parse_shard_spec("100", 10) == {9}
    assert parse_shard_spec("50:", 10) == set()

    # Very negative indices
    assert parse_shard_spec("-100:", 10) == set(range(10))
    assert parse_shard_spec("-100", 10) == {0}


def test_parse_shard_spec_empty() -> None:
    """Test parsing empty specifications."""
    assert parse_shard_spec("", 10) == set()
    assert parse_shard_spec("  ", 10) == set()


def test_parse_shard_spec_whitespace() -> None:
    """Test that whitespace is handled correctly."""
    assert parse_shard_spec(" 1 , 3 , 5 ", 10) == {1, 3, 5}
    assert parse_shard_spec(" 1 : 5 ", 10) == {1, 2, 3, 4}


def test_parse_shard_spec_edge_cases() -> None:
    """Test edge cases."""
    # Single shard dataset
    assert parse_shard_spec("0", 1) == {0}
    assert parse_shard_spec(":", 1) == {0}

    # Empty range
    assert parse_shard_spec("5:5", 10) == set()
    assert parse_shard_spec("5:3", 10) == set()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
