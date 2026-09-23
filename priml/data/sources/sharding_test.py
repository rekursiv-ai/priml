"""Tests for deterministic worker sharding."""

from __future__ import annotations

import pytest

from priml.data.sources.sharding import shard_and_shuffle


def test_balanced_slices_partition_exactly() -> None:
    """Union of all worker slices equals the full set, no dup, no gap."""
    items = list(range(10))
    num_workers = 4
    union: list[int] = []
    for w in range(num_workers):
        union.extend(shard_and_shuffle(items, worker_slice=(w, num_workers)))
    assert sorted(union) == items
    assert len(union) == len(items)  # No duplicates.


def test_no_starved_worker_when_fewer_items_than_workers() -> None:
    """With L < N every item lands on exactly one worker; no remainder dump."""
    items = list(range(3))
    num_workers = 5
    union: list[int] = []
    for w in range(num_workers):
        union.extend(shard_and_shuffle(items, worker_slice=(w, num_workers)))
    assert sorted(union) == items


def test_shuffle_before_slice_is_seed_consistent_across_workers() -> None:
    """All workers shuffle identically (shared seed), so slices still partition."""
    items = list(range(20))
    num_workers = 4
    union: list[int] = []
    for w in range(num_workers):
        union.extend(
            shard_and_shuffle(
                items,
                worker_slice=(w, num_workers),
                shuffle=True,
                epoch_seed=7,
            ),
        )
    assert sorted(union) == items
    assert len(union) == len(items)  # No overlap, no loss.


def test_shuffle_actually_reorders() -> None:
    """Shuffle with a seed produces a non-identity permutation."""
    items = list(range(20))
    shuffled = shard_and_shuffle(items, shuffle=True, epoch_seed=1)
    assert sorted(shuffled) == items
    assert shuffled != items


def test_epoch_seed_changes_permutation() -> None:
    """Different epoch seeds yield different permutations."""
    items = list(range(50))
    a = shard_and_shuffle(items, shuffle=True, epoch_seed=1)
    b = shard_and_shuffle(items, shuffle=True, epoch_seed=2)
    assert a != b


def test_no_slice_returns_all() -> None:
    """worker_slice=None returns every item."""
    items = list(range(5))
    assert shard_and_shuffle(items) == items


def test_invalid_slice_raises() -> None:
    """worker_id out of range raises ValueError."""
    with pytest.raises(ValueError, match="Invalid slice"):
        shard_and_shuffle([1, 2, 3], worker_slice=(2, 2))


def test_does_not_mutate_input() -> None:
    """Input list is not shuffled in place."""
    items = list(range(20))
    original = list(items)
    shard_and_shuffle(items, shuffle=True, epoch_seed=3)
    assert items == original


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
