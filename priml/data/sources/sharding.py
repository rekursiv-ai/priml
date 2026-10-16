"""Deterministic sharding of items across distributed workers.

Sources slice their items across data-loading workers. Two correctness
requirements drive this module:

- Shuffle must happen BEFORE slicing, with a seed shared by all workers.
  Otherwise each worker (a separate fork) shuffles with a different RNG
  state and the per-worker contiguous slices overlap and miss items --
  silent data loss and duplication.
- Slicing must be balanced and exhaustive. With ``start = w * L // N`` and
  ``end = (w + 1) * L // N`` the union of all worker slices is exactly the
  full item list with no gaps or overlaps, and no worker is starved even
  when ``L < N``.
"""

from __future__ import annotations

from collections import deque
from itertools import islice
from typing import TYPE_CHECKING

import random


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


__all__ = ["interleave_shards", "shard_and_shuffle"]


def shard_and_shuffle[T](
    items: list[T],
    *,
    worker_slice: tuple[int, int] | None = None,
    shuffle: bool = False,
    epoch_seed: int = 0,
) -> list[T]:
    """Shuffle items with a shared seed then return this worker's slice.

    The shuffle is applied before slicing using ``epoch_seed`` so every
    worker produces the same permutation and the contiguous slices
    partition the items exactly.

    Args:
      items: Full list of items, identical across all workers.
      worker_slice: ``(worker_id, num_workers)`` for this worker, or None for
        the full list. ``worker_id`` must satisfy ``0 <= worker_id < num_workers``.
      shuffle: Shuffle the items before slicing.
      epoch_seed: Seed shared by all workers; vary per epoch to reshuffle.

    Returns:
      worker_items: The contiguous slice assigned to this worker.

    Raises:
      ValueError: If ``worker_slice`` is out of range.

    """
    items = list(items)

    if shuffle:
        random.Random(epoch_seed).shuffle(items)  # noqa: S311 -- Shard assignment is seeded for reproducibility, not security.

    if worker_slice is None:
        return items

    worker_id, num_workers = worker_slice
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(
            f"Invalid slice: worker_id={worker_id}, num_workers={num_workers}",
        )

    length = len(items)
    start = worker_id * length // num_workers
    end = (worker_id + 1) * length // num_workers
    return items[start:end]


def interleave_shards[S, T](
    shards: list[S],
    read: Callable[[S], Iterator[T]],
    *,
    num_concurrently_read_shards: int,
) -> Iterator[T]:
    """Round-robin over at most ``num_concurrently_read_shards`` open shards.

    A shard opens only when one in flight is exhausted, so at most that many
    readers hold a file at once, and ``1`` reads the shards in order.

    Args:
      shards: Shards in read order, already sliced for this worker.
      read: Opens one shard as an iterator of its items.
      num_concurrently_read_shards: Readers in flight; at least 1.

    Yields:
      item: The next item of the next open shard, in round-robin order.

    Raises:
      ValueError: If ``num_concurrently_read_shards`` is below 1, since no
        reader would ever open and the stream would be silently empty.

    """
    if num_concurrently_read_shards < 1:
        raise ValueError(
            "num_concurrently_read_shards must be >= 1, got "
            f"{num_concurrently_read_shards}.",
        )
    active = deque(read(shard) for shard in shards[:num_concurrently_read_shards])
    pending = iter(shards[num_concurrently_read_shards:])
    while active:
        reader = active.popleft()
        try:
            item = next(reader)
        except StopIteration:
            active.extend(read(shard) for shard in islice(pending, 1))
            continue
        yield item
        active.append(reader)
