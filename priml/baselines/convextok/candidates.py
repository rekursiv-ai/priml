"""Count candidate tokens: every substring of two or more bytes of every pretoken.

A candidate's count is the number of times it occurs across the corpus: its
occurrences within each pretoken, weighted by that pretoken's frequency.
Candidates seen only once are dropped, and the rest are ordered as Python
orders their ByteLevel strings -- the order upstream ConvexTok numbers its token
variables in and breaks rounding ties by.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import zlib


def count_candidates(
    pretokens: dict[str, int],
    *,
    min_count: int = 2,
    num_workers: int = 1,
    chunk_size: int = 100_000,
    num_partitions: int = 1,
) -> dict[str, int]:
    """Count every substring of two or more bytes and keep the frequent ones.

    Args:
      pretokens: Pretoken to frequency.
      min_count: Smallest weighted occurrence count a candidate keeps.
      num_workers: Processes sharing the counting and merging; 1 runs inline.
      chunk_size: Pretokens per independently counted chunk.
      num_partitions: Disjoint key ranges merged independently of each other.

    Returns:
      candidates: Candidate to weighted count, in Python string order.

    """
    items = list(pretokens.items())
    chunks = [
        items[start : start + chunk_size] for start in range(0, len(items), chunk_size)
    ]
    count = partial(count_substrings, num_partitions=num_partitions)
    merge = partial(merge_partition, min_count=min_count)
    if num_workers == 1:
        tables = list(map(count, chunks))
        kept = list(map(merge, _by_partition(tables, num_partitions)))
    else:
        with ProcessPoolExecutor(num_workers) as pool:
            tables = list(pool.map(count, chunks))
            kept = list(pool.map(merge, _by_partition(tables, num_partitions)))
    merged = {token: total for part in kept for token, total in part.items()}
    return {token: merged[token] for token in sorted(merged)}


def count_substrings(
    pretokens: Sequence[tuple[str, int]],
    *,
    num_partitions: int,
) -> list[dict[str, int]]:
    """Count one chunk's substrings of two or more bytes, split by partition.

    Args:
      pretokens: Pretoken and frequency pairs of the chunk.
      num_partitions: Number of disjoint key sets to split the counts into.

    Returns:
      tables: One substring-to-count table per partition.

    """
    counts: dict[str, int] = {}
    for piece, frequency in pretokens:
        size = len(piece)
        for length in range(2, size + 1):
            for start in range(size - length + 1):
                token = piece[start : start + length]
                counts[token] = counts.get(token, 0) + frequency
    tables: list[dict[str, int]] = [{} for _ in range(num_partitions)]
    for token, total in counts.items():
        # A process-independent hash: forkserver workers each draw their own
        # string-hash seed, so `hash(token)` would scatter a token across partitions.
        tables[zlib.crc32(token.encode()) % num_partitions][token] = total
    return tables


def merge_partition(
    tables: Sequence[dict[str, int]],
    *,
    min_count: int,
) -> dict[str, int]:
    """Sum one partition's tables and keep candidates counted at least `min_count` times."""
    merged: dict[str, int] = {}
    for table in tables:
        for token, total in table.items():
            merged[token] = merged.get(token, 0) + total
    return {token: total for token, total in merged.items() if total >= min_count}


def _by_partition(
    tables: Sequence[list[dict[str, int]]],
    num_partitions: int,
) -> list[list[dict[str, int]]]:
    return [[chunk[part] for chunk in tables] for part in range(num_partitions)]
