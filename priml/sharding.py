"""Utilities for parsing and working with shard specifications."""

from __future__ import annotations


def parse_shard_spec(spec: str, total_shards: int) -> set[int]:
    """Parse shard specification string into set of shard IDs.

    Uses full Python-style slicing syntax with start:end:step, where end is exclusive.
    Supports negative indices. Results are automatically deduped via set.

    Args:
        spec: Shard specification string (e.g., "1,4:9,3,:10:2,20:,-5:")
        total_shards: Total number of shards available

    Returns:
        Set of shard IDs to process

    Examples:
        >>> parse_shard_spec("1,3,5", 10)
        {1, 3, 5}
        >>> parse_shard_spec("1:5", 10)
        {1, 2, 3, 4}
        >>> parse_shard_spec(":5", 10)
        {0, 1, 2, 3, 4}
        >>> parse_shard_spec("5:", 10)
        {5, 6, 7, 8, 9}
        >>> parse_shard_spec("::2", 10)
        {0, 2, 4, 6, 8}
        >>> parse_shard_spec("1:10:2", 20)
        {1, 3, 5, 7, 9}
        >>> parse_shard_spec("-5:", 30)
        {25, 26, 27, 28, 29}
        >>> parse_shard_spec(":-5", 30)
        {0, 1, 2, ..., 24}
        >>> parse_shard_spec("1,4:9,3", 20)
        {1, 3, 4, 5, 6, 7, 8}

    """
    if not spec.strip():
        return set()

    shard_ids: set[int] = set()

    for part in spec.split(","):
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if ":" in part_stripped:
            # Delegate to Python's own slice semantics so negative steps
            # (e.g. "::-1") and negative indices behave exactly as slicing
            # a list of shard IDs would -- including clamping to bounds.
            parts = part_stripped.split(":")
            start_str = parts[0].strip()
            end_str = parts[1].strip() if len(parts) > 1 else ""
            step_str = parts[2].strip() if len(parts) > 2 else ""

            sl = slice(
                int(start_str) if start_str else None,
                int(end_str) if end_str else None,
                int(step_str) if step_str else None,
            )
            shard_ids.update(range(*sl.indices(total_shards)))
        else:
            idx = int(part_stripped)
            if idx < 0:
                idx = max(0, total_shards + idx)
            else:
                idx = min(idx, total_shards - 1)
            if 0 <= idx < total_shards:
                shard_ids.add(idx)

    return shard_ids
