"""Count a corpus's unique ByteLevel pretokens in first-occurrence order.

Upstream ConvexTok splits each document with the nanochat regular expression,
maps every byte to its GPT-2 ByteLevel character, and counts the pieces. The
pretoken order it produces is the order of first occurrence in the corpus:
batches are counted separately and merged in batch order, and a merge keeps
each piece at its earliest position. That order seeds the candidate list and
the linear program's variables, so it is reproduced exactly.
"""

from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tokenizers import Regex, pre_tokenizers


def count_pretokens(
    texts: Sequence[str],
    *,
    split_pattern: str,
    num_workers: int = 1,
    chunk_size: int = 10_000,
) -> dict[str, int]:
    """Count unique pretokens across documents in first-occurrence order.

    Args:
      texts: Documents in corpus order.
      split_pattern: Regular expression that splits a document into pieces.
      num_workers: Processes counting chunks concurrently; 1 counts inline.
      chunk_size: Documents per independently counted chunk.

    Returns:
      counts: Pretoken to frequency, ordered by first occurrence.

    """
    chunks = [
        texts[start : start + chunk_size] for start in range(0, len(texts), chunk_size)
    ]
    count = partial(count_chunk, split_pattern=split_pattern)
    if num_workers == 1:
        return merge_counts(map(count, chunks))
    with ProcessPoolExecutor(num_workers) as pool:
        return merge_counts(pool.map(count, chunks))


def count_chunk(texts: Iterable[str], *, split_pattern: str) -> dict[str, int]:
    """Count one chunk's pretokens in first-occurrence order.

    Args:
      texts: Documents of the chunk.
      split_pattern: Regular expression that splits a document into pieces.

    Returns:
      counts: Pretoken to frequency, ordered by first occurrence.

    """
    split = pretokenizer(split_pattern)
    counts: dict[str, int] = {}
    for text in texts:
        for piece, _ in split.pre_tokenize_str(text):
            # Upstream skips empty pieces rather than counting an empty pretoken.
            if piece:
                counts[piece] = counts.get(piece, 0) + 1
    return counts


def merge_counts(tables: Iterable[dict[str, int]]) -> dict[str, int]:
    """Sum ordered count tables, keeping each key at its earliest position."""
    merged: dict[str, int] = {}
    for table in tables:
        for piece, count in table.items():
            merged[piece] = merged.get(piece, 0) + count
    return merged


def pretokenizer(split_pattern: str) -> pre_tokenizers.Sequence:
    """Build the regex split followed by the reversible GPT-2 byte mapping.

    Args:
      split_pattern: Regular expression that splits a document into pieces.

    Returns:
      pretokenizer: Splits text and maps every byte to one ByteLevel character.

    """
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(split_pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ],
    )
