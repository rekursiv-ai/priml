"""Pretokens must match upstream ConvexTok's: strings, counts, and order.

The order is load-bearing. Candidate tokens and the linear program's variables
are both built by walking pretokens in this order, so a table with the right
counts in a different order yields a different (equally valid) LP whose solver
trajectory, and therefore vocabulary, differs from upstream's.
"""

from pathlib import Path
from typing import Final, cast

import json

import pytest

from priml.baselines.convextok.pretokens import (
    count_pretokens,
    merge_counts,
)
from priml.lib.custom_json import DictCodec, ListCodec, StrCodec


_CWD: Final = Path(__file__).resolve().parent


def test_counts_match_upstream_golden() -> None:
    corpus = _read_json("corpus.json")
    golden = _read_json("pretokens.json")
    texts = ListCodec.coerce(corpus.get("texts"), str, default=None)
    counts = count_pretokens(texts, split_pattern=_split_pattern())
    assert list(counts) == ListCodec.coerce(golden.get("pretokens"), str, default=None)
    assert list(counts.values()) == ListCodec.coerce(
        golden.get("frequencies"),
        int,
        default=None,
    )


def test_chunks_merge_into_first_occurrence_order() -> None:
    texts = ["b a b", "c a", "", "d c b"]
    whole = count_pretokens(texts, split_pattern=_split_pattern())
    chunked = count_pretokens(texts, split_pattern=_split_pattern(), chunk_size=1)
    assert list(chunked.items()) == list(whole.items())
    assert list(whole.items()) == [
        ("b", 1),
        ("Ġa", 2),
        ("Ġb", 2),
        ("c", 1),
        ("d", 1),
        ("Ġc", 1),
    ]


def test_merge_counts_sums_and_keeps_earliest_position() -> None:
    merged = merge_counts([{"x": 1, "y": 2}, {"z": 5, "x": 3}])
    assert list(merged.items()) == [("x", 4), ("y", 2), ("z", 5)]


def test_bytes_map_to_the_byte_level_alphabet() -> None:
    counts = count_pretokens(["é\n"], split_pattern=_split_pattern())
    assert list(counts) == ["Ã©", "Ċ"]


@pytest.mark.cli_python_subprocess
def test_worker_processes_match_serial_counts() -> None:
    corpus = _read_json("corpus.json")
    texts = ListCodec.coerce(corpus.get("texts"), str, default=None)
    serial = count_pretokens(texts, split_pattern=_split_pattern())
    parallel = count_pretokens(
        texts,
        split_pattern=_split_pattern(),
        num_workers=2,
        chunk_size=3,
    )
    assert list(parallel.items()) == list(serial.items())


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


def _split_pattern() -> str:
    """Upstream's nanochat regular expression, as recorded when the goldens were minted."""
    return StrCodec.coerce(_read_json("corpus.json").get("split_pattern"), default=None)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
