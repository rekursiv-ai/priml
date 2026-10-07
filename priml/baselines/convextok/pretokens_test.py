"""Pretokens must match upstream ConvexTok's: strings, counts, and order.

The order is load-bearing. Candidate tokens and the linear program's variables
are both built by walking pretokens in this order, so a table with the right
counts in a different order yields a different (equally valid) LP whose solver
trajectory, and therefore vocabulary, differs from upstream's.
"""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast

import json

import pytest

from priml.baselines.convextok import pretokens
from priml.baselines.convextok.pretokens import (
    count_pretokens,
    merge_counts,
)
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


_CWD: Final = Path(__file__).resolve().parent


def test_count_pretokens_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    parameters = signature(count_pretokens).parameters
    num_workers_default = cast(object, parameters["num_workers"].default)
    chunk_size_default = cast(object, parameters["chunk_size"].default)
    assert isinstance(num_workers_default, int)
    assert isinstance(chunk_size_default, int)
    assert num_workers_default == 1
    assert chunk_size_default == 10_000

    def unexpected_executor(*args: object) -> object:
        pytest.fail(f"default serial count constructed a process pool: {args}")

    monkeypatch.setattr(pretokens, "ProcessPoolExecutor", unexpected_executor)
    assert count_pretokens([], split_pattern=_split_pattern()) == {}


def test_default_chunk_size_counts_ten_thousand_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingPool:
        def __init__(self, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            del exc_type, exc_value, traceback

        def map(
            self,
            function: Callable[[Sequence[str]], dict[str, int]],
            chunks: Iterable[Sequence[str]],
        ) -> list[dict[str, int]]:
            chunk_list = list(chunks)
            assert [len(chunk) for chunk in chunk_list] == [10_000, 1]
            return [function(chunk) for chunk in chunk_list]

    monkeypatch.setattr(pretokens, "ProcessPoolExecutor", RecordingPool)
    assert (
        count_pretokens(
            [""] * 10_001,
            split_pattern=_split_pattern(),
            num_workers=2,
        )
        == {}
    )


def test_counts_match_upstream_golden() -> None:
    corpus = _read_json("corpus.json")
    golden = _read_json("pretokens.json")
    texts = from_plain(corpus.get("texts"), list[str])
    counts = count_pretokens(texts, split_pattern=_split_pattern())
    assert list(counts) == from_plain(golden.get("pretokens"), list[str])
    assert list(counts.values()) == from_plain(
        golden.get("frequencies"),
        list[int],
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
    texts = from_plain(corpus.get("texts"), list[str])
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
    return dict(from_plain(raw, dict[str, object]))


def _split_pattern() -> str:
    """Upstream's nanochat regular expression, as recorded when the goldens were minted."""
    return from_plain(_read_json("corpus.json").get("split_pattern"), str)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
