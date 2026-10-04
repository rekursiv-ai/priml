"""Candidate tokens must match upstream ConvexTok's: strings, counts, and order.

Every substring of two or more bytes is a candidate, weighted by how often its
pretoken occurs, and kept only if its total reaches two. The kept list is in
Python string order; that order numbers the linear program's token variables
and breaks ties when rounding, so it must match upstream exactly.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final, Self

import pytest

from priml.baselines.convextok import candidates
from priml.baselines.convextok.candidates import count_candidates
from priml.lib.custom_json import DictCodec, ListCodec, loads


_CWD: Final = Path(__file__).resolve().parent


def test_candidates_match_upstream_golden() -> None:
    pretokens = _read_json("pretokens.json")
    golden = _read_json("candidates.json")
    counts = dict(
        zip(
            ListCodec.coerce(pretokens.get("pretokens"), str, default=None),
            ListCodec.coerce(pretokens.get("frequencies"), int, default=None),
            strict=True,
        ),
    )
    candidates = count_candidates(counts)
    assert list(candidates) == ListCodec.coerce(golden.get("tokens"), str, default=None)
    assert list(candidates.values()) == ListCodec.coerce(
        golden.get("counts"),
        int,
        default=None,
    )


def test_occurrences_are_weighted_by_pretoken_frequency() -> None:
    candidates = count_candidates({"abab": 1, "ba": 3})
    assert candidates == {"ab": 2, "ba": 4}


def test_single_occurrences_are_dropped_unless_repeated() -> None:
    assert count_candidates({"abc": 1}) == {}
    assert count_candidates({"abc": 2}) == {"ab": 2, "abc": 2, "bc": 2}


def test_order_is_python_string_order_of_the_byte_level_spelling() -> None:
    candidates = count_candidates({"Ġzz": 2, "zz": 2, "ab": 2})
    assert list(candidates) == sorted(candidates)
    assert list(candidates)[:3] == ["ab", "zz", "Ġz"]


def test_partitions_merge_to_the_same_counts() -> None:
    counts = {"abab": 1, "ba": 3, "bab": 2, "Ġab": 5}
    assert count_candidates(counts, num_partitions=3) == count_candidates(counts)


def test_threshold_is_inclusive_and_applies_after_all_chunks() -> None:
    candidates = count_candidates(
        {"abab": 1, "baba": 1},
        min_count=3,
        chunk_size=1,
        num_partitions=3,
    )

    assert candidates == {"ab": 3, "ba": 3}


def test_default_chunk_size_and_partition_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_shapes: list[tuple[int, int]] = []

    def count_chunk(
        pretokens: list[tuple[str, int]],
        *,
        num_partitions: int,
    ) -> list[dict[str, int]]:
        chunk_shapes.append((len(pretokens), num_partitions))
        return [{} for _ in range(num_partitions)]

    monkeypatch.setattr(
        candidates,
        "count_substrings",
        count_chunk,
    )
    pretokens = {chr(0x10000 + index): 1 for index in range(100_001)}

    assert count_candidates(pretokens) == {}
    assert chunk_shapes == [(100_000, 1), (1, 1)]


def test_default_counting_stays_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_process_pool(max_workers: int) -> object:
        del max_workers
        pytest.fail("the default count must stay in-process")

    monkeypatch.setattr(
        candidates,
        "ProcessPoolExecutor",
        unexpected_process_pool,
    )
    assert count_candidates({"abc": 2}) == {"ab": 2, "abc": 2, "bc": 2}


def test_process_pool_path_maps_counts_and_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_args: list[int | None] = []
    map_calls: list[tuple[Callable[[object], object], list[object]]] = []

    class InlinePool:
        def __init__(self, max_workers: int | None) -> None:
            pool_args.append(max_workers)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def map(
            self,
            function: Callable[[object], object],
            values: Iterable[object],
        ) -> list[object]:
            items = list(values)
            map_calls.append((function, items))
            return [function(item) for item in items]

    monkeypatch.setattr(
        candidates,
        "ProcessPoolExecutor",
        InlinePool,
    )
    result = count_candidates(
        {"abab": 1, "ba": 3, "bab": 2, "Ġab": 2},
        min_count=2,
        num_workers=3,
        chunk_size=2,
        num_partitions=2,
    )

    assert pool_args == [3]
    assert len(map_calls) == 2
    assert [len(values) for _, values in map_calls] == [2, 2]
    assert result == count_candidates(
        {"abab": 1, "ba": 3, "bab": 2, "Ġab": 2},
        min_count=2,
        chunk_size=2,
        num_partitions=2,
    )


@pytest.mark.cli_python_subprocess
def test_worker_processes_match_serial_counts() -> None:
    pretokens = _read_json("pretokens.json")
    counts = dict(
        zip(
            ListCodec.coerce(pretokens.get("pretokens"), str, default=None),
            ListCodec.coerce(pretokens.get("frequencies"), int, default=None),
            strict=True,
        ),
    )
    parallel = count_candidates(counts, num_workers=2, chunk_size=7, num_partitions=4)
    assert list(parallel.items()) == list(count_candidates(counts).items())


def _read_json(name: str) -> dict[str, object]:
    return dict(
        DictCodec.coerce(
            loads((_CWD / "testdata" / name).read_text()),
            default=None,
        ),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
