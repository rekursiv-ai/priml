"""Candidate tokens must match upstream ConvexTok's: strings, counts, and order.

Every substring of two or more bytes is a candidate, weighted by how often its
pretoken occurs, and kept only if its total reaches two. The kept list is in
Python string order; that order numbers the linear program's token variables
and breaks ties when rounding, so it must match upstream exactly.
"""

from pathlib import Path
from typing import Final, cast

import json

import pytest

from priml.baselines.convextok.candidates import count_candidates
from priml.lib.custom_json import DictCodec, ListCodec


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
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
