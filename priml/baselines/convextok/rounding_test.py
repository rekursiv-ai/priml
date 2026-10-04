"""Rounding must choose exactly the vocabulary upstream ConvexTok chooses from a solution.

Upstream ranks only the candidates with a positive indicator, with Python's stable
sort, so ties keep candidate order. Given upstream's own solution, each scheme
must reproduce upstream's vocabulary set exactly.
"""

from pathlib import Path
from typing import Final, cast

import json

from tokenizers import pre_tokenizers

import numpy as np
import pytest
import torch

from priml.baselines.convextok.rounding import (
    RoundingFn,
    biased_rounding,
    deterministic_rounding,
    integral_rounding,
)
from priml.lib.custom_json import DictCodec, IntCodec, ListCodec


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("rounding", "scheme"),
    [
        (deterministic_rounding, "det"),
        (biased_rounding, "bias"),
        (integral_rounding, "all_ones"),
    ],
)
def test_rounding_matches_upstream_golden(
    rounding: RoundingFn,
    scheme: str,
) -> None:
    candidates = ListCodec.coerce(
        _read_json("candidates.json").get("tokens"),
        str,
        default=None,
    )
    budget = IntCodec.coerce(_read_json("corpus.json").get("budget"), default=None)
    with _npz("program.npz") as program:
        token_edges, byte_edges, _ = (
            int(size) for size in torch.from_numpy(program["sizes"])
        )
    # cuOpt's solution of the fixture LP on upstream's default path (PSLP presolve).
    with _npz("cuopt_solution.npz") as solved:
        solution = torch.from_numpy(solved["x"])
    chosen = rounding(solution[token_edges + byte_edges :], candidates, budget=budget)
    vocabulary = {candidates[int(position)] for position in chosen}
    vocabulary |= set(pre_tokenizers.ByteLevel.alphabet())
    golden = ListCodec.coerce(_read_json("vocab.json").get(scheme), str, default=None)
    assert vocabulary == set(golden)


def test_deterministic_ties_keep_candidate_order() -> None:
    indicators = torch.tensor([0.5, 1.0, 0.5, 0.5], dtype=torch.float64)
    chosen = deterministic_rounding(indicators, ["a", "b", "c", "d"], budget=2)
    assert chosen.tolist() == [0, 1]


def test_many_ties_keep_candidate_order_at_budget_boundary() -> None:
    indicators = torch.ones(32, dtype=torch.float64)
    chosen = deterministic_rounding(indicators, [str(i) for i in range(32)], budget=16)
    assert chosen.tolist() == list(range(16))


def test_only_positive_indicators_are_ranked() -> None:
    indicators = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert deterministic_rounding(indicators, ["a", "b", "c"], budget=2).tolist() == [2]
    assert biased_rounding(indicators, ["a", "b", "c"], budget=2).tolist() == [2]


def test_biased_rounding_divides_by_length() -> None:
    indicators = torch.tensor([0.6, 1.0], dtype=torch.float64)
    assert biased_rounding(indicators, ["ab", "abcd"], budget=1).tolist() == [0]
    assert deterministic_rounding(indicators, ["ab", "abcd"], budget=1).tolist() == [1]


def test_integral_rounding_keeps_values_from_point_nine_nine() -> None:
    indicators = torch.tensor([0.99, 0.989_999, 1.0, 0.0], dtype=torch.float64)
    chosen = integral_rounding(indicators, ["a", "b", "c", "d"], budget=1)
    assert chosen.tolist() == [0, 2]


def _npz(name: str) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(_CWD / "testdata" / name))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
