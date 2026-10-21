"""Checks for the fusion module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import json
import math
import sys

from pyarrow import Table, parquet
from treekle import ReadError, to_plain

import numpy as np
import pytest

from priml.baselines.casmi2026.custom_types import GapGraph, MetricIdentity
from priml.baselines.casmi2026.fusion import (
    FragmentGap,
    FusionAudit,
    FusionInputs,
    FusionPolicy,
    GapGraphTable,
    RankFusion,
    ScoreColumn,
    SimulatorFallback,
    final_fusion,
    finite_score,
    fragment_levels,
    guarded_z,
    insert_engine_second,
    insert_second,
    merge,
    promote,
    rerank_groups,
    safe_extend,
    simulator_candidates,
    standardized,
)
from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, SIMULATOR_RDKIT


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from priml.baselines.casmi2026.custom_types import GapSpectrum


def identity(candidate: str) -> str | None:
    """Model equal metric identities with differently spelled representatives."""
    return None if candidate == "invalid" else candidate.lower()


def vote(*, alpha: float = 0.6, offset: float = 3.0, window: int = 40) -> RankFusion:
    """Build the notebook's second-engine vote with optional overrides."""
    config = RankFusion.Config()
    config.alpha, config.offset, config.window = alpha, offset, window
    return config.make()


def test_merge_and_promotion_match_notebook_self_test() -> None:
    base, keys = ["A", "B", "C"], ["ka", "kb", "kc"]
    pc, pc_keys = ["X", "Y"], ["kx", "ky"]
    slots = [2, 4]
    assert merge(base, keys, pc, pc_keys, slots=slots, limit=25) == [
        "A",
        "X",
        "B",
        "Y",
        "C",
    ]
    assert promote(base, keys, pc, pc_keys, slots=slots, limit=25) == [
        "X",
        "A",
        "B",
        "Y",
        "C",
    ]
    assert merge(base, keys, [], [], slots=slots, limit=25) == base
    assert promote(base, keys, [], [], slots=slots, limit=25) == base


def test_metric_fusion_compacts_duplicate_votes_before_ranking() -> None:
    candidates, scores, collapsed = vote().metric(
        ["A", "a", "invalid", "B"],
        ["b", "C", "c"],
        identity=identity,
    )
    assert candidates == ["B", "A", "C"]
    assert scores == [1 / 5 + 0.6 / 4, 1 / 4, 0.6 / 5]
    assert collapsed == 2


def test_safe_extend_keeps_first_valid_identity_and_order() -> None:
    assert safe_extend(
        ["A", "invalid", "a", "B"],
        ["b", "C"],
        identity=identity,
        limit=25,
    ) == ["A", "B", "C"]


def test_engine_insertion_uses_engine_key_and_preserves_top_one() -> None:
    assert insert_engine_second(
        ["A", "B", "C", "D"],
        ["c"],
        key=identity,
        limit=25,
    ) == ["A", "c", "B", "D"]
    assert insert_engine_second(["A", "B"], ["a"], key=identity, limit=25) == [
        "A",
        "B",
    ]
    assert insert_engine_second(["CCO"], ["B"], key=identity, limit=25) == ["CCO"]
    assert insert_engine_second(
        ["A", "B"],
        ["invalid", "C"],
        key=identity,
        limit=25,
    ) == ["A", "B"]


def test_tail_limit_and_exhausted_base() -> None:
    assert safe_extend(["A"], ["B", "C"], identity=identity, limit=2) == ["A", "B"]
    assert merge([], [], ["B", "C"], ["b", "c"], slots=[], limit=25) == ["B", "C"]


@pytest.mark.parametrize("limit", [0, -1])
def test_every_ranking_entry_rejects_nonpositive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        _ = merge([], [], [], [], slots=[], limit=limit)
    with pytest.raises(ValueError, match="limit"):
        _ = promote([], [], [], [], slots=[], limit=limit)
    with pytest.raises(ValueError, match="window"):
        _ = vote(window=limit).metric([], [], identity=identity)
    with pytest.raises(ValueError, match="limit"):
        _ = safe_extend([], [], identity=identity, limit=limit)
    with pytest.raises(ValueError, match="limit"):
        _ = insert_engine_second([], [], key=identity, limit=limit)


def rerank(column: ScoreColumn, *, top_n: int = 9, min_covered: int = 2) -> list[int]:
    """Rerank nine candidates in formula groups a {0,2,5}, b {1,4,6} and d {7,8}."""
    return rerank_groups(
        [str(index) for index in range(9)],
        [f"k{index}" for index in range(9)],
        [9.0, 8.0, 5.0, 6.0, 4.0, 3.0, 0.0, 2.0, 1.0],
        ["a", "b", "a", "c", "b", "a", "b", "d", "d"],
        columns=[column],
        weights=[2.0],
        top_n=top_n,
        min_covered=min_covered,
        tie_eps=1e-9,
    )


def test_reranking_permutes_only_covered_same_formula_groups() -> None:
    # Group a reads 0, k2 and 5; group b reads k1 and 4; group d has no scores.
    covered: ScoreColumn = {
        "0": 0.0,
        "k2": 1.0,
        "5": 5.0,
        "1": None,
        "k1": 0.0,
        "4": 5.0,
    }
    # Base z (1.09, -0.22, -0.87) plus twice z (-0.76, -0.38, 1.13) ranks 5, 0, 2;
    # base z (1, 0, -1) plus twice (-0.71, 0.71, 0) ranks 4, 1, 6.
    assert rerank(covered) == [5, 4, 0, 3, 1, 2, 6, 7, 8]
    assert rerank(covered, top_n=5) == [2, 4, 0, 3, 1, 5, 6, 7, 8]
    assert rerank(covered, min_covered=3) == [5, 1, 0, 3, 4, 2, 6, 7, 8]
    assert rerank([5.0, None, 0.0, None, 8.0, 9.0]) == [0, 1, 5, 3, 4, 2, 6, 7, 8]
    identity = list(range(9))
    assert rerank({}) == rerank({"0": None, "2": None}) == identity
    assert rerank({"0": math.nan, "2": math.inf, "5": 1.0}) == identity
    second: ScoreColumn = {"0": 8.0, "2": 4.0, "5": 0.0, "1": "bad", "6": "6"}
    for epsilon in (0.0, 1e-9):
        assert rerank_groups(
            [str(index) for index in range(9)],
            [f"k{index}" for index in range(9)],
            [9.0, 8.0, 5.0, 6.0, 4.0, 3.0, 0.0, 2.0, 1.0],
            ["a", "b", "a", "c", "b", "a", "b", "d", "d"],
            columns=[covered, second],
            weights=[2.0, 0.5],
            top_n=60,
            min_covered=2,
            tie_eps=epsilon,
        ) == [5, 4, 0, 3, 1, 2, 6, 7, 8]
    assert not rerank_groups(
        [],
        [],
        [],
        [],
        columns=[{}],
        weights=[0.5],
        top_n=60,
        min_covered=2,
        tie_eps=1e-9,
    )


def test_scores_standardize_finite_values_and_admit_repeated_formulas() -> None:
    assert [finite_score(value) for value in (None, "bad", "2.5", math.inf, 0.0)] == [
        None,
        None,
        2.5,
        None,
        0.0,
    ]
    assert standardized([]) == []
    assert standardized([1.0, None, math.nan, math.inf]) == [0.0] * 4
    assert standardized([2.0, 2.0, None]) == [0.0] * 3
    # The finite scores 0, 2 and 4 have mean 2 and sample deviation 2.
    assert standardized([0.0, None, 2.0, 4.0, math.nan, math.inf]) == [
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    candidates = [str(index) for index in range(9)]
    formulas = ["a", "b", "a", "c", "b", "a", "b", "d", "d"]
    assert simulator_candidates(candidates, formulas, top_n=5) == ["0", "1", "2", "4"]
    assert simulator_candidates(candidates, formulas, top_n=9) == [
        "0",
        "1",
        "2",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    assert simulator_candidates(["0", "0", "2"], ["a"] * 3, top_n=40) == ["0", "2"]


def test_guarded_z_centers_by_population_deviation_and_tolerates_a_flat_group() -> None:
    np.testing.assert_array_equal(
        guarded_z(np.array([1.0, 3.0])),
        np.array([-1.0, 1.0]) / (1.0 + 1e-9),
    )
    np.testing.assert_array_equal(guarded_z(np.full(3, 2.5)), np.zeros(3))


def test_each_simulator_requires_one_weight() -> None:
    with pytest.raises(ValueError, match="one weight"):
        _ = rerank_groups(
            [],
            [],
            [],
            [],
            columns=[{}],
            weights=[],
            top_n=60,
            min_covered=2,
            tie_eps=1e-9,
        )


def test_score_lists_and_keys_are_optional() -> None:
    assert rerank_groups(
        ["a", "b"],
        None,
        [1.0, 0.0],
        ["a", "a"],
        columns=[[0.0, 1.0], None],
        weights=[2.0, 0.5],
        top_n=60,
        min_covered=2,
        tie_eps=1e-9,
    ) == [1, 0]


def fuse(
    molecule_id: str,
    original: Sequence[str],
    *,
    prepared: Prepared,
    prefusion: Sequence[str] | None,
    engine: Sequence[str] | None,
    ranking: RankFusion | None = None,
    fallback: bool = True,
    ice: Mapping[str, float | None] | None = None,
    glacier: Mapping[str, float | None] | None = None,
    gap: Mapping[str, float | None] | None = None,
    processed: bool = True,
) -> tuple[list[str], FusionAudit]:
    """Apply the notebook's final policy to prepared identities."""
    return final_fusion(
        molecule_id,
        original,
        prefusion=prefusion,
        engine=engine,
        identity=prepared.identity,
        key=prepared.key,
        formula=prepared.formula,
        vote=vote() if ranking is None else ranking,
        fallback=SimulatorFallback.Config(gap_weight=0.5).make() if fallback else None,
        length=25,
        min_covered=2,
        tie_eps=1e-9,
        ice_weight=1.0,
        glacier_weight=1.0,
        ice=ice,
        glacier=glacier,
        gap=gap,
        processed=processed,
    )


def test_printed_policy_keeps_legacy_control_and_final_votes_independent() -> None:
    config = FusionPolicy.Config()
    config.final.alpha, config.final.offset, config.final.window = 2.0, 3.0, 40
    config.control.alpha, config.control.offset, config.control.window = 0.6, 3.0, 40
    config.legacy.alpha, config.legacy.offset, config.legacy.window = 3.0, 3.0, 1
    policy = config.make()
    options: FusionInputs = {
        "prefusion": ["a", "b"],
        "engine": ["b", "a"],
        "identity": lambda value: (value, ()),
        "key": lambda value: value,
        "formula": lambda value: value,
        "ice": {},
        "glacier": {},
        "gap": {},
    }
    assert policy.fuse("m", ["a", "b"], options=options)[0] == ["b", "a"]
    assert policy.control("m", ["a", "b"], options=options)[0] == ["a", "b"]
    assert policy.legacy(["a"], ["b"], ["b"], key=lambda value: value) == (
        ["b"],
        ["b"],
        [0.75],
    )


def test_fallback_is_a_slot_that_only_the_final_policy_fills() -> None:
    config = FusionPolicy.Config()
    config.fallback = SimulatorFallback.Config()
    formulas = {"a": "same", "b": "same"}
    options: FusionInputs = {
        "prefusion": ["a", "b"],
        "engine": ["a", "b"],
        "identity": lambda value: (value, ()),
        "key": lambda value: value,
        "formula": formulas.__getitem__,
        "ice": {"a": 0.0, "b": 1.0},
        "glacier": None,
        "gap": None,
    }
    policy = config.make()
    assert policy.fuse("m", ["a", "b"], options=options)[1]["branch"] == "ice_fallback"
    assert policy.control("m", ["a", "b"], options=options)[1]["branch"] == "unrescored"
    config.fallback = None
    assert config.make().fuse("m", ["a", "b"], options=options)[1]["branch"] == (
        "unrescored"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Prepared:
    """Supply every reached independent identity/key/formula with strict lookup."""

    names: tuple[str, ...]
    identities: dict[str, MetricIdentity | None]
    keys: dict[str, str | None]
    formulas: dict[str, str]

    def identity(self, value: str) -> MetricIdentity | None:
        return self.identities[value]

    def key(self, value: str) -> str | None:
        return self.keys[value]

    def formula(self, value: str) -> str:
        return self.formulas[value]


def fusion_inputs() -> Prepared:
    """Stage aliases, a same-formula trio, an invalid string and the placeholder."""
    rows: dict[str, tuple[MetricIdentity | None, str | None, str]] = {
        "OCC": (("ethanol", (("C", 2), ("O", 1))), "LFQ", "C2H6O"),
        "C(C)O": (("ethanol", (("C", 2), ("O", 1))), "LFQ", "C2H6O"),
        "COC": (("ether", (("C", 2), ("O", 1))), "LCG", "C2H6O"),
        "NCC": (("ethylamine", (("C", 2), ("N", 1))), "QUS", "C2H7N"),
        "C(C)N": (("ethylamine", (("C", 2), ("N", 1))), "QUS", "C2H7N"),
        "c1ccccc1": (("benzene", (("C", 6),)), "UHO", "C6H6"),
        "invalid": (None, None, "invalid"),
        "OC(C)": (("isomer", (("C", 2), ("O", 1))), "XYZ", "C2H6O"),
        "CCO": (("ethanol", (("C", 2), ("O", 1))), "LFQ", "C2H6O"),
        "": (None, None, ""),
    }
    return Prepared(
        names=tuple(rows)[:8],
        identities={name: row[0] for name, row in rows.items()},
        keys={name: row[1] for name, row in rows.items()},
        formulas={name: row[2] for name, row in rows.items()},
    )


PREFUSION: Final = (
    "OCC",
    "COC",
    "C(C)O",
    "NCC",
    "c1ccccc1",
    "invalid",
    "OC(C)",
    "CCO",
    "",
)
ENGINE: Final = ("NCC", "COC", "OCC")
ORIGINAL: Final = ("OCC", "COC", "NCC", "OC(C)", "C(C)N")
# Votes 0.35, 0.32, 0.317, 0.143 and 0.125 after the alias C(C)O collapses.
UNRESCORED: Final = ("OCC", "COC", "NCC", "c1ccccc1", "OC(C)")
# Within C2H6O, base z (0.70, 0.45, -1.15) plus z (-0.58, -0.58, 1.15) ranks OC(C)
# above COC; half that weight cannot.
RERANKED: Final = ("OCC", "OC(C)", "NCC", "c1ccccc1", "COC")


def trio() -> dict[str, float | None]:
    """Score only the lowest-voted C2H6O member."""
    return {"OCC": 0.0, "COC": 0.0, "OC(C)": 1.0}


@pytest.mark.parametrize(
    ("ice", "glacier", "gap", "fallback", "branch", "expected"),
    [
        ({}, {}, None, True, "unrescored", UNRESCORED),
        ({"OCC": 0.1}, trio(), None, False, "glacier", RERANKED),
        ({}, trio(), None, True, "unrescored", UNRESCORED),
        (trio(), {}, None, True, "ice_fallback", RERANKED),
        (trio(), {}, None, False, "unrescored", UNRESCORED),
        (
            {"LFQ": 0.0, "LCG": 0.0, "XYZ": 1.0},
            {},
            None,
            True,
            "ice_fallback",
            RERANKED,
        ),
        ({"OCC": None}, {}, None, True, "ice_fallback", UNRESCORED),
        ({"OCC": None}, {}, trio(), True, "frag_gap", UNRESCORED),
        (
            {"OCC": None},
            {"OCC": math.nan, "COC": math.inf},
            None,
            True,
            "glacier",
            UNRESCORED,
        ),
    ],
)
def test_final_policy_selects_its_branch_and_reranks_the_same_formula_trio(
    *,
    ice: dict[str, float | None],
    glacier: dict[str, float | None],
    gap: dict[str, float | None] | None,
    fallback: bool,
    branch: str,
    expected: tuple[str, ...],
) -> None:
    prepared = fusion_inputs()
    candidates, audit = fuse(
        "m",
        ORIGINAL,
        prepared=prepared,
        prefusion=PREFUSION,
        engine=ENGINE,
        fallback=fallback,
        ice=ice,
        glacier=glacier,
        gap=gap,
    )
    assert candidates == list(expected)
    assert audit["branch"] == branch
    assert (audit["candidate_count"], audit["collapsed_source_votes"]) == (5, 1)
    assert audit["fallback_changed"] == (
        branch == "ice_fallback" and expected == RERANKED
    )
    assert not audit["fallback_changed_top1"]
    # Engine 2 leads with NCC, which moves to rank two below the fused leader.
    final, inserted = insert_second(candidates, ENGINE, key=prepared.key, limit=25)
    assert inserted
    assert final[:2] == ["OCC", "NCC"]
    assert sorted(final) == sorted(candidates)


def test_unprocessed_rows_and_absent_engines_keep_the_original_list() -> None:
    prepared = fusion_inputs()
    for processed, engine, branch in (
        (False, ENGINE, "placeholder"),
        (True, None, "no_engine"),
    ):
        candidates, audit = fuse(
            "m",
            ORIGINAL,
            prepared=prepared,
            prefusion=PREFUSION,
            engine=engine,
            processed=processed,
        )
        assert candidates == list(ORIGINAL)
        assert audit["branch"] == branch
        assert audit["processed"] == processed
    _, covered = fuse(
        "m",
        ORIGINAL,
        prepared=prepared,
        prefusion=PREFUSION,
        engine=ENGINE,
        ice={"LFQ": 0.0, "COC": 0.5},
        glacier=trio(),
    )
    assert (covered["ice_finite_count"], covered["gl_finite_count"]) == (2, 3)


def test_insertion_flag_is_true_when_rank_two_is_already_occupied() -> None:
    prepared = fusion_inputs()
    first, second = prepared.names[0], prepared.names[2]
    assert insert_second([first, second], [second], key=prepared.key, limit=25) == (
        [first, second],
        True,
    )
    assert insert_second([first, second], [first], key=prepared.key, limit=25) == (
        [first, second],
        False,
    )
    assert insert_second(["CCO"], [second], key=prepared.key, limit=25) == (
        ["CCO"],
        False,
    )
    assert insert_second(
        [first],
        ["invalid", second],
        key=prepared.key,
        limit=25,
    ) == ([first], False)
    assert insert_second([], [second], key=prepared.key, limit=25) == ([], False)
    assert insert_second([first], [], key=prepared.key, limit=25) == ([first], False)


def test_final_policy_keeps_missing_staging_and_window_boundaries() -> None:
    prepared = fusion_inputs()
    with pytest.raises(ValueError, match="window must be positive"):
        fuse(
            "m",
            [],
            prepared=prepared,
            prefusion=["absent"],
            engine=prepared.names[:2],
            ranking=vote(window=0),
        )
    with pytest.raises(KeyError, match="absent"):
        fuse(
            "m",
            [],
            prepared=prepared,
            prefusion=["absent"],
            engine=prepared.names[:2],
        )
    with pytest.raises(KeyError, match="absent"):
        vote().legacy(["absent"], [], [], key=prepared.key)
    with pytest.raises(KeyError, match="absent"):
        insert_second(prepared.names[:1], ["absent"], key=prepared.key, limit=25)


def test_legacy_cap_ties_missing_keys_and_both_unequal_vote_lengths() -> None:
    names = [f"candidate/{index}" for index in range(42)]
    candidates, keys, scores = vote().legacy(names, [], [], key=lambda value: value)
    assert candidates == keys == names[:40]
    assert len(scores) == 40
    assert vote(window=0).legacy(names, [], [], key=lambda value: value) == (
        [],
        [],
        [],
    )
    assert vote().legacy(
        [],
        ["empty", "none", "kept", "tail"],
        ["", None, "K"],
        key=lambda value: value,
    ) == (["kept"], ["K"], [0.6 / 6.0])
    assert vote().legacy(
        [],
        ["kept"],
        ["K", "unused"],
        key=lambda value: value,
    ) == (["kept"], ["K"], [0.6 / 4.0])
    assert vote(alpha=1.0).legacy(
        ["first"],
        ["second"],
        ["second"],
        key=lambda value: value,
    ) == (["first", "second"], ["first", "second"], [0.25, 0.25])
    assert vote(offset=9007199254740992.0).legacy(
        [],
        ["skip", "kept"],
        [None, "K"],
        key=lambda value: value,
    ) == (["kept"], ["K"], [0.6 / 9007199254740994.0])


def test_full_eighty_candidate_window_can_promote_a_late_same_formula_member() -> None:
    names = [f"candidate/{index}" for index in range(80)]
    identities: dict[str, MetricIdentity | None] = {
        value: (value, ()) for value in names
    }
    keys: dict[str, str | None] = {value: value for value in names}
    formulas = {value: value for value in names}
    formulas.update({names[index]: "same" for index in (0, 40, 79)})
    prepared = Prepared(
        names=tuple(names),
        identities=identities,
        keys=keys,
        formulas=formulas,
    )
    for window, first in ((40, names[0]), (80, names[79])):
        candidates, audit = fuse(
            "wide",
            [],
            prepared=prepared,
            prefusion=names,
            engine=names[:1],
            ice={names[0]: -1.0, names[40]: 0.0, names[79]: 1.0},
            ranking=vote(window=window, alpha=0.0, offset=1e300),
        )
        assert candidates == [first, *names[1:25]]
        assert audit["candidate_count"] == window
        assert audit["ice_finite_count"] == (1 if window == 40 else 3)
        assert audit["fallback_changed"] == (window == 80)
        assert audit["fallback_changed_top1"] == (window == 80)


def test_insertion_caps_deduplicates_and_keeps_placeholder_tail() -> None:
    names = [f"candidate/{index}" for index in range(30)]
    candidates, inserted = insert_second(
        names,
        ["proposal"],
        key=lambda value: value,
        limit=25,
    )
    assert candidates == [names[0], "proposal", *names[1:24]]
    assert inserted
    keys = {"first": "A", "CCO": "B", "OCC": "B"}
    assert insert_second(
        ["", "first", "CCO", "OCC", "OCC"],
        ["", "CCO", "OCC"],
        key=keys.__getitem__,
        limit=25,
    ) == (["first", "OCC", "CCO"], True)


def test_absent_lists_empty_fusion_and_deferred_formula_staging() -> None:
    prepared = fusion_inputs()
    original = ["CCO"]
    for prefusion, engine in ((None, prepared.names[:1]), ((), ())):
        candidates, audit = fuse(
            "missing",
            original,
            prepared=prepared,
            prefusion=prefusion,
            engine=engine,
        )
        assert candidates == original
        assert candidates is not original
        assert audit["branch"] == "no_engine"
    candidates, audit = fuse(
        "invalid",
        original,
        prepared=prepared,
        prefusion=["invalid"],
        engine=["invalid"],
        ice={"invalid": None},
    )
    assert candidates == original
    assert audit["candidate_count"] == 0
    assert audit["branch"] == "ice_fallback"
    assert not audit["fallback_changed"]
    missing: dict[str, str] = {}
    deferred = Prepared(
        names=prepared.names,
        identities=prepared.identities,
        keys=prepared.keys,
        formulas=missing,
    )
    candidates, audit = fuse(
        "deferred",
        [],
        prepared=deferred,
        prefusion=[],
        engine=prepared.names[:1],
    )
    assert candidates == list(prepared.names[:1])
    assert audit["branch"] == "unrescored"
    with pytest.raises(KeyError, match=prepared.names[0]):
        fuse(
            "reached",
            [],
            prepared=deferred,
            prefusion=[],
            engine=prepared.names[:1],
            ice={prepared.names[0]: None},
        )


def test_cleavage_levels_skip_unsplit_rings_and_cap_long_chains() -> None:
    triangle = GapGraph(
        weights=np.array([1.0, 2.0, 4.0]),
        bonds=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        pieces=1,
    )
    levels = fragment_levels(triangle, max_bonds=[100])
    assert levels is not None
    # One cut leaves the ring whole; two cuts free one atom and the bonded pair.
    assert [level.tolist() for level in levels] == [
        [7.0],
        [],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    chain = GapGraph(
        weights=np.full(102, 14.0),
        bonds=np.column_stack((np.arange(101), np.arange(1, 102))),
        pieces=1,
    )
    # 101 bonds exceed the 100-bond ceiling, so enumeration stops at one cut.
    levels = fragment_levels(chain, max_bonds=[100])
    assert levels is not None
    assert levels[1].tolist() == [14.0 * count for count in range(1, 102)]
    assert len(levels[2]) == 0


def test_gap_credits_each_peak_by_its_least_cleaved_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    proton = 1.00782503207 - 0.00054857990
    graphs = _Graphs(
        {
            "water": GapGraph(
                weights=np.array([16.0, 2.0]),
                bonds=np.array([[0, 1]], dtype=np.int64),
                pieces=1,
            ),
            "invalid": None,
        },
    )
    # The intact 18 and the one-cut 16 explain the first two peaks; 100 is unexplained.
    spectrum: GapSpectrum = {
        "mz": [18.0 + proton, 16.0 + proton, 100.0, math.nan],
        "it": [1.0, 1.0, 2.0, 5.0],
        "adduct": "[M+H]+",
        "prec": 0.0,
        "mode": 1,
    }
    halves = FragmentGap.Config(level_weights=[0.5, 0.25]).make()
    candidates = ["water", "invalid", "water"]
    assert halves.score([spectrum], candidates, graphs=graphs) == [0.375, None, 0.375]
    silent: GapSpectrum = {**spectrum, "it": [0.0, -1.0, 0.0, 0.0]}
    assert halves.score([silent], ["water"], graphs=graphs) == [0.0]
    # The explained peaks alone credit (1 + 0.5) / 2; spectra average to 0.5625.
    explained: GapSpectrum = {**spectrum, "mz": spectrum["mz"][:2], "it": [1.0, 1.0]}
    assert halves.score([spectrum, explained], ["water"], graphs=graphs) == [0.5625]
    # Weighting by m/z over the largest mass favours the fully explained heavier peak.
    heavy = FragmentGap.Config(level_weights=[0.5, 0.25], mz_power=1.0).make()
    (score,) = heavy.score([explained], ["water"], graphs=graphs)
    heavier, lighter = 18.0 + proton, 16.0 + proton
    # The closed form rounds 1 ulp below the exact rational credit; the score is exact.
    assert score == pytest.approx(
        (heavier + 0.5 * lighter) / (heavier + lighter),
        rel=2**-52,
        abs=0,
    )


def test_default_caps_rounding_and_minimum_cleavage_order() -> None:
    graph = GapGraph(
        weights=np.array([0.1, 0.100003, 0.1000001]),
        bonds=np.array([[0, 1], [0, 2]], dtype=np.int64),
        pieces=1,
    )
    levels = fragment_levels(graph, max_bonds=[100])
    assert levels is not None
    np.testing.assert_array_equal(levels[0], [graph.weights.sum()])
    np.testing.assert_array_equal(levels[1], [0.100003, 0.1 + 0.1000001])
    assert len(levels[2]) == 0
    disconnected = GapGraph(
        weights=np.array([1.0, 2.0, 4.0]),
        bonds=np.array([[0, 1]], dtype=np.int64),
        pieces=2,
    )
    levels = fragment_levels(disconnected, max_bonds=[100])
    assert levels is not None
    np.testing.assert_array_equal(levels[0], [3.0, 4.0, 7.0])
    np.testing.assert_array_equal(levels[1], [1.0, 2.0])
    star = GapGraph(
        weights=np.array([1000.0, 1.0, 10.0, 100.0]),
        bonds=np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64),
        pieces=1,
    )
    levels = fragment_levels(star, max_bonds=[100, 3])
    assert levels is not None
    np.testing.assert_array_equal(levels[3], [1000.0])
    limited = fragment_levels(star, max_bonds=[100, 2])
    assert limited is not None
    assert len(limited[3]) == 0
    large = GapGraph(
        weights=np.arange(1, 43, dtype=np.float64),
        bonds=np.column_stack((np.zeros(41, dtype=np.int64), np.arange(1, 42))),
        pieces=1,
    )
    levels = fragment_levels(large, max_bonds=[100, 40])
    assert levels is not None
    assert len(levels[3]) == 0


def test_carrier_fallbacks_invalid_configs_and_explicit_coverage() -> None:
    graphs = _Graphs(
        {
            "candidate": GapGraph(
                weights=np.array([17.0]),
                bonds=np.empty((0, 2), dtype=np.int64),
                pieces=1,
            ),
            "invalid": None,
        },
    )
    config = FragmentGap.Config()
    config.h_shifts = [0]
    config.tol = 0.0
    model = config.make()
    proton = 1.00782503207 - 0.00054857990
    for adduct, mode, carrier in (
        ("unknown-", 1, -proton),
        ("unknown+", -1, proton),
        (None, -1, -proton),
        (None, 0, proton),
    ):
        spectra: list[GapSpectrum] = [
            {
                "mz": [17.0 + carrier],
                "it": [1.0],
                "adduct": adduct,
                "prec": 0.0,
                "mode": mode,
            },
        ]
        assert model.score(
            spectra,
            ["candidate", "invalid", "candidate"],
            graphs=graphs,
        ) == [1.0, None, 1.0]
    with pytest.raises(KeyError, match=r"^'unstaged'$"):
        _ = model.score([], ["unstaged"], graphs=graphs)
    for weights in (list[float](), [1.0, 1.0, 1.0, 1.0]):
        config.level_weights = weights
        with pytest.raises(ValueError, match=r"^level_weights must hold one, two"):
            _ = config.make()
    config.level_weights = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match=r"^max_bonds needs one entry per level"):
        _ = config.make()
    config.level_weights = [1.0, 0.6]
    config.intensity_power = float("nan")
    invalid_weights: list[GapSpectrum] = [{**spectra[0], "it": [2.0]}]
    assert config.make().score(invalid_weights, ["candidate"], graphs=graphs) == [0.0]


class _Graphs:
    def __init__(self, values: dict[str, GapGraph | None]) -> None:
        self.values = values

    def for_smiles(self, smiles: str) -> GapGraph | None:
        """Require complete prepared data even for invalid source inputs."""
        return self.values[smiles]


def test_complete_prepared_rows_keep_first_duplicate_and_need_no_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["mixture", "invalid", "mixture"],
                "valid": [True, False, True],
                "weights": [[1.0, 2.0, 4.0], [], [19.0]],
                "bonds": [[[0, 1]], [], []],
                "pieces": [2, 0, 1],
            },
        ),
        tmp_path / "graphs.parquet",
    )
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )
    table = GapGraphTable.Config(working_dir=tmp_path).make()
    first = table.for_smiles("mixture")
    assert first is not None
    assert first.pieces == 2
    assert first is table.row(0)
    assert first is table.for_smiles("mixture")
    assert table.for_smiles("invalid") is None
    assert table.row(1) is None
    last = table.row(2)
    assert last is not None
    np.testing.assert_array_equal(last.weights, [19.0])
    assert last.bonds.shape == (0, 2)
    assert first.weights.dtype == np.float64
    assert first.bonds.dtype == np.int64
    assert not first.weights.flags.writeable
    assert not first.bonds.flags.writeable
    levels = fragment_levels(first, max_bonds=[100])
    assert levels is not None
    np.testing.assert_array_equal(levels[0], [3.0, 4.0, 7.0])
    np.testing.assert_array_equal(levels[1], [1.0, 2.0])
    assert len(levels[2]) == 0
    with pytest.raises(KeyError, match=r"^'unstaged'$"):
        _ = table.for_smiles("unstaged")
    for index in (-1, 3):
        with pytest.raises(IndexError, match=f"^{index}$"):
            _ = table.row(index)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 2, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(
        ValueError,
        match=r"^Gap graphs require format 1 and RDKit 2026\.03\.3\.$",
    ):
        _ = GapGraphTable.Config(working_dir=tmp_path).make()
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": SIMULATOR_RDKIT})),
    )
    with pytest.raises(
        ValueError,
        match=r"^Gap graphs require format 1 and RDKit 2026\.03\.3\.$",
    ):
        _ = GapGraphTable.Config(working_dir=tmp_path).make()
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": 20_260_303})),
    )
    with pytest.raises(ReadError):
        _ = GapGraphTable.Config(working_dir=tmp_path).make()


@pytest.mark.parametrize(
    ("weights", "bonds", "pieces"),
    [
        ([], [], 1),
        ([1.0], [], 0),
        ([1.0], [], 2),
        ([1.0], [[-1, 0]], 1),
        ([1.0], [[0, 1]], 1),
    ],
)
def test_gap_reader_rejects_inconsistent_declared_graphs(
    tmp_path: Path,
    weights: list[float],
    bonds: list[list[int]],
    pieces: int,
) -> None:
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["inconsistent"],
                "valid": [True],
                "weights": [weights],
                "bonds": [bonds],
                "pieces": [pieces],
            },
        ),
        tmp_path / "graphs.parquet",
    )
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ValueError, match=r"^Inconsistent prepared gap graph\.$"):
        _ = GapGraphTable.Config(working_dir=tmp_path).make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
