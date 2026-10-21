"""Checks for the fusion module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import ast
import json
import math
import sys

from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, from_plain, to_plain

import numpy as np
import pytest
import torch

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
from priml.baselines.casmi2026.testing import float32_comparands
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import GapSpectrum


_BASELINE: Final = _CWD


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


GOLDEN: Final = _BASELINE / "testdata" / "reranking.pt"


def replay() -> dict[str, Tensor]:
    """Exercise all coverage, fallback-key and tie branches."""
    candidates = [str(index) for index in range(9)]
    keys = [f"k{index}" for index in range(9)]
    formulas = ["a", "b", "a", "c", "b", "a", "b", "d", "d"]
    scores = [9.0, 8.0, 5.0, 6.0, 4.0, 3.0, 0.0, 2.0, 1.0]
    covered = {"0": 0.0, "k2": 1.0, "5": 5.0, "1": None, "k1": 0.0, "4": 5.0}
    second = {"0": 8.0, "2": 4.0, "5": 0.0, "1": "bad", "6": "6"}
    cases: list[tuple[str, ScoreColumn, int, int]] = [
        ("complete", covered, 9, 2),
        ("prefix", covered, 5, 2),
        ("minimum", covered, 9, 3),
        ("list", [5.0, None, 0.0, None, 8.0, 9.0], 9, 2),
        ("nonfinite", {"0": math.nan, "2": math.inf, "5": 1.0}, 9, 2),
        ("all_none", {"0": None, "2": None}, 9, 2),
        ("empty", {}, 9, 2),
    ]
    record: dict[str, Tensor] = {}
    arguments = (candidates, keys, scores, formulas)
    for name, column, prefix, minimum in cases:
        record[f"single/{name}"] = torch.tensor(
            rerank_groups(
                *arguments,
                columns=[column],
                weights=[2.0],
                top_n=prefix,
                min_covered=minimum,
                tie_eps=1e-9,
            ),
        )
    record["multi"] = torch.tensor(
        rerank_groups(
            *arguments,
            columns=[covered, second],
            weights=[2.0, 0.5],
            top_n=60,
            min_covered=2,
            tie_eps=1e-9,
        ),
    )
    record["multi/no_grid"] = torch.tensor(
        rerank_groups(
            *arguments,
            columns=[covered, second],
            weights=[2.0, 0.5],
            top_n=60,
            min_covered=2,
            tie_eps=0,
        ),
    )
    record["multi/empty"] = torch.tensor(
        rerank_groups(
            [],
            [],
            [],
            [],
            columns=[],
            weights=[],
            top_n=60,
            min_covered=2,
            tie_eps=1e-9,
        ),
    )
    record["single/empty_rows"] = torch.tensor(
        rerank_groups(
            [],
            [],
            [],
            [],
            columns=[{}],
            weights=[0.5],
            top_n=60,
            min_covered=2,
            tie_eps=1e-9,
        ),
    )
    for name, values in (
        ("empty", list[float | None]()),
        ("one", [1.0, None, math.nan, math.inf]),
        ("constant", [2.0, 2.0, None]),
        ("mixed", [0.0, None, 2.0, 5.0, math.nan, math.inf]),
    ):
        record[f"z/{name}"] = torch.tensor(standardized(values), dtype=torch.float64)
    for prefix in (5, 9):
        record[f"admit/{prefix}"] = torch.tensor(
            [
                int(value)
                for value in simulator_candidates(candidates, formulas, top_n=prefix)
            ],
        )
    record["admit/duplicates"] = torch.tensor(
        [
            int(value)
            for value in simulator_candidates(
                ["0", "0", "2"],
                ["a", "a", "a"],
                top_n=40,
            )
        ],
    )
    opposing = {"0": 0.0, "1": 1.0}
    for epsilon in (0.0, 1e-9):
        record[f"ties/{epsilon}"] = torch.tensor(
            rerank_groups(
                ["0", "1"],
                ["0", "1"],
                [1.0, 0.0],
                ["a", "a"],
                columns=[opposing, opposing],
                weights=[0.5, 0.5],
                top_n=60,
                min_covered=2,
                tie_eps=epsilon,
            ),
        )
    cleaned = [
        finite_score(value) for value in (None, "bad", "2.5", math.inf, math.nan, 0.0)
    ]
    record["clean"] = torch.tensor(
        [-1.0 if value is None else value for value in cleaned],
        dtype=torch.float64,
    )
    return record


def test_all_simulator_reranking_values() -> None:
    assert_tensor_golden(GOLDEN, float32_comparands(replay()))


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


def test_every_fusion_stage_and_all_twenty_final_branches() -> None:
    golden = _BASELINE / "testdata/fusion.pt"
    expected = read_tensors(golden)
    prepared = source_inputs(expected)
    names = prepared.names
    actual: dict[str, Tensor] = {}
    for index, value in enumerate(names):
        prefix = f"identities/{index}"
        actual[prefix + "/input"] = text(value)
        actual[prefix + "/metric"] = text(str(prepared.identity(value)))
        actual[prefix + "/old"] = text(str(prepared.key(value)))
        actual[prefix + "/formula"] = text(prepared.formula(value))
    first = [*names[:7], "CCO", ""]
    second = [names[3], names[2], names[7], names[4]]
    fused, keys, scores = vote().legacy(
        first,
        second,
        [prepared.key(value) for value in second],
        key=prepared.key,
    )
    actual["legacy/smiles"], actual["legacy/keys"] = text(str(fused)), text(str(keys))
    actual["legacy/scores"] = torch.tensor(scores, dtype=torch.float64)
    original = [names[0], names[2], names[3], names[7], names[4]]
    for case in (
        "placeholder",
        "no_engine",
        "unrescored",
        "glacier",
        "gl_without_ice",
        "ice_fallback",
        "all_none",
        "fragment_gap",
        "nonfinite_gl",
        "key_lookup",
    ):
        ice: dict[str, float | None] = {}
        glacier: dict[str, float | None] = {}
        gap: dict[str, float | None] = {}
        if case in {"glacier", "ice_fallback", "key_lookup"}:
            ice = {names[0]: 0.1, names[2]: 0.9, names[3]: 0.8}
        if case in {"glacier", "gl_without_ice"}:
            glacier = {names[0]: 0.0, names[2]: 0.5, names[3]: 1.0}
        if case in {"all_none", "fragment_gap", "nonfinite_gl"}:
            ice = {names[0]: None}
        if case == "nonfinite_gl":
            glacier = {names[0]: math.nan, names[2]: math.inf}
        if case == "fragment_gap":
            gap = {names[0]: 0.0, names[2]: 0.7, names[3]: 1.0}
        if case == "key_lookup":
            ice = {str(prepared.key(value)): score for value, score in ice.items()}
        for fallback in (False, True):
            candidates, audit = fuse(
                "m",
                original,
                prepared=prepared,
                prefusion=first,
                engine=None if case == "no_engine" else second,
                fallback=fallback,
                ice=ice,
                glacier=glacier,
                gap=gap,
                processed=case != "placeholder",
            )
            label = f"{case}/{fallback}"
            actual[label + "/candidates"] = text(str(candidates))
            actual[label + "/audit"] = text(str(audit))
            final, inserted = insert_second(
                candidates,
                second,
                key=prepared.key,
                limit=25,
            )
            actual[label + "/final"] = text(str(final))
            actual[label + "/inserted"] = torch.tensor(inserted)
    assert_tensor_golden(golden, float32_comparands(actual))


def test_insertion_flag_is_true_when_rank_two_is_already_occupied() -> None:
    prepared = source_inputs(read_tensors(_BASELINE / "testdata/fusion.pt"))
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
    prepared = source_inputs(read_tensors(_BASELINE / "testdata/fusion.pt"))
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
    prepared = source_inputs(read_tensors(_BASELINE / "testdata/fusion.pt"))
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


def source_inputs(expected: dict[str, Tensor]) -> Prepared:
    """Load the stored scientific inputs from the fusion golden."""
    names = tuple(decode(expected[f"identities/{index}/input"]) for index in range(8))
    identities: dict[str, MetricIdentity | None] = {"": None}
    keys: dict[str, str | None] = {"": None}
    formulas: dict[str, str] = {"": ""}
    for index, value in enumerate(names):
        metric = decode(expected[f"identities/{index}/metric"])
        identities[value] = (
            None
            if metric == "None"
            else from_plain(
                cast(object, ast.literal_eval(metric)),
                tuple[str, tuple[tuple[str, int], ...]],
            )
        )
        old_key = decode(expected[f"identities/{index}/old"])
        keys[value] = None if old_key == "None" else old_key
        formulas[value] = decode(expected[f"identities/{index}/formula"])
    legacy_values = from_plain(
        cast(object, ast.literal_eval(decode(expected["legacy/smiles"]))),
        list[str],
    )
    legacy_keys = from_plain(
        cast(object, ast.literal_eval(decode(expected["legacy/keys"]))),
        list[str],
    )
    keys.update(
        {
            value: key
            for value, key in zip(legacy_values, legacy_keys, strict=True)
            if value == "CCO"
        },
    )
    return Prepared(names=names, identities=identities, keys=keys, formulas=formulas)


def decode(value: Tensor) -> str:
    """Read an entire stored string without changing its contents."""
    return bytes(from_plain(value.tolist(), list[int])).decode()


def text(value: str) -> Tensor:
    """Encode a native result string as golden bytes."""
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


def test_complete_graph_cleavage_levels_and_fallback_scores_without_rdkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    golden = _BASELINE / "testdata/fragment_gap.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    graphs = _graphs(expected)
    for index, graph in enumerate(graphs.values()):
        prefix = str(index)
        for name in ("input", "valid"):
            actual[prefix + "/" + name] = expected[prefix + "/" + name].clone()
        if graph is not None:
            for name in ("weights", "left", "right", "pieces"):
                actual[prefix + "/" + name] = expected[prefix + "/" + name].clone()
        for name, max_bonds in (
            ("default", [100]),
            ("one", list[int]()),
            ("three", [100, 40]),
            ("caps", [2, 1]),
        ):
            levels = fragment_levels(graph, max_bonds=max_bonds)
            actual[f"{prefix}/{name}/valid"] = torch.tensor(levels is not None)
            if levels is not None:
                for level, values in enumerate(levels):
                    actual[f"{prefix}/{name}/{level}"] = torch.from_numpy(values)
    prepared = _Graphs(graphs)
    spectra = gap_spectra()
    for name in ("default", "baseline", "weighted", "empty", "filtered"):
        config = FragmentGap.Config()
        if name == "baseline":
            config.level_weights = [1.0, 1.0]
            config.intensity_power = 0.5
            config.tol = 0.01
            config.h_shifts = [-2, -1, 0, 1, 2]
        elif name == "weighted":
            config.level_weights = [1.0, 0.6, 0.3]
            config.max_bonds = [100, 40]
            config.intensity_power = 0.7
            config.mz_power = 1.2
        values = config.make().score(
            [] if name == "empty" else spectra[-1:] if name == "filtered" else spectra,
            tuple(graphs),
            graphs=prepared,
        )
        actual[name + "/scores/valid"] = torch.tensor(
            [value is not None for value in values],
        )
        # The powers run on the host's libm; one rounding to float32 is portable.
        actual[name + "/scores/value"] = torch.tensor(
            [value if value is not None else 0.0 for value in values],
            dtype=torch.float64,
        ).float()
    assert_tensor_golden(golden, float32_comparands(actual, inputs=("*/weights",)))


def test_default_caps_rounding_and_minimum_cleavage_order() -> None:
    record = read_tensors(_BASELINE / "testdata/fragment_gap.pt")
    levels = fragment_levels(_graphs(record)["C" * 102], max_bonds=[100])
    assert levels is not None
    assert len(levels) == 3
    assert len(levels[2]) == 0
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


def gap_spectra() -> list[GapSpectrum]:
    """Return every raw spectrum stored in the golden."""
    return [
        {
            "mz": [31.018389717, 47.049141, float("nan"), 2.0],
            "it": [1.0, 0.4, 0.2, -1.0],
            "adduct": "[M+H]+",
            "prec": 47.049141,
            "mode": 1,
        },
        {
            "mz": [15.023475, 17.038576, 28.030752],
            "it": [0.3, 1.0, 0.7],
            "adduct": "[M-H]-",
            "prec": 45.0,
            "mode": -1,
        },
        {"mz": [61.023], "it": [1.0], "adduct": "[M+Na]+", "prec": 61.023, "mode": 1},
        {"mz": [10.0, 20.0], "it": [0.0], "adduct": None, "prec": 0.0, "mode": -1},
    ]


class _Graphs:
    def __init__(self, values: dict[str, GapGraph | None]) -> None:
        self.values = values

    def for_smiles(self, smiles: str) -> GapGraph | None:
        """Require complete prepared data even for invalid source inputs."""
        return self.values[smiles]


def _graphs(record: dict[str, Tensor]) -> dict[str, GapGraph | None]:
    result: dict[str, GapGraph | None] = {}
    for index in range(8):
        prefix = str(index)
        value = record[prefix + "/input"].numpy().tobytes().decode()
        graph = None
        if bool(record[prefix + "/valid"]):
            graph = GapGraph(
                weights=cast(
                    "NDArray[np.float64]",
                    record[prefix + "/weights"].numpy(),
                ),
                bonds=np.column_stack(
                    (
                        record[prefix + "/left"].numpy(),
                        record[prefix + "/right"].numpy(),
                    ),
                ),
                pieces=int(record[prefix + "/pieces"].item()),
            )
        result[value] = graph
    return result


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
