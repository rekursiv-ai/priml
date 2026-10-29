"""Declared record schemas and dataclass defaults shared across packages."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from priml.baselines.casmi2026.custom_types import (
    Analog,
    CandidateSet,
    EngineDiagnostics,
    EngineResult,
    FragmentMassRow,
    GapGraph,
    GapGraphRow,
    ProposalPath,
    ProposalRow,
    ReactionHeader,
    ReactionRequest,
    SpectrumRequest,
)


def _hints(typed_dict: type) -> dict[str, str]:
    return {
        name: hint if isinstance(hint, str) else hint.__forward_arg__
        for name, hint in typed_dict.__annotations__.items()
    }


def test_typed_dict_key_sets() -> None:
    request_hints = _hints(ReactionRequest)
    assert request_hints.keys() == ReactionHeader.__annotations__.keys()
    assert {k for k, v in request_hints.items() if "NotRequired" in v} == (
        request_hints.keys() - {"smiles", "delta"}
    )
    assert ReactionHeader.__optional_keys__ == frozenset()
    assert len(ReactionHeader.__required_keys__) == 12
    spectrum_hints = _hints(SpectrumRequest)
    assert {k for k, v in spectrum_hints.items() if "NotRequired" in v} == {
        "instrument",
        "instr",
        "ce",
        "ce_n",
        "ce_units",
        "ce_orig",
        "ce_ev",
    }
    assert spectrum_hints.keys() >= {"mz", "it", "prec", "mode", "adduct"}


def test_typed_dict_inheritance_adds_fields() -> None:
    assert ProposalRow.__required_keys__ == ProposalPath.__required_keys__ | {
        "request",
        "position",
    }
    assert GapGraphRow.__required_keys__ == FragmentMassRow.__required_keys__ | {
        "pieces",
    }


def test_gap_graph_is_frozen_slotted_keyword_only() -> None:
    graph = GapGraph(weights=np.ones(2), bonds=np.zeros((0, 2), np.int64), pieces=1)
    assert graph.pieces == 1
    assert not hasattr(graph, "__dict__")
    assert all(field.kw_only for field in dataclasses.fields(graph))
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataclasses.replace(graph, pieces=2).__setattr__("pieces", 3)


def test_candidate_and_engine_result_hold_given_fields() -> None:
    candidates = CandidateSet(
        smiles=("C",),
        keys=("k",),
        formulas=("CH4",),
        pool_rows=np.array([0], np.int64),
        generated=np.array([False]),
        sources=("lib",),
    )
    diagnostics = EngineDiagnostics(n_cand=1, lib_max=0.5, top_sim=0.25, fzmax=2.0)
    result = EngineResult(
        candidates=candidates,
        features=np.zeros((1, 2), np.float32),
        feature_names=("a", "b"),
        scores=np.array([0.1]),
        diagnostics=diagnostics,
    )
    assert result.candidates is candidates
    assert result.diagnostics.n_gen == 0
    assert result.feature_names == ("a", "b")
    assert diagnostics == EngineDiagnostics(
        n_cand=1,
        lib_max=0.5,
        top_sim=0.25,
        fzmax=2.0,
    )


def test_analog_defaults() -> None:
    analog = Analog(sid=3, similarity=0.9)
    assert (analog.shift, analog.natural_product) == (0.0, False)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
