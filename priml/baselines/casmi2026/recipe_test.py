"""Checks for the recipe module."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Final, Literal, TypedDict, Unpack, cast

import hashlib
import json
import math
import pickle
import sys
import types
import weakref

from configgle import Fig, InlineConfig, traverse
from pyarrow import Table, parquet
from torch import Tensor
from treekle import from_plain, json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026 import recipe
from priml.baselines.casmi2026.chemistry import CandidatePool, IdentityTable
from priml.baselines.casmi2026.custom_types import (
    CandidateSet,
    EngineDiagnostics,
    EngineResult,
    GapGraph,
    QueryRow,
    SpectrumRequest,
)
from priml.baselines.casmi2026.dreams import DreamsStore, MainBank
from priml.baselines.casmi2026.dreams_test import save_bank
from priml.baselines.casmi2026.experiments import exp000
from priml.baselines.casmi2026.fusion import (
    PLACEHOLDER,
    FragmentGap,
    FusionAudit,
    FusionInputs,
    FusionPolicy,
    GapGraphTable,
)
from priml.baselines.casmi2026.isolated import (
    ADMISSION_DIR,
    DATASET_ROOT,
    ELEMENT_MASSES,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    ReactionTable,
    formula_vector,
    normalize_request,
)
from priml.baselines.casmi2026.isolated_test import _reaction_data_write
from priml.baselines.casmi2026.main_engine import MainRetrieval, ReactionScreen
from priml.baselines.casmi2026.main_engine_test import (
    RANKER_NAMES,
    _inputs,
    pipeline_stage_a,
    pipeline_stage_b,
    ranker_texts,
)
from priml.baselines.casmi2026.pubchem import (
    PubChemAssets,
    PubChemGate,
    PubChemResult,
    PubChemStore,
    ScreenedCandidates,
)
from priml.baselines.casmi2026.pubchem_test import write_pool
from priml.baselines.casmi2026.recipe import (
    DEVICE_OWNERS,
    BaseList,
    GenerationRequest,
    HasBaseDir,
    ListBudget,
    MainAssets,
    MainResource,
    MoleculeQueries,
    PreparedPubChem,
    PriorRequest,
    PubChemResource,
    PublicationContext,
    PublicationProvenance,
    PublicRecipe,
    QueryDataset,
    RankedMain,
    RankedSecond,
    RecipeInputs,
    RecipePlan,
    RecipeResults,
    RerankedLists,
    SecondResource,
    SimulationRunner,
    SimulatorStage,
    StageMetadata,
    WindowRequest,
    _gap_scores,
    _legacy,
    _screen,
    assemble_results,
    glacier_stage,
    plan_requests,
    product_inputs,
    rank_arms,
    rank_main,
    rank_second,
    read_queries,
    rerank_pubchem,
    run_recipe,
    simulator_inputs,
    tail_target,
    write_results,
)
from priml.baselines.casmi2026.scripts.recipe_plan import (
    write_plan,
    write_products,
    write_reaction_inputs,
)
from priml.baselines.casmi2026.scripts.recipe_preparation import (
    PlanningPreparation,
    ProductPreparation,
    RawProductPreparation,
    prepare_product_requests,
    prepare_raw_products,
    prepare_requests,
)
from priml.baselines.casmi2026.second_engine import SecondAssets, SimulationRows
from priml.baselines.casmi2026.second_engine_test import (
    engine_keys,
    second_config,
    second_states,
)
from priml.baselines.casmi2026.simulators import (
    GlacierBackend,
    GlacierScheduling,
    IceScheduling,
    SimulationJobState,
    SimulatorResult,
)
from priml.baselines.casmi2026.simulators_test import (
    _Boundary,
    _Clock,
    _molecules,
    _write,
    admission_rows,
)
from priml.baselines.casmi2026.testing import (
    identity_rows,
    retrieval_inputs,
    write_identity,
)


if TYPE_CHECKING:
    from priml.baselines.casmi2026.custom_types import ReactionRequest


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        GapSpectrum,
        MetricIdentity,
        ReactionRequest,
    )
    from priml.baselines.casmi2026.dreams import ModelViews
    from priml.baselines.casmi2026.fusion import GapGraphs
    from priml.baselines.casmi2026.pubchem import PubChemGate
    from priml.baselines.casmi2026.recipe import (
        ListBudget,
        MainRanker,
        PubChemRanker,
        PublicationProvenance,
        RecipeResults,
        ResultIdentities,
        SecondRanker,
    )
    from priml.baselines.casmi2026.simulators import (
        PredictedPeaks,
        SimulationJob,
        SimulationMolecule,
    )


def released_policy() -> FusionPolicy:
    """Build the fusion votes and simulator fallback that exp000 prints."""
    return exp000().policy.make()


def released_budget() -> ListBudget:
    """Build the candidate prefixes and cut-offs that exp000 prints."""
    return exp000().budget.make()


def released_gate() -> PubChemGate:
    """Build the PubChem admission gate that exp000 prints."""
    return exp000().gate.make()


def released_provenance() -> PublicationProvenance:
    """Build the source experiments and digests that exp000 prints."""
    return exp000().provenance.make()


def engine_result(
    smiles: Sequence[str],
    *,
    keys: Sequence[str] | None = None,
    formulas: Sequence[str] | None = None,
    pids: Sequence[int] | None = None,
    scores: Sequence[float],
    library_max: float = 0.0,
) -> EngineResult:
    """Wrap literal candidates and ranker scores in the engine result record."""
    count = len(smiles)
    rows = np.array(range(count) if pids is None else pids, np.int64)
    return EngineResult(
        candidates=CandidateSet(
            smiles=tuple(smiles),
            keys=tuple(smiles if keys is None else keys),
            formulas=tuple(("unused",) * count if formulas is None else formulas),
            pool_rows=rows,
            generated=rows < 0,
            sources=("pool",) * count,
        ),
        features=np.zeros((count, 0), np.float32),
        feature_names=(),
        scores=np.array(scores, np.float64),
        diagnostics=EngineDiagnostics(
            n_cand=count,
            lib_max=library_max,
            top_sim=0.0,
            fzmax=0.0,
        ),
    )


def test_main_shortlist_takes_unique_keys_then_permutes_pool_slots_by_popularity() -> (
    None
):
    popularity = np.array([0.0, 10.0, 20.0])
    # Equal scores standardize to zero, so popularity alone orders the pool slots.
    flat = rank_main(
        engine_result(
            ("g", "p1", "p2"),
            formulas=("C0H0", "C1H1", "C2H2"),
            pids=(-1, 1, 2),
            scores=(4.0, 4.0, 4.0),
            library_max=0.72,
        ),
        popularity=popularity,
        limit=60,
        weight=0.15,
    )
    assert flat.library_max == 0.72
    assert flat.popularity_applied
    assert flat.smiles == ("g", "p2", "p1")
    assert flat.formulas == ("C0H0|gen", "C2H2", "C1H1")
    assert flat.scores == pytest.approx((0.0, 3.0, 1.5))
    # Key k0 repeats; the limit of two keeps the best k0 and then k2.
    capped = rank_main(
        engine_result(
            ("a", "b", "c", "d"),
            keys=("k0", "k0", "k1", "k2"),
            pids=(0, 0, 1, 2),
            scores=(3.0, 2.5, 1.0, 2.0),
        ),
        popularity=popularity,
        limit=2,
        weight=0.15,
    )
    assert capped.keys == ("k2", "k0")
    assert capped.smiles == ("d", "a")
    # z-scores of 3 and 2 are +-1; the second row adds 0.15 * 20 popularity.
    assert capped.scores == pytest.approx((2.0, 1.0), abs=1e-8)
    for smiles in (("only",), ()):
        unchanged = rank_main(
            engine_result(smiles, scores=(1.0,) * len(smiles)),
            popularity=popularity,
            limit=60,
            weight=0.15,
        )
        assert not unchanged.popularity_applied
        assert unchanged.smiles == smiles
        assert unchanged.scores == (1.0,) * len(smiles)


def test_second_shortlist_recomputes_keys_retaining_the_first_empty_key() -> None:
    keys = engine_keys()
    result = engine_result(
        ("*C", "invalid", "C", "CC", "CCO", "C"),
        scores=(8.0, 9.0, 7.0, 4.0, 4.0, 3.0),
    )
    ranked = rank_second(result, key=keys.__getitem__, prescreen=80, limit=40)
    assert ranked is not None
    # The invalid structure has no key; the repeated methane key is skipped.
    assert ranked.smiles == ("*C", "C", "CC", "CCO")
    assert ranked.keys == ("", keys["C"], keys["CC"], keys["CCO"])
    limited = rank_second(result, key=keys.__getitem__, prescreen=80, limit=2)
    assert limited is not None
    assert limited.smiles == ("*C", "C")
    screened = rank_second(result, key=keys.__getitem__, prescreen=2, limit=40)
    assert screened is not None
    assert screened.smiles == ("*C",)
    none_valid = rank_second(
        engine_result(("invalid",) * 3, scores=(3.0, 2.0, 4.0)),
        key=keys.__getitem__,
        prescreen=80,
        limit=40,
    )
    assert none_valid == RankedSecond(smiles=(), keys=())
    assert (
        rank_second(
            engine_result((), scores=()),
            key=keys.__getitem__,
            prescreen=80,
            limit=40,
        )
        is None
    )


def test_default_prescreen_stops_before_the_eighty_first_novel_structure() -> None:
    keys = engine_keys()
    strings = ("CCO",) * 80 + ("CCCC",)
    result = engine_result(
        strings,
        keys=("unused",) * len(strings),
        scores=np.arange(len(strings), 0, -1, dtype=np.float64).tolist(),
    )
    actual = rank_second(result, key=keys.__getitem__, prescreen=80, limit=40)
    assert actual is not None
    assert actual.smiles == ("CCO",)
    assert actual.keys == (keys["CCO"],)
    expanded = rank_second(result, key=keys.__getitem__, prescreen=81, limit=40)
    assert expanded is not None
    assert expanded.smiles == ("CCO", "CCCC")


def test_disabled_popularity_keeps_generated_formulas_and_raw_scores() -> None:
    result = EngineResult(
        candidates=CandidateSet(
            smiles=("a", "b"),
            keys=("a", "b"),
            formulas=("C2H6O", "C2H6O"),
            pool_rows=np.array([-1, 0], np.int64),
            generated=np.array([True, False]),
            sources=("generated", "pool"),
        ),
        features=np.zeros((2, 0), np.float32),
        feature_names=(),
        scores=np.array([2.0, 1.0]),
        diagnostics=EngineDiagnostics(n_cand=2, lib_max=0.0, top_sim=0.0, fzmax=0.0),
    )
    actual = rank_main(
        result,
        popularity=np.array([100.0]),
        limit=60,
        weight=0.0,
    )
    assert actual.smiles == ("a", "b")
    assert actual.scores == (2.0, 1.0)
    assert actual.formulas == ("C2H6O", "C2H6O")
    assert not actual.popularity_applied


def _text(value: object) -> Tensor:
    return torch.tensor(
        list(json.dumps(to_plain(value), separators=(",", ":")).encode()),
        dtype=torch.uint8,
    )


@dataclass(slots=True, kw_only=True)
class Resource[T]:
    """Retain actual rank functions and observe the arm's ownership boundary."""

    rank: T
    stage: str
    calls: list[tuple[str, str, int]]

    def close(self) -> None:
        """Record the release after every complete or unsuccessful arm loop."""
        self.calls.append((f"close_{self.stage}", "", 0))


def test_both_query_orders_null_targets_and_complete_views() -> None:
    queries = _arms_queries()
    calls: list[tuple[str, str, int]] = []

    def main(
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        calls.append(("main", mid, len(spectra)))
        assert spectra == queries.molecules[0].main
        assert target == 100.0
        return _ranked()

    def second(mid: str, spectra: tuple[SpectrumRequest, ...]) -> RankedSecond | None:
        calls.append(("second", mid, len(spectra)))
        if mid == "a":
            assert spectra == queries.molecules[1].raw
            return None
        assert spectra == queries.molecules[0].raw
        return RankedSecond(smiles=("second",), keys=("key2",))

    def pubchem(mid: str, spectra: tuple[SpectrumRequest, ...]) -> PubChemResult | None:
        calls.append(("pubchem", mid, len(spectra)))
        if mid == "a":
            return None
        assert spectra == queries.molecules[0].raw
        return _proposal()

    result = rank_arms(
        queries,
        main=Resource(rank=main, stage="main", calls=calls),
        second=Resource(rank=second, stage="second", calls=calls),
        pubchem=Resource(rank=pubchem, stage="pubchem", calls=calls),
    )
    assert calls == [
        ("main", "z", 2),
        ("close_main", "", 0),
        ("second", "a", 1),
        ("second", "z", 3),
        ("close_second", "", 0),
        ("pubchem", "z", 3),
        ("pubchem", "a", 1),
        ("close_pubchem", "", 0),
    ]
    assert tuple(result.main) == ("z", "a")
    assert result.main["z"] == _ranked()
    assert result.main["a"] == RankedMain(
        smiles=(),
        keys=(),
        scores=(),
        formulas=(),
        library_max=0,
        popularity_applied=False,
    )
    assert result.second == {"z": RankedSecond(smiles=("second",), keys=("key2",))}
    assert result.pubchem == {"z": _proposal()}
    assert result.errors == ()


def test_main_failure_retains_empty_row_and_exact_error_but_other_stages_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def main(
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        del mid, spectra, target
        raise KeyError("unstaged product")

    def second(mid: str, spectra: tuple[SpectrumRequest, ...]) -> RankedSecond:
        del spectra
        return RankedSecond(smiles=(mid,), keys=(mid,))

    def pubchem(mid: str, spectra: tuple[SpectrumRequest, ...]) -> PubChemResult | None:
        del mid, spectra
        return None

    calls: list[tuple[str, str, int]] = []
    result = rank_arms(
        _arms_queries(),
        main=Resource(rank=main, stage="main", calls=calls),
        second=Resource(rank=second, stage="second", calls=calls),
        pubchem=Resource(rank=pubchem, stage="pubchem", calls=calls),
    )
    assert result.errors == (("z", "KeyError('unstaged product')"),)
    assert caplog.messages == ["Main inference failed for z."]
    assert not result.main["z"].smiles
    assert tuple(result.second) == ("a", "z")
    assert not result.pubchem
    assert calls == [
        (f"close_{stage}", "", 0) for stage in ("main", "second", "pubchem")
    ]


@pytest.mark.parametrize("stage", ["second", "pubchem"])
def test_unstaged_secondary_data_is_not_converted_to_an_empty_result(
    stage: str,
) -> None:
    def main(
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        del mid, spectra, target
        return _ranked()

    def second(mid: str, spectra: tuple[SpectrumRequest, ...]) -> RankedSecond | None:
        del mid, spectra
        if stage == "second":
            raise KeyError("missing second")
        return None

    def pubchem(mid: str, spectra: tuple[SpectrumRequest, ...]) -> PubChemResult | None:
        del mid, spectra
        raise KeyError("missing pubchem")

    calls: list[tuple[str, str, int]] = []
    with pytest.raises(KeyError) as error:
        rank_arms(
            _arms_queries(),
            main=Resource(rank=main, stage="main", calls=calls),
            second=Resource(rank=second, stage="second", calls=calls),
            pubchem=Resource(rank=pubchem, stage="pubchem", calls=calls),
        )
    assert error.value.args == (f"missing {stage}",)
    assert calls == [
        (f"close_{name}", "", 0)
        for name in (
            ("main", "second") if stage == "second" else ("main", "second", "pubchem")
        )
    ]


def test_main_interrupt_releases_only_the_started_resource() -> None:
    def main(
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        del mid, spectra, target
        raise KeyboardInterrupt("cancelled")

    def unused(mid: str, spectra: tuple[SpectrumRequest, ...]) -> None:
        del mid, spectra
        pytest.fail("A later resource started after cancellation.")

    calls: list[tuple[str, str, int]] = []
    with pytest.raises(KeyboardInterrupt, match=r"^cancelled$"):
        rank_arms(
            _arms_queries(),
            main=Resource(rank=main, stage="main", calls=calls),
            second=Resource(rank=unused, stage="second", calls=calls),
            pubchem=Resource(rank=unused, stage="pubchem", calls=calls),
        )
    assert calls == [("close_main", "", 0)]


def _arms_queries() -> QueryDataset:
    row: SpectrumRequest = {
        "mz": [10.0, 20.0],
        "it": [1.0, 0.5],
        "prec": 101.0,
        "adduct": "[M+H]+",
        "mode": 1,
        "ce": 25.0,
        "ce_n": 1,
    }
    return QueryDataset(
        rows=(),
        engine_order=("a", "z"),
        molecules=(
            MoleculeQueries(
                mid="z",
                source_id="z",
                main=(row, row),
                raw=(row, row, row),
                finite_mass=np.array([100.0, 100.0]),
                target=100.0,
            ),
            MoleculeQueries(
                mid="a",
                source_id="a",
                main=(),
                raw=(row,),
                finite_mass=np.zeros(0),
                target=None,
            ),
        ),
    )


def _ranked() -> RankedMain:
    return RankedMain(
        smiles=("main",),
        keys=("key",),
        scores=(0.4,),
        formulas=("CH4",),
        library_max=0.2,
        popularity_applied=False,
    )


def _proposal() -> PubChemResult:
    return {
        "pc": ["pc"],
        "pc_fz": [2.0],
        "pc_keys": ["pc_key"],
        "fz_top": 3.0,
        "S": 7.0,
        "top_pop": 5.0,
        "best_pool_fz": 1.0,
        "target": 100.0,
    }


def query_rows() -> list[QueryRow]:
    """Return six query spectra covering every polarity, energy and empty-field form.

    Returns:
      rows: Three molecules plus an unparseable one, in non-sorted encounter order.

    """
    return [
        {
            "molecule_id": "m02",
            "spectrum_id": "s0",
            "precursor_mz": 101.00727645217,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "collision_energy_ev": None,
            "collision_energy_orig_units": None,
            "collision_energy_orig": None,
            "instrument_type": "timsTOF",
            "ms2_mzs": [30.0, 60.0, 104.0],
            "ms2_normalized_intensities": [1.0, 0.5, 0.01],
        },
        {
            "molecule_id": "m01",
            "spectrum_id": "s1",
            "precursor_mz": 98.99272354783,
            "adduct": "[M-H]-",
            "ionization_mode": "negative",
            "collision_energy_ev": [],
            "collision_energy_orig_units": "eV",
            "collision_energy_orig": "",
            "instrument_type": None,
            "ms2_mzs": [],
            "ms2_normalized_intensities": [],
        },
        {
            "molecule_id": "m02",
            "spectrum_id": "s2",
            "precursor_mz": 123.989220701,
            "adduct": "[M+Na]+",
            "ionization_mode": "positive",
            "collision_energy_ev": [20.0, 40.0],
            "collision_energy_orig_units": "eV",
            "collision_energy_orig": "20/40",
            "instrument_type": "QTOF",
            "ms2_mzs": [44.0, 70.0],
            "ms2_normalized_intensities": [0.3, 1.0],
        },
        {
            "molecule_id": "bad",
            "spectrum_id": "s3",
            "precursor_mz": 70.0,
            "adduct": "unknown",
            "ionization_mode": None,
            "collision_energy_ev": [None],
            "collision_energy_orig_units": "NCE",
            "collision_energy_orig": "20/40 NCE",
            "instrument_type": "Orbitrap",
            "ms2_mzs": [80.0],
            "ms2_normalized_intensities": [1.0],
        },
        {
            "molecule_id": "m03",
            "spectrum_id": "s4",
            "precursor_mz": 151.00727645217,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "collision_energy_ev": [-10.0, -20.0],
            "collision_energy_orig_units": "eV",
            "collision_energy_orig": "-10/-20",
            "instrument_type": "Orbitrap",
            "ms2_mzs": [10.0, 20.0, 30.0, 40.0],
            "ms2_normalized_intensities": [0.1, 0.2, 0.4, 1.0],
        },
        {
            "molecule_id": "m01",
            "spectrum_id": "s5",
            "precursor_mz": 134.9694012599,
            "adduct": "[M+Cl]-",
            "ionization_mode": "negative",
            "collision_energy_ev": [10.0, None, 30.0],
            "collision_energy_orig_units": "NCE",
            "collision_energy_orig": "10,30%",
            "instrument_type": "TOF",
            "ms2_mzs": [20.0, 80.0],
            "ms2_normalized_intensities": [1.0, 0.25],
        },
    ]


def test_every_query_field_mass_order_and_raw_row_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    rows = query_rows()
    path = tmp_path / "test.parquet"
    parquet.write_table(Table.from_pylist(rows), path)
    data = read_queries(path)
    assert torch.equal(_text(list(data.rows)), _text(rows))
    assert [molecule.mid for molecule in data.molecules] == ["m02", "m01", "bad", "m03"]
    assert data.engine_order == ("bad", "m01", "m02", "m03")
    expected: tuple[tuple[list[float], float | None, list[SpectrumRequest]], ...] = (
        (
            [100.0, 101.0],
            100.5,
            [
                _main([30.0, 60.0, 104.0], [1.0, 0.5, 0.01], prec=101.00727645217),
                _main(
                    [44.0, 70.0],
                    [0.3, 1.0],
                    adduct="[M+Na]+",
                    prec=123.989220701,
                    ce=(30.0, 2),
                ),
            ],
        ),
        (
            [100.0, 100.0],
            100.0,
            [
                _main([], [], adduct="[M-H]-", prec=98.99272354783),
                _main(
                    [20.0, 80.0],
                    [1.0, 0.25],
                    adduct="[M+Cl]-",
                    prec=134.9694012599,
                    ce=(math.nan, 3),
                ),
            ],
        ),
        ([], None, []),
        (
            [150.0],
            150.0,
            [
                _main(
                    [10.0, 20.0, 30.0, 40.0],
                    [0.1, 0.2, 0.4, 1.0],
                    prec=151.00727645217,
                    ce=(-15.0, 2),
                ),
            ],
        ),
    )
    for molecule, (finite, target, main) in zip(data.molecules, expected, strict=True):
        assert molecule.finite_mass.dtype == np.float64
        np.testing.assert_array_equal(molecule.finite_mass, finite)
        assert molecule.target == target
        assert torch.equal(_text(list(molecule.main)), _text(main))
        inputs = [row for row in rows if row["molecule_id"] == molecule.source_id]
        assert len(molecule.raw) == len(inputs)
        for raw, original in zip(molecule.raw, inputs, strict=True):
            assert raw["mz"] == original["ms2_mzs"]
            assert raw["it"] == original["ms2_normalized_intensities"]
            assert raw["prec"] == original["precursor_mz"]
            assert raw["adduct"] == original["adduct"]
            assert raw["mode"] == (
                1 if original["ionization_mode"] == "positive" else -1
            )
            assert "instrument" in raw
            assert "ce_units" in raw
            assert "ce_orig" in raw
            assert "ce_ev" in raw
            assert raw["instrument"] == original["instrument_type"]
            assert raw["ce_units"] == original["collision_energy_orig_units"]
            assert raw["ce_orig"] == original["collision_energy_orig"]
            values = original["collision_energy_ev"]
            np.testing.assert_equal(
                raw["ce_ev"],
                [math.nan if value is None else value for value in values]
                if values is not None
                else None,
            )


def _main(
    mz: list[float],
    it: list[float],
    *,
    prec: float,
    adduct: str = "[M+H]+",
    ce: tuple[float, int] = (math.nan, 1),
) -> SpectrumRequest:
    return {
        "mz": mz,
        "it": it,
        "mode": 1 if adduct.endswith("+") else -1,
        "adduct": adduct,
        "prec": prec,
        "ce": ce[0],
        "ce_n": ce[1],
    }


def test_numeric_ids_keep_encounter_order_and_sort_numerically(tmp_path: Path) -> None:
    row = query_rows()[0]
    rows = [{**row, "molecule_id": mid} for mid in (10, 2, 10, 1)]
    path = tmp_path / "numeric.parquet"
    parquet.write_table(Table.from_pylist(rows), path)
    data = read_queries(path)
    assert tuple(molecule.source_id for molecule in data.molecules) == (10, 2, 1)
    assert tuple(len(molecule.raw) for molecule in data.molecules) == (2, 1, 1)
    assert data.engine_order == ("1", "2", "10")
    parquet.write_table(Table.from_pylist(rows).slice(0, 0), path)
    empty = read_queries(path)
    assert empty.rows == empty.molecules == empty.engine_order == ()


def test_tail_mass_preserves_infinities_before_its_later_finite_gate() -> None:
    precursor = query_rows()[0]["precursor_mz"]
    raw: tuple[SpectrumRequest, ...] = tuple(
        {
            "mz": [],
            "it": [],
            "prec": value,
            "mode": 1,
            "adduct": "[M+H]+",
        }
        for value in (precursor, precursor + 2, math.inf, math.nan)
    )
    molecule = MoleculeQueries(
        mid="tail",
        source_id="tail",
        main=(),
        raw=raw,
        finite_mass=np.zeros(0, np.float64),
        target=None,
    )
    assert tail_target(molecule) == precursor + 2 - 1.00727645217
    for values, expected in (
        ((math.inf,), math.inf),
        ((math.nan,), math.nan),
        ((), math.nan),
    ):
        query = MoleculeQueries(
            mid="tail",
            source_id="tail",
            main=(),
            raw=tuple({**raw[0], "prec": value} for value in values),
            finite_mass=np.zeros(0, np.float64),
            target=None,
        )
        if math.isnan(expected):
            assert math.isnan(tail_target(query))
        else:
            assert tail_target(query) == expected


def outer_main() -> dict[str, RankedMain]:
    """Return top lists of a gated pair, an ungated molecule and an empty row."""
    return {
        mid: RankedMain(
            smiles=smiles,
            keys=keys,
            library_max=maximum,
            scores=scores,
            formulas=formulas,
            popularity_applied=True,
        )
        for mid, smiles, keys, maximum, scores, formulas in (
            (
                "m02",
                ("CCCO", "CC(C)O", "CCCC", "CCO"),
                ("b0", "b1", "b2", "b3"),
                0.2,
                (4.0, 3.0, 2.0, 1.0),
                ("C3H8O", "C3H8O", "C4H10", "C2H6O|gen"),
            ),
            (
                "m01",
                ("CCN", "CCO", "OCC"),
                ("n0", "n1", "n2"),
                0.9,
                (3.0, 2.0, 1.0),
                ("C2H7N", "C2H6O", "C2H6O"),
            ),
            ("bad", (), (), 0.0, (), ()),
            ("m03", ("CCO",), ("x0",), 0.2, (2.0,), ("C2H6O",)),
        )
    }


def outer_second() -> dict[str, RankedSecond]:
    """Return the second engine's lists, one repeating a main key."""
    return {
        "m02": RankedSecond(smiles=("CCCO", "COCC", "CCN"), keys=("b0", "e1", "e2")),
        "m01": RankedSecond(smiles=("COC",), keys=("e3",)),
        "bad": RankedSecond(smiles=(), keys=()),
        "m03": RankedSecond(smiles=(), keys=()),
    }


def outer_pubchem() -> dict[str, PreparedPubChem]:
    """Return PubChem proposals for two gated molecules and one ungated molecule."""
    return {
        "m02": {
            "pc": ["CCO", "COC", "CCN"],
            "pc_keys": ["p0", "p1", "p2"],
            "pc_fz": [1.0, 2.0, 0.0],
            "fz_top": 700.0,
            "S": 7.0,
            "top_pop": 5.0,
            "best_pool_fz": 0.0,
            "target": 100.0,
        },
        "m01": {
            "pc": ["CCCO", "CC(C)O"],
            "pc_keys": ["p3", "p4"],
            "pc_fz": [2.0, 1.0],
            "fz_top": 0.0,
            "S": 0.0,
            "top_pop": 0.0,
            "best_pool_fz": 0.0,
            "target": 100.0,
        },
        "m03": {
            "pc": ["CCO", "COC"],
            "pc_keys": ["p0", "p1"],
            "pc_fz": [0.0, 1.0],
            "fz_top": None,
            "S": None,
            "top_pop": 0.0,
            "best_pool_fz": math.nan,
            "target": 150.0,
        },
    }


OUTER_FORMULAS: Final = {
    "CC(C)O": "C3H8O",
    "CCCO": "C3H8O",
    "CCN": "C2H7N",
    "CCO": "C2H6O",
    "COC": "C2H6O",
    "COCC": "C3H8O",
}


def outer_scores() -> tuple[
    dict[str, dict[str, float | None]],
    dict[str, dict[str, float | None]],
]:
    """Return ICE and GLACIER scores, with missing ICE values."""
    return (
        {
            "m02": {"CCO": 1.0, "COC": 0.1, "CCN": None},
            "m03": {"CCO": None, "COC": None},
        },
        {"m02": {"CCO": 1.0, "COC": 0.1}, "m03": {"CCO": 0.9, "COC": 0.2}},
    )


@pytest.mark.parametrize("missing", ["", "engine", "pubchem"])
def test_unions_order_molecules_by_library_evidence_and_gate_pubchem(
    tmp_path: Path,
    missing: str,
) -> None:
    path = tmp_path / "queries.parquet"
    parquet.write_table(Table.from_pylist(query_rows()), path)
    queries = read_queries(path)
    main = outer_main()
    second = {} if missing == "engine" else outer_second()
    pubchem = {} if missing == "pubchem" else outer_pubchem()
    inputs = simulator_inputs(
        main,
        second=second,
        pubchem=pubchem,
        queries=queries,
        formula=OUTER_FORMULAS.__getitem__,
        budget=released_budget(),
        gate=released_gate(),
    )
    # Ties at library maximum 0.2 keep encounter order.
    assert inputs.priority == ("bad", "m02", "m03", "m01")
    assert "bad" not in inputs.candidates
    for mid, values in inputs.candidates.items():
        assert len(set(values)) == len(values)
        sources = {
            *main[mid].smiles,
            *(second[mid].smiles if mid in second else ()),
            *(pubchem[mid]["pc"] if mid in pubchem else ()),
        }
        assert set(values) <= sources
    assert ("COCC" in inputs.candidates["m02"]) == (missing != "engine")
    assert ("m03" in inputs.candidates) == (missing != "pubchem")
    # Only molecules below the library gate prepare PubChem formulas.
    for mid, row in inputs.proposals.items():
        if mid == "m01":
            assert "pc_form" not in row
        else:
            assert row.get("pc_form") == [OUTER_FORMULAS[value] for value in row["pc"]]
    assert all("pc_form" not in row for row in pubchem.values())
    assert tuple(molecule.mid for molecule in inputs.ice) == tuple(inputs.candidates)
    assert all(
        row["adduct"] == "[M+H]+"
        for molecule in inputs.glacier
        for row in molecule.spectra
    )
    ice, glacier = outer_scores()
    ranked = rerank_pubchem(
        main,
        proposals=inputs.proposals,
        ice=ice,
        glacier={} if missing else glacier,
        policy=released_policy(),
        budget=released_budget(),
    )
    for mid, row in ranked.main.items():
        assert (row.smiles, row.keys) == (main[mid].smiles, main[mid].keys)
    assert set(ranked.proposals) == set(inputs.proposals)
    assert set(ranked.ice_stats) <= {"pc_changed"}


def test_empty_reordering_keeps_all_maps_and_counters_absent() -> None:
    assert rerank_pubchem(
        {},
        proposals={},
        ice={},
        glacier={},
        policy=released_policy(),
        budget=released_budget(),
    ) == RerankedLists(
        main={},
        proposals={},
        ice_stats={},
        glacier_stats={},
    )


def test_union_caps_each_source_prefix_and_retains_empty_query_groups() -> None:
    base = tuple(f"b{index}" for index in range(61))
    engine = tuple(f"e{index}" for index in range(41))
    main = RankedMain(
        smiles=base,
        keys=base,
        library_max=0.0,
        scores=(1.0,) * len(base),
        formulas=("group",) * len(base),
        popularity_applied=False,
    )
    pc: PubChemResult = {
        "pc": [f"p{index}" for index in range(26)],
        "pc_keys": [f"p{index}" for index in range(26)],
        "pc_fz": [0.0] * 26,
        "fz_top": None,
        "S": None,
        "top_pop": 0.0,
        "best_pool_fz": 0.0,
        "target": 100.0,
    }
    formulas = dict.fromkeys([*engine, *pc["pc"]], "group")
    inputs = simulator_inputs(
        {"mid": main},
        second={"mid": RankedSecond(smiles=engine, keys=engine)},
        pubchem={"mid": pc},
        queries=QueryDataset(rows=(), molecules=(), engine_order=()),
        formula=formulas.__getitem__,
        budget=released_budget(),
        gate=released_gate(),
    )
    assert inputs.candidates == {"mid": (*base[:60], *engine[:40], *pc["pc"][:25])}
    assert len(inputs.ice) == 1
    assert inputs.ice[0].spectra == ()
    assert inputs.glacier == ()
    assert len(inputs.proposals["mid"].get("pc_form", [])) == 26
    assert "pc_form" not in pc


def test_reorder_prefix_key_fallback_weights_and_accumulation() -> None:
    labels = (
        "single",
        "key",
        "ice_weight",
        "gl_weight",
        "flip1",
        "flip2",
        "absent",
        "absent_empty",
        "empty",
        "edge",
        "edge_single",
    )
    main = {
        mid: RankedMain(
            smiles=("A", "B") * 13,
            keys=("ka", "kb") * 13,
            library_max=0.1,
            scores=(2.0, 1.0) * 13,
            formulas=("f",) * 26,
            popularity_applied=False,
        )
        for mid in labels
    }
    proposals: dict[str, PreparedPubChem] = {
        mid: {
            "pc": ["A", "B"],
            "pc_keys": ["ka", "kb"],
            "pc_fz": [1.0, 2.0]
            if mid in ("key", "ice_weight", "gl_weight")
            else [2.0, 1.0],
            "pc_form": [] if mid == "empty" else ["f", "f"],
            "fz_top": 123.0,
            "S": 7.0,
            "top_pop": 6.0,
            "best_pool_fz": 4.0,
            "target": 100.0,
        }
        for mid in labels
    }
    edge = [f"p{index}" for index in range(26)]
    for label, boundary in (("edge", 24), ("edge_single", 23)):
        proposals[label] = {
            **proposals[label],
            "pc": edge,
            "pc_keys": edge,
            "pc_fz": [float(index) for index in range(26, 0, -1)],
            "pc_form": [
                *(f"f{index}" for index in range(boundary)),
                *["edge"] * (26 - boundary),
            ],
        }
    ice = {
        mid: {"ka": 1.0, "kb": 2.0}
        for mid in ("single", "key", "gl_weight", "flip1", "flip2", "empty")
    }
    ice.update(
        ice_weight={"ka": 2.0, "kb": 1.0},
        absent_empty={},
        edge={"p24": 0.0, "p25": 1.0},
        edge_single={"p23": 0.0, "p24": 10.0, "p25": 0.0},
    )
    glacier = {mid: {"ka": 1.0, "kb": 2.0} for mid in ("ice_weight", "flip1", "flip2")}
    glacier.update(gl_weight={"ka": 2.0, "kb": 1.0}, edge={"p24": 0.0, "p25": 1.0})
    actual = rerank_pubchem(
        main,
        proposals=proposals,
        ice=ice,
        glacier=glacier,
        policy=released_policy(),
        budget=released_budget(),
    )
    for mid in ("key", "ice_weight", "gl_weight", "flip1", "flip2"):
        assert actual.proposals[mid]["pc"] == ["B", "A"]
        assert actual.proposals[mid]["pc_keys"] == ["kb", "ka"]
        assert actual.proposals[mid]["pc_fz"] == (
            [1.0, 1.0] if mid in ("key", "ice_weight", "gl_weight") else [2.0, 2.0]
        )
        assert proposals[mid]["pc"] == ["A", "B"]
    for mid in ("single", "absent", "absent_empty", "empty"):
        assert actual.proposals[mid]["pc"] == ["A", "B"]
    assert actual.proposals["edge"]["pc"] == edge
    assert actual.proposals["edge_single"]["pc"] == edge
    assert actual.ice_stats == {"pc_changed": 5}
    assert actual.glacier_stats == {
        "pc_molecules": 5,
        "pc_changed_vs_ice_top25": 3,
        "pc_changed_vs_ice_top1": 3,
    }
    for row in actual.main.values():
        assert row.smiles == (("A", "B") * 13)[:25]
        assert row.keys == (("ka", "kb") * 13)[:25]
    empty_ice = rerank_pubchem(
        {"absent_empty": main["absent_empty"]},
        proposals={"absent_empty": proposals["absent_empty"]},
        ice={"absent_empty": {}},
        glacier={},
        policy=released_policy(),
        budget=released_budget(),
    )
    assert empty_ice.ice_stats == empty_ice.glacier_stats == {}
    assert empty_ice.proposals == {"absent_empty": proposals["absent_empty"]}


def _items(molecules: Sequence[SimulationMolecule]) -> list[dict[str, object]]:
    return [
        {
            "mid": molecule.mid,
            "spectra": [
                {
                    "mz": row["mz"],
                    "it": row["it"],
                    "prec": row["prec"],
                    "adduct": row["adduct"],
                    "mode": "positive" if row["mode"] == 1 else "negative",
                    "instrument": row.get("instrument"),
                    "ce_ev": row.get("ce_ev"),
                    "ce_orig": row.get("ce_orig"),
                    "ce_units": row.get("ce_units"),
                }
                for row in molecule.spectra
            ],
            "cands": list(molecule.candidates),
        }
        for molecule in molecules
    ]


class _Identity(TypedDict):
    key: str | None
    identity: MetricIdentity | None
    formula: str


def _gap_graph(groups: Sequence[tuple[str, int]]) -> GapGraph:
    """Weigh a three-atom chain's heavy atoms with their hydrogens."""
    masses = ELEMENT_MASSES
    return GapGraph(
        weights=np.array(
            [masses[symbol] + count * masses["H"] for symbol, count in groups],
        ),
        bonds=np.array([[0, 1], [1, 2]], np.int64),
        pieces=1,
    )


class Prepared:
    """Serve literal metric identities and fragment-gap graphs."""

    def __init__(self) -> None:
        self.identities: dict[str, _Identity] = {
            smiles: {
                "key": key,
                "identity": (key, composition),
                "formula": formula,
            }
            for smiles, key, composition, formula in (
                ("CC(C)O", "KFZMGEQAYNKOFK", (("C", 3), ("O", 1)), "C3H8O"),
                ("CCCC", "IJDNQMDRQITEOD", (("C", 4),), "C4H10"),
                ("CCCO", "BDERNNFJNOPAEC", (("C", 3), ("O", 1)), "C3H8O"),
                ("CCN", "QUSNBJAOOMFDIB", (("C", 2), ("N", 1)), "C2H7N"),
                ("CCO", "LFQSCWFLJHTTHZ", (("C", 2), ("O", 1)), "C2H6O"),
                ("COC", "LCGLNKUTAGEVQW", (("C", 2), ("O", 1)), "C2H6O"),
                ("COCC", "XOBKSJJDNFUZPF", (("C", 3), ("O", 1)), "C3H8O"),
                # Ethanol written backwards shares ethanol's identity.
                ("OCC", "LFQSCWFLJHTTHZ", (("C", 2), ("O", 1)), "C2H6O"),
            )
        }
        self.graphs: dict[str, GapGraph | None] = {
            "CCN": _gap_graph((("C", 3), ("C", 2), ("N", 2))),
            "OCC": _gap_graph((("O", 1), ("C", 2), ("C", 3))),
            "COC": _gap_graph((("C", 3), ("O", 0), ("C", 3))),
        }

    def key(self, smiles: str) -> str | None:
        """Return the staged older key or raise for undeclared strings."""
        return self.identities[smiles]["key"]

    def formula(self, smiles: str) -> str:
        """Return the staged raw formula or original fallback."""
        return self.identities[smiles]["formula"]

    def identity(self, smiles: str) -> MetricIdentity | None:
        """Return the staged official identity."""
        return self.identities[smiles]["identity"]

    def for_smiles(self, smiles: str) -> GapGraph | None:
        """Return the staged numeric graph, preserving explicit invalidity."""
        return self.graphs[smiles]


SAMPLE_IDS: Final = ("m03", "bad", "m01", "m02")


def outer_sample(path: Path) -> Path:
    """Write the public sample submission, which proposes ethanol for every row."""
    path.write_text(
        "molecule_id,smiles\n" + "".join(f"{mid},CCO\n" for mid in SAMPLE_IDS),
    )
    return path


def outer_tail() -> dict[str, list[str]]:
    """Return the main engine's tail candidates per molecule."""
    return {
        "m02": ["CCCO", "CC(C)O", "CCCC", "CCO"],
        "m01": ["CCN", "CCO", "OCC"],
        "bad": [],
        "m03": ["CCO"],
    }


def outer_bases() -> dict[str, BaseList]:
    """Strip the ranked main lists to their published base form."""
    return {
        mid: BaseList(smiles=row.smiles, keys=row.keys, library_max=row.library_max)
        for mid, row in outer_main().items()
    }


def outer_proposals() -> dict[str, PreparedPubChem]:
    """Attach prepared formulas to the gated PubChem proposals."""
    proposals = outer_pubchem()
    for mid in ("m02", "m03"):
        proposals[mid]["pc_form"] = [
            OUTER_FORMULAS[value] for value in proposals[mid]["pc"]
        ]
    return proposals


def replay_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case: str = "complete",
) -> RecipeResults:
    """Assemble and publish the literal stage outputs, or one malformed input."""
    monkeypatch.setitem(sys.modules, "rdkit", None)
    prepared = Prepared()
    query_path = tmp_path / "test.parquet"
    parquet.write_table(Table.from_pylist(query_rows()), query_path)
    queries = read_queries(query_path)
    sample_ids = list(SAMPLE_IDS)
    sample = outer_sample(tmp_path / "sample_submission.csv")
    base = outer_bases()
    second = outer_second()
    tail = outer_tail()
    ice, glacier = outer_scores()
    if case == "missing_engine":
        del second["bad"]
    elif case == "missing_base":
        del base["bad"]
    elif case == "duplicate_id":
        sample_ids.append(sample_ids[0])
    elif case == "missing_id":
        sample_ids.remove("bad")
    elif case == "empty_final":
        fuse = FusionPolicy.fuse

        def fuse_nothing(
            self: FusionPolicy,
            molecule_id: str,
            original: Sequence[str],
            *,
            options: FusionInputs,
        ) -> tuple[list[str], FusionAudit]:
            return [], fuse(self, molecule_id, original, options=options)[1]

        monkeypatch.setattr(FusionPolicy, "fuse", fuse_nothing)
    elif case == "no_tail_audit":

        def no_audit(
            *args: object,
            **kwargs: object,
        ) -> tuple[list[str], None]:
            del args, kwargs
            return [], None

        monkeypatch.setattr(recipe, "safe_tail", no_audit)
    result = assemble_results(
        sample_ids,
        base=base,
        base_tail=tail,
        pubchem=outer_proposals(),
        engine=second,
        queries=queries,
        identities=prepared,
        ice=ice,
        glacier=glacier,
        graphs=prepared,
        fragment_gap=FragmentGap.Config().make(),
        base_errors=int(case == "base_error"),
        ice_status="budget" if case == "ice_budget" else "ok",
        glacier_status="error" if case == "glacier_error" else "ok",
        policy=released_policy(),
        budget=released_budget(),
        gate=released_gate(),
    )
    metadata: StageMetadata = {
        "status": "ok",
        "device": "cpu",
        "rdkit": SIMULATOR_RDKIT,
        "n_mols_scored": 2,
        "n_scored": 4,
        "errors": [],
    }
    write_results(
        tmp_path / "output",
        results=result,
        sample_ids=sample_ids,
        base_tail=tail,
        pubchem=outer_proposals(),
        engine=second,
        context=PublicationContext(
            sample_csv=sample,
            ice=metadata,
            glacier=metadata.copy(),
            elapsed=0.0,
        ),
        policy=released_policy(),
        provenance=released_provenance(),
    )
    return result


def assert_outer_publication(result: RecipeResults, output: Path) -> None:
    """Check each published list against identities, sources and the sample rows."""
    prepared = Prepared()
    pubchem = outer_pubchem()
    sources = {
        mid: {
            *outer_main()[mid].smiles,
            *outer_second()[mid].smiles,
            *(pubchem[mid]["pc"] if mid in pubchem else ()),
            PLACEHOLDER,
        }
        for mid in SAMPLE_IDS
    }
    rows = (output / "submission.csv").read_text().splitlines()
    assert rows[0] == "molecule_id,smiles"
    assert [row.split(",")[0] for row in rows[1:]] == list(SAMPLE_IDS)
    for row in rows[1:]:
        mid, listed = row.split(",")
        final = listed.split(";")
        assert final == list(result.submission[mid])
        # No published list repeats a metric identity.
        identities = [prepared.identity(value) for value in final]
        assert len(set(identities)) == len(identities)
        assert set(final) <= sources[mid]
        assert set(result.prefusion[mid]) <= sources[mid]
    assert result.submission["bad"] == [PLACEHOLDER]
    assert set(result.overflow) <= set(SAMPLE_IDS)
    assert set(result.gap) <= set(SAMPLE_IDS)
    for name in (
        "baseline_submission.csv",
        "candidate_audit.csv",
        "experiment_audit.csv",
        "experiment_audit_factorial.csv",
        "submission_base.csv",
        "submission_e004_control.csv",
        "submission_reference.csv",
        "candidate_audit.json",
        "experiment_audit.json",
        "experiment_audit_factorial.json",
        "experiment_verification.json",
        "e004_candidate_inputs.json",
        "e004_verification.json",
    ):
        lines = (output / name).read_text().splitlines()
        assert lines, name
    for name in (
        "submission_base.csv",
        "submission_reference.csv",
        "baseline_submission.csv",
    ):
        published = (output / name).read_text().splitlines()
        assert [row.split(",")[0] for row in published[1:]] == list(SAMPLE_IDS), name
    probe = json_loads((output / "probe_applied.txt").read_text(), dict[str, int])
    assert probe["rows"] == len(SAMPLE_IDS)
    manifest = json_loads((output / "run_manifest.json").read_text(), dict[str, object])
    assert manifest["n_errors"] == 0
    assert manifest["ice_health"] == manifest["gl_health"] == "ok"
    assert manifest["families"] == [
        "derivation",
        "frag2",
        "analog_struct",
        "model_views",
        "fragnet",
    ]


def test_complete_outer_lists_control_gap_audits_and_every_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = replay_outputs(tmp_path, monkeypatch)
    assert_outer_publication(result, tmp_path / "output")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("base_error", r"^Base inference errors: 1$"),
        ("missing_engine", r"^Engine 2 did not cover all active molecules$"),
        ("ice_budget", r"^ICE stage is not complete: budget$"),
        ("glacier_error", r"^GLACIER stage is not complete: error$"),
        (
            "missing_base",
            r"^Every test molecule requires one complete publication row\.$",
        ),
        (
            "duplicate_id",
            r"^Every test molecule requires one complete publication row\.$",
        ),
        (
            "missing_id",
            r"^Every test molecule requires one complete publication row\.$",
        ),
        ("empty_final", r"^Invalid final metric identities: "),
    ],
)
def test_unhealthy_or_incomplete_stages_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        replay_outputs(tmp_path, monkeypatch, case=case)
    assert not (tmp_path / "output").exists()


def test_a_processed_tail_without_audit_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"^Expected audit is not None\.$"):
        replay_outputs(tmp_path, monkeypatch, case="no_tail_audit")


@pytest.mark.parametrize(
    "case",
    ["none", "missing_gl", "null_gl", "empty_ice", "triple", "pair"],
)
def test_legacy_gate_key_fallback_full_forty_window_and_prefixes(
    case: str,
) -> None:
    prepared = Prepared()
    for count in range(1, 46):
        smiles = "C" * count
        key = f"ALKANE{count:08d}"
        prepared.identities[smiles] = {
            "key": key,
            "identity": (key, (("C", count),)),
            "formula": f"C{count if count > 1 else ''}H{2 * count + 2}",
        }
    first: list[str] = ["C" * count for count in range(1, 26)]
    second = (
        ["C" * count for count in range(26, 38)]
        + ["CCCO", "CC(C)O", "COCC"]
        + ["C" * count for count in range(38, 46)]
    )
    if case == "pair":
        second = (
            ["C" * count for count in range(26, 39)]
            + ["COC", "CCO"]
            + ["C" * count for count in range(39, 46)]
        )
    ranked = RankedSecond(
        smiles=tuple(second),
        keys=tuple(from_plain(prepared.key(value), str) for value in second),
    )
    scores: dict[str, float | None] = {
        from_plain(prepared.key("CCCO"), str): 0.0,
        from_plain(prepared.key("CC(C)O"), str): 1.0,
        from_plain(prepared.key("COCC"), str): 0.0,
    }
    if case == "pair":
        scores = {
            from_plain(prepared.key("COC"), str): 0.6 / 18,
            from_plain(prepared.key("CCO"), str): 0.6 / 17,
        }
    glacier: dict[str, dict[str, float | None]] = (
        {}
        if case in ("none", "missing_gl")
        else {"m": dict.fromkeys(scores) if case == "null_gl" else scores}
    )
    ice: dict[str, dict[str, float | None]] = {
        "m": {} if case == "empty_ice" else {"sentinel": None},
    }
    source_first: list[str] = ["CCO", "CCN"] if case == "none" else first
    baseline, overflow = _legacy(
        {"m": source_first},
        policy=released_policy(),
        budget=released_budget(),
        engine={} if case == "none" else {"m": ranked},
        identities=prepared,
        ice=ice,
        glacier=glacier,
    )
    if case == "none":
        assert baseline == {"m": ["CCO", "CCN"]}
        assert overflow == {}
        return
    before, _, _ = released_policy().legacy(
        first,
        second,
        ranked.keys,
        key=prepared.key,
    )
    assert len(before) == 40
    expected = before.copy()
    if case == "triple":
        positions = [
            index
            for index, value in enumerate(before)
            if value in ("CCCO", "CC(C)O", "COCC")
        ]
        for position, smiles in zip(positions, ["CC(C)O", "CCCO", "COCC"], strict=True):
            expected[position] = smiles
    assert baseline == {"m": expected[:25]}
    assert overflow == {"m": expected[25:]}


_NAN: Final = float("nan")


def _molecule(
    mid: str,
    *modes: int | str,
    adduct: str = "[M+H]+",
) -> MoleculeQueries:
    raw = tuple(
        SpectrumRequest(
            mz=[float(index)],
            it=[0.5],
            prec=101.007276,
            mode=mode,
            adduct=adduct,
        )
        for index, mode in enumerate(modes or (1,))
    )
    return MoleculeQueries(
        mid=mid,
        source_id=mid,
        main=raw,
        raw=raw,
        finite_mass=np.zeros(0),
        target=None,
    )


class _NoGraphs:
    """Reject every graph lookup the recording model never makes."""

    def for_smiles(self, smiles: str) -> GapGraph | None:
        del smiles
        return None


class _Gap:
    """Record every scoring call and drop the score of any candidate named drop."""

    config = types.SimpleNamespace(max_spectra=2)

    def __init__(self) -> None:
        self.calls: list[tuple[list[GapSpectrum], list[str]]] = []
        self.short = False

    def score(
        self,
        spectra: list[GapSpectrum],
        candidates: list[str],
        *,
        graphs: GapGraphs,
    ) -> list[float | None]:
        del graphs
        self.calls.append((spectra, candidates))
        scores = [
            None if value == "drop" else float(index)
            for index, value in enumerate(candidates)
        ]
        return scores[1:] if self.short else scores


def _gap_inputs(
    ice: dict[str, dict[str, float | None]],
    glacier: dict[str, dict[str, float | None]],
    gap: _Gap,
) -> dict[str, dict[str, float]]:
    prefusion = {
        "ok": ["x", "", PLACEHOLDER, "drop", "x"],
        "pair": ["x", "y"],
        "single": ["x", ""],
        "cut": ["x", "y"],
    }
    queries = {
        "ok": _molecule("ok", 1, "positive", -1, "negative"),
        "pair": _molecule("pair"),
        "single": _molecule("single"),
        "cut": _molecule("cut"),
    }
    engine = {
        mid: RankedSecond(smiles=("y", "z") if mid == "ok" else (), keys=())
        for mid in prefusion
    }
    return _gap_scores(
        prefusion,
        engine=engine,
        queries=queries,
        ice=ice,
        glacier=glacier,
        model=cast("FragmentGap", gap),
        graphs=_NoGraphs(),
    )


def test_gap_scores_cover_every_unscored_molecule_with_two_candidates() -> None:
    gap = _Gap()
    got = _gap_inputs(
        {"cut": {"s": 0.5}, "pair": {"s": None}},
        {"ok": {"s": None}},
        gap,
    )
    assert got == {"ok": {"x": 0.0, "y": 2.0, "z": 3.0}, "pair": {"x": 0.0, "y": 1.0}}
    by_candidates = {tuple(candidates): spectra for spectra, candidates in gap.calls}
    assert list(by_candidates) == [("x", "drop", "y", "z"), ("x", "y")]
    capped = by_candidates["x", "drop", "y", "z"]
    assert capped == [
        {"mz": [0.0], "it": [0.5], "adduct": "[M+H]+", "prec": 101.007276, "mode": 1},
        {"mz": [1.0], "it": [0.5], "adduct": "[M+H]+", "prec": 101.007276, "mode": 1},
    ]


def test_gap_scores_skip_any_molecule_with_a_published_simulator_score() -> None:
    assert _gap_inputs({"pair": {"s": 0.5}}, {}, _Gap()).keys() == {"ok", "cut"}
    assert _gap_inputs({}, {"pair": {"s": 0.5}}, _Gap()).keys() == {"ok", "cut"}
    assert _gap_inputs({}, {}, _Gap()).keys() == {"ok", "pair", "cut"}


def test_gap_scores_encode_every_spectrum_mode_as_plus_or_minus_one() -> None:
    gap = _Gap()
    gap.config = types.SimpleNamespace(max_spectra=4)
    _ = _gap_inputs({}, {}, gap)
    modes = [row["mode"] for row in gap.calls[0][0]]
    assert modes == [1, 1, -1, -1]


def test_gap_scores_reject_a_model_that_misses_a_candidate() -> None:
    gap = _Gap()
    gap.short = True
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        _ = _gap_inputs({}, {}, gap)


class _Gate:
    """Record every screening call and return a mode derived from the molecule."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def screen(self, *args: object, **kwargs: object) -> ScreenedCandidates:
        self.calls.append((args, kwargs))
        mode = "gentle" if args[0] else "no_pc"
        return ScreenedCandidates(
            candidates=("p", "q"),
            mode=mode,
            promoted=bool(args[0]),
        )


def test_screen_forwards_every_proposal_field_and_counts_modes() -> None:
    gate = _Gate()
    proposal = cast(
        "PreparedPubChem",
        {
            "pc": ["c"],
            "pc_keys": ["kc"],
            "pc_fz": [0.25],
            "fz_top": 0.75,
            "best_pool_fz": 0.5,
            "S": 0.125,
            "top_pop": 3.0,
            "target": 100.0,
        },
    )
    rows, counts = _screen(
        ["with", "none"],
        base={"with": BaseList(smiles=("a",), keys=("ka",), library_max=0.4)},
        pubchem={"with": proposal},
        gate=cast("PubChemGate", gate),
    )
    assert rows == {"with": ["p", "q"], "none": ["p", "q"]}
    assert counts == {
        "untouched": 0,
        "gentle": 1,
        "aggressive": 0,
        "no_pc": 1,
        "promoted": 1,
    }
    assert gate.calls[0] == (
        (("a",), ("ka",)),
        {
            "library_max": 0.4,
            "proposals": ["c"],
            "proposal_keys": ["kc"],
            "proposal_scores": [0.25],
            "top_fit": 0.75,
            "pool_fit": 0.5,
            "outlier": 0.125,
            "popularity": 3.0,
        },
    )
    args, kwargs = gate.calls[1]
    pool_fit = kwargs.pop("pool_fit")
    assert isinstance(pool_fit, float)
    assert math.isnan(pool_fit)
    assert args == ((), ())
    assert kwargs == {
        "library_max": 0.0,
        "proposals": (),
        "proposal_keys": (),
        "proposal_scores": (),
        "top_fit": None,
        "outlier": None,
        "popularity": None,
    }


def test_legacy_reranks_with_the_policy_weights_and_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def spy(*args: object, **kwargs: object) -> list[int]:
        seen.update(kwargs, args=args)
        return [1, 0]

    class Policy:
        config = types.SimpleNamespace(
            length=1,
            glacier_weight=0.5,
            min_covered=7,
            tie_eps=0.25,
        )

        def legacy(
            self,
            first: list[str],
            second: Sequence[str],
            keys: Sequence[str],
            *,
            key: object,
        ) -> tuple[list[str], list[str], list[float]]:
            del second, keys, key
            return [*first, "e"][:2], ["k0", "k1"], [0.0, 1.0]

    class Budget:
        config = types.SimpleNamespace(legacy_rerank_top_n=9)

    monkeypatch.setattr(recipe, "rerank_groups", spy)
    baseline, overflow = _legacy(
        {"m": ["a", PLACEHOLDER, ""]},
        engine={"m": RankedSecond(smiles=("e",), keys=("ke",))},
        identities=_Names(),
        ice={"m": {"s": None}},
        glacier={"m": {"s": 0.5}},
        policy=cast("FusionPolicy", Policy()),
        budget=cast("ListBudget", Budget()),
    )
    assert (baseline, overflow) == ({"m": ["e"]}, {"m": ["a"]})
    assert seen == {
        "args": (["a", "e"], ["k0", "k1"], [0.0, 1.0], ["fa", "fe"]),
        "columns": [{"s": 0.5}],
        "weights": [0.5],
        "top_n": 9,
        "min_covered": 7,
        "tie_eps": 0.25,
    }


class _Names:
    """Prepared identities that reject the string bad and invent the rest."""

    def key(self, smiles: str) -> str | None:
        return "k" + smiles

    def formula(self, smiles: str) -> str:
        return "f" + smiles

    def identity(self, smiles: str) -> MetricIdentity | None:
        return None if smiles in ("bad", "bad2") else (smiles, ())


class _Passthrough:
    """Admit exactly the main list."""

    def screen(
        self,
        base: Sequence[str],
        *args: object,
        **kwargs: object,
    ) -> ScreenedCandidates:
        del args, kwargs
        return ScreenedCandidates(candidates=tuple(base), mode="untouched")


class _Policy:
    """Fuse to fixed lists and record every molecule and option the recipe passes."""

    def __init__(
        self,
        final: dict[str, list[str]],
        *,
        length: int = 3,
        extra: tuple[str, ...] = (),
    ) -> None:
        self.extra = extra
        self.config = types.SimpleNamespace(
            length=length,
            glacier_weight=1.0,
            min_covered=2,
            tie_eps=0.0,
        )
        self.final = final
        self.controlled: list[str] = []
        self.options: dict[str, FusionInputs] = {}

    def legacy(
        self,
        first: list[str],
        second: Sequence[str],
        keys: Sequence[str],
        *,
        key: object,
    ) -> tuple[list[str], list[str], list[float]]:
        del keys, key
        merged = [
            *first,
            *self.extra,
            *(value for value in second if value not in first),
        ]
        return merged, ["k" + value for value in merged], [0.0] * len(merged)

    def control(
        self,
        molecule_id: str,
        original: Sequence[str],
        *,
        options: FusionInputs,
    ) -> tuple[list[str], dict[str, int]]:
        del options
        self.controlled.append(molecule_id)
        return list(original), {"collapsed_source_votes": 2}

    def fuse(
        self,
        molecule_id: str,
        original: Sequence[str],
        *,
        options: FusionInputs,
    ) -> tuple[list[str], FusionAudit]:
        self.options[molecule_id] = options
        audit = cast("FusionAudit", {"molecule_id": molecule_id})
        return self.final.get(molecule_id, list(original)), audit


class _Dreams:
    def __init__(self) -> None:
        self.targets: list[float] = []

    def candidates(self, target: float) -> tuple[str, ...]:
        self.targets.append(target)
        return ("d1", "d2")


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResultsStage:
    """One molecule's main list, second-engine list and optional main tail."""

    base: tuple[str, ...]
    second: tuple[str, ...] = ()
    tail: tuple[str, ...] = ()


class _Health(TypedDict, total=False):
    ice_status: str
    glacier_status: str
    base_errors: int


def _assemble(
    policy: _Policy,
    stages: dict[str, _ResultsStage],
    *,
    adduct: str = "[M+H]+",
    dreams: _Dreams | None = None,
    **health: Unpack[_Health],
) -> RecipeResults:
    mids = list(stages)
    molecules = tuple(_molecule(mid, adduct=adduct) for mid in mids)
    queries = QueryDataset(
        rows=cast(
            "tuple[QueryRow, ...]",
            tuple({"adduct": adduct} for _ in mids),
        ),
        molecules=molecules,
        engine_order=tuple(mids),
    )
    return assemble_results(
        mids,
        base={
            mid: BaseList(
                smiles=stage.base,
                keys=tuple("k" + value for value in stage.base),
                library_max=0.0,
            )
            for mid, stage in stages.items()
        },
        base_tail={mid: stage.tail for mid, stage in stages.items() if stage.tail},
        pubchem={},
        engine={
            mid: RankedSecond(smiles=stage.second, keys=())
            for mid, stage in stages.items()
        },
        queries=queries,
        identities=_Names(),
        ice={},
        glacier={},
        graphs=_NoGraphs(),
        fragment_gap=cast("FragmentGap", _Gap()),
        dreams=cast("DreamsStore | None", dreams),
        policy=cast("FusionPolicy", policy),
        budget=cast(
            "ListBudget",
            types.SimpleNamespace(
                config=types.SimpleNamespace(insertion_limit=3, legacy_rerank_top_n=60),
            ),
        ),
        gate=cast("PubChemGate", _Passthrough()),
        **health,  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
    )


_PAIR: Final = {
    "a": _ResultsStage(base=("a1",), second=("e1",)),
    "b": _ResultsStage(base=("b1",)),
}


def test_assemble_defaults_publish_healthy_stages_and_fill_the_tail_from_dreams() -> (
    None
):
    policy = _Policy({})
    dreams = _Dreams()
    got = _assemble(policy, _PAIR, dreams=dreams)
    assert policy.controlled == ["a", "b"]
    assert got.base_nonempty == 2
    assert got.baseline == {"a": ["a1", "e1"], "b": ["b1"]}
    assert got.reference == {"a": ["a1", "e1", "d1"], "b": ["b1", "d1", "d2"]}
    assert dreams.targets == [pytest.approx(100.0, abs=1e-3)] * 2
    assert policy.options["a"]["engine"] == ["e1"]
    assert policy.options["b"]["engine"] == []
    assert policy.options["a"]["prefusion"] == ["a1"]
    assert got.control_audit[0] == {
        "molecule_id": "a",
        "collapsed_source_votes": 2,
        "changed": False,
        "changed_top1": False,
    }
    assert got.insertion == {"rows": 2, "e2_ins": 1, "e1_ins": 0}


def test_assemble_tail_prefers_overflow_then_main_tail_then_the_second_engine() -> None:
    policy = _Policy({}, length=2, extra=("o3",))
    got = _assemble(
        policy,
        {
            "a": _ResultsStage(base=("a1", "bad"), second=("e2",)),
            "b": _ResultsStage(base=("b1", "bad"), tail=("bt",)),
        },
    )
    assert got.overflow == {"a": ["o3", "e2"]}
    assert got.reference == {"a": ["a1", "o3"], "b": ["b1", "bt"]}


@pytest.mark.parametrize(
    "final",
    [[], ["a1", "e1", "a3", "a4"], ["a1", "bad"], ["a1", "a1"]],
)
def test_assemble_rejects_each_invalid_final_list_alone(final: list[str]) -> None:
    with pytest.raises(AssertionError, match=r"^Invalid final metric identities: a$"):
        _ = _assemble(_Policy({"a": final}), _PAIR)


def test_assemble_accepts_a_final_list_exactly_at_the_published_length() -> None:
    got = _assemble(_Policy({"a": ["a1", "e1", "a3"]}), _PAIR)
    assert got.preinsert["a"] == ["a1", "e1", "a3"]


def test_assemble_requires_glacier_health_only_for_protonated_queries() -> None:
    with pytest.raises(AssertionError, match=r"^GLACIER stage is not complete: x$"):
        _ = _assemble(_Policy({}), _PAIR, glacier_status="x")
    got = _assemble(_Policy({}), _PAIR, adduct="[M-H]-", glacier_status="x")
    assert got.base_nonempty == 2


def test_assemble_rejects_unhealthy_ice_and_base_errors() -> None:
    with pytest.raises(AssertionError, match=r"^ICE stage is not complete: x$"):
        _ = _assemble(_Policy({}), _PAIR, ice_status="x")
    with pytest.raises(AssertionError, match=r"^Base inference errors: 1$"):
        _ = _assemble(_Policy({}), _PAIR, base_errors=1)


def test_main_missing_prepared_pool_is_an_error_at_read(tmp_path: Path) -> None:
    config = MainAssets.Config()
    config.pool_dir = tmp_path / "missing-pool"
    assets = config.make()
    with pytest.raises(FileNotFoundError, match="missing-pool"):
        _ = assets.read()


def test_complete_main_stage_is_lazy_ranks_and_releases_models(
    main_assets: MainAssets.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    original = MainBank.views

    def views(
        self: MainBank,
        spectra: Sequence[SpectrumRequest],
        **kwargs: Unpack[_ViewOptions],
    ) -> ModelViews:
        calls.append((id(self), len(spectra)))
        return original(self, spectra, **kwargs)  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.

    monkeypatch.setattr(MainBank, "views", views)
    config = MainResource.Config()
    config.assets = main_assets
    config.engine.max_query_spectra = 2
    resource = config.make()
    with torch.random.fork_rng(devices=[]):
        ranked = resource.rank("molecule", _resources_queries(), 100.0)
        assert resource.rank("again", _resources_queries(), 100.0) == ranked
    assert ranked.smiles
    assert ranked.popularity_applied
    assert ranked.library_max > 0
    assert len(ranked.smiles) == len(ranked.keys) == len(ranked.formulas)
    assert resource._inputs is not None
    assert (
        calls
        == [
            (id(resource._inputs.bank), 2),
            (id(resource._inputs.views_bank), 2),
        ]
        * 2
    )
    references = [
        weakref.ref(model)
        for model in (
            *resource._inputs.bank.models,
            *resource._inputs.views_bank.models,
        )
    ]
    assert all(reference() is not None for reference in references)
    resource.close()
    resource.close()
    assert all(reference() is None for reference in references)
    with pytest.raises(RuntimeError, match=r"^Main resource is closed\.$"):
        _ = resource.rank("closed", _resources_queries(), 100.0)


def test_main_data_reads_all_positions_and_casts_each_popularity_before_sum(
    main_assets: MainAssets.Config,
) -> None:
    config = main_assets
    with torch.random.fork_rng(devices=[]):
        inputs = config.make().read()
    assert inputs.families.pool is inputs.pool
    assert inputs.families.train_smiles == inputs.library.smiles
    assert inputs.train_fingerprints.shape == (3, 1)
    assert len(inputs.bank.models) == 1
    assert len(inputs.views_bank.models) == 3
    assert len(inputs.rankers.models) == 4
    assert all(len(model.trees) == 3 for model in inputs.rankers.models)
    np.testing.assert_array_equal(
        inputs.popularity,
        np.array([2049, 5, 9], np.float64),
        strict=True,
    )
    assert inputs.popularity.dtype == np.float64
    np.testing.assert_array_equal(
        inputs.families.generated_fragments.base_for_smiles("CCO"),
        np.array([70.1230000001, 71.1230000001]),
        strict=True,
    )
    assert inputs.families.generated_fragments.for_smiles("CCO")[0].dtype == np.float32


def _resources_queries() -> tuple[SpectrumRequest, ...]:
    return tuple(
        {
            "mz": [20.0, 30.0],
            "it": [1.0, 0.5],
            "prec": 101.00727645217,
            "mode": mode,
            "adduct": "[M+H]+",
            "ce": 10.0,
        }
        for mode in (1, -1, 1)
    )


def test_second_stage_reaches_native_index_models_and_accepts_empty_raw_key(
    tmp_path: Path,
) -> None:
    config = SecondResource.Config()
    assets = second_config(tmp_path)
    np.save(Path(assets.pool_dir) / "fp_bits.npy", np.arange(8, dtype=np.int64))
    states = second_states()
    assets.bank.model.channels_out = 8
    for state, pin in zip(
        states,
        (*assets.bank.merged, *assets.bank.single),
        strict=True,
    ):
        path = Path(pin.path)
        state["head.3.weight"] = torch.cat(
            # SecondBank adds one output row to the seven-bit source head.
            [state["head.3.weight"], torch.zeros((1, 9))],
        )
        state["head.3.bias"] = torch.cat([state["head.3.bias"], torch.zeros(1)])
        torch.save({"model": state, "nbits": 8, "d": 8, "layers": 1}, path)
        pin.sha256 = _seal(path)
    for path, width, group in (
        (Path(assets.own_training), 51, "S"),
        (Path(assets.pv_training), 31, "M"),
    ):
        data = SimulationRows.load(path, group)
        np.savez(
            path,
            X=np.resize(data.data, (len(data.data), width)),
            Y=data.labels,
            S=data.groups,
            M=data.groups,
            blocks=np.array(["base", "libx", "anapol", "fragx"]),
        )
    config.assets = assets
    resource = config.make()
    with torch.random.fork_rng(devices=[]):
        ranked = resource.rank("engine", _resources_queries())
        assert ranked is not None
        assert ranked.keys == ("",)
        assert len(ranked.smiles) == 1
        assert resource.rank("invalid", ()) is None
    assert resource._inputs is not None
    references = [
        weakref.ref(model)
        for model in (
            *resource._inputs.bank.merged,
            *resource._inputs.bank.single,
        )
    ]
    resource.close()
    resource.close()
    assert all(reference() is None for reference in references)
    with pytest.raises(RuntimeError, match=r"^Second resource is closed\.$"):
        _ = resource.rank("closed", ())


@pytest.mark.parametrize(("limit", "weight", "count"), [(1, 0.15, 1), (60, 0.0, 2)])
def test_main_resource_honors_shortlist_and_popularity_overrides(
    main_assets: MainAssets.Config,
    limit: int,
    weight: float,
    count: int,
) -> None:
    config = MainResource.Config()
    config.assets = main_assets
    config.limit = limit
    config.popularity_weight = weight
    resource = config.make()
    with torch.random.fork_rng(devices=[]):
        ranked = resource.rank("override", _resources_queries(), 100.0)
    assert len(ranked.keys) == count
    assert ranked.popularity_applied is False
    resource.close()


def test_pubchem_stage_uses_both_v3_models_and_releases_them(tmp_path: Path) -> None:
    config = PubChemResource.Config()
    config.assets = _pubchem_assets(tmp_path)
    resource = config.make()
    with torch.random.fork_rng(devices=[]):
        ranked = resource.rank("pubchem", _resources_queries())
        assert ranked is not None
        assert ranked["pc"] == ["CCO"]
        assert ranked["pc_keys"] == ["ethanol"]
        assert ranked["target"] == 100.0
        assert resource.rank("invalid", ()) is None
    assert resource._inputs is not None
    references = [weakref.ref(model) for model in resource._inputs.bank.models]
    resource.close()
    resource.close()
    assert resource._inputs is None
    assert resource._open is False
    assert all(reference() is None for reference in references)
    with pytest.raises(RuntimeError, match=r"^PubChem resource is closed\.$"):
        _ = resource.rank("closed", ())


def test_all_arm_constructors_are_io_free_and_close_before_first_rank() -> None:
    for config in (
        MainResource.Config(),
        SecondResource.Config(),
        PubChemResource.Config(),
    ):
        resource = config.copy_tree().finalize().make()
        assert resource._inputs is None
        resource.close()


def test_main_rejects_incomplete_sid_popularity_manifest_and_ranker_data(
    tmp_path: Path,
) -> None:
    config = _main_assets(tmp_path)
    manifest = Path(config.manifest.path)
    config.manifest.sha256 = "0" * 64
    with pytest.raises(RuntimeError, match=r"hashes to"):
        _ = config.make().read()
    config.manifest.sha256 = _seal(manifest)
    fingerprints = Path(config.fingerprints)
    # MainAssets.read requires uint8 rows with one byte for seven selected bits.
    np.save(fingerprints, np.zeros((3, 1), np.int8))
    with pytest.raises(TypeError, match=r"expected uint8, got int8\.$"):
        _ = config.make().read()
    # MainAssets.read pairs one packed row with each of the three training SIDs.
    np.save(fingerprints, np.zeros((2, 1), np.uint8))
    with pytest.raises(
        ValueError,
        match=r"^Main training fingerprints and structures differ from SID order\.$",
    ):
        _ = config.make().read()
    # MainAssets.read consumes the three-row, seven-bit packed training layout.
    np.save(fingerprints, np.zeros((3, 1), np.uint8))
    popularity = Path(config.popularity_dir)
    np.save(popularity / "pool_lsid.npy", np.zeros(3, np.int16))
    with pytest.raises(TypeError, match=r"expected floating, got int16\.$"):
        _ = config.make().read()
    np.save(popularity / "pool_lsid.npy", np.zeros(2, np.float16))
    with pytest.raises(
        ValueError,
        match=r"^Main popularity arrays differ from complete pool positions\.$",
    ):
        _ = config.make().read()
    np.save(popularity / "pool_lsid.npy", np.zeros(3, np.float16))
    text = ranker_texts()[0]
    names = list(RANKER_NAMES)
    rankers = Path(config.rankers.path)
    rankers.write_bytes(pickle.dumps({"features": names, "boosters": [text] * 3}))
    config.rankers.sha256 = _seal(rankers)
    with pytest.raises(
        ValueError,
        match=r"^Main inference requires all four rankers\.$",
    ):
        _ = config.make().read()


@pytest.mark.parametrize("second", [True, False])
def test_missing_secondary_inputs_fail_on_first_rank(
    tmp_path: Path,
    *,
    second: bool,
) -> None:
    if second:
        config = SecondResource.Config()
        assert isinstance(config.assets, SecondAssets.Config)
    else:
        config = PubChemResource.Config()
        assert isinstance(config.assets, PubChemAssets.Config)
    config.assets.pool_dir = tmp_path / "missing"
    resource = config.make()
    with pytest.raises(FileNotFoundError, match="missing"):
        _ = resource.rank("missing", ())
    resource.close()


def test_main_defaults_keep_generation_geometry() -> None:
    config = MainResource.Config().copy_tree().finalize()
    assert isinstance(config.assets, MainAssets.Config)
    assert config.limit == 60
    assert config.popularity_weight == 0.15
    assert config.generation.parents == 6
    assert config.generation.min_similarity == 0.3
    assert config.generation.per_parent == 60
    assert config.generation.total == 150
    assert config.generation.tolerance == 0.004
    assert config.generation.ppm == 10
    assert config.generation.max_steps == 2
    assert config.generation.application_budget == 800
    assert config.generation.stage_parents == 24
    assert config.generation.first_products == 32
    assert config.generation.later_products == 16
    assert SecondResource.Config().prescreen == 80
    assert SecondResource.Config().limit == 40
    screen = PubChemResource.Config().screen
    assert screen.pass_one == 5000
    assert screen.popularity_weight == 0.25
    assert screen.popularity_union == 200
    assert screen.limit == 25
    assert screen.max_spectra == 16


def test_main_generation_keeps_fallback_pool_aliases_and_all_support_logs(
    tmp_path: Path,
) -> None:
    config = MainResource.Config()
    assets = _main_assets(tmp_path)
    config.assets = assets
    config.engine.max_query_spectra = 2
    pool_dir = Path(assets.pool_dir)
    pool = CandidatePool.load(pool_dir)
    write_pool(pool_dir, pool=replace(pool, mass=np.full(3, 100.002)))
    chemistry_dir = Path(assets.chemistry.working_dir)
    chemistry = IdentityTable.Config(working_dir=chemistry_dir).make()
    reactions_dir = Path(assets.reactions_dir)
    derivation_dir = Path(assets.derivation_dir)
    product = {
        "smiles": "raw-product",
        "standardized": "CCO",
        "key": "raw-key",
        "standard_key": "a",
        "formula": "C2H6O",
        "metric": "null",
        "standard_metric": "null",
        "standard_formula_lookup": "C2H6O",
        "properties_valid": True,
        "standard_formula": "C2H6O",
        "standard_mass": 100.0,
        "standard_heavy": 3,
        "fingerprint_valid": True,
    }
    parquet.write_table(
        Table.from_pylist([product, *[dict(row) for row in chemistry.metadata]]),
        chemistry_dir / "molecules.parquet",
    )
    packed = np.zeros(2069, np.uint8)
    packed[0] = 32
    np.save(
        chemistry_dir / "fingerprints.npy",
        np.vstack((packed, chemistry.packed)),
    )
    _reaction_data_write(
        reactions_dir,
        headers=[
            normalize_request({"smiles": parent, "delta": 0.0, "tolerance": 0.004})
            for parent in ("parent0", "parent1")
        ],
        responses=[
            {
                "input_smiles": parent,
                "products": ["raw-product", "raw-product"],
                "plans": [[0], [1]],
                "matches": [],
                "applications": "both complete paths",
            }
            for parent in ("parent0", "parent1")
        ],
        manifest={"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
    )
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O", "demethyl_O"]}),
        reactions_dir / "rules.parquet",
    )
    parquet.write_table(
        Table.from_pylist(
            [
                {
                    "parent_smiles": parent,
                    "combo": [combo],
                    "max_first": 32,
                    "max_second": 16,
                    "max_mols": 24,
                    "terminal_count": 0,
                    "applications": "explicit zero terminal paths",
                }
                for parent in ("parent0", "parent1")
                for combo in ("methyl_O", "demethyl_O")
            ],
        ),
        derivation_dir / "requests.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["CCO", "C", "raw-product"],
                "key": ["a", "b", "a"],
                "skeleton": ["s0", "s1", "s0"],
            },
        ),
        derivation_dir / "identities.parquet",
    )
    names = list(RANKER_NAMES)
    column = names.index("derivation__dg_n_combo")
    lines = [
        "num_class=1",
        "num_tree_per_iteration=1",
        f"max_feature_idx={len(names) - 1}",
        "feature_names=" + " ".join(names),
        "objective=lambdarank",
    ]
    for step in range(3):
        lines.extend(
            (
                f"Tree={step}",
                "num_leaves=2",
                "num_cat=0",
                f"split_feature={column}",
                "decision_type=2",
                "left_child=-1",
                "right_child=-2",
                "leaf_count=2 2",
                "internal_count=4",
                "is_linear=0",
                "split_gain=1",
                "threshold=1.5",
                "leaf_value=0 1",
                "leaf_weight=2 2",
                "internal_value=0",
                "internal_weight=4",
                "shrinkage=0.1",
            ),
        )
    text = "\n".join([*lines, "end of trees"])
    rankers = Path(assets.rankers.path)
    rankers.write_bytes(pickle.dumps({"features": names, "boosters": [text] * 4}))
    assets.rankers.sha256 = _seal(rankers)
    resource = config.make()
    with torch.random.fork_rng(devices=[]):
        ranked = resource.rank("generation", _resources_queries(), 100.0)
    assert ranked.keys[0] == "a"
    assert ranked.formulas[0] == "C2H6O|gen"
    assert ranked.smiles[0] == "CCO"
    assert resource._inputs is not None
    assert len(resource._inputs.pool.window(100, ppm=10)) == 0
    assert len(resource._inputs.pool.window(100, ppm=30)) == 3
    resource.close()


def _pubchem_assets(directory: Path) -> PubChemAssets.Config:
    config = PubChemAssets.Config()
    config.pool_dir = config.store_dir = directory
    config.popularity_dir = config.chemistry_dir = directory
    _, pool, _ = retrieval_inputs()
    write_pool(directory, pool=replace(pool, bits=np.arange(7, dtype=np.int64)))
    for name, values in (
        ("pc_mass", np.array([100, 100], np.float64)),
        ("pc_off", np.array([0, 3, 4], np.int64)),
        ("pc_smiles", np.frombuffer(b"CCOC", np.uint8)),
        ("pc_lsid", np.array([0.5, 1.5], np.float16)),
        ("pc_lpmid", np.array([2.5, 3.5], np.float16)),
        ("ecfp4", np.arange(1024, dtype=np.uint8).reshape(2, 512)),
        ("raw_fp", np.arange(4138, dtype=np.uint8).reshape(2, 2069)),
    ):
        np.save(directory / f"{name}.npy", values)
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["CCO", "C"],
                "parsed": [True, True],
                "raw_valid": [True, False],
                "key": ["ethanol", None],
            },
        ),
        directory / "molecules.parquet",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {
                    "format_version": 1,
                    "rdkit": IDENTITY_RDKIT,
                    "raw_bits": 16_551,
                    "ecfp4_bits": 4096,
                },
            ),
        ),
    )
    config.bank = save_bank(directory, [0, 1])
    return config


@pytest.fixture(scope="module")
def main_assets(tmp_path_factory: pytest.TempPathFactory) -> MainAssets.Config:
    """Share one read-only tiny release across the tests that only read it."""
    return _main_assets(tmp_path_factory.mktemp("main_assets"))


def _main_assets(directory: Path) -> MainAssets.Config:
    families, _ = _inputs(directory)
    config = MainAssets.Config()
    config.pool_dir = config.library_dir = config.popularity_dir = directory
    manifest = directory / "release.json"
    manifest.write_text(
        json.dumps(to_plain({"fixture": "complete tiny resource release"})),
    )
    config.manifest.path = manifest
    config.manifest.sha256 = _seal(manifest)
    config.fingerprints = directory / "train_fp_sel.npy"
    config.pool_structures_dir = directory / "pool"
    config.train_structures_dir = directory / "train"
    config.generated_structures_dir = directory / "generated"
    config.fragments_dir = directory / "fragments"
    config.intrinsic_dir = directory / "intrinsic"
    config.fragnet_dir = directory / "fragnet"
    families_dir = directory / "families"
    config.pool_families_dir = families_dir
    families_dir.mkdir()
    pool = families.pool
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": [100.0, 100.0, 100.0],
                "key": pool.keys,
                "smiles": pool.smiles,
                "formula": pool.formulas,
                "n_heavy": pool.heavy_atoms,
                "src": pool.source,
                "train_sid": pool.train_sid,
            },
        ),
        directory / "pool_meta.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 1, 2],
                "smiles": families.train_smiles,
                "mass": [100.0, 100.0, 100.0],
                "key": ["p0", "p1", "p0"],
            },
        ),
        directory / "train_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 1, 2],
                "lib": [2, 3, 2],
                "mode": [1, -1, 1],
                "instr": [0, 0, 0],
                "adduct_ix": [0, 0, 0],
                "precursor_mz": [101.00727645217] * 3,
                "n_clean": [2, 2, 2],
            },
        ),
        directory / "spec_meta.parquet",
    )
    for filename, array in (
        ("spec_off.npy", np.arange(4, dtype=np.int64) * 2),
        ("spec_mz.npy", np.tile(np.array([20, 30], np.float32), 3)),
        ("spec_it.npy", np.tile(np.array([1, 0.5], np.float32), 3)),
        ("pool_fp.npy", pool.packed),
        ("fp_bits.npy", np.arange(7, dtype=np.int64)),
        ("pool_frag_off.npy", pool.fragment_offsets),
        ("pool_frag_mass.npy", pool.fragment_mass),
        ("train_fp_sel.npy", np.array([[32], [64], [32]], np.uint8)),
        ("pool_lsid.npy", np.array([2048, 2, 4], np.float16)),
        ("pool_lpmid.npy", np.array([1, 3, 5], np.float16)),
    ):
        np.save(directory / filename, array)
    for name, array in (
        ("pool_f1_off.npy", families.pool_families.offsets),
        ("pool_f1_mass.npy", families.pool_families.mass),
        ("pool_cap.npy", families.pool_families.capabilities),
    ):
        np.save(families_dir / name, array)
    (families_dir / "caps.txt").write_text(
        "\n".join(families.pool_families.names),
    )
    np.save(
        Path(config.fragments_dir) / "base_mass.npy",
        np.array([70.1230000001, 71.1230000001]),
    )
    chemistry_dir = directory / "identities"
    config.chemistry.working_dir = chemistry_dir
    chemistry_dir.mkdir()
    write_identity(chemistry_dir)
    reactions_dir = directory / "reactions"
    config.reactions_dir = reactions_dir
    reactions_dir.mkdir()
    _reaction_data_write(
        reactions_dir,
        headers=[
            normalize_request({"smiles": s, "delta": 0.0, "tolerance": 0.004})
            for s in ("parent0", "parent1")
        ],
        responses=[
            {
                "input_smiles": s,
                "products": [],
                "plans": [],
                "matches": [],
                "applications": "complete empty result",
            }
            for s in ("parent0", "parent1")
        ],
        manifest={"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
    )
    derivation_dir = directory / "derivation"
    config.derivation_dir = derivation_dir
    derivation_dir.mkdir()
    (derivation_dir / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )
    parquet.write_table(
        Table.from_pydict(
            {
                name: []
                for name in (
                    "parent_smiles",
                    "combo",
                    "max_first",
                    "max_second",
                    "max_mols",
                    "terminal_count",
                    "applications",
                )
            },
        ),
        derivation_dir / "requests.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                name: []
                for name in (
                    "request",
                    "position",
                    "raw",
                    "standardized",
                    "key",
                    "skeleton",
                    "environments",
                )
            },
        ),
        derivation_dir / "paths.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {"smiles": ["CCO", "C"], "key": ["a", "b"], "skeleton": ["s0", "s1"]},
        ),
        derivation_dir / "identities.parquet",
    )
    config.priors.path = directory / "priors.pkl"
    Path(config.priors.path).write_bytes(
        pickle.dumps({"glob": 0.2, "tf": {}, "env": {1: {}, 2: {}, 3: {}}}),
    )
    config.priors.sha256 = _seal(Path(config.priors.path))
    stage_a = pipeline_stage_a()
    arrays: dict[str, NDArray[np.generic]] = {"alpha": np.array(stage_a.alpha)}
    chains: list[NDArray[np.object_]] = []
    for ci, (chain, tables) in enumerate(
        zip(stage_a.chains, stage_a.tabs, strict=True),
    ):
        chains.append(np.array([np.array(level) for level in chain], dtype=object))
        for li, (keys, values) in enumerate(tables):
            arrays[f"k_{ci}_{li}"], arrays[f"l_{ci}_{li}"] = keys, values
    arrays["chains"] = np.array(chains, dtype=object)
    config.stage_a.path = directory / "stageA.npz"
    np.savez(Path(config.stage_a.path), allow_pickle=True, **arrays)
    config.stage_a.sha256 = _seal(Path(config.stage_a.path))
    config.stage_b.path = directory / "stageB.txt"
    Path(config.stage_b.path).write_text(pipeline_stage_b())
    config.stage_b.sha256 = _seal(Path(config.stage_b.path))
    config.rankers.path = directory / "rankers.pkl"
    Path(config.rankers.path).write_bytes(
        pickle.dumps({"features": list(RANKER_NAMES), "boosters": ranker_texts()}),
    )
    config.rankers.sha256 = _seal(Path(config.rankers.path))
    config.bank = save_bank(directory, [0])
    config.views_bank = save_bank(directory, [0, 1, 2])
    return config


def _seal(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ViewOptions(TypedDict, total=False):
    merged_counts: Sequence[int]


@dataclass(slots=True, kw_only=True)
class _Arm[T]:
    values: dict[str, T]
    queries: QueryDataset
    name: str
    calls: SimpleQueue[str]
    same_instance: bool = True

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float | None = None,
    ) -> T:
        molecule = next(value for value in self.queries.molecules if value.mid == mid)
        expected = molecule.main if self.name == "main" else molecule.raw
        if self.same_instance:
            assert spectra is expected
        else:
            assert torch.equal(_text(spectra), _text(expected))
        assert target == (molecule.target if self.name == "main" else None)
        self.calls.put(f"{self.name}:{mid}")
        return self.values[mid]

    def close(self) -> None:
        self.calls.put(f"close:{self.name}")


@dataclass(slots=True, kw_only=True)
class _RunStage:
    name: str
    items: Tensor
    scores: dict[str, dict[str, float | None]]
    directory: Path
    calls: SimpleQueue[str]
    status: Literal["ok", "budget", "error"] = "ok"

    def run(
        self,
        molecules: Sequence[SimulationMolecule],
        *,
        publish: Callable[[dict[str, dict[str, float | None]]], None],
    ) -> SimulatorResult:
        assert torch.equal(_text(_items(molecules)), self.items)
        self.calls.put(self.name)
        publish({value.mid: dict.fromkeys(value.candidates) for value in molecules})
        path = self.directory / f"{self.name}_work/{self.name}_out.json"
        initial = json_loads(path.read_text(), dict[str, dict[str, float | None]])
        assert initial == {
            value.mid: dict.fromkeys(value.candidates) for value in molecules
        }
        publish(self.scores)
        assert (
            json_loads(path.read_text(), dict[str, dict[str, float | None]])
            == self.scores
        )
        return SimulatorResult(
            scores=self.scores,
            jobs=SimulationJobState(batch_size=96 if self.name == "ice" else 32),
            status=self.status,
            scored_molecules=2,
            scored_candidates=4,
            errors=[]
            if self.status == "ok"
            else ["complete retained failure evidence"],
        )


@pytest.mark.parametrize("precreated", [False, True])
def test_complete_native_outer_join_preserves_every_field_and_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    precreated: bool,
) -> None:
    result = replay_recipe(tmp_path, monkeypatch, precreated=precreated)
    assert_recipe_output(result, output=tmp_path / "output")


def assert_recipe_output(result: RecipeResults, *, output: Path) -> None:
    """Check every publication and both simulator receipts of a recipe run."""
    assert_outer_publication(result, output)
    assert not tuple(output.rglob("*.pending"))
    for name in ("ice", "gl"):
        assert json_loads(
            (output / f"{name}_work/{name}_out.json.meta.json").read_text(),
            dict[str, object],
        ) == {
            "status": "ok",
            "device": "cpu",
            "rdkit": SIMULATOR_RDKIT,
            "n_mols_scored": 2,
            "n_scored": 4,
            "errors": [],
        }


@pytest.mark.parametrize("stage", ["ice", "gl"])
def test_incomplete_simulator_maps_remain_durable_without_final_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    expected_message = (
        "ICE stage is not complete: budget"
        if stage == "ice"
        else "GLACIER stage is not complete: error"
    )
    with pytest.raises(AssertionError, match=f"^{expected_message}$"):
        replay_recipe(tmp_path, monkeypatch, unhealthy=stage)
    output = tmp_path / "output"
    assert not (output / "submission.csv").exists()
    assert not (output / "experiment_verification.json").exists()
    for name in ("ice", "gl"):
        metadata = json_loads(
            (output / f"{name}_work/{name}_out.json.meta.json").read_text(),
            dict[str, object],
        )
        expected_status = (
            ("budget" if name == "ice" else "error") if name == stage else "ok"
        )
        assert metadata["status"] == expected_status
        assert metadata["errors"] == (
            ["complete retained failure evidence"] if name == stage else []
        )


def replay_recipe(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unhealthy: str = "",
    precreated: bool = False,
    native_entry: bool = False,
) -> RecipeResults:
    """Bind the literal arm outputs at each qualified seam of the recipe."""
    monkeypatch.setitem(sys.modules, "rdkit", None)
    query_path = root / "queries.parquet"
    parquet.write_table(Table.from_pylist(query_rows()), query_path)
    queries = read_queries(query_path)
    sample = outer_sample(root / "sample.csv")
    main, second, pc = outer_main(), outer_second(), outer_pubchem()
    prepared = Prepared()
    union = simulator_inputs(
        main,
        second=second,
        pubchem=pc,
        queries=queries,
        formula=prepared.formula,
        budget=released_budget(),
        gate=released_gate(),
    )
    directory = root / "output"
    directory.mkdir()
    if precreated:
        for stage_name in ("ice", "gl"):
            (directory / f"{stage_name}_work").mkdir()
    calls = SimpleQueue[str]()
    ice, gl = (
        _RunStage(
            name=name,
            items=_text(_items(molecules)),
            scores=scores,
            directory=directory,
            calls=calls,
            status=("budget" if name == "ice" else "error")
            if name == unhealthy
            else "ok",
        )
        for name, molecules, scores in zip(
            ("ice", "gl"),
            (union.ice, union.glacier),
            outer_scores(),
            strict=True,
        )
    )
    result = (native_run if native_entry else run_recipe)(
        queries,
        sample_csv=sample,
        directory=directory,
        main=_Arm(values=main, queries=queries, name="main", calls=calls),
        second=_Arm(values=second, queries=queries, name="second", calls=calls),
        pubchem=_Arm(
            values={
                molecule.mid: pc.get(molecule.mid) for molecule in queries.molecules
            },
            queries=queries,
            name="pubchem",
            calls=calls,
        ),
        ice=ice,
        glacier=gl,
        identities=prepared,
        graphs=prepared,
        fragment_gap=FragmentGap.Config().make(),
        started=0.0,
        clock=lambda: 0.0,
        device="cpu",
        policy=released_policy(),
        budget=released_budget(),
        gate=released_gate(),
        provenance=released_provenance(),
    )
    expected_calls = [
        *(
            f"main:{molecule.mid}"
            for molecule in queries.molecules
            if molecule.target is not None
        ),
        "close:main",
        *(f"second:{mid}" for mid in queries.engine_order),
        "close:second",
        *(f"pubchem:{molecule.mid}" for molecule in queries.molecules),
        "close:pubchem",
        "ice",
        "gl",
    ]
    assert [calls.get_nowait() for _ in expected_calls] == expected_calls
    assert calls.empty()
    return result


class _EntryStage:
    class Config(Fig["_EntryStage"]):
        runner: SimulationRunner | None = None
        """Stored simulator seam, preserving every panel and map."""

    def __init__(self, config: Config) -> None:
        assert config.runner is not None
        self.runner = config.runner

    def run(
        self,
        molecules: Sequence[SimulationMolecule],
        *,
        publish: Callable[[dict[str, dict[str, float | None]]], None],
    ) -> SimulatorResult:
        return self.runner.run(molecules, publish=publish)


def native_run(
    queries: QueryDataset,
    *,
    sample_csv: Path,
    directory: Path,
    main: MainRanker,
    second: SecondRanker,
    pubchem: PubChemRanker,
    ice: SimulationRunner,
    glacier: SimulationRunner,
    identities: ResultIdentities,
    graphs: GapGraphs,
    fragment_gap: FragmentGap,
    started: float,
    device: str,
    policy: FusionPolicy,
    budget: ListBudget,
    gate: PubChemGate,
    provenance: PublicationProvenance,
    dreams: DreamsStore | None = None,
    gpu: str | None = None,
    clock: Callable[[], float],
) -> RecipeResults:
    """Run the actual native entry with stored numerical stage boundaries."""
    del queries, started
    assert dreams is None
    assert gpu is None
    directory.rmdir()
    config = PublicRecipe.Config()
    config.base_dir = None
    config.dataset_dir = directory.parent
    config.queries = "queries.parquet"
    config.sample_csv = sample_csv.relative_to(directory.parent)
    config.working_dir = directory
    config.input_dir = "input"
    config.protected_inputs = ["input"]
    config.device = device
    config.policy = policy.config
    config.budget = budget.config
    config.gate = gate.config
    config.provenance = provenance.config
    for arm in (main, second, pubchem):
        assert isinstance(arm, _Arm)
        arm.same_instance = False
    config.main = InlineConfig(_same, value=main)
    config.second = InlineConfig(_same, value=second)
    config.pubchem = InlineConfig(_same, value=pubchem)
    ice_config, gl_config = _EntryStage.Config(), _EntryStage.Config()
    ice_config.runner, gl_config.runner = ice, glacier
    config.ice, config.glacier = ice_config, gl_config
    config.identities = InlineConfig(_same, value=identities)
    config.graphs = InlineConfig(_same, value=graphs)
    config.fragment_gap = InlineConfig(_same, value=fragment_gap)
    config.clock = clock
    results = config.make().run()
    assert (directory / "config.txt").is_file()
    return results


def _same[T](value: T) -> T:
    return value


class FixtureBackend:
    """Inject only the already-qualified numerical prediction/clock boundary."""

    class Config(Fig["FixtureBackend"]):
        boundary: _Boundary = field(default_factory=lambda: _Boundary(glacier=False))

    def __init__(self, config: Config) -> None:
        self.inner = config.boundary.load()
        self.feature_failures = self.inner.feature_failures

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        try:
            return self.inner.predict(jobs)
        finally:
            self.feature_failures = self.inner.feature_failures

    def release_cache(self) -> None:
        self.inner.release_cache()


class _Time:
    def __init__(self) -> None:
        self.clock = _Clock()

    def perf_counter(self) -> float:
        return self.clock.time()


class _DeviceCounters:
    def __init__(self) -> None:
        self.predictions = 0
        self.releases = 0


class DeviceFailureBackend:
    """Count allocator callbacks without using a real accelerator."""

    class Config(Fig["DeviceFailureBackend"]):
        counters: _DeviceCounters = field(default_factory=_DeviceCounters)

    def __init__(self, config: Config) -> None:
        self.counters = config.counters
        self.feature_failures = 0

    def predict(self, jobs: list[SimulationJob]) -> list[PredictedPeaks | None]:
        del jobs
        self.counters.predictions += 1
        raise RuntimeError("device prediction failure")

    def release_cache(self) -> None:
        self.counters.releases += 1


def _finite(result: SimulatorResult) -> int:
    return sum(
        value is not None for row in result.scores.values() for value in row.values()
    )


def test_complete_stages_all_publications_caches_order_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _write(
        tmp_path,
        admission_rows(),
    )
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"complete declared scientific data")
    for family in ("iceberg", "glacier"):
        results: dict[str, SimulatorResult] = {}
        for case, budget in (("complete", 0.0), ("before", 0.5), ("partial", 2.75)):
            boundary = _Boundary(glacier=family == "glacier")
            backend = FixtureBackend.Config()
            backend.boundary = boundary
            config = (
                glacier_stage(backend=GlacierBackend.Config())
                if family == "glacier"
                else SimulatorStage.Config()
            )
            config.admission_dir = tmp_path
            config.backend = backend
            config.budget.seconds = budget or float("inf")
            config.budget.clock = _Time().perf_counter
            config.batch_size = config.chunk_jobs = 2
            config.asset_seals = {asset: hashlib.sha256(asset.read_bytes()).hexdigest()}
            result = config.make().run(_molecules(), publish=boundary.publish)
            results[case] = result
            # The stage opens with every admitted candidate unscored and ends on its map.
            assert all(
                value is None
                for row in boundary.publications[0].values()
                for value in row.values()
            )
            assert boundary.publications[-1] == result.scores
            assert result.jobs.batch_size == 2
        complete, before, partial = (
            results[name] for name in ("complete", "before", "partial")
        )
        assert complete.status == "ok"
        assert before.status == partial.status == "budget"
        assert (
            before.jobs.predicted <= partial.jobs.predicted <= complete.jobs.predicted
        )
        assert _finite(before) <= _finite(partial) <= _finite(complete)
        assert set(complete.scores) == set(before.scores) == set(partial.scores)


def test_defaults_and_bad_checkpoint_pin_cannot_load_a_backend(
    tmp_path: Path,
) -> None:
    ice, gl = SimulatorStage.Config(), glacier_stage(backend=GlacierBackend.Config())
    assert isinstance(ice.scheduling, IceScheduling.Config)
    assert (ice.budget.seconds, ice.batch_size, ice.chunk_jobs) == (5400.0, 96, 3000)
    assert isinstance(gl.scheduling, GlacierScheduling.Config)
    assert (gl.budget.seconds, gl.batch_size, gl.chunk_jobs) == (4000.0, 32, 3000)
    _write(
        tmp_path,
        admission_rows(),
    )
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"corrupted data")
    boundary = _Boundary(glacier=False)
    backend = FixtureBackend.Config()
    backend.boundary = boundary
    ice.admission_dir = tmp_path
    ice.backend = backend
    ice.asset_seals = {asset: "wrong pin"}
    ice.budget.clock = _Time().perf_counter
    result = ice.make().run(_molecules(), publish=boundary.publish)
    assert result.status == "error"
    assert result.errors == [f"ValueError('Simulator asset digest differs: {asset}')"]
    assert len(boundary.publications) == 2
    assert boundary.publications[0] == boundary.publications[1]
    assert boundary.backend.calls == []


def test_every_failed_prediction_batch_releases_the_backend_cache(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        admission_rows(),
    )
    backend = DeviceFailureBackend.Config()
    config = SimulatorStage.Config()
    config.admission_dir = tmp_path
    config.backend = backend
    config.budget.clock = _Time().perf_counter
    boundary = _Boundary(glacier=False)
    molecule = replace(_molecules()[0], candidates=("CCO", "OCC"))
    result = config.make().run((molecule,), publish=boundary.publish)
    assert backend.counters.predictions == 3
    assert backend.counters.releases == 3
    assert result.status == "ok"
    assert result.jobs.predicted == result.jobs.failed == 2
    assert result.scores == {"a": {"CCO": None, "OCC": None}}


def test_a_backend_without_a_base_directory_inherits_the_stage_dataset_root(
    tmp_path: Path,
) -> None:
    backend = GlacierBackend.Config()
    backend.base_dir = None
    stage = glacier_stage(backend=backend)
    stage.base_dir = tmp_path
    assert stage.finalize().backend is backend
    assert backend.base_dir == tmp_path


def test_the_default_admission_directory_is_where_the_preparer_writes() -> None:
    assert SimulatorStage.Config().admission_dir == ADMISSION_DIR


def test_native_configuration_is_io_free_and_declares_every_stage() -> None:
    config = PublicRecipe.Config().copy_tree().finalize()
    assert Path(config.queries).name == "test.parquet"
    assert Path(config.sample_csv).name == "sample_submission.csv"
    assert config.working_dir == Path("/opt/scratch/runs")
    assert isinstance(config.ice, SimulatorStage.Config)
    assert isinstance(config.glacier, SimulatorStage.Config)
    assert config.dreams is None
    assert "compile_notebook" not in config.pformat(hide_default_values=False)
    assert config.make().config == config


def test_run_identity_names_the_output_directory() -> None:
    config = PublicRecipe.Config()
    config.base_dir = "/root"
    config.working_dir = "/runs/{study_name}/{experiment_name}"
    config.study_name, config.experiment_name = "study", "exp007"
    assert config.copy_tree().finalize().working_dir == Path("/root/runs/study/exp007")


def test_declared_root_reaches_recipe_tables_and_every_relative_input(
    tmp_path: Path,
) -> None:
    config = PublicRecipe.Config()
    config.base_dir = tmp_path
    config.dreams = "dreams"
    config.sample_csv = "separate/sample.csv"
    config = config.finalize()
    dataset = tmp_path / "datasets/casmi2026"
    assert config.dataset_dir == dataset
    assert (
        config.queries
        == dataset / "inputs/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
    )
    assert config.sample_csv == dataset / "separate/sample.csv"
    assert config.input_dir == dataset / "inputs"
    assert config.dreams == dataset / "dreams"
    assert config.protected_inputs == [dataset]
    assert isinstance(config.identities, IdentityTable.Config)
    assert isinstance(config.graphs, GapGraphTable.Config)
    assert config.identities.working_dir == dataset / "native-chemistry/identities"
    assert config.graphs.working_dir == dataset / "native-chemistry/fragment-gap"


def test_native_job_snapshots_config_before_missing_input(tmp_path: Path) -> None:
    config = native_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        config.make().run()
    assert (tmp_path / "runs/native/output/config.txt").read_text() == (
        config.pformat(hide_default_values=False) + "\n"
    )
    assert not (tmp_path / "runs/native/output/submission.csv").exists()
    with pytest.raises(FileExistsError):
        config.make().run()
    with pytest.raises(
        ValueError,
        match=r"^PublicRecipe accepts configuration fields only\.$",
    ):
        config.make().run("--truncate")


def test_declared_root_and_device_reach_every_resource_and_pinned_checkpoint(
    tmp_path: Path,
) -> None:
    config = exp000()
    config.base_dir = tmp_path
    config.device = "meta"
    resolved = config.copy_tree().finalize()
    dataset = tmp_path / "datasets/casmi2026"
    for resource in (
        resolved.main,
        resolved.second,
        resolved.pubchem,
        resolved.ice,
        resolved.glacier,
    ):
        owners = [match.config for match in traverse(resource, HasBaseDir)]
        assert owners
        assert all(owner.base_dir == dataset for owner in owners)
        paths = [match.config for match in traverse(resource, Path)]
        assert paths
        assert all(path.is_relative_to(dataset) for path in paths)
        devices = [match.config for match in traverse(resource, DEVICE_OWNERS)]
        assert devices
        assert all(owner.device == "meta" for owner in devices)
    for stage in (resolved.ice, resolved.glacier):
        assert isinstance(stage, SimulatorStage.Config)
        assert all(Path(path).is_relative_to(dataset) for path in stage.asset_seals)


@pytest.mark.parametrize("contains", [False, True])
def test_native_job_rejects_output_inside_or_above_protected_inputs(
    tmp_path: Path,
    *,
    contains: bool,
) -> None:
    config = native_config(tmp_path)
    config.input_dir = "external-inputs"
    config.working_dir = tmp_path / "input" if contains else tmp_path / "input/run"
    config.protected_inputs = ["input/data"] if contains else ["input"]
    with pytest.raises(
        ValueError,
        match=r"^Recipe output overlaps protected inputs\.$",
    ):
        config.make().run()
    assert not Path(config.working_dir).exists()


def test_native_entry_runs_the_complete_outer_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = replay_recipe(tmp_path, monkeypatch, native_entry=True)
    assert_recipe_output(result, output=tmp_path / "output")
    assert (tmp_path / "output/dreams.json").read_text() == (
        '{"used": false, "path": "None"}\n'
    )


def native_config(root: Path) -> PublicRecipe.Config:
    """Declare only private test inputs and an unowned output directory."""
    config = PublicRecipe.Config()
    config.base_dir = None
    config.dataset_dir = root
    config.queries = "queries.parquet"
    config.sample_csv = "sample.csv"
    config.working_dir = root / "runs/native/output"
    config.input_dir = "input"
    config.protected_inputs = ["prepared"]
    return config


def test_complete_native_parent_headers_and_external_window_positions(
    tmp_path: Path,
) -> None:
    library, _, spectra = retrieval_inputs()
    raw: tuple[SpectrumRequest, ...] = tuple(
        {**spectrum, "prec": 101.00727645217, "adduct": "[M+H]+"}
        for spectrum in spectra
    )
    molecule = MoleculeQueries(
        mid="q",
        source_id="q",
        main=tuple(spectra),
        raw=raw,
        finite_mass=np.array([100.0]),
        target=100.0,
    )
    queries = QueryDataset(
        rows=(),
        molecules=(molecule, replace(molecule, mid="unknown", target=None, raw=())),
        engine_order=("q", "unknown"),
    )
    search = MainRetrieval.Config()
    search.n_analog = 2
    retrieval = search.make()
    generator = ReactionScreen.Config().make()
    text = ("outside", " A ", "B", "B", "outside")
    encoded = [value.encode("ascii") for value in text]
    store = PubChemStore(
        mass=np.array([99, 99.999, 100, 100.001, 101], np.float64),
        offsets=np.concatenate(
            [[0], np.cumsum([len(value) for value in encoded])],
        ).astype(np.int64),
        buffer=np.frombuffer(b"".join(encoded), np.uint8),
    )
    np.save(tmp_path / "dreams_np_mass.npy", np.array([100.0005, 100, 101], np.float32))
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["dream0", "dream1", "far"]))
    dreams = DreamsStore.load(tmp_path)
    assert dreams is not None
    plan = plan_requests(
        queries,
        library=library,
        retrieval=retrieval,
        generator=generator,
        pubchem=store,
        ppm=10.0,
        max_spectra=16,
        dreams=dreams,
    )
    parents = retrieval.analogs(spectra, library=library, target=100.0)
    assert len(plan.reactions) == 2
    for row, parent in zip(plan.reactions, parents, strict=True):
        assert row.molecule_id == "q"
        assert row.sid == parent.sid
        assert row.similarity == parent.similarity
        assert row.target == 100.0
        assert row.header == {
            "smiles": library.smiles[parent.sid],
            "delta": 100.0 - library.structure_mass.item(parent.sid),
            "operation": "derive",
            "rule": -1,
            "tolerance": 0.004,
            "max_steps": 2,
            "max_products": 60,
            "allow_isomerization": True,
            "application_budget": 800,
            "stage_parents": 24,
            "first_products": 32,
            "later_products": 16,
        }
    assert plan.pubchem == tuple(
        WindowRequest(molecule_id="q", row=i, smiles=text[i]) for i in (1, 2, 3)
    )
    assert plan.dreams == tuple(
        WindowRequest(molecule_id="q", row=i, smiles=f"dream{i}") for i in (1, 0)
    )
    empty = plan_requests(
        QueryDataset(rows=(), molecules=(), engine_order=()),
        library=library,
        retrieval=retrieval,
        generator=generator,
        pubchem=store,
        ppm=10.0,
        max_spectra=16,
    )
    assert empty.reactions == empty.pubchem == empty.dreams == ()
    absent = plan_requests(
        replace(queries, molecules=(replace(molecule, main=(), raw=(), target=None),)),
        library=library,
        retrieval=retrieval,
        generator=generator,
        pubchem=store,
        ppm=10.0,
        max_spectra=16,
        dreams=dreams,
    )
    assert absent.reactions == absent.pubchem == absent.dreams == ()
    limited = plan_requests(
        queries,
        library=library,
        retrieval=retrieval,
        generator=generator,
        pubchem=store,
        ppm=10.0,
        max_spectra=0,
    )
    assert limited.reactions == ()
    assert limited.pubchem == plan.pubchem
    assert limited.dreams == ()
    capped = plan_requests(
        replace(
            queries,
            molecules=(replace(molecule, main=(spectra[2],) * 16 + (spectra[0],)),),
        ),
        library=library,
        retrieval=retrieval,
        generator=generator,
        pubchem=store,
        ppm=10.0,
        max_spectra=16,
    )
    assert capped.reactions == ()
    assert capped.pubchem == plan.pubchem


def test_tail_planning_keeps_positive_subdalton_and_never_opens_invalid_targets(
    tmp_path: Path,
) -> None:
    library, _, spectra = retrieval_inputs()
    molecule = MoleculeQueries(
        mid="small",
        source_id="small",
        main=(),
        raw=(),
        finite_mass=np.zeros(0, np.float64),
        target=None,
    )
    store = PubChemStore(
        mass=np.zeros(0, np.float64),
        offsets=np.zeros(1, np.int64),
        buffer=np.zeros(0, np.uint8),
    )
    for target in (0.0, -1.0, float("inf"), float("nan")):
        shifted = spectra[0].copy()
        shifted["prec"] = target + 1.00727645217
        raw = (shifted,)
        result = plan_requests(
            QueryDataset(
                rows=(),
                molecules=(replace(molecule, raw=raw),),
                engine_order=(),
            ),
            library=library,
            retrieval=MainRetrieval.Config().make(),
            generator=ReactionScreen.Config().make(),
            pubchem=store,
            ppm=10.0,
            max_spectra=16,
            dreams=DreamsStore(tmp_path / "unread"),
        )
        assert result.reactions == result.pubchem == result.dreams == ()
    np.save(tmp_path / "dreams_np_mass.npy", np.array([0.5], np.float64))
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["small-row"]))
    shifted = spectra[0].copy()
    shifted["prec"] = 0.5 + 1.00727645217
    raw = (shifted,)
    result = plan_requests(
        QueryDataset(rows=(), molecules=(replace(molecule, raw=raw),), engine_order=()),
        library=library,
        retrieval=MainRetrieval.Config().make(),
        generator=ReactionScreen.Config().make(),
        pubchem=store,
        ppm=10.0,
        max_spectra=16,
        dreams=DreamsStore(tmp_path),
    )
    assert result.dreams == (
        WindowRequest(molecule_id="small", row=0, smiles="small-row"),
    )


class _Standardizations:
    def standardize(self, smiles: str) -> str | None:
        return {"raw": "standard", "invalid": None, " raw2 ": "standard"}[smiles]


def test_every_product_path_alias_empty_header_and_stripped_identity_is_retained(
    tmp_path: Path,
) -> None:
    requests: list[ReactionRequest] = [
        {"smiles": "parent", "delta": 14},
        {"smiles": "zero", "delta": 0},
    ]
    headers = [normalize_request(request) for request in requests]
    responses: list[dict[str, object]] = [
        {
            "input_smiles": "parent",
            "products": ["raw", "invalid", " raw2 ", "raw"],
            "plans": [[0], [1], [], [0]],
            "matches": [],
            "applications": "all",
        },
        {
            "input_smiles": "zero",
            "products": [],
            "plans": [],
            "matches": [],
            "applications": "zero",
        },
    ]
    parquet.write_table(Table.from_pylist(headers), tmp_path / "requests.parquet")
    parquet.write_table(Table.from_pylist(responses), tmp_path / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O", "dehexose2"]}),
        tmp_path / "rules.parquet",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
            ),
        ),
    )
    table = ReactionTable(tmp_path)
    result = product_inputs(
        table,
        chemistry=_Standardizations(),
        pool=("pool", "pool"),
        train=("parent",),
        second=("engine",),
        windows=(WindowRequest(molecule_id="q", row=3, smiles=" pc "),),
        max_first=32,
        max_second=16,
        max_mols=24,
    )
    assert result.generated == (
        "raw",
        "standard",
        "invalid",
        " raw2 ",
        "standard",
        "raw",
        "standard",
    )
    prefix = (
        "pool",
        "pool",
        "parent",
        "engine",
        *result.generated,
        " pc ",
        "parent",
        "zero",
        "CCO",
    )
    assert result.candidates == (
        *prefix,
        *(value.strip() for value in prefix if value.strip()),
    )
    assert result.derivations == (
        PriorRequest(
            parent_smiles="parent",
            combo="methyl_O",
            max_first=32,
            max_second=16,
            max_mols=24,
        ),
        PriorRequest(
            parent_smiles="parent",
            combo="dehexose2",
            max_first=32,
            max_second=16,
            max_mols=24,
        ),
        PriorRequest(
            parent_smiles="parent",
            combo="methyl_O",
            max_first=32,
            max_second=16,
            max_mols=24,
        ),
    )
    with pytest.raises(KeyError, match=r"^'missing'$"):
        product_inputs(
            table,
            chemistry=_Missing(),
            pool=(),
            train=(),
            second=(),
            windows=(),
            max_first=32,
            max_second=16,
            max_mols=24,
        )


class _Missing:
    def standardize(self, smiles: str) -> str | None:
        del smiles
        raise KeyError("missing")


_SCRIPTS_DIR: Final = _CWD / "scripts"


# recipe_plan_test.py.
def test_complete_tables_vectors_duplicate_windows_and_byte_receipt_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _, pool, _ = retrieval_inputs()
    request: ReactionRequest = {"smiles": "parent", "delta": 14.0}
    header = normalize_request(request)
    plan = RecipePlan(
        reactions=(
            GenerationRequest(
                molecule_id="q",
                sid=2,
                similarity=0.5,
                target=100.0,
                header=header,
            ),
        ),
        pubchem=(WindowRequest(molecule_id="q", row=5, smiles=" pc "),) * 2,
        dreams=(WindowRequest(molecule_id="r", row=4, smiles="dream"),),
    )
    original = tmp_path / "provided"
    original.write_bytes(b"complete input bytes")
    output = tmp_path / "requests"
    vectors = np.array([[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], [-1] * 10, [0] * 10], np.int16)
    write_plan(
        plan,
        pool=pool,
        train_smiles=("train0", "bad", "zero"),
        train_formula=vectors,
        output=output,
        inputs=[original],
    )
    assert parquet.read_table(output / "reaction-requests.parquet").to_pylist() == [
        dict(header),
    ]
    assert parquet.read_table(output / "generation.parquet").to_pylist() == [
        {
            "molecule_id": "q",
            "request": 0,
            "sid": 2,
            "similarity": 0.5,
            "target": 100.0,
        },
    ]
    assert parquet.read_table(output / "analog-pool.parquet").to_pylist() == [
        {"smiles": smiles, "formula": formula}
        for smiles, formula in zip(pool.smiles, pool.formulas, strict=True)
    ]
    training = parquet.read_table(output / "analog-train.parquet")
    assert training.column("sid").to_pylist() == [0, 1, 2]
    assert training.column("smiles").to_pylist() == ["train0", "bad", "zero"]
    formulas = from_plain(training.column("formula").to_pylist(), list[str | None])
    np.testing.assert_array_equal(
        np.stack([formula_vector(formula) for formula in formulas]),
        vectors,
    )
    assert (
        parquet.read_table(output / "pubchem-requests.parquet").to_pylist()
        == [{"molecule_id": "q", "row": 5, "smiles": " pc "}] * 2
    )
    assert parquet.read_table(output / "dreams-requests.parquet").to_pylist() == [
        {"molecule_id": "r", "row": 4, "smiles": "dream"},
    ]
    _receipt(output, source=original, count=6)
    with pytest.raises(FileExistsError):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a", "b", "c"),
            train_formula=vectors,
            output=output,
            inputs=[original],
        )
    with pytest.raises(
        ValueError,
        match=r"^Training formula vectors differ from complete SID rows\.$",
    ):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a",),
            train_formula=vectors,
            output=tmp_path / "bad-shape",
            inputs=[original],
        )
    assert not (tmp_path / "bad-shape").exists()
    with pytest.raises(
        ValueError,
        match=r"^Request output contains a protected input\.$",
    ):
        write_plan(
            plan,
            pool=pool,
            train_smiles=("a", "b", "c"),
            train_formula=vectors,
            output=tmp_path,
            inputs=[original],
        )


class _Formulas:
    def formula(self, smiles: str) -> str:
        return {"raw": "raw", "standard": "C2H6O", "invalid": "invalid"}[smiles]


def test_every_raw_proposal_and_final_request_including_empty_tables(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    requests: list[ReactionRequest] = [
        {"smiles": "parent", "delta": 14.0},
        {"smiles": "empty", "delta": 0},
    ]
    parquet.write_table(
        Table.from_pylist([normalize_request(row) for row in requests]),
        source / "requests.parquet",
    )
    rows: list[dict[str, object]] = [
        {
            "input_smiles": "parent",
            "products": ["raw", "invalid", "raw"],
            "plans": [[0], [], [1, 0]],
            "matches": [],
            "applications": "all",
        },
        {
            "input_smiles": "empty",
            "products": [],
            "plans": [],
            "matches": [],
            "applications": "zero",
        },
    ]
    parquet.write_table(Table.from_pylist(rows), source / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O", "dehexose2"]}),
        source / "rules.parquet",
    )
    (source / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
            ),
        ),
    )
    inputs = sorted(source.iterdir())
    reactions = ReactionTable(source)
    output = tmp_path / "raw"
    write_reaction_inputs(reactions, output=output, inputs=inputs)
    assert parquet.read_table(output / "raw-products.parquet").to_pylist() == [
        {
            "smiles": "raw",
            "request": 0,
            "position": 0,
            "parent_smiles": "parent",
            "combo": "methyl_O",
        },
        {
            "smiles": "invalid",
            "request": 0,
            "position": 1,
            "parent_smiles": "parent",
            "combo": None,
        },
        {
            "smiles": "raw",
            "request": 0,
            "position": 2,
            "parent_smiles": "parent",
            "combo": "dehexose2+methyl_O",
        },
    ]
    _receipt(output, source=inputs[0], count=1)
    records = RecipeInputs(
        generated=("raw", "standard", "invalid", "raw"),
        candidates=("pool", "parent", " raw ", "raw"),
        derivations=(
            PriorRequest(
                parent_smiles="parent",
                combo="methyl_O",
                max_first=32,
                max_second=16,
                max_mols=24,
            ),
        )
        * 2,
    )
    final = tmp_path / "final"
    write_products(records, chemistry=_Formulas(), output=final, inputs=inputs)
    assert parquet.read_table(final / "generated-requests.parquet").to_pylist() == [
        {"smiles": value, "formula": _Formulas().formula(value)}
        for value in records.generated
    ]
    assert parquet.read_table(final / "candidate-requests.parquet").column(
        "smiles",
    ).to_pylist() == list(records.candidates)
    assert (
        parquet.read_table(final / "derivation-requests.parquet").to_pylist()
        == [
            {
                "parent_smiles": "parent",
                "combo": "methyl_O",
                "max_first": 32,
                "max_second": 16,
                "max_mols": 24,
            },
        ]
        * 2
    )
    _receipt(final, source=inputs[0], count=3)
    empty = tmp_path / "empty"
    write_products(
        RecipeInputs(generated=(), candidates=(), derivations=()),
        chemistry=_Formulas(),
        output=empty,
        inputs=[],
    )
    for name in ("generated", "candidate", "derivation"):
        assert parquet.read_table(empty / f"{name}-requests.parquet").num_rows == 0
    with pytest.raises(KeyError, match=r"^'missing'$"):
        write_products(
            RecipeInputs(generated=("missing",), candidates=(), derivations=()),
            chemistry=_Formulas(),
            output=tmp_path / "missing",
            inputs=[],
        )
    assert not (tmp_path / "missing/manifest.json").exists()


def _receipt(directory: Path, *, source: Path, count: int) -> None:
    manifest = json_loads((directory / "manifest.json").read_text(), dict[str, object])
    assert manifest["format_version"] == 1
    assert manifest["implementation"] == "independent Priml preparation requests"
    inputs = from_plain(manifest["inputs"], dict[str, str])
    assert (
        inputs[str(source.resolve())] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    files = from_plain(manifest["files"], dict[str, str])
    assert len(files) == count
    for name, digest in files.items():
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
    generator = from_plain(manifest["generator"], dict[str, str])
    package = _SCRIPTS_DIR.parent
    for name, digest in generator.items():
        assert digest == hashlib.sha256((package / name).read_bytes()).hexdigest()
    assert "recipe.py" in generator
    assert "scripts/recipe_plan.py" in generator


# recipe_preparation_test.py.
_RECIPE_PREPARATION_BASELINE: Final = _SCRIPTS_DIR.parent


def test_launcher_factories_are_pure_and_preserve_complete_public_budgets() -> None:
    config = prepare_requests().copy_tree().finalize()
    assert config.max_spectra == 16
    assert config.retrieval.n_analog == 100
    assert config.generator.parents == 6
    assert config.generator.min_similarity == 0.3
    assert config.generator.total == 150
    assert config.generator.per_parent == 60
    assert config.generator.application_budget == 800
    assert config.generator.stage_parents == 24
    assert config.generator.first_products == 32
    assert config.generator.later_products == 16
    assert config.generator.max_steps == 2
    assert isinstance(prepare_raw_products(), RawProductPreparation.Config)
    assert isinstance(prepare_product_requests(), ProductPreparation.Config)


def test_job_directories_resolve_beneath_the_dataset_root_and_agree() -> None:
    root = Path(DATASET_ROOT)
    planning = PlanningPreparation.Config().copy_tree().finalize()
    raw = RawProductPreparation.Config().copy_tree().finalize()
    product = ProductPreparation.Config().copy_tree().finalize()
    assert planning.base_dir == raw.base_dir == product.base_dir == DATASET_ROOT
    assert planning.queries == root / (
        "inputs/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
    )
    assert planning.library == root / "training/work"
    assert planning.dreams is None
    assert planning.output == product.plan
    assert planning.output == root / "native-chemistry/preparation/requests"
    assert raw.reactions == product.reactions == root / "native-chemistry/reactions"
    assert product.identities == root / "native-chemistry/identities"
    assert product.output == root / "native-chemistry/preparation/products"
    explicit = PlanningPreparation.Config()
    explicit.dreams = "inputs/dreams"
    assert explicit.copy_tree().finalize().dreams == root / "inputs/dreams"


def test_planning_job_uses_native_library_and_all_declared_data_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = _planning(tmp_path)
    job = config.make()
    job.run()
    headers = parquet.read_table(
        Path(config.output) / "reaction-requests.parquet",
    ).to_pylist()
    assert len(headers) == 1
    assert headers[0]["max_products"] == 60
    assert headers[0]["application_budget"] == 800
    windows = from_plain(
        parquet.read_table(Path(config.output) / "dreams-requests.parquet").to_pylist(),
        list[WindowRequest],
    )
    assert windows == [WindowRequest(molecule_id="q", row=0, smiles="first-dream")]
    pc = from_plain(
        parquet.read_table(Path(config.output) / "pubchem-requests.parquet")
        .column("smiles")
        .to_pylist(),
        list[str],
    )
    assert pc == [" CCO ", "CCO"]
    manifest = json_loads(
        (Path(config.output) / "manifest.json").read_text(),
        dict[str, object],
    )
    inputs = from_plain(manifest["inputs"], dict[str, str])
    assert str(Path(config.queries).resolve()) in inputs
    assert str(Path(config.train_formula).resolve()) in inputs
    assert config.dreams is not None
    assert str((Path(config.dreams) / "dreams_np_mass.npy").resolve()) in inputs
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^PlanningPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")
    config.output = tmp_path / "no-optional"
    config.dreams = None
    config.make().run()
    assert (
        parquet.read_table(Path(config.output) / "dreams-requests.parquet").num_rows
        == 0
    )


def test_raw_product_job_retains_all_paths_and_refuses_passthrough(
    tmp_path: Path,
) -> None:
    source, _ = _reactions(tmp_path)
    config = RawProductPreparation.Config()
    config.base_dir = None
    config.reactions = source
    config.output = tmp_path / "raw"
    job = config.make()
    job.run()
    rows = parquet.read_table(Path(config.output) / "raw-products.parquet")
    assert rows.num_rows == 2
    assert rows.column("position").to_pylist() == [0, 1]
    assert rows.column("combo").to_pylist() == ["methyl_O", "methyl_O"]
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^RawProductPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")


def test_product_job_joins_complete_sid_pool_proposal_and_window_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    source, raw = _reactions(tmp_path)
    config = ProductPreparation.Config()
    config.base_dir = None
    config.reactions = source
    config.identities = tmp_path / "identity"
    config.identities.mkdir()
    write_identity(config.identities)
    config.pool_metadata = tmp_path / "pool.parquet"
    config.training = tmp_path / "train.parquet"
    config.second_metadata = tmp_path / "second.parquet"
    config.plan = tmp_path / "plan"
    config.plan.mkdir()
    config.output = tmp_path / "products"
    parquet.write_table(
        Table.from_pydict({"smiles": ["pool", "pool"]}),
        config.pool_metadata,
    )
    parquet.write_table(
        Table.from_pydict({"sid": [1, 0], "smiles": ["sid1", "sid0"]}),
        config.training,
    )
    parquet.write_table(
        Table.from_pydict({"smiles": ["second"]}),
        config.second_metadata,
    )
    for name, values in (("pubchem", [" pc ", " pc "]), ("dreams", ["dream"])):
        parquet.write_table(
            Table.from_pydict(
                {
                    "molecule_id": ["q"] * len(values),
                    "row": list(range(len(values))),
                    "smiles": values,
                },
            ),
            config.plan / f"{name}-requests.parquet",
        )
    job = config.make()
    job.run()
    chemistry = IdentityTable.Config(working_dir=config.identities).make()
    generated = (raw, chemistry.standardize(raw)) * 2
    assert None not in generated
    assert parquet.read_table(
        Path(config.output) / "generated-requests.parquet",
    ).column(
        "smiles",
    ).to_pylist() == list(generated)
    requests = (
        parquet.read_table(Path(config.output) / "candidate-requests.parquet")
        .column("smiles")
        .to_pylist()
    )
    prefix = [
        "pool",
        "pool",
        "sid0",
        "sid1",
        "second",
        *generated,
        " pc ",
        " pc ",
        "dream",
        "parent",
        "CCO",
    ]
    assert requests == prefix + [str(value).strip() for value in prefix]
    assert (
        parquet.read_table(
            Path(config.output) / "derivation-requests.parquet",
        ).to_pylist()
        == [
            {
                "parent_smiles": "parent",
                "combo": "methyl_O",
                "max_first": 32,
                "max_second": 16,
                "max_mols": 24,
            },
        ]
        * 2
    )
    with pytest.raises(FileExistsError):
        job.run()
    with pytest.raises(
        ValueError,
        match=r"^ProductPreparation accepts configuration fields only\.$",
    ):
        job.run("--truncate")


def _planning(root: Path) -> PlanningPreparation.Config:
    config = PlanningPreparation.Config()
    config.base_dir = None
    config.queries = root / "test.parquet"
    config.library = root / "data"
    config.library.mkdir()
    config.pool = config.library
    config.pubchem = root / "pc"
    config.pubchem.mkdir()
    config.train_formula = root / "train_fvec.npy"
    config.output = root / "requests"
    config.retrieval.n_analog = 2
    config.generator.parents = 1
    library, pool, _ = retrieval_inputs()
    for name, values in (
        ("spec_off", library.offsets),
        ("spec_mz", library.mz),
        ("spec_it", library.intensity),
        ("pool_fp", pool.packed),
        ("fp_bits", pool.bits),
        ("pool_frag_off", pool.fragment_offsets),
        ("pool_frag_mass", pool.fragment_mass),
    ):
        np.save(config.library / f"{name}.npy", values)
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": list(range(len(library.smiles))),
                "smiles": library.smiles,
                "key": library.keys,
                "mass": library.structure_mass,
            },
        ),
        config.library / "train_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": library.sid,
                "lib": library.library,
                "mode": library.mode,
                "instr": library.instrument,
                "adduct_ix": library.adduct,
                "precursor_mz": library.precursor,
                "n_clean": library.peak_count,
            },
        ),
        config.library / "spec_meta.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": pool.mass,
                "key": pool.keys,
                "smiles": pool.smiles,
                "formula": pool.formulas,
                "n_heavy": pool.heavy_atoms,
                "src": pool.source,
                "train_sid": pool.train_sid,
            },
        ),
        config.pool / "pool_meta.parquet",
    )
    np.save(config.train_formula, np.ones((len(library.smiles), 10), np.int16))
    query: QueryRow = {
        "molecule_id": "q",
        "spectrum_id": "s",
        "precursor_mz": 101.00727645217,
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
        "collision_energy_ev": None,
        "collision_energy_orig_units": None,
        "collision_energy_orig": None,
        "instrument_type": None,
        "ms2_mzs": [20, 30, 40],
        "ms2_normalized_intensities": [1, 2, 3],
    }
    parquet.write_table(Table.from_pylist([query]), config.queries)
    np.save(config.pubchem / "pc_mass.npy", np.array([100, 100], np.float64))
    text = [b" CCO ", b"CCO"]
    np.save(
        config.pubchem / "pc_off.npy",
        np.concatenate([[0], np.cumsum([len(value) for value in text])]).astype(
            np.int64,
        ),
    )
    np.save(config.pubchem / "pc_smiles.npy", np.frombuffer(b"".join(text), np.uint8))
    config.dreams = root / "optional"
    config.dreams.mkdir()
    np.save(config.dreams / "dreams_np_mass.npy", np.array([100], np.float64))
    np.save(config.dreams / "dreams_np_smiles.npy", np.array(["first-dream"]))
    return config


def _reactions(root: Path) -> tuple[Path, str]:
    source = root / "reactions"
    source.mkdir()
    raw = identity_rows()[0]["smiles"]
    request: ReactionRequest = {"smiles": "parent", "delta": 14.0}
    parquet.write_table(
        Table.from_pylist([normalize_request(request)]),
        source / "requests.parquet",
    )
    row: dict[str, object] = {
        "input_smiles": "parent",
        "products": [raw, raw],
        "plans": [[0], [0]],
        "matches": [],
        "applications": "all",
    }
    parquet.write_table(Table.from_pylist([row]), source / "results.parquet")
    parquet.write_table(
        Table.from_pydict({"name": ["methyl_O"]}),
        source / "rules.parquet",
    )
    (source / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {"format_version": 1, "rdkit": IDENTITY_RDKIT, "default_products": 60},
            ),
        ),
    )
    return source, raw


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
