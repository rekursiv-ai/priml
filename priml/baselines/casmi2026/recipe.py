"""The public recipe: resources, arms, lists, results and publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    override,
    runtime_checkable,
)

import csv
import hashlib
import json
import logging
import math
import shutil
import time

from configgle import Fig, Makeable, traverse
from numpy.typing import NDArray
from treekle import from_plain, json_dumps, to_plain

import numpy as np

from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    IdentityTable,
    SpectralLibrary,
)
from priml.baselines.casmi2026.custom_types import (
    QueryMasses,
    QueryRow,
    ReactionHeader,
    SpectrumRequest,
)
from priml.baselines.casmi2026.dreams import (
    DreamsStore,
    MainBank,
    TailAudit,
    safe_tail,
)
from priml.baselines.casmi2026.fragnet import FragnetTable, StageA
from priml.baselines.casmi2026.fusion import (
    PLACEHOLDER,
    FragmentGap,
    FusionAudit,
    FusionInputs,
    FusionPolicy,
    GapGraphs,
    GapGraphTable,
    guarded_z,
    insert_second,
    rerank_groups,
    simulator_candidates,
)
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    AnalogTable,
    IntrinsicTable,
    ReactionTable,
    neutral_mass,
    normalize_request,
)
from priml.baselines.casmi2026.main_engine import (
    DerivationPriors,
    DerivationTable,
    MainEngine,
    MainFamilyData,
    MainFragmentTable,
    MainPoolFamilies,
    ReactionScreen,
    TreeBank,
    load_tree,
    training_smiles,
)
from priml.baselines.casmi2026.pubchem import (
    PubChemAssetProvider,
    PubChemAssets,
    PubChemGate,
    PubChemInputs,
    PubChemResult,
    PubChemScreen,
)
from priml.baselines.casmi2026.second_engine import (
    SecondAssetProvider,
    SecondAssets,
    SecondBank,
    SecondEngine,
    SecondInputs,
)
from priml.baselines.casmi2026.simulators import (
    DatasetRooted,
    GlacierBackend,
    GlacierScheduling,
    IceBackend,
    IceScheduling,
    JobScheduling,
    PredictionBackend,
    QueryScoring,
    SimulationMolecule,
    SimulatorAdmissionTable,
    covered,
    run_simulator,
)
from priml.data.arrays import load_array
from priml.model.ae.checkpoint import LocalFile
from priml.model.custom_types import propagate_attr
from priml.paths import resolve_working_dir, validated_output_path
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyarrow import parquet

    import torch

    from priml.baselines.casmi2026.custom_types import (
        EngineResult,
        GapSpectrum,
        MetricIdentity,
    )
    from priml.baselines.casmi2026.main_engine import MainRetrieval
    from priml.baselines.casmi2026.pubchem import PubChemStore
    from priml.baselines.casmi2026.second_engine import SecondIndex
    from priml.baselines.casmi2026.simulators import SimulatorResult
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")
    torch = lazy_import("torch")


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedMain:
    """Carry the complete source BASE record before simulator scheduling."""

    smiles: tuple[str, ...]
    keys: tuple[str, ...]
    scores: tuple[float, ...]
    formulas: tuple[str, ...]
    library_max: float
    popularity_applied: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedSecond:
    """Carry the complete recomputed-key second-engine shortlist."""

    smiles: tuple[str, ...]
    keys: tuple[str, ...]


def rank_main(
    result: EngineResult,
    *,
    popularity: NDArray[np.float64],
    limit: int,
    weight: float,
) -> RankedMain:
    """Select unique keys, then permute pool slots using standardized scores.

    Args:
      result: Complete main-arm candidates and ranker outputs in candidate order.
      popularity: Full PID-aligned F64 log-SID plus log-PMID popularity values.
      limit: Unique-key shortlist cap, applied before popularity reranking.
      weight: Population prior coefficient; nonpositive values disable reranking.

    Returns:
      ranked: Complete BASE record, including generated formula suffixes.

    """
    selected: list[int] = []
    seen: set[str] = set()
    ranked = np.argsort(-result.scores, stable=True)
    for rank in range(len(ranked)):
        position = ranked.item(rank)
        key = result.candidates.keys[position]
        if key in seen:
            continue
        seen.add(key)
        selected.append(position)
        if len(selected) >= limit:
            break
    rows = np.array(selected, np.int64)
    scores = result.scores[rows]
    pids = result.candidates.pool_rows[rows]
    generated = pids < 0
    applied = weight > 0 and len(rows) > 1
    order = np.arange(len(rows))
    if applied:
        scores = guarded_z(scores) + weight * np.where(
            generated,
            0.0,
            popularity[np.maximum(pids, 0)],
        )
        pool = np.flatnonzero(~generated)
        order[~generated] = pool[np.argsort(-scores[pool], stable=True)]
    positions = rows[order]
    return RankedMain(
        smiles=tuple(
            result.candidates.smiles[positions.item(i)] for i in range(len(rows))
        ),
        keys=tuple(result.candidates.keys[positions.item(i)] for i in range(len(rows))),
        scores=tuple(scores.item(order.item(i)) for i in range(len(rows))),
        formulas=tuple(
            result.candidates.formulas[positions.item(i)]
            + ("|gen" if applied and generated.item(order.item(i)) else "")
            for i in range(len(rows))
        ),
        library_max=result.diagnostics.lib_max,
        popularity_applied=applied,
    )


def rank_second(
    result: EngineResult,
    *,
    key: Callable[[str], str | None],
    prescreen: int,
    limit: int,
) -> RankedSecond | None:
    """Recompute legacy keys inside the sorted top80, retaining the first empty key.

    Args:
      result: Complete second-arm candidates and blended scores.
      key: Explicit prepared E2 key lookup; missing rows must raise.
      prescreen: Sorted candidate prefix before canonical-key deduplication.
      limit: Maximum number of admitted distinct canonical keys.

    Returns:
      ranked: Complete shortlist, or None when no candidate scores existed.

    """
    if not len(result.scores):
        return None
    smiles: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    ranked = np.argsort(-result.scores, stable=True)[:prescreen]
    for rank in range(len(ranked)):
        position = ranked.item(rank)
        value = result.candidates.smiles[position]
        canonical = key(value)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        smiles.append(value)
        keys.append(canonical)
        if len(smiles) >= limit:
            break
    return RankedSecond(smiles=tuple(smiles), keys=tuple(keys))


logger = logging.getLogger(__name__)


class RankedResource(Protocol):
    """Release one completed ranking arm before constructing the next arm."""

    def close(self) -> None:
        """Drop owned models and numerical data, including after failures."""
        ...


class MainRanker(RankedResource, Protocol):
    """Bind the complete main recipe to its finite target and main query view."""

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        """Rank every native candidate and retain the full top60 record."""
        ...


class SecondRanker(RankedResource, Protocol):
    """Bind the independent second arm to every original raw spectrum."""

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
    ) -> RankedSecond | None:
        """Return the complete source shortlist, preserving absent evidence."""
        ...


class PubChemRanker(RankedResource, Protocol):
    """Bind the complete independent PC channel to original raw spectra."""

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
    ) -> PubChemResult | None:
        """Return every original proposal field or its explicit absence."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedArms:
    """Retain every complete source row and exact main inference failure evidence."""

    main: dict[str, RankedMain]
    second: dict[str, RankedSecond]
    pubchem: dict[str, PubChemResult]
    errors: tuple[tuple[str, str], ...]


def rank_arms(
    queries: QueryDataset,
    *,
    main: MainRanker,
    second: SecondRanker,
    pubchem: PubChemRanker,
) -> RankedArms:
    """Run the public molecule orders without narrowing either query view.

    Args:
      queries: Complete original rows and both independent molecule orders.
      main: Native main ranker including generation, all families and popularity.
      second: Native complete second ranker with its independently fitted ensembles.
      pubchem: Native V3 screen with explicitly prepared chemical primitives.

    Returns:
      arms: Every main row, available secondary rows and exact main failures.

    """
    first: dict[str, RankedMain] = {}
    errors: list[tuple[str, str]] = []
    try:
        for molecule in queries.molecules:
            ranked = RankedMain(
                smiles=(),
                keys=(),
                scores=(),
                formulas=(),
                library_max=0.0,
                popularity_applied=False,
            )
            if molecule.target is not None:
                try:
                    ranked = main.rank(molecule.mid, molecule.main, molecule.target)
                except Exception as error:
                    logger.exception("Main inference failed for %s.", molecule.mid)
                    errors.append((molecule.mid, repr(error)))
            first[molecule.mid] = ranked
    finally:
        main.close()
    groups = {molecule.mid: molecule for molecule in queries.molecules}
    engine: dict[str, RankedSecond] = {}
    try:
        for mid in queries.engine_order:
            value = second.rank(mid, groups[mid].raw)
            if value is not None:
                engine[mid] = value
    finally:
        second.close()
    proposals: dict[str, PubChemResult] = {}
    try:
        for molecule in queries.molecules:
            proposal = pubchem.rank(molecule.mid, molecule.raw)
            if proposal is not None:
                proposals[molecule.mid] = proposal
    finally:
        pubchem.close()
    return RankedArms(
        main=first,
        second=engine,
        pubchem=proposals,
        errors=tuple(errors),
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MoleculeQueries:
    """Retain source identity, both query views and complete finite neutral masses."""

    mid: str
    source_id: str | int
    main: tuple[SpectrumRequest, ...]
    raw: tuple[SpectrumRequest, ...]
    finite_mass: QueryMasses
    target: float | None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class QueryDataset:
    """Keep every original row, encounter-order groups and sorted engine IDs."""

    rows: tuple[QueryRow, ...]
    molecules: tuple[MoleculeQueries, ...]
    engine_order: tuple[str, ...]


def tail_target(molecule: MoleculeQueries) -> float:
    """Return the source safe-tail median after removing only unknown masses.

    Args:
      molecule: Every original raw spectrum, including nonfinite precursors.

    Returns:
      target: NaN when no known mass exists; infinity remains in the median.

    """
    masses = neutral_mass(
        np.array([row["prec"] for row in molecule.raw], np.float64),
        [row["adduct"] for row in molecule.raw],
    )
    masses = masses[~np.isnan(masses)]
    return float(np.median(masses)) if len(masses) else math.nan


def read_queries(path: Path) -> QueryDataset:
    """Read and condition the complete Parquet input without dataframe coercion.

    Args:
      path: Competition test Parquet with every original spectrum column.

    Returns:
      dataset: Every input row and both public molecule orderings. Main inputs
        exist only for groups with a finite target; raw inputs always exist.

    """
    rows = tuple(from_plain(parquet.read_table(path).to_pylist(), list[QueryRow]))
    groups: dict[str | int, list[QueryRow]] = {}
    for row in rows:
        groups.setdefault(row["molecule_id"], []).append(row)
    return QueryDataset(
        rows=rows,
        molecules=tuple(
            _molecule(source_id, rows=members) for source_id, members in groups.items()
        ),
        engine_order=tuple(str(source_id) for source_id in sorted(groups)),
    )


class ListBudget:
    """Fix the candidate prefixes and cut-offs of the simulator and published lists."""

    class Config(Fig["ListBudget"]):
        main_candidates: int = 60
        """Main prefix admitted to simulator candidate unions."""

        second_candidates: int = 40
        """Independent second-engine prefix in simulator unions."""

        pubchem_candidates: int = 25
        """PubChem prefix admitted to same-formula simulator groups."""

        legacy_rerank_top_n: int = 60
        """Legacy same-formula reranking prefix."""

        pubchem_rerank_top_n: int = 25
        """PubChem same-formula reranking prefix."""

        main_limit: int = 25
        """Main prefix retained before final fusion and official publication."""

        insertion_limit: int = 25
        """Maximum candidates after engine-two rank-two insertion."""

    def __init__(self, config: Config) -> None:
        self.config = config


class PreparedPubChem(PubChemResult):
    """Keep the conditional raw-formula column alongside every original field."""

    pc_form: NotRequired[list[str]]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SimulatorInputs:
    """Retain full molecule priority, candidate unions and both query panels."""

    priority: tuple[str, ...]
    candidates: dict[str, tuple[str, ...]]
    ice: tuple[SimulationMolecule, ...]
    glacier: tuple[SimulationMolecule, ...]
    proposals: dict[str, PreparedPubChem]


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseList:
    """Retain the truncated source base identity and direct-library protection."""

    smiles: tuple[str, ...]
    keys: tuple[str, ...]
    library_max: float


@dataclass(frozen=True, slots=True, kw_only=True)
class RerankedLists:
    """Retain complete list changes and every simulator reorder counter."""

    main: dict[str, BaseList]
    proposals: dict[str, PreparedPubChem]
    ice_stats: dict[str, int]
    glacier_stats: dict[str, int]


def simulator_inputs(
    main: Mapping[str, RankedMain],
    *,
    second: Mapping[str, RankedSecond],
    pubchem: Mapping[str, PubChemResult],
    queries: QueryDataset,
    formula: Callable[[str], str],
    budget: ListBudget,
    gate: PubChemGate,
) -> SimulatorInputs:
    """Build source-priority candidate unions with ICE and proton-only GL queries.

    Args:
      main: Complete top60 main ranks in molecule encounter order.
      second: Available engine2 top40 lists and their distinct legacy keys.
      pubchem: Complete V3 proposals before conditional formula preparation.
      queries: Every original spectrum grouped by its public molecule identity.
      formula: Strict independent raw-formula lookup, including known fallbacks.
      budget: Printed simulator source budgets.
      gate: Printed library admission gate.

    Returns:
      inputs: Every priority/candidate/formula decision and both complete panels.

    """
    settings = budget.config
    priority = tuple(sorted(main, key=lambda mid: main[mid].library_max))
    proposals: dict[str, PreparedPubChem] = {
        mid: {**row}  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
        for mid, row in pubchem.items()
    }
    candidates: dict[str, tuple[str, ...]] = {}
    for mid in priority:
        base = main[mid]
        selected: list[str] = []
        engine = second.get(mid)
        if engine is not None and engine.smiles:
            keys = set(base.keys[: settings.main_candidates])
            extra = [
                engine.smiles[index]
                for index in range(
                    min(
                        settings.second_candidates,
                        len(engine.smiles),
                        len(engine.keys),
                    ),
                )
                if engine.keys[index] not in keys
            ]
            combined = [*base.smiles[: settings.main_candidates], *extra]
            selected = simulator_candidates(
                combined,
                [
                    *base.formulas[: settings.main_candidates],
                    *(formula(value) for value in extra),
                ],
                top_n=len(combined),
            )
        proposal = proposals.get(mid)
        if (
            proposal is not None
            and proposal["pc"]
            and base.library_max < gate.config.library_gate
        ):
            forms = [formula(value) for value in proposal["pc"]]
            proposal["pc_form"] = forms
            selected.extend(
                simulator_candidates(
                    proposal["pc"],
                    forms,
                    top_n=settings.pubchem_candidates,
                ),
            )
        if selected:
            candidates[mid] = tuple(dict.fromkeys(selected))
    by_mid = {molecule.mid: molecule.raw for molecule in queries.molecules}
    ice = tuple(
        SimulationMolecule(
            mid=mid,
            candidates=values,
            spectra=tuple(row for row in by_mid.get(mid, ()) if covered(row)),
        )
        for mid, values in candidates.items()
    )
    glacier: list[SimulationMolecule] = []
    for molecule in ice:
        spectra = tuple(row for row in molecule.spectra if row["adduct"] == "[M+H]+")
        if spectra:
            glacier.append(
                SimulationMolecule(
                    mid=molecule.mid,
                    candidates=molecule.candidates,
                    spectra=spectra,
                ),
            )
    return SimulatorInputs(
        priority=priority,
        candidates=candidates,
        ice=ice,
        glacier=tuple(glacier),
        proposals=proposals,
    )


def rerank_pubchem(
    main: Mapping[str, RankedMain],
    *,
    proposals: Mapping[str, PreparedPubChem],
    ice: Mapping[str, Mapping[str, float | None]],
    glacier: Mapping[str, Mapping[str, float | None]],
    policy: FusionPolicy,
    budget: ListBudget,
) -> RerankedLists:
    """Keep base ICE disabled and rerank proposals without changing their gate fit.

    Args:
      main: Complete source main ranks before their first25 publication.
      proposals: Original PubChem proposals with conditionally reached raw formulas.
      ice: Complete ICE score maps, including truthy all-null maps.
      glacier: Complete GL score maps; all-null maps behave as absent.
      policy: Printed reranking weights, coverage and tie policy.
      budget: Printed reranking prefixes and main list cut-off.

    Returns:
      lists: Truncated main lists, complete reordered proposals and source counters.

    """
    settings = policy.config
    output: dict[str, PreparedPubChem] = {
        mid: {**proposal} for mid, proposal in proposals.items()
    }
    ice_stats: dict[str, int] = {}
    gl_stats: dict[str, int] = {}
    for mid in main:
        proposal = output.get(mid)
        ice_scores = ice.get(mid)
        if proposal is None:
            continue
        forms = proposal.get("pc_form")
        if forms is None or not forms or ice_scores is None or not ice_scores:
            continue
        candidates, keys, scores = (
            proposal["pc"],
            proposal["pc_keys"],
            proposal["pc_fz"],
        )
        order = rerank_groups(
            candidates,
            keys,
            scores,
            forms,
            columns=[ice_scores],
            weights=[settings.ice_weight],
            top_n=budget.config.pubchem_rerank_top_n,
            min_covered=settings.min_covered,
            tie_eps=settings.tie_eps,
        )
        gl_scores = glacier.get(mid, {})
        if any(value is not None for value in gl_scores.values()):
            revised = rerank_groups(
                candidates,
                keys,
                scores,
                forms,
                columns=[ice_scores, gl_scores],
                weights=[settings.ice_weight, settings.glacier_weight],
                top_n=budget.config.pubchem_rerank_top_n,
                min_covered=settings.min_covered,
                tie_eps=settings.tie_eps,
            )
            for name, value in (
                ("pc_molecules", 1),
                (
                    "pc_changed_vs_ice_top25",
                    int(revised[: settings.length] != order[: settings.length]),
                ),
                ("pc_changed_vs_ice_top1", int(revised[:1] != order[:1])),
            ):
                gl_stats[name] = gl_stats.get(name, 0) + value
            order = revised
        proposal["pc"] = [candidates[index] for index in order]
        proposal["pc_keys"] = [keys[index] for index in order]
        proposal["pc_fz"] = [scores[0], *[scores[index] for index in order][1:]]
        ice_stats["pc_changed"] = ice_stats.get("pc_changed", 0) + int(order[:1] != [0])
    return RerankedLists(
        main={
            mid: BaseList(
                smiles=row.smiles[: budget.config.main_limit],
                keys=row.keys[: budget.config.main_limit],
                library_max=row.library_max,
            )
            for mid, row in main.items()
        },
        proposals=output,
        ice_stats=ice_stats,
        glacier_stats=gl_stats,
    )


class ResultIdentities(Protocol):
    """Require every reached older key, raw formula and official metric identity."""

    def key(self, smiles: str) -> str | None:
        """Read the explicitly prepared raw older key, preserving known rejection."""
        ...

    def formula(self, smiles: str) -> str:
        """Read the explicitly prepared raw formula or original invalid fallback."""
        ...

    def identity(self, smiles: str) -> MetricIdentity | None:
        """Read the official identity without changing its source representative."""
        ...


class ControlAudit(TypedDict):
    """Retain the complete independent E004 control comparison."""

    molecule_id: str
    collapsed_source_votes: int
    changed: bool
    changed_top1: bool


class FactorialAudit(FusionAudit):
    """Retain all final fields and both changes against the reconstructed control."""

    changed: bool
    changed_top1: bool


class MergeCounts(TypedDict):
    """Retain every source PubChem admission counter in publication order."""

    untouched: int
    gentle: int
    aggressive: int
    no_pc: int
    promoted: int


class InsertionCounts(TypedDict):
    """Retain all three rank-two overlay counters."""

    rows: int
    e2_ins: int
    e1_ins: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RecipeResults:
    """Keep every list, audit and fallback before source-compatible publication."""

    prefusion: dict[str, list[str]]
    overflow: dict[str, list[str]]
    baseline: dict[str, list[str]]
    reference: dict[str, list[str]]
    control: dict[str, list[str]]
    preinsert: dict[str, list[str]]
    submission: dict[str, list[str]]
    gap: dict[str, dict[str, float]]
    tail_audit: tuple[TailAudit, ...]
    control_audit: tuple[ControlAudit, ...]
    final_audit: tuple[FactorialAudit, ...]
    merge: MergeCounts
    insertion: InsertionCounts
    base_nonempty: int


def assemble_results(
    sample_ids: Sequence[str],
    *,
    base: Mapping[str, BaseList],
    base_tail: Mapping[str, Sequence[str]],
    pubchem: Mapping[str, PreparedPubChem],
    engine: Mapping[str, RankedSecond],
    queries: QueryDataset,
    identities: ResultIdentities,
    ice: Mapping[str, Mapping[str, float | None]],
    glacier: Mapping[str, Mapping[str, float | None]],
    graphs: GapGraphs,
    fragment_gap: FragmentGap,
    dreams: DreamsStore | None = None,
    ice_status: str = "ok",
    glacier_status: str = "ok",
    base_errors: int = 0,
    policy: FusionPolicy,
    budget: ListBudget,
    gate: PubChemGate,
) -> RecipeResults:
    """Compose screening, both fusions, safe tail, gap fallback and rank-two insertion.

    Args:
      sample_ids: Complete publication order from the competition sample CSV.
      base: Main top25 lists with source direct-library protection.
      base_tail: Complete main top60 lists, before truncation.
      pubchem: Complete conditionally reranked PubChem lists and original gate fits.
      engine: Every second-arm shortlist, including present empty lists.
      queries: Every original test spectrum and source group identity.
      identities: Strict independent key, formula and official-identity preparation.
      ice: Complete published ICE scores, including all-null maps.
      glacier: Complete published GL scores, including all-null maps.
      graphs: Strict independent final-gap graphs for every reached string.
      fragment_gap: Complete native gap-scoring recipe.
      dreams: Optional lazily loaded source mass store.
      ice_status: Observed completed stage health.
      glacier_status: Observed completed stage health.
      base_errors: Number of main-arm inference failures.
      policy: Printed fusion, reranking and fallback policies.
      budget: Printed candidate prefixes and list cut-offs.
      gate: Printed PubChem admission gate.

    Returns:
      results: Every intermediate/final candidate, control, score and audit field.

    """
    prefusion, merge = _screen(sample_ids, base=base, pubchem=pubchem, gate=gate)
    baseline, overflow = _legacy(
        prefusion,
        engine=engine,
        identities=identities,
        ice=ice,
        glacier=glacier,
        policy=policy,
        budget=budget,
    )
    active = {molecule.mid for molecule in queries.molecules}
    if base_errors:
        raise AssertionError(f"Base inference errors: {base_errors}")
    if active - engine.keys():
        raise AssertionError("Engine 2 did not cover all active molecules")
    if ice_status != "ok":
        raise AssertionError(f"ICE stage is not complete: {ice_status}")
    if (
        any(row["adduct"] == "[M+H]+" for row in queries.rows)
        and glacier_status != "ok"
    ):
        raise AssertionError(f"GLACIER stage is not complete: {glacier_status}")
    if (
        len(set(sample_ids)) != len(sample_ids)
        or set(sample_ids) != active
        or set(base) != active
    ):
        raise AssertionError(
            "Every test molecule requires one complete publication row.",
        )
    by_mid = {molecule.mid: molecule for molecule in queries.molecules}
    reference: dict[str, list[str]] = {}
    tails: list[TailAudit] = []
    for mid, original in baseline.items():
        second = engine[mid]
        proposal = pubchem.get(mid)
        revised, audit = safe_tail(
            mid,
            original,
            [
                *overflow.get(mid, ()),
                *base_tail.get(mid, ()),
                *second.smiles,
                *(proposal["pc"] if proposal is not None else ()),
            ],
            target=tail_target(by_mid[mid]),
            identity=identities.identity,
            limit=policy.config.length,
            dreams=dreams,
        )
        if audit is None:
            raise ValueError("Expected audit is not None.")
        reference[mid] = revised
        tails.append(audit)
    gap = _gap_scores(
        prefusion,
        engine=engine,
        queries=by_mid,
        ice=ice,
        glacier=glacier,
        model=fragment_gap,
        graphs=graphs,
    )
    control: dict[str, list[str]] = {}
    before_insert: dict[str, list[str]] = {}
    submission: dict[str, list[str]] = {}
    control_audit: list[ControlAudit] = []
    final_audit: list[FactorialAudit] = []
    inserted = 0
    for mid, original in reference.items():
        second = engine[mid]
        options: FusionInputs = {
            "prefusion": prefusion.get(mid),
            "engine": list(second.smiles),
            "identity": identities.identity,
            "key": identities.key,
            "formula": identities.formula,
            "ice": ice.get(mid),
            "glacier": glacier.get(mid),
            "gap": gap.get(mid),
        }
        prior, prior_audit = policy.control(mid, original, options=options)
        final, audit = policy.fuse(mid, original, options=options)
        official = [identities.identity(value) for value in final]
        if (
            len(final) < 1
            or len(final) > policy.config.length
            or None in official
            or len(set(official)) != len(official)
        ):
            raise AssertionError(f"Invalid final metric identities: {mid}")
        control[mid], before_insert[mid] = prior, final
        control_audit.append(
            {
                "molecule_id": mid,
                "collapsed_source_votes": prior_audit["collapsed_source_votes"],
                "changed": prior != original,
                "changed_top1": prior[:1] != original[:1],
            },
        )
        final_audit.append(
            {  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
                **audit,
                "changed": final != prior,
                "changed_top1": final[:1] != prior[:1],
            },
        )
        submission[mid], changed = insert_second(
            final,
            second.smiles,
            key=identities.key,
            limit=budget.config.insertion_limit,
        )
        inserted += int(changed)
    return RecipeResults(
        prefusion=prefusion,
        overflow=overflow,
        baseline=baseline,
        reference=reference,
        control=control,
        preinsert=before_insert,
        submission=submission,
        gap=gap,
        tail_audit=tuple(tails),
        control_audit=tuple(control_audit),
        final_audit=tuple(final_audit),
        merge=merge,
        insertion={"rows": len(sample_ids), "e2_ins": inserted, "e1_ins": 0},
        base_nonempty=sum(bool(row.smiles) for row in base.values()),
    )


class StageMetadata(TypedDict):
    """Keep observed stage health and its separately prepared scientific version."""

    status: str
    device: str
    rdkit: str
    n_mols_scored: int
    n_scored: int
    errors: list[str]


class PublicationProvenance:
    """Name the source experiments, notebook versions and digests that reports cite."""

    class Config(Fig["PublicationProvenance"]):
        control_experiment: str = ""
        """Identifier of the fusion-without-fallback control experiment."""

        final_experiment: str = ""
        """Identifier of the experiment adding the simulator fallback."""

        recipe_sha256: str = ""
        """SHA256 of the complete source recipe notebook."""

        notebook_sha256: str = ""
        """SHA256 of the base notebook that the final experiment extends."""

        source_code_sha256: str = ""
        """SHA256 of the base notebook's source code."""

        helper_source_sha256: str = ""
        """SHA256 of the final experiment's helper source."""

        metric_source_sha256: str = ""
        """SHA256 of the official metric source."""

        source_version_id: int = 0
        """Kaggle script version identifier of the base notebook."""

        source_version: int = 0
        """Kaggle script version number of the base notebook."""

        ice_policy_source_version_id: int = 0
        """Kaggle script version identifier of the ICE fallback policy."""

        ice_policy_source: str = ""
        """Kaggle script that defines the ICE fallback policy."""

    def __init__(self, config: Config) -> None:
        self.config = config


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicationContext:
    """Carry actual stage observations, source input bytes and elapsed wall time."""

    sample_csv: Path
    ice: StageMetadata
    glacier: StageMetadata
    elapsed: float
    gpu: str | None = None


def write_results(
    directory: Path,
    *,
    results: RecipeResults,
    sample_ids: Sequence[str],
    base_tail: Mapping[str, Sequence[str]],
    pubchem: Mapping[str, PreparedPubChem],
    engine: Mapping[str, RankedSecond],
    context: PublicationContext,
    policy: FusionPolicy,
    provenance: PublicationProvenance,
) -> None:
    """Publish every complete list/audit and its same-run control verification.

    Args:
      directory: Owned run directory, separate from the original sample input.
      results: Complete independently reconstructed lists and all policy audits.
      sample_ids: Original complete sample order used for every candidate CSV.
      base_tail: Complete original top60 main lists before any truncation.
      pubchem: Every final proposal field, including the original admission fit.
      engine: Every original engine2 shortlist and its raw keys.
      context: Observed stage counts, preparation versions, device and elapsed time.
      policy: Printed fusion policy whose window and fallback the reports state.
      provenance: Source experiments, versions and digests that the reports cite.

    """
    names = provenance.config
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(context.sample_csv, directory / "sample_submission.csv")
    for filename, values in (
        ("baseline_submission.csv", results.baseline),
        ("submission_reference.csv", results.reference),
        ("submission_e004_control.csv", results.control),
        ("submission_base.csv", results.preinsert),
        ("submission.csv", results.submission),
    ):
        _csv(
            directory / filename,
            rows=[
                {"molecule_id": mid, "smiles": ";".join(values[mid])}
                for mid in sample_ids
            ],
            fields=["molecule_id", "smiles"],
        )
    _csv(
        directory / "candidate_audit.csv",
        rows=results.tail_audit,
        fields=list(TailAudit.__annotations__),
    )
    _csv(
        directory / "experiment_audit.csv",
        rows=results.control_audit,
        fields=list(ControlAudit.__annotations__),
    )
    _csv(
        directory / "experiment_audit_factorial.csv",
        rows=results.final_audit,
        fields=list(FactorialAudit.__annotations__),
    )
    _json(directory / "probe_applied.txt", results.insertion)
    reference_sha = _sha(directory / "submission_reference.csv")
    control_sha = _sha(directory / "submission_e004_control.csv")
    ice_meta: StageMetadata = {**context.ice, "errors": list(context.ice["errors"])}
    glacier_meta: StageMetadata = {
        **context.glacier,
        "errors": list(context.glacier["errors"]),
    }
    families = ["derivation", "frag2", "analog_struct", "model_views", "fragnet"]
    _json(
        directory / "run_manifest.json",
        {
            "secs": round(context.elapsed),
            "merge": results.merge,
            "n_errors": 0,
            "ice_health": "ok",
            "gl_health": "ok",
            "ice_n_scored": context.ice["n_mols_scored"],
            "gl_n_scored": context.glacier["n_mols_scored"],
            "families": families,
            "gpu": context.gpu,
        },
        indent=1,
    )
    summary = {
        "experiment": "V1.1 metric-safe tail",
        "rdkit": IDENTITY_RDKIT,
        "hidden_rerun": True,
        "active_molecules": len(sample_ids),
        "output_rows": len(sample_ids),
        "changed_molecules": sum(row["changed"] for row in results.tail_audit),
        "invalid_removed": sum(row["invalid_count"] for row in results.tail_audit),
        "equivalent_duplicates_removed": sum(
            row["duplicate_count"] for row in results.tail_audit
        ),
        "existing_candidates_promoted": sum(
            row["promoted_original"] for row in results.tail_audit
        ),
        "ranked_candidates_added": sum(
            row["ranked_added"] for row in results.tail_audit
        ),
        "dreams_candidates_added": sum(
            row["dreams_added"] for row in results.tail_audit
        ),
        "monotone_rank_assertions_passed": True,
        "metric_source_sha256": names.metric_source_sha256,
        "baseline_sha256": _sha(directory / "baseline_submission.csv"),
        "submission_sha256": reference_sha,
        "stages": {"ice": ice_meta, "glacier": glacier_meta},
        "runtime_seconds": round(context.elapsed, 1),
        "validation_note": "Visible test is a smoke check only and overlaps training; no generalization score claimed.",
    }
    _json(directory / "candidate_audit.json", summary, indent=2)
    _json(
        directory / "experiment_audit.json",
        {
            "experiment": "metric_identity_fusion",
            "changed_molecules": sum(row["changed"] for row in results.control_audit),
            "changed_top1": sum(row["changed_top1"] for row in results.control_audit),
            "reference_sha256": reference_sha,
            "submission_sha256": control_sha,
            "measured_score": None,
        },
        indent=2,
    )
    processed = sorted(sample_ids)
    pc_inputs: dict[str, PreparedPubChem | dict[str, None]] = {
        mid: pubchem.get(mid, dict[str, None]()) for mid in processed
    }
    candidate_inputs = {
        "prefusion": {mid: results.prefusion[mid] for mid in processed},
        "engine2": {
            mid: {"smiles": list(engine[mid].smiles), "keys": list(engine[mid].keys)}
            for mid in processed
        },
        "pubchem": pc_inputs,
        "base_tail": {mid: list(base_tail[mid]) for mid in processed},
        "fusion_overflow": {mid: results.overflow.get(mid, []) for mid in processed},
    }
    _json(
        directory / "e004_candidate_inputs.json",
        candidate_inputs,
        indent=2,
        newline=True,
    )
    hashes = {
        filename: control_sha
        if filename == "submission.csv"
        else _sha(directory / filename)
        for filename in (
            "submission.csv",
            "baseline_submission.csv",
            "submission_reference.csv",
            "candidate_audit.json",
            "experiment_audit.json",
            "e004_candidate_inputs.json",
        )
    }
    e004 = {
        "experiment": names.control_experiment,
        "source_version_id": names.source_version_id,
        "source_version": names.source_version,
        "source_code_sha256": names.source_code_sha256,
        "input_ids": processed,
        "is_full_run": True,
        "processed_ids": list(processed),
        "processed_molecules": len(sample_ids),
        "input_molecules": len(sample_ids),
        "output_rows": len(sample_ids),
        "placeholder_fill_rows": 0,
        "base_errors": 0,
        "base_nonempty": results.base_nonempty,
        "engine2_nonempty": sum(bool(engine[mid].smiles) for mid in sample_ids),
        "pubchem_nonempty": sum(
            bool(pubchem[mid]["pc"]) for mid in sample_ids if mid in pubchem
        ),
        "main_rdkit": IDENTITY_RDKIT,
        "stages": {"ice": ice_meta, "gl": glacier_meta},
        "hashes": hashes,
        "elapsed_seconds": round(context.elapsed, 2),
        "limitations": [
            "Visible smoke is not independent validation.",
            "Author 0.436 is not this reproduction score.",
            "Whole v80 recipe differs from E003 in multiple factors.",
        ],
    }
    _json(directory / "e004_verification.json", e004, indent=2, newline=True)
    report = {
        "experiment": names.final_experiment,
        "window": policy.config.final.window,
        "ice_fallback": policy.config.fallback is not None,
        "source_experiment": names.control_experiment,
        "base_report_file": "e004_verification.json",
        "control_file": "submission_e004_control.csv",
        "control_sha256": control_sha,
        "submission_sha256": _sha(directory / "submission_base.csv"),
        "processed_ids": processed,
        "output_rows": len(sample_ids),
        "changed_rows": sum(row["changed"] for row in results.final_audit),
        "changed_top1": sum(row["changed_top1"] for row in results.final_audit),
        "fallback_rows": sum(
            row["branch"] == "ice_fallback" for row in results.final_audit
        ),
        "fallback_finite_rows": sum(
            row["branch"] == "ice_fallback" and row["ice_finite_count"] > 0
            for row in results.final_audit
        ),
        "fallback_changed_rows": sum(
            row["fallback_changed"] for row in results.final_audit
        ),
        "fallback_changed_top1": sum(
            row["fallback_changed_top1"] for row in results.final_audit
        ),
        "same_run_control_reconstructed": True,
        "provenance": {
            "base_notebook_sha256": names.notebook_sha256,
            "base_source_code_sha256": names.source_code_sha256,
            "helper_source_sha256": names.helper_source_sha256,
            "source_script_version_id": names.source_version_id,
            "ice_policy_source_script_version_id": names.ice_policy_source_version_id,
            "ice_policy_source": names.ice_policy_source,
        },
        "limitations": [
            "Visible smoke is not independent validation.",
            "An inactive fallback in visible smoke says nothing about hidden benefit.",
        ],
    }
    _json(
        directory / "experiment_audit_factorial.json",
        {**report, "rows": list(results.final_audit)},
        indent=2,
        newline=True,
        finite=True,
    )
    _json(
        directory / "experiment_verification.json",
        report,
        indent=2,
        newline=True,
        finite=True,
    )
    _json(
        directory / "native_manifest.json",
        {
            "implementation": "independent Priml",
            "runtime_chemistry_imports": False,
            "prepared_identity_rdkit": IDENTITY_RDKIT,
            "prepared_simulator_rdkit": SIMULATOR_RDKIT,
            "source_recipe_sha256": names.recipe_sha256,
            "checkpoint_training_released": False,
        },
        indent=2,
        newline=True,
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainInputs:
    """Own every complete native main-arm data and model resource."""

    pool: CandidatePool
    library: SpectralLibrary
    train_fingerprints: NDArray[np.uint8]
    families: MainFamilyData
    reactions: ReactionTable
    chemistry: IdentityTable
    bank: MainBank
    views_bank: MainBank
    stage_a: StageA
    stage_b: LightGBMModel
    rankers: TreeBank
    popularity: NDArray[np.float64]


class MainAssetProvider(Protocol):
    """Allocate a complete independently prepared native main stage."""

    def read(self) -> MainInputs:
        """Read every required resource, preserving missing-input failures."""
        ...


class MainAssets:
    """Read complete main-arm native data from explicitly declared paths."""

    class Config(Fig["MainAssets"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root; every relative path below resolves beneath it."""

        pool_dir: Path | str = "inputs/datasets/ahmedberatozer/casmi26-v2-pool"
        """Complete provided main candidate pool."""

        manifest: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="inputs/datasets/ahmedberatozer/casmi26-v4b-models/MANIFEST.json",
                sha256="e64b960037769971e146a35932e29ed3f042de1716e4dbe281ebd84a7cfc3b6a",
            ),
        )
        """Pinned complete V4 release manifest."""

        library_dir: Path | str = "training/work"
        """Complete native cleaned spectral cache and SID-ordered structures."""

        fingerprints: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-v2-pool/train_fp_sel.npy"
        )
        """Complete packed selected-bit fingerprints in training SID order."""

        popularity_dir: Path | str = (
            "inputs/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior"
        )
        """Complete PID-aligned pool_lsid.npy and pool_lpmid.npy."""

        chemistry: IdentityTable.Config = field(default_factory=IdentityTable.Config)
        """Independent uncoupled product identities, properties and raw fingerprints."""

        reactions_dir: Path | str = "native-chemistry/reactions"
        """Complete ordered reaction requests, products and application logs."""

        pool_structures_dir: Path | str = "native-chemistry/analog-pool"
        """Complete PID-aligned molecular family rows."""

        train_structures_dir: Path | str = "native-chemistry/analog-train"
        """Complete training SID-aligned molecular family rows."""

        generated_structures_dir: Path | str = "native-chemistry/analog-generated"
        """Independent generated-product molecular family rows."""

        pool_families_dir: Path | str = (
            "inputs/datasets/ahmedberatozer/casmi26-v4b-models/code/fe_v4/res/frag2"
        )
        """Provided full PID-aligned depth-one fragments and capabilities."""

        fragments_dir: Path | str = "native-chemistry/ahmed-fragments"
        """Separate complete F32 family fragments and pre-cast F64 MAIN94 masses."""

        intrinsic_dir: Path | str = "native-chemistry/intrinsic"
        """Complete generated intrinsic descriptors and capability vocabulary."""

        fragnet_dir: Path | str = "native-chemistry/fragnet"
        """Complete independently prepared Fragnet molecular records."""

        derivation_dir: Path | str = "native-chemistry/derivation"
        """Every ordered derivation path and independent identity."""

        priors: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="inputs/datasets/ahmedberatozer/casmi26-v4b-models/code/fe_v4/res/derivation/prior_tables.pkl",
                sha256="217905818dd187da47cb458a60aeae2a5f2b0139904b4e320f70e41a9bd74c90",
            ),
        )
        """Pinned complete provided plain-data transform and site counts."""

        stage_a: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="inputs/datasets/ahmedberatozer/casmi26-v4b-models/code/fe_v4/res/fragnet/stageA.npz",
                sha256="3a501ecbfae647d86d7fda87605eda2d573475201b2482ca02a130aa2a3d7a68",
            ),
        )
        """Pinned complete Fragnet StageA categorical bank."""

        stage_b: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="inputs/datasets/ahmedberatozer/casmi26-v4b-models/code/fe_v4/res/fragnet/stageB.txt",
                sha256="6916c25c6f32fdea4a20f5c0b4b5cf45a3f6bbbfa625552195a7909e8ae8685b",
            ),
        )
        """Pinned complete Fragnet StageB numerical text model."""

        rankers: LocalFile.Config = field(
            default_factory=lambda: LocalFile.Config(
                path="inputs/datasets/ahmedberatozer/casmi26-v4b-models/ranker_v4b.pkl",
                sha256="6172e926fb2de244f0b9189ed54e1ed8a15e76d166592b724129a4d6a34931da",
            ),
        )
        """Pinned complete four-ranker file with its complete ordered feature schema."""

        bank: Makeable[MainBank] = field(default_factory=MainBank.Config)
        """Complete pinned Full1 bank; injectable native checkpoint configuration."""

        views_bank: Makeable[MainBank] = field(default_factory=MainBank.Config)
        """Separate complete pinned A/B/D bank in its original model order."""

        @override
        def finalize(self) -> Self:
            self.pool_dir = resolve_working_dir(self.base_dir, self.pool_dir)
            self.library_dir = resolve_working_dir(self.base_dir, self.library_dir)
            self.fingerprints = resolve_working_dir(self.base_dir, self.fingerprints)
            self.popularity_dir = resolve_working_dir(
                self.base_dir,
                self.popularity_dir,
            )
            self.reactions_dir = resolve_working_dir(self.base_dir, self.reactions_dir)
            self.pool_structures_dir = resolve_working_dir(
                self.base_dir,
                self.pool_structures_dir,
            )
            self.train_structures_dir = resolve_working_dir(
                self.base_dir,
                self.train_structures_dir,
            )
            self.generated_structures_dir = resolve_working_dir(
                self.base_dir,
                self.generated_structures_dir,
            )
            self.pool_families_dir = resolve_working_dir(
                self.base_dir,
                self.pool_families_dir,
            )
            self.fragments_dir = resolve_working_dir(self.base_dir, self.fragments_dir)
            self.intrinsic_dir = resolve_working_dir(self.base_dir, self.intrinsic_dir)
            self.fragnet_dir = resolve_working_dir(self.base_dir, self.fragnet_dir)
            self.derivation_dir = resolve_working_dir(
                self.base_dir,
                self.derivation_dir,
            )
            for pinned in (
                self.manifest,
                self.priors,
                self.stage_a,
                self.stage_b,
                self.rankers,
            ):
                pinned.path = resolve_working_dir(self.base_dir, pinned.path)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def read(self) -> MainInputs:
        """Read complete native data before allocating the four neural models.

        Returns:
          inputs: Every aligned main resource, owned by the calling stage.

        """
        config = self.config
        pool = CandidatePool.load(Path(config.pool_dir))
        _ = config.manifest.make().path()
        library = SpectralLibrary.load(Path(config.library_dir))
        train_smiles = training_smiles(
            Path(config.library_dir) / "train_structs.parquet",
        )
        fingerprints = load_array(Path(config.fingerprints), np.uint8, mmap=True)
        if (
            fingerprints.shape
            != (
                len(train_smiles),
                (len(pool.bits) + 7) // 8,
            )
            or library.smiles != train_smiles
        ):
            raise ValueError(
                "Main training fingerprints and structures differ from SID order.",
            )
        popularity = _pool_popularity(Path(config.popularity_dir), rows=len(pool.mass))
        fragments = MainFragmentTable(Path(config.fragments_dir))
        families = MainFamilyData(
            pool=pool,
            train_smiles=train_smiles,
            pool_structures=AnalogTable(Path(config.pool_structures_dir)),
            train_structures=AnalogTable(Path(config.train_structures_dir)),
            generated_structures=AnalogTable(Path(config.generated_structures_dir)),
            pool_families=MainPoolFamilies.load(
                Path(config.pool_families_dir),
                count=len(pool.mass),
            ),
            generated_fragments=fragments,
            generated_intrinsic=IntrinsicTable.load(Path(config.intrinsic_dir)),
            fragnet=FragnetTable(Path(config.fragnet_dir)),
            priors=DerivationPriors.load(
                Path(config.priors.path),
                expected_sha256=config.priors.sha256,
                proposals=DerivationTable(Path(config.derivation_dir)),
            ),
        )
        reactions = ReactionTable(Path(config.reactions_dir))
        chemistry = config.chemistry.make()
        stage_a = StageA.load(Path(config.stage_a.path), config.stage_a.sha256)
        stage_b = load_tree(Path(config.stage_b.path), config.stage_b.sha256)
        rankers = TreeBank.load(Path(config.rankers.path), config.rankers.sha256)
        if len(rankers.models) != 4:
            raise ValueError("Main inference requires all four rankers.")
        return MainInputs(
            pool=pool,
            library=library,
            train_fingerprints=fingerprints,
            families=families,
            reactions=reactions,
            chemistry=chemistry,
            bank=config.bank.make(),
            views_bank=config.views_bank.make(),
            stage_a=stage_a,
            stage_b=stage_b,
            rankers=rankers,
            popularity=popularity,
        )


class MainResource:
    """Own one lazy main stage until the outer controller explicitly closes it."""

    class Config(Fig["MainResource"]):
        assets: Makeable[MainAssetProvider] = field(default_factory=MainAssets.Config)
        """Complete independent native assets, allocated only at first ranking."""

        engine: MainEngine.Config = field(default_factory=MainEngine.Config)
        """Complete Full1 and five-family engine recipe."""

        generation: ReactionScreen.Config = field(default_factory=ReactionScreen.Config)
        """Complete source reaction admission and enumeration budgets."""

        limit: int = 60
        """Unique-key main shortlist cap before popularity permutation."""

        popularity_weight: float = 0.15
        """Main pool-slot popularity coefficient."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = config.engine.make()
        self.generation = config.generation.make()
        self._inputs: MainInputs | None = None
        self._open = True

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
        target: float,
    ) -> RankedMain:
        """Rank complete main queries with every generated and family input.

        Args:
          mid: Query identity owned by the outer error-recording controller.
          spectra: Complete main query view before the engine's spectrum cap.
          target: Molecule-level finite neutral mass.

        Returns:
          ranked: Complete top60 record after source pool popularity permutation.

        """
        del mid
        if not self._open:
            raise RuntimeError("Main resource is closed.")
        if self._inputs is None:
            self._inputs = self.config.assets.make().read()
        inputs = self._inputs
        analogs = self.engine.retrieval.analogs(
            spectra[: self.engine.config.max_query_spectra],
            library=inputs.library,
            target=target,
        )
        initial = inputs.pool.window(target, ppm=self.engine.config.ppm)
        generated = self.generation.run(
            analogs,
            target=target,
            excluded={inputs.pool.keys[initial.item(i)] for i in range(len(initial))},
            train_smiles=inputs.families.train_smiles,
            train_mass=inputs.library.structure_mass,
            train_fingerprints=inputs.train_fingerprints,
            bits=inputs.pool.bits,
            proposals=inputs.reactions,
            chemistry=inputs.chemistry,
            fragments=inputs.families.generated_fragments,
        )
        result = self.engine.run(
            spectra,
            target=target,
            library=inputs.library,
            pool=inputs.pool,
            train_fingerprints=inputs.train_fingerprints,
            bank=inputs.bank,
            views_bank=inputs.views_bank,
            provider=inputs.families,
            stage_a=inputs.stage_a,
            stage_b=inputs.stage_b,
            rankers=inputs.rankers,
            generated=generated.candidates,
            logs=generated.logs,
        )
        return rank_main(
            result,
            popularity=inputs.popularity,
            limit=self.config.limit,
            weight=self.config.popularity_weight,
        )

    def close(self) -> None:
        """Release all owned data and model references permanently."""
        self._inputs = None
        self._open = False


class SecondResource:
    """Own one lazy complete second stage until explicitly closed."""

    class Config(Fig["SecondResource"]):
        assets: Makeable[SecondAssetProvider] = field(
            default_factory=SecondAssets.Config,
        )
        """Every independent second-arm raw resource and complete fitted bank."""

        engine: SecondEngine.Config = field(default_factory=SecondEngine.Config)
        """Complete native retrieval, fingerprint and ranking recipe."""

        prescreen: int = 80
        """Sorted source prefix before recomputed-key deduplication."""

        limit: int = 40
        """Maximum distinct raw second-engine canonical keys."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = config.engine.make()
        self._inputs: SecondInputs | None = None
        self._index: SecondIndex | None = None
        self._open = True

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
    ) -> RankedSecond | None:
        """Rank every raw query row using the complete independent second arm.

        Args:
          mid: Query identity retained by the outer result mapping.
          spectra: Every raw query row before native second-engine preparation.

        Returns:
          ranked: Top40 recomputed raw keys, or None when no score exists.

        """
        del mid
        if not self._open:
            raise RuntimeError("Second resource is closed.")
        if self._inputs is None or self._index is None:
            self._inputs = self.config.assets.make().read()
            self._index = self.engine.retrieval.index(self._inputs.library)
        inputs = self._inputs
        result = self.engine.run(
            spectra,
            pool=inputs.pool,
            library=inputs.library,
            index=self._index,
            fragments=inputs.fragments,
            bank=inputs.bank,
            rankers=inputs.rankers,
            formulas=inputs.formulas,
        )
        return rank_second(
            result,
            key=inputs.identities.key,
            prescreen=self.config.prescreen,
            limit=self.config.limit,
        )

    def close(self) -> None:
        """Release all second-stage data, indices and model references permanently."""
        self._inputs = None
        self._index = None
        self._open = False


class PubChemResource:
    """Own one lazy complete independent V3 stage until explicitly closed."""

    class Config(Fig["PubChemResource"]):
        assets: Makeable[PubChemAssetProvider] = field(
            default_factory=PubChemAssets.Config,
        )
        """Full provided store/popularity, independent chemistry, pool and V3 bank."""

        screen: PubChemScreen.Config = field(default_factory=PubChemScreen.Config)
        """Complete public two-pass and popularity-union screen recipe."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.screen = config.screen.make()
        self._inputs: PubChemInputs | None = None
        self._open = True

    def rank(
        self,
        mid: str,
        spectra: tuple[SpectrumRequest, ...],
    ) -> PubChemResult | None:
        """Screen every raw query row through the full independent V3 channel.

        Args:
          mid: Query identity passed unchanged to source diagnostics.
          spectra: Every raw query row before the inner screen's spectrum cap.

        Returns:
          result: Complete public channel mapping or an invalid-query skip.

        """
        if not self._open:
            raise RuntimeError("PubChem resource is closed.")
        if self._inputs is None:
            self._inputs = self.config.assets.make().read()
        inputs = self._inputs
        return self.screen.run(
            mid,
            spectra,
            store=inputs.store,
            chemistry=inputs.chemistry,
            pool=inputs.pool,
            bank=inputs.bank,
        )

    def close(self) -> None:
        """Release all PubChem arrays, chemistry and model references permanently."""
        self._inputs = None
        self._open = False


def _pool_popularity(directory: Path, *, rows: int) -> NDArray[np.float64]:
    values = [
        load_array(directory / name, np.floating, mmap=True)
        for name in ("pool_lsid.npy", "pool_lpmid.npy")
    ]
    if any(value.shape != (rows,) for value in values):
        raise ValueError("Main popularity arrays differ from complete pool positions.")
    return np.asarray(values[0], np.float64) + np.asarray(values[1], np.float64)


class SimulationRunner(Protocol):
    """Run a complete native simulator stage and publish every finished score map."""

    def run(
        self,
        molecules: Sequence[SimulationMolecule],
        *,
        publish: Callable[[dict[str, dict[str, float | None]]], None],
    ) -> SimulatorResult:
        """Retain full candidate maps, stage health and budget/failure evidence."""
        ...


def run_recipe(
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
    clock: Callable[[], float] = time.perf_counter,
) -> RecipeResults:
    """Run every native public stage and publish only a complete healthy result.

    Args:
      queries: Every original query and both independent molecule orderings.
      sample_csv: Original competition publication order, copied without coercion.
      directory: Already owned run directory, separate from all input artifacts.
      main: Complete lazy main arm, released after its entire molecule loop.
      second: Complete lazy second arm, including actual native HGB fitting.
      pubchem: Complete lazy V3 channel with strict prepared molecular primitives.
      ice: Complete timed native ICE stage with durable whole-map publications.
      glacier: Complete timed native proton-only GL stage, including empty panels.
      identities: Strict prepared raw keys, formulas and official identities.
      graphs: Strict prepared final-gap molecular graphs.
      fragment_gap: Unchanged complete native final fallback scorer.
      started: Observed job origin before any model or numerical asset loading.
      device: Actual configured prediction device of both simulators.
      policy: Complete printed fusion and reranking choices from the owning recipe.
      budget: Printed candidate prefixes and list cut-offs.
      gate: Printed PubChem admission gate.
      provenance: Source experiments and digests that the reports cite.
      dreams: Optional lazy prepared tail store.
      gpu: Observed GPU name, or None for complete CPU execution.
      clock: Job wall clock, independently injectable for whole-record replay.

    Returns:
      results: Every source-compatible candidate list and complete policy audit.

    """
    with sample_csv.open(newline="", encoding="utf-8") as handle:
        sample_ids = [row["molecule_id"] for row in csv.DictReader(handle)]
    arms = rank_arms(queries, main=main, second=second, pubchem=pubchem)
    tail = {mid: row.smiles for mid, row in arms.main.items()}
    panels = simulator_inputs(
        arms.main,
        second=arms.second,
        pubchem=arms.pubchem,
        queries=queries,
        formula=identities.formula,
        budget=budget,
        gate=gate,
    )
    ice_result, ice_meta = _simulate(
        ice,
        molecules=panels.ice,
        path=directory / "ice_work/ice_out.json",
        device=device,
    )
    gl_result, gl_meta = _simulate(
        glacier,
        molecules=panels.glacier,
        path=directory / "gl_work/gl_out.json",
        device=device,
    )
    lists = rerank_pubchem(
        arms.main,
        proposals=panels.proposals,
        ice=ice_result.scores,
        glacier=gl_result.scores,
        policy=policy,
        budget=budget,
    )
    results = assemble_results(
        sample_ids,
        base=lists.main,
        base_tail=tail,
        pubchem=lists.proposals,
        engine=arms.second,
        queries=queries,
        identities=identities,
        ice=ice_result.scores,
        glacier=gl_result.scores,
        graphs=graphs,
        fragment_gap=fragment_gap,
        dreams=dreams,
        ice_status=ice_result.status,
        glacier_status=gl_result.status,
        base_errors=len(arms.errors),
        policy=policy,
        budget=budget,
        gate=gate,
    )
    write_results(
        directory,
        results=results,
        sample_ids=sample_ids,
        base_tail=tail,
        pubchem=lists.proposals,
        engine=arms.second,
        context=PublicationContext(
            sample_csv=sample_csv,
            ice=ice_meta,
            glacier=gl_meta,
            elapsed=clock() - started,
            gpu=gpu,
        ),
        policy=policy,
        provenance=provenance,
    )
    return results


class SimulatorStage:
    """Load native admission/models inside the budget and retain full publications."""

    class Config(Fig["SimulatorStage"]):
        base_dir: Path | str | None = None
        """Dataset root beneath which the stage and its backend resolve their paths."""

        admission_dir: Path | str = "native-chemistry/simulator-admission"
        """Independent complete cheap simulator decisions."""

        backend: Makeable[PredictionBackend] = field(default_factory=IceBackend.Config)
        """Complete native prepared-graph and released-state model backend."""

        scheduling: Makeable[JobScheduling] = field(
            default_factory=IceScheduling.Config,
        )
        """Admission decision and job ordering of the backend's model family."""

        scoring: QueryScoring.Config = field(default_factory=QueryScoring.Config)
        """Query cleaning and entropy similarity of the stage."""

        budget: WallBudget.Config = field(
            default_factory=lambda: WallBudget.Config(seconds=5400.0),
        )
        """Complete stage allowance, including admission and checkpoint loading."""

        batch_size: int = 96
        """Initial native prediction batch geometry."""

        chunk_jobs: int = 3000
        """Flush only after complete molecules reach this job count."""

        asset_seals: dict[Path | str, str] = field(
            default_factory=dict[Path | str, str],
        )
        """Complete checkpoint digests verified inside the stage budget."""

        @override
        def finalize(self) -> Self:
            self.admission_dir = resolve_working_dir(self.base_dir, self.admission_dir)
            if (
                isinstance(self.backend, DatasetRooted)
                and self.backend.base_dir is None
            ):
                self.backend.base_dir = self.base_dir
            self.asset_seals = {
                resolve_working_dir(self.base_dir, path): seal
                for path, seal in self.asset_seals.items()
            }
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(
        self,
        molecules: Sequence[SimulationMolecule],
        *,
        publish: Callable[[dict[str, dict[str, float | None]]], None],
    ) -> SimulatorResult:
        """Run complete native scheduling with the configured scientific assets.

        Args:
          molecules: Complete ordered query and candidate groups.
          publish: Durable complete-map sink, called before model loading.

        Returns:
          result: Every completed cache, score, failure and stage-health field.

        """
        config = self.config
        return run_simulator(
            molecules,
            scheduling=config.scheduling.make(),
            scoring=config.scoring.make(),
            admission_factory=partial(
                SimulatorAdmissionTable,
                Path(config.admission_dir),
            ),
            backend_factory=self._backend,
            publish=publish,
            budget=config.budget,
            batch_size=config.batch_size,
            chunk_jobs=config.chunk_jobs,
        )

    def _backend(self) -> PredictionBackend:
        for path, expected in self.config.asset_seals.items():
            with Path(path).open("rb") as handle:
                actual = hashlib.file_digest(handle, hashlib.sha256).hexdigest()
            if actual != expected:
                raise ValueError(f"Simulator asset digest differs: {path}")
        return self.config.backend.make()


def glacier_stage(*, backend: Makeable[PredictionBackend]) -> SimulatorStage.Config:
    """Return the distinct complete public GLACIER stage configuration.

    Args:
      backend: Declared prediction resource.

    Returns:
      config: Canonical scheduling with its original model, budget and batch size.

    """
    config = SimulatorStage.Config()
    config.backend = backend
    config.scheduling = GlacierScheduling.Config()
    config.budget.seconds = 4000.0
    config.batch_size = 32
    return config


@runtime_checkable
class HasBaseDir(Protocol):
    """A path-owning Config that resolves its relative paths beneath a parent root."""

    base_dir: Path | str | None


@runtime_checkable
class HasDevice(Protocol):
    """A Config naming the device on which its models and inputs run."""

    device: str


DEVICE_OWNERS: Final = (
    MainBank.Config,
    SecondBank.Config,
    IceBackend.Config,
    GlacierBackend.Config,
)
"""Configs that place models on the recipe device; their layers keep their own."""


class PublicRecipe:
    """Own a new output and compose every independently native public stage."""

    class Config(Fig["PublicRecipe"]):
        study_name: str = ""
        """Run-family name, the first part of the output directory."""

        experiment_name: str = ""
        """Experiment name, the second part of the output directory."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root beneath which the run output and the dataset root resolve."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """New owned output; existing runs and overlapping inputs are rejected."""

        dataset_dir: Path | str = "/datasets/casmi2026"
        """Dataset root beneath which every relative input path resolves."""

        queries: Path | str = (
            "inputs/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
        )
        """Complete original competition query rows."""

        sample_csv: Path | str = (
            "inputs/enveda-CASMI26-molecule-id-mass-spectra/sample_submission.csv"
        )
        """Complete competition publication order, with string identifiers."""

        input_dir: Path | str = "inputs"
        """Protected asset root of the provided competition and model downloads."""

        dreams: Path | str | None = None
        """Prepared DREAMS mass store directory for the safe tail; None disables it."""

        protected_inputs: list[Path | str] = field(default_factory=lambda: [Path()])
        """All additional input roots that this output must never overwrite."""

        main: Makeable[MainRanker] = field(default_factory=MainResource.Config)
        """Full main bank, generation, five feature families and four rankers."""

        second: Makeable[SecondRanker] = field(default_factory=SecondResource.Config)
        """Independent raw retrieval and all twelve natively fitted HGB rankers."""

        pubchem: Makeable[PubChemRanker] = field(default_factory=PubChemResource.Config)
        """Full V3 bank, original mass store, popularity and prepared chemistry."""

        ice: Makeable[SimulationRunner] = field(default_factory=SimulatorStage.Config)
        """Full native ICE model and original timed scheduling policy."""

        glacier: Makeable[SimulationRunner] = field(
            default_factory=SimulatorStage.Config,
        )
        """Full native GL model and its distinct original budget and batches."""

        identities: Makeable[ResultIdentities] = field(
            default_factory=IdentityTable.Config,
        )
        """Strict independent raw keys, formulas and official metric identities."""

        graphs: Makeable[GapGraphs] = field(
            default_factory=GapGraphTable.Config,
        )
        """Strict complete prepared graph rows for the final fragment-gap fallback."""

        fragment_gap: Makeable[FragmentGap] = field(default_factory=FragmentGap.Config)
        """Complete native fallback mass matching and source candidate policy."""

        policy: FusionPolicy.Config = field(default_factory=FusionPolicy.Config)
        """Independent legacy/control/final votes and the simulator fallback."""

        budget: ListBudget.Config = field(default_factory=ListBudget.Config)
        """Candidate prefixes and cut-offs of the simulator and published lists."""

        gate: PubChemGate.Config = field(default_factory=PubChemGate.Config)
        """Library score and fit margins admitting PubChem proposals."""

        provenance: PublicationProvenance.Config = field(
            default_factory=PublicationProvenance.Config,
        )
        """Source experiments, versions and digests that the reports cite."""

        device: str = "cpu"
        """Device of every neural model and numerical input; also names the GPU."""

        clock: Callable[[], float] = time.perf_counter
        """Whole-job wall clock, beginning before asset and model construction."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.working_dir, str):
                self.working_dir = self.working_dir.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            dataset = resolve_working_dir(self.base_dir, self.dataset_dir)
            self.dataset_dir = dataset
            self.queries = resolve_working_dir(dataset, self.queries)
            self.sample_csv = resolve_working_dir(dataset, self.sample_csv)
            self.input_dir = resolve_working_dir(dataset, self.input_dir)
            if self.dreams is not None:
                self.dreams = resolve_working_dir(dataset, self.dreams)
            self.protected_inputs = [
                resolve_working_dir(dataset, path) for path in self.protected_inputs
            ]
            for part in (
                self.main,
                self.second,
                self.pubchem,
                self.ice,
                self.glacier,
                self.identities,
                self.graphs,
            ):
                for match in traverse(part, HasBaseDir, recurse=True):
                    if match.config.base_dir is None:
                        match.config.base_dir = dataset
                for match in traverse(part, DEVICE_OWNERS):
                    propagate_attr(
                        match.config,
                        "device",
                        self.device,
                        protocol=HasDevice,
                    )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> RecipeResults:
        """Run every native stage and preserve complete failures before publication.

        Args:
          *args: No passthrough arguments are accepted.

        Returns:
          results: Complete source-compatible lists, fallback scores and audits.

        """
        if args:
            raise ValueError("PublicRecipe accepts configuration fields only.")
        config = self.config
        started = config.clock()
        queries_path, sample_csv = Path(config.queries), Path(config.sample_csv)
        protected = (
            Path(config.input_dir),
            queries_path,
            sample_csv,
            *map(Path, config.protected_inputs),
        )
        output = validated_output_path(config.working_dir, protected=protected)
        if any(
            output.resolve().is_relative_to(path.resolve())
            or path.resolve().is_relative_to(output.resolve())
            for path in protected
        ):
            raise ValueError("Recipe output overlaps protected inputs.")
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.txt").write_text(
            config.pformat(hide_default_values=False) + "\n",
        )
        queries = read_queries(queries_path)
        identities, graphs = config.identities.make(), config.graphs.make()
        dreams = (
            None if config.dreams is None else DreamsStore.load(Path(config.dreams))
        )
        (output / "dreams.json").write_text(
            json.dumps({"used": dreams is not None, "path": str(config.dreams)}) + "\n",
        )
        gpu = (
            torch.cuda.get_device_name(config.device)
            if torch.device(config.device).type == "cuda"
            else None
        )
        return run_recipe(
            queries,
            sample_csv=sample_csv,
            directory=output,
            main=config.main.make(),
            second=config.second.make(),
            pubchem=config.pubchem.make(),
            ice=config.ice.make(),
            glacier=config.glacier.make(),
            identities=identities,
            graphs=graphs,
            fragment_gap=config.fragment_gap.make(),
            dreams=dreams,
            device=config.device,
            gpu=gpu,
            started=started,
            clock=config.clock,
            policy=config.policy.make(),
            budget=config.budget.make(),
            gate=config.gate.make(),
            provenance=config.provenance.make(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationRequest:
    """Retain one potential parent with the complete original enumeration budget."""

    molecule_id: str
    sid: int
    similarity: float
    target: float
    header: ReactionHeader


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowRequest:
    """Retain an original external row, spelling and query-window provenance."""

    molecule_id: str
    row: int
    smiles: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RecipePlan:
    """Keep every potential reaction and complete external mass-window row."""

    reactions: tuple[GenerationRequest, ...]
    pubchem: tuple[WindowRequest, ...]
    dreams: tuple[WindowRequest, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorRequest:
    """Retain a reached parent/combo pair and complete site-enumeration limits."""

    parent_smiles: str
    combo: str
    max_first: int
    max_second: int
    max_mols: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RecipeInputs:
    """Retain raw/standardized proposals and every potential consumer spelling."""

    generated: tuple[str, ...]
    candidates: tuple[str, ...]
    derivations: tuple[PriorRequest, ...]


class Standardizations(Protocol):
    """Require explicit independent standardization, including known invalid rows."""

    def standardize(self, smiles: str) -> str | None:
        """Read the prepared decision or raise for missing raw strings."""
        ...


def plan_requests(
    queries: QueryDataset,
    *,
    library: SpectralLibrary,
    retrieval: MainRetrieval,
    generator: ReactionScreen,
    pubchem: PubChemStore,
    ppm: float,
    max_spectra: int,
    dreams: DreamsStore | None = None,
) -> RecipePlan:
    """Declare complete potential chemistry without loading neural checkpoints.

    Args:
      queries: Every original query and both complete molecule input views.
      library: Native prepared training spectra in their original row order.
      retrieval: The same analog retrieval used by the main numerical arm.
      generator: The same complete parent and reaction budgets used at inference.
      pubchem: Original complete mass and ASCII stores.
      ppm: PubChem mass-window half-width in parts per million.
      max_spectra: Main-arm query cap before analog selection.
      dreams: Optional original mass/string store, queried only at a valid tail mass.

    Returns:
      plan: Every potential parent and every positional external-window row.

    """
    reactions: list[GenerationRequest] = []
    pc: list[WindowRequest] = []
    tail: list[WindowRequest] = []
    for molecule in queries.molecules:
        if molecule.target is not None:
            analogs = retrieval.analogs(
                molecule.main[:max_spectra],
                library=library,
                target=molecule.target,
            )
            reactions.extend(
                GenerationRequest(
                    molecule_id=molecule.mid,
                    sid=analog.sid,
                    similarity=analog.similarity,
                    target=molecule.target,
                    header=normalize_request(request),
                )
                for analog, request in generator.plan(
                    analogs,
                    target=molecule.target,
                    train_smiles=library.smiles,
                    train_mass=library.structure_mass,
                )
            )
            begin, strings = pubchem.window(molecule.target, ppm=ppm)
            pc.extend(
                WindowRequest(molecule_id=molecule.mid, row=begin + i, smiles=value)
                for i, value in enumerate(strings)
            )
        target = tail_target(molecule)
        if dreams is not None and math.isfinite(target) and target > 0:
            selection = dreams.select(target)
            tail.extend(
                WindowRequest(
                    molecule_id=molecule.mid,
                    row=selection.rows.item(i),
                    smiles=value,
                )
                for i, value in enumerate(selection.smiles)
            )
    return RecipePlan(reactions=tuple(reactions), pubchem=tuple(pc), dreams=tuple(tail))


def product_inputs(
    reactions: ReactionTable,
    *,
    chemistry: Standardizations,
    pool: Sequence[str],
    train: Sequence[str],
    second: Sequence[str],
    windows: Sequence[WindowRequest],
    max_first: int,
    max_second: int,
    max_mols: int,
) -> RecipeInputs:
    """Expand complete proposals and strict identity coverage before preparation.

    Args:
      reactions: Every prepared request, including explicit zero-product headers.
      chemistry: Strict independent standardization for every raw reaction product.
      pool: Complete main pool strings in original PID order.
      train: Complete training strings in original SID order.
      second: Complete second-engine strings in its independently sorted pool order.
      windows: Every PubChem and optional DREAMS row from the complete mass windows.
      max_first: First-reaction site limit of every derivation request.
      max_second: Second-reaction site limit of every derivation request.
      max_mols: Molecule limit of every derivation request.

    Returns:
      inputs: All raw products, standardized aliases, prior pairs and consumer strings.

    """
    generated: list[str] = []
    derivations: list[PriorRequest] = []
    for request in reactions.requests:
        for product in reactions.products(request):
            raw = product["raw"]
            generated.append(raw)
            standardized = chemistry.standardize(raw)
            if standardized is not None:
                generated.append(standardized)
            combo = product["combo"]
            if combo is not None:
                derivations.append(
                    PriorRequest(
                        parent_smiles=request["smiles"],
                        combo="+".join(combo),
                        max_first=max_first,
                        max_second=max_second,
                        max_mols=max_mols,
                    ),
                )
    candidates = (
        *pool,
        *train,
        *second,
        *generated,
        *(row.smiles for row in windows),
        *(row["smiles"] for row in reactions.requests),
        PLACEHOLDER,
    )
    return RecipeInputs(
        generated=tuple(generated),
        candidates=(
            *candidates,
            *(value.strip() for value in candidates if value.strip()),
        ),
        derivations=tuple(derivations),
    )


def _molecule(source_id: str | int, *, rows: list[QueryRow]) -> MoleculeQueries:
    masses = neutral_mass(
        np.array([row["precursor_mz"] for row in rows], np.float64),
        [row["adduct"] for row in rows],
    )
    finite = masses[np.isfinite(masses)]
    main: list[SpectrumRequest] = []
    raw: list[SpectrumRequest] = []
    for row in rows:
        values = row["collision_energy_ev"]
        energy = np.asarray(values, np.float64) if values is not None else np.zeros(0)
        spectrum: SpectrumRequest = {
            "mz": row["ms2_mzs"],
            "it": row["ms2_normalized_intensities"],
            "mode": 1 if row["ionization_mode"] == "positive" else -1,
            "adduct": row["adduct"],
            "prec": row["precursor_mz"],
            "ce": float(energy.mean()) if len(energy) else float("nan"),
            "ce_n": len(energy) or 1,
        }
        main.append(spectrum)
        raw.append(
            {
                **spectrum,
                "instrument": row["instrument_type"],
                "ce_units": row["collision_energy_orig_units"],
                "ce_orig": row["collision_energy_orig"],
                "ce_ev": [float(value) for value in energy]
                if values is not None
                else None,
            },
        )
    return MoleculeQueries(
        mid=str(source_id),
        source_id=source_id,
        main=tuple(main) if len(finite) else (),
        raw=tuple(raw),
        finite_mass=finite,
        target=float(np.median(finite)) if len(finite) else None,
    )


def _screen(
    sample_ids: Sequence[str],
    *,
    base: Mapping[str, BaseList],
    pubchem: Mapping[str, PreparedPubChem],
    gate: PubChemGate,
) -> tuple[dict[str, list[str]], MergeCounts]:
    rows: dict[str, list[str]] = {}
    counts: MergeCounts = {
        "untouched": 0,
        "gentle": 0,
        "aggressive": 0,
        "no_pc": 0,
        "promoted": 0,
    }
    for mid in sample_ids:
        main, proposal = base.get(mid), pubchem.get(mid)
        screened = gate.screen(
            main.smiles if main is not None else (),
            main.keys if main is not None else (),
            library_max=main.library_max if main is not None else 0.0,
            proposals=proposal["pc"] if proposal is not None else (),
            proposal_keys=proposal["pc_keys"] if proposal is not None else (),
            proposal_scores=proposal["pc_fz"] if proposal is not None else (),
            top_fit=proposal["fz_top"] if proposal is not None else None,
            pool_fit=proposal["best_pool_fz"] if proposal is not None else math.nan,
            outlier=proposal["S"] if proposal is not None else None,
            popularity=proposal["top_pop"] if proposal is not None else None,
        )
        rows[mid] = list(screened.candidates)
        counts[screened.mode] += 1
        counts["promoted"] += int(screened.promoted)
    return rows, counts


def _legacy(
    prefusion: Mapping[str, list[str]],
    *,
    engine: Mapping[str, RankedSecond],
    identities: ResultIdentities,
    ice: Mapping[str, Mapping[str, float | None]],
    glacier: Mapping[str, Mapping[str, float | None]],
    policy: FusionPolicy,
    budget: ListBudget,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    baseline: dict[str, list[str]] = {}
    overflow: dict[str, list[str]] = {}
    for mid, original in prefusion.items():
        first = (
            [value for value in original if value and value != PLACEHOLDER]
            if engine
            else list(original)
        )
        second = engine.get(mid)
        if second is not None and second.smiles:
            first, keys, scores = policy.legacy(
                first,
                second.smiles,
                second.keys,
                key=identities.key,
            )
            ice_scores = ice.get(mid)
            if ice_scores is not None and len(ice_scores) > 0:
                forms = [identities.formula(value) for value in first]
                gl = glacier.get(mid, {})
                if any(value is not None for value in gl.values()):
                    order = rerank_groups(
                        first,
                        keys,
                        scores,
                        forms,
                        columns=[gl],
                        weights=[policy.config.glacier_weight],
                        top_n=budget.config.legacy_rerank_top_n,
                        min_covered=policy.config.min_covered,
                        tie_eps=policy.config.tie_eps,
                    )
                    first = [first[index] for index in order]
            overflow[mid] = first[policy.config.length :]
            first = first[: policy.config.length]
        baseline[mid] = first or [PLACEHOLDER]
    return baseline, overflow


def _gap_scores(
    prefusion: Mapping[str, list[str]],
    *,
    engine: Mapping[str, RankedSecond],
    queries: Mapping[str, MoleculeQueries],
    ice: Mapping[str, Mapping[str, float | None]],
    glacier: Mapping[str, Mapping[str, float | None]],
    model: FragmentGap,
    graphs: GapGraphs,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for mid, first in prefusion.items():
        if any(value is not None for value in glacier.get(mid, {}).values()) or any(
            value is not None for value in ice.get(mid, {}).values()
        ):
            continue
        second, query = engine[mid], queries[mid]
        candidates = [
            value
            for value in dict.fromkeys(
                [*first, *second.smiles],
            )
            if value and value != PLACEHOLDER
        ]
        if len(candidates) < 2:
            continue
        spectra: list[GapSpectrum] = [
            {
                "mz": row["mz"],
                "it": row["it"],
                "adduct": row["adduct"],
                "prec": row["prec"],
                "mode": 1 if row["mode"] in (1, "positive") else -1,
            }
            for row in query.raw[: model.config.max_spectra]
        ]
        scores = model.score(spectra, candidates, graphs=graphs)
        result[mid] = {
            smiles: score
            for smiles, score in zip(candidates, scores, strict=True)
            if score is not None
        }
    return result


def _csv(
    path: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json(
    path: Path,
    value: object,
    *,
    indent: int | None = None,
    newline: bool = False,
    finite: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            to_plain(value, dialect="python"),
            indent=indent,
            allow_nan=not finite,
        )
        + ("\n" if newline else ""),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _simulate(
    stage: SimulationRunner,
    *,
    molecules: Sequence[SimulationMolecule],
    path: Path,
    device: str,
) -> tuple[SimulatorResult, StageMetadata]:
    path.parent.mkdir(exist_ok=True)
    result = stage.run(molecules, publish=partial(_write, path=path))
    metadata: StageMetadata = {
        "status": result.status,
        "device": device,
        "rdkit": SIMULATOR_RDKIT,
        "n_mols_scored": result.scored_molecules,
        "n_scored": result.scored_candidates,
        "errors": list(result.errors),
    }
    _write(metadata, path=path.with_suffix(path.suffix + ".meta.json"))
    return result, metadata


def _write(value: object, *, path: Path) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json_dumps(value, dialect="python"))
    pending.replace(path)
