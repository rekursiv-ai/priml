"""Rank fusion, reranking and fragment-gap scoring."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import math

from configgle import Fig
from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.custom_types import (
    GapGraph,
    GapGraphRow,
    PreparationHeader,
)
from priml.baselines.casmi2026.isolated import (
    ELECTRON_MASS,
    ELEMENT_MASSES,
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray
    from pyarrow import parquet

    from priml.baselines.casmi2026.custom_types import GapSpectrum, MetricIdentity
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


PLACEHOLDER: Final = "CCO"
"""Single-molecule list that the source publishes when no candidate exists."""


class RankFusion:
    """Fuse a main ranking with a down-weighted second ranking by reciprocal rank."""

    class Config(Fig["RankFusion"]):
        alpha: float = 1.0
        """Second-engine vote weight."""

        offset: float = 0.0
        """Reciprocal-rank denominator offset."""

        window: int = 25
        """Maximum number of fused representatives."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def legacy(
        self,
        first: Sequence[str],
        second: Sequence[str],
        second_keys: Sequence[str | None],
        *,
        key: Callable[[str], str | None],
    ) -> tuple[list[str], list[str], list[float]]:
        """Fuse every raw source vote, retaining duplicate votes and invalid fallbacks.

        Args:
          first: Main-engine representatives, including positional duplicates.
          second: Second-engine representatives in source rank order.
          second_keys: Aligned optional older keys; the shorter sequence bounds votes.
          key: Strict prepared lookup of the older chemical key.

        Returns:
          candidates: First-encountered representative of each ranked key.
          keys: Older keys or original strings for missing main-engine keys.
          scores: Complete aligned reciprocal-rank vote sums.

        """
        config = self.config
        scores: dict[str, float] = {}
        representatives: dict[str, str] = {}
        for rank, candidate in enumerate(first, start=1):
            candidate_key = key(candidate) or candidate
            scores[candidate_key] = scores.get(candidate_key, 0.0) + 1.0 / (
                config.offset + rank
            )
            representatives.setdefault(candidate_key, candidate)
        for index in range(min(len(second), len(second_keys))):
            candidate_key = second_keys[index]
            if candidate_key is not None and candidate_key != "":
                rank = index + 1
                scores[candidate_key] = scores.get(
                    candidate_key,
                    0.0,
                ) + config.alpha / (config.offset + rank)
                representatives.setdefault(candidate_key, second[index])
        ordered = sorted(scores, key=lambda candidate_key: -scores[candidate_key])[
            : config.window
        ]
        return (
            [representatives[candidate_key] for candidate_key in ordered],
            ordered,
            [scores[candidate_key] for candidate_key in ordered],
        )

    def metric[Identity: Hashable](
        self,
        first: Sequence[str],
        second: Sequence[str],
        *,
        identity: Callable[[str], Identity | None],
    ) -> tuple[list[str], list[float], int]:
        """Fuse one compacted vote per identity per engine in stable encounter order.

        Args:
          first: Main-engine candidates.
          second: Second-engine candidates.
          identity: Official metric canonicalization.

        Returns:
          candidates: Ranked first-encountered representatives.
          values: Reciprocal-rank scores in the same order.
          collapsed: Duplicate source votes removed before compacting ranks.

        Raises:
          ValueError: The window is not positive.

        """
        config = self.config
        if config.window <= 0:
            raise ValueError("window must be positive.")
        scores: dict[Identity, float] = {}
        representatives: dict[Identity, str] = {}
        collapsed = 0
        for candidates, weight in ((first, 1.0), (second, config.alpha)):
            seen: set[Identity] = set()
            for candidate in candidates:
                key = identity(candidate)
                if key is None:
                    continue
                if key in seen:
                    collapsed += 1
                    continue
                seen.add(key)
                scores[key] = scores.get(key, 0.0) + weight / (
                    config.offset + len(seen)
                )
                representatives.setdefault(key, candidate)
        ordered = sorted(scores, key=lambda key: -scores[key])[: config.window]
        return (
            [representatives[key] for key in ordered],
            [scores[key] for key in ordered],
            collapsed,
        )


def merge(
    base: Sequence[str],
    base_keys: Sequence[str],
    pc: Sequence[str],
    pc_keys: Sequence[str],
    *,
    slots: Sequence[int],
    limit: int,
) -> list[str]:
    """Interleave novel PubChem candidates at the reference's one-based slots.

    Args:
      base: Ranked main-engine representatives.
      base_keys: Main-engine chemical keys.
      pc: Ranked PubChem-only representatives.
      pc_keys: PubChem chemical keys.
      slots: One-based insertion positions.
      limit: Maximum output length.

    Returns:
      candidates: The interleaved list.

    """
    if limit <= 0:
        raise ValueError("limit must be positive.")
    base_set = set(base_keys)
    proposals = [s for s, key in zip(pc, pc_keys, strict=True) if key not in base_set]
    out: list[str] = []
    bi, pj = 0, 0
    for position in range(1, limit + 1):
        if position in slots and pj < len(proposals):
            out.append(proposals[pj])
            pj += 1
        elif bi < len(base):
            out.append(base[bi])
            bi += 1
        elif pj < len(proposals):
            out.append(proposals[pj])
            pj += 1
    return out


def promote(
    base: Sequence[str],
    base_keys: Sequence[str],
    pc: Sequence[str],
    pc_keys: Sequence[str],
    *,
    slots: Sequence[int],
    limit: int,
) -> list[str]:
    """Promote PubChem's first proposal ahead of the usual merged list."""
    usual = merge(base, base_keys, pc, pc_keys, slots=slots, limit=limit)
    if not pc:
        return usual
    return ([pc[0]] + [s for s in usual if s != pc[0]])[:limit]


def safe_extend[Identity: Hashable](
    original: Sequence[str],
    extra: Sequence[str],
    *,
    identity: Callable[[str], Identity | None],
    limit: int,
) -> list[str]:
    """Keep the first spelling of each valid identity, then fill vacant ranks.

    Args:
      original: Candidates whose identity order takes precedence.
      extra: Ranked tail candidates.
      identity: Official metric canonicalization.
      limit: Maximum output length.

    Returns:
      candidates: Unique valid representatives in encounter order.

    """
    if limit <= 0:
        raise ValueError("limit must be positive.")
    out: list[str] = []
    seen: set[Identity] = set()
    for raw in [*original, *extra]:
        candidate = raw.strip()
        key = identity(candidate)
        if key is not None and key not in seen:
            seen.add(key)
            out.append(candidate)
            if len(out) == limit:
                break
    return out


def insert_engine_second[Identity: Hashable](
    candidates: Sequence[str],
    engine: Sequence[str],
    *,
    key: Callable[[str], Identity | None],
    limit: int,
) -> list[str]:
    """Insert engine 2's valid leading candidate at rank two exactly once.

    Args:
      candidates: Completed fused submission list.
      engine: Second-engine list in its original rank order.
      key: The engine's older tautomer key, without stereo removal.
      limit: Maximum output length.

    Returns:
      candidates: The rank-two insertion result, retaining fused rank one.

    """
    if limit <= 0:
        raise ValueError("limit must be positive.")
    tokens = list(candidates)
    if not tokens or tokens[0] == PLACEHOLDER:
        return tokens
    leading = [s for s in engine if s and s != PLACEHOLDER]
    if not leading or key(leading[0]) is None:
        return tokens
    proposal = leading[0]
    proposal_key = key(proposal)
    if proposal_key == key(tokens[0]):
        return tokens
    keys = [key(token) or token for token in tokens]
    rest = [token for token, k in zip(tokens, keys, strict=True) if k != proposal_key]
    return [*rest[:1], proposal, *rest[1:]][:limit]


type Score = float | int | str | None
type ScoreColumn = Mapping[str, Score] | Sequence[Score] | None


def finite_score(value: Score) -> float | None:
    """Read a finite optional simulator score.

    Args:
      value: A numeric value or optional numeric text from a simulator.

    Returns:
      score: Finite numeric value, or None for missing/nonfinite/malformed text.

    """
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def guarded_z(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardize by the population deviation, guarded against a zero spread.

    Args:
      values: Scores to center and scale.

    Returns:
      scores: Values minus their mean, divided by their deviation plus 1e-9.

    """
    return (values - values.mean()) / (values.std() + 1e-9)


def standardized(values: Sequence[float | None]) -> list[float]:
    """Center and scale finite scores using sample standard deviation.

    Args:
      values: Scores in current candidate order, including missing values.

    Returns:
      scores: Sample z-scores; missing and zero-variance groups contribute zero.

    """
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if len(valid) < 2:
        return [0.0] * len(values)
    mean = sum(valid) / len(valid)
    variance = sum((value - mean) ** 2 for value in valid) / (len(valid) - 1)
    deviation = variance**0.5
    if deviation <= 0:
        return [0.0] * len(values)
    return [
        (value - mean) / deviation
        if value is not None and math.isfinite(value)
        else 0.0
        for value in values
    ]


def simulator_candidates(
    candidates: Sequence[str],
    formulas: Sequence[str],
    *,
    top_n: int,
) -> list[str]:
    """Admit distinct members of repeated-formula groups in the ranked prefix.

    Args:
      candidates: Ranked molecular representatives.
      formulas: Aligned molecular formulas, including generation-group suffixes.
      top_n: Prefix whose repeated formulas can be reranked.

    Returns:
      candidates: Eligible representatives in encounter order, without duplicates.

    """
    prefix = min(top_n, len(candidates))
    counts: dict[str, int] = {}
    for formula in formulas[:prefix]:
        counts[formula] = counts.get(formula, 0) + 1
    return list(
        dict.fromkeys(
            candidate
            for index, candidate in enumerate(candidates[:prefix])
            if counts[formulas[index]] >= 2
        ),
    )


def rerank_groups(
    candidates: Sequence[str],
    keys: Sequence[str] | None,
    scores: Sequence[float],
    formulas: Sequence[str],
    *,
    columns: Sequence[ScoreColumn],
    weights: Sequence[float],
    top_n: int,
    min_covered: int,
    tie_eps: float,
) -> list[int]:
    """Fuse standardized evidence within same-formula groups, preserving slots.

    Args:
      candidates: Molecular representatives in current ranking order.
      keys: Optional older chemical keys for simulator lookup fallback.
      scores: Aligned base-ranker evidence.
      formulas: Aligned formula groups; generated suffixes remain distinct.
      columns: Optional simulator mappings or aligned sequences of scores.
      weights: One weight per simulator, applied after within-group scaling.
      top_n: Prefix eligible for permutation.
      min_covered: Required finite coverage from at least one simulator.
      tie_eps: Multiple contributing simulators use this tie grid; zero disables it.

    Returns:
      order: Source candidate index for each original output slot.

    """
    if len(columns) != len(weights):
        raise ValueError("Each simulator requires one weight.")
    active = [
        (column, weights[index])
        for index, column in enumerate(columns)
        if column is not None and len(column) > 0
    ]
    order = list(range(len(candidates)))
    groups: dict[str, list[int]] = {}
    for index in range(min(top_n, len(candidates))):
        groups.setdefault(formulas[index], []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        evidence = [
            (
                [_column_score(column, candidates, keys, index) for index in indices],
                weight,
            )
            for column, weight in active
        ]
        counts = [sum(value is not None for value in values) for values, _ in evidence]
        if not any(count >= max(2, min_covered) for count in counts):
            continue
        fused = standardized([scores[index] for index in indices])
        contributors = 0
        for column_index, (values, weight) in enumerate(evidence):
            if counts[column_index] >= 2:
                normalized = standardized(values)
                fused = [
                    value + weight * normalized[index]
                    for index, value in enumerate(fused)
                ]
                contributors += 1
        if contributors >= 2 and tie_eps > 0:
            fused = [round(value / tie_eps) for value in fused]
        ranked = sorted(range(len(indices)), key=lambda index: (-fused[index], index))
        for index, slot in enumerate(indices):
            order[slot] = indices[ranked[index]]
    return order


type FusionBranch = Literal[
    "placeholder",
    "no_engine",
    "unrescored",
    "glacier",
    "frag_gap",
    "ice_fallback",
]


class FusionAudit(TypedDict):
    """Retain all nine source audit fields in their original insertion order."""

    molecule_id: str
    processed: bool
    branch: FusionBranch
    candidate_count: int
    collapsed_source_votes: int
    ice_finite_count: int
    gl_finite_count: int
    fallback_changed: bool
    fallback_changed_top1: bool


class FusionInputs(TypedDict):
    """Complete scientific evidence shared by final fusion and its control."""

    prefusion: list[str] | None
    engine: list[str] | None
    identity: Callable[[str], MetricIdentity | None]
    key: Callable[[str], str | None]
    formula: Callable[[str], str]
    ice: Mapping[str, float | None] | None
    glacier: Mapping[str, float | None] | None
    gap: Mapping[str, float | None] | None


class SimulatorFallback:
    """Rescue a molecule without GLACIER scores from its ICE or fragment-gap scores."""

    class Config(Fig["SimulatorFallback"]):
        gap_weight: float = 1.0
        """Standardized fragment-gap evidence weight."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def evidence(
        self,
        *,
        ice: Mapping[str, float | None],
        glacier: Mapping[str, float | None],
        gap: Mapping[str, float | None] | None,
        ice_weight: float,
    ) -> tuple[FusionBranch, list[Mapping[str, float | None]], list[float]] | None:
        """Select the fallback branch and its weighted score columns.

        Args:
          ice: Complete ICE scores; empty when absent.
          glacier: GLACIER scores; empty when absent or all missing.
          gap: Optional fragment-gap scores.
          ice_weight: Standardized ICE evidence weight.

        Returns:
          branch: Fallback branch name.
          columns: Score columns of the branch.
          weights: One weight per column.

        """
        if glacier:
            return None
        if (
            gap is not None
            and len(gap) > 0
            and not any(value is not None for value in ice.values())
        ):
            return "frag_gap", [gap], [self.config.gap_weight]
        if ice:
            return "ice_fallback", [ice], [ice_weight]
        return None


class FusionPolicy:
    """Own the printed public list, reciprocal-rank and simulator policies."""

    class Config(Fig["FusionPolicy"]):
        legacy: RankFusion.Config = field(default_factory=RankFusion.Config)
        """Vote of the legacy list before official-identity reconstruction."""

        control: RankFusion.Config = field(default_factory=RankFusion.Config)
        """Official-identity vote of the independent E004 control, without fallback."""

        final: RankFusion.Config = field(default_factory=RankFusion.Config)
        """Official-identity vote of the final published policy."""

        fallback: SimulatorFallback.Config | None = None
        """ICE and fragment-gap fallback of the final policy; None disables it."""

        length: int = 25
        """Published list length, the metric cutoff."""

        min_covered: int = 2
        """Finite within-formula simulator values required for reranking."""

        tie_eps: float = 1e-9
        """Tie grid when multiple simulators contribute."""

        ice_weight: float = 1.0
        """Standardized ICE evidence weight."""

        glacier_weight: float = 1.0
        """Standardized GLACIER evidence weight."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._legacy = config.legacy.make()
        self._control = config.control.make()
        self._final = config.final.make()
        self._fallback = None if config.fallback is None else config.fallback.make()

    def legacy(
        self,
        first: Sequence[str],
        second: Sequence[str],
        second_keys: Sequence[str | None],
        *,
        key: Callable[[str], str | None],
    ) -> tuple[list[str], list[str], list[float]]:
        """Apply the configured legacy vote policy to complete ranked inputs.

        Args:
          first: Complete main ranking.
          second: Complete second-engine ranking.
          second_keys: Original raw keys aligned with the second ranking.
          key: Canonical identity lookup for the main ranking.

        Returns:
          fused: Ranked candidates, their keys and complete vote values.

        """
        return self._legacy.legacy(first, second, second_keys, key=key)

    def fuse(
        self,
        molecule_id: str,
        original: Sequence[str],
        *,
        options: FusionInputs,
    ) -> tuple[list[str], FusionAudit]:
        """Apply the final policy with its ICE and fragment-gap fallback.

        Args:
          molecule_id: Original query identity.
          original: Complete original candidate ordering.
          options: Every source candidate channel and evidence table.

        Returns:
          fused: Complete selected ranking and its admission audit.

        """
        return self._fuse(
            molecule_id,
            original,
            options=options,
            vote=self._final,
            fallback=self._fallback,
        )

    def control(
        self,
        molecule_id: str,
        original: Sequence[str],
        *,
        options: FusionInputs,
    ) -> tuple[list[str], FusionAudit]:
        """Apply the independent control policy, which has no fallback.

        Args:
          molecule_id: Original query identity.
          original: Complete original candidate ordering.
          options: Every source candidate channel and evidence table.

        Returns:
          fused: Complete control ranking and its admission audit.

        """
        return self._fuse(
            molecule_id,
            original,
            options=options,
            vote=self._control,
            fallback=None,
        )

    def _fuse(
        self,
        molecule_id: str,
        original: Sequence[str],
        *,
        options: FusionInputs,
        vote: RankFusion,
        fallback: SimulatorFallback | None,
    ) -> tuple[list[str], FusionAudit]:
        config = self.config
        return final_fusion(
            molecule_id,
            original,
            **options,  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            vote=vote,
            fallback=fallback,
            length=config.length,
            min_covered=config.min_covered,
            tie_eps=config.tie_eps,
            ice_weight=config.ice_weight,
            glacier_weight=config.glacier_weight,
        )


def final_fusion(
    molecule_id: str,
    original: Sequence[str],
    *,
    prefusion: Sequence[str] | None,
    engine: Sequence[str] | None,
    identity: Callable[[str], MetricIdentity | None],
    key: Callable[[str], str | None],
    formula: Callable[[str], str],
    vote: RankFusion,
    fallback: SimulatorFallback | None,
    length: int,
    min_covered: int,
    tie_eps: float,
    ice_weight: float,
    glacier_weight: float,
    ice: Mapping[str, float | None] | None = None,
    glacier: Mapping[str, float | None] | None = None,
    gap: Mapping[str, float | None] | None = None,
    processed: bool = True,
) -> tuple[list[str], FusionAudit]:
    """Apply the complete final shortlist, simulator, safe-tail and audit policy.

    Args:
      molecule_id: Source molecule identifier retained in the audit.
      original: Metric-safe reference before the final policy overlay.
      prefusion: Same-run first-engine list, or None when absent.
      engine: Same-run second-engine list, or None when absent.
      identity: Strict prepared lookup of the official metric identity.
      key: Strict prepared lookup of the older chemical key.
      formula: Strict prepared lookup of the raw formula or known invalid string.
      vote: Reciprocal-rank vote selecting the fusion shortlist.
      fallback: ICE and fragment-gap fallback; None disables its branches.
      length: Published list length.
      min_covered: Minimum finite simulator coverage within a formula group.
      tie_eps: Tie grid for multiple contributing simulators.
      ice_weight: Standardized ICE evidence weight.
      glacier_weight: Standardized GLACIER evidence weight.
      ice: Optional ICE scores by SMILES or older key.
      glacier: Optional GL scores; all-None maps count as absent.
      gap: Optional fragment-gap fallback scores.
      processed: Whether this molecule entered the source processing loop.

    Returns:
      candidates: Complete metric-safe list after same-formula reranking and fill.
      audit: All nine source fields in their original insertion order.

    """
    original = list(original)
    audit: FusionAudit = {
        "molecule_id": molecule_id,
        "processed": processed,
        "branch": "placeholder",
        "candidate_count": 0,
        "collapsed_source_votes": 0,
        "ice_finite_count": 0,
        "gl_finite_count": 0,
        "fallback_changed": False,
        "fallback_changed_top1": False,
    }
    if not processed:
        return original, audit
    audit["branch"] = "no_engine"
    if prefusion is None or engine is None or len(engine) == 0:
        return original, audit
    candidates, scores, collapsed = vote.metric(
        [
            candidate
            for candidate in prefusion
            if candidate and candidate != PLACEHOLDER
        ],
        engine,
        identity=identity,
    )
    keys = [key(candidate) or candidate for candidate in candidates]
    absent = dict[str, float | None]()
    ice_scores = absent if ice is None else ice
    gl_scores = (
        glacier
        if glacier is not None and any(v is not None for v in glacier.values())
        else absent
    )
    audit.update(
        branch="unrescored",
        candidate_count=len(candidates),
        collapsed_source_votes=collapsed,
        ice_finite_count=_finite_coverage(ice_scores, candidates, keys),
        gl_finite_count=_finite_coverage(gl_scores, candidates, keys),
    )
    before = list(candidates)
    columns: list[Mapping[str, float | None]] = []
    weights: list[float] = []
    if ice_scores and gl_scores:
        columns, weights = [absent, gl_scores], [ice_weight, glacier_weight]
        audit["branch"] = "glacier"
    elif fallback is not None:
        selected = fallback.evidence(
            ice=ice_scores,
            glacier=gl_scores,
            gap=gap,
            ice_weight=ice_weight,
        )
        if selected is not None:
            audit["branch"], columns, weights = selected
    if columns:
        order = rerank_groups(
            candidates,
            keys,
            scores,
            [formula(candidate) for candidate in candidates],
            columns=columns,
            weights=weights,
            top_n=len(candidates),
            min_covered=min_covered,
            tie_eps=tie_eps,
        )
        candidates = [candidates[index] for index in order]
    result = (
        safe_extend(
            candidates[:length],
            candidates[length:] + original,
            identity=identity,
            limit=length,
        )
        if candidates
        else original
    )
    if audit["branch"] == "ice_fallback":
        without_ice = (
            safe_extend(
                before[:length],
                before[length:] + original,
                identity=identity,
                limit=length,
            )
            if before
            else original
        )
        audit["fallback_changed"] = result != without_ice
        audit["fallback_changed_top1"] = result[:1] != without_ice[:1]
    return result, audit


def insert_second(
    candidates: Sequence[str],
    engine: Sequence[str],
    *,
    key: Callable[[str], str | None],
    limit: int,
) -> tuple[list[str], bool]:
    """Insert the leading valid engine key at rank two and retain the source flag.

    Args:
      candidates: Final fused submission tokens, before rank-two insertion.
      engine: Complete second-engine ranking in its original order.
      key: Strict prepared lookup of the older chemical key.
      limit: Maximum completed submission length.

    Returns:
      candidates: Submission with the leading eligible proposal at rank two.
      inserted: Whether the source insertion branch ran, even for an equal list.

    """

    def insertion_key(candidate: str) -> str | None:
        return None if not candidate or candidate == PLACEHOLDER else key(candidate)

    tokens = [candidate for candidate in candidates if candidate]
    result = insert_engine_second(tokens, engine, key=insertion_key, limit=limit)
    leading = [
        candidate for candidate in engine if candidate and candidate != PLACEHOLDER
    ]
    if not tokens or tokens[0] == PLACEHOLDER or not leading:
        return result, False
    proposal_key = insertion_key(leading[0])
    inserted = proposal_key is not None and proposal_key != insertion_key(tokens[0])
    return result, inserted


class GapGraphs(Protocol):
    """Supply explicitly prepared complete graphs, retaining known invalid rows."""

    def for_smiles(self, smiles: str) -> GapGraph | None:
        """Return the declared graph; missing strings raise KeyError."""
        ...


class FragmentGap:
    """Cache cleavage levels and score complete raw spectra without chemistry imports."""

    class Config(Fig["FragmentGap"]):
        level_weights: list[float] = field(default_factory=lambda: [1.0, 0.6])
        """Credit for peaks explained by one, two and up to three cleavages; the
        length is the maximum cleavage count."""

        max_bonds: list[int] = field(default_factory=lambda: [100])
        """Bond count above which enumeration stops one cleavage short; entry k
        guards cleavage level k + 2, so the length is one less than the levels."""

        intensity_power: float = 1.0
        """Exponent on peak intensity before normalization; one is linear."""

        mz_power: float = 0.0
        """Additional exponent on mass-to-charge divided by precursor."""

        tol: float = 0.005
        """Inclusive absolute neutral-mass matching tolerance in daltons."""

        h_shifts: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2, 3])
        """Complete ordered hydrogen rearrangements per charge carrier."""

        max_spectra: int = 16
        """Leading original spectra of a molecule that are scored."""

    def __init__(self, config: Config) -> None:
        levels = len(config.level_weights)
        if levels < 1 or levels > 3:
            raise ValueError("level_weights must hold one, two or three weights.")
        if len(config.max_bonds) != levels - 1:
            raise ValueError("max_bonds needs one entry per level beyond the first.")
        self.config = config
        self._levels: dict[
            tuple[str | int, ...],
            tuple[NDArray[np.float64], ...] | None,
        ] = {}

    def score(
        self,
        spectra: Sequence[GapSpectrum],
        candidates: Sequence[str],
        *,
        graphs: GapGraphs,
    ) -> list[float | None]:
        """Return complete candidate mean scores, preserving explicit invalid inputs.

        Args:
          spectra: Complete original peaks and per-spectrum adduct metadata.
          candidates: Ordered candidate strings, including duplicates and invalid rows.
          graphs: Explicit prepared graphs; an unstaged string raises KeyError.

        Returns:
          scores: Mean credited intensity per candidate, or None for invalid graphs.

        """
        config = self.config
        prepared = [
            value
            for spectrum in spectra
            if (value := _prepare(spectrum, config=config)) is not None
        ]
        weights = (1.0, *config.level_weights)
        scores: list[float | None] = []
        for smiles in candidates:
            key = (smiles, *config.max_bonds)
            if key not in self._levels:
                self._levels[key] = fragment_levels(
                    graphs.for_smiles(smiles),
                    max_bonds=config.max_bonds,
                )
            levels = self._levels[key]
            scores.append(
                None
                if levels is None
                else float(
                    np.mean(
                        [
                            _credited(
                                levels,
                                weights=weights,
                                prep=prep,
                                tol=config.tol,
                            )
                            for prep in prepared
                        ],
                    ),
                )
                if prepared
                else 0.0,
            )
        return scores


def fragment_levels(
    graph: GapGraph | None,
    *,
    max_bonds: Sequence[int],
) -> tuple[NDArray[np.float64], ...] | None:
    """Return complete neutral masses grouped by their minimum cleavage count.

    Args:
      graph: Prepared atom weights, ordered bonds and intact component count.
      max_bonds: Bond-count ceiling of each cleavage level beyond the first; the
        length plus one is the number of output cleavage levels after the intact
        level.

    Returns:
      levels: Every sorted unrounded mass, deduplicated at five decimal places,
        or None for an explicitly invalid input graph.

    """
    if graph is None:
        return None
    count = len(graph.weights)
    edges = from_plain(cast(object, graph.bonds.tolist()), list[tuple[int, int]])
    intact = np.array([graph.weights.sum()])
    roots = _parents(count, edges=edges, removed=())
    groups: dict[int, list[int]] = {}
    for atom in range(count):
        groups.setdefault(_root(roots, atom=atom), []).append(atom)
    intact = np.unique(
        np.concatenate(
            [intact, np.array([graph.weights[row].sum() for row in groups.values()])],
        ),
    )
    max_breaks = len(max_bonds) + 1
    if len(edges) == 0:
        return (intact, *(np.zeros(0) for _ in range(max_breaks)))
    depth = max_breaks
    for level in range(max_breaks, 1, -1):
        if depth >= level and len(edges) > max_bonds[level - 2]:
            depth = level - 1
    masses = list(intact)
    cuts = [0] * len(intact)
    atom_weights = from_plain(cast(object, graph.weights.tolist()), list[float])
    for removed in _removals(len(edges), depth=depth):
        parents = _parents(count, edges=edges, removed=removed)
        if sum(parent == atom for atom, parent in enumerate(parents)) < 2:
            continue
        totals = [0.0] * count
        for atom, weight in enumerate(atom_weights):
            totals[_root(parents, atom=atom)] += weight
        for atom, parent in enumerate(parents):
            if parent == atom:
                masses.append(totals[atom])
                cuts.append(len(removed))
    values = np.asarray(masses)
    depths = np.asarray(cuts)
    keys = np.round(values, 5)
    order = np.lexsort((depths, keys))
    keys, values, depths = keys[order], values[order], depths[order]
    first = np.ones(len(keys), dtype=np.bool_)
    first[1:] = keys[1:] != keys[:-1]
    return tuple(
        np.sort(values[first & (depths == depth)]) for depth in range(max_breaks + 1)
    )


def _parents(
    count: int,
    *,
    edges: list[tuple[int, int]],
    removed: tuple[int, ...],
) -> list[int]:
    parents = list(range(count))
    for index, (left, right) in enumerate(edges):
        if index not in removed:
            root_left = _root(parents, atom=left)
            root_right = _root(parents, atom=right)
            if root_left != root_right:
                parents[root_left] = root_right
    return parents


def _root(parents: list[int], *, atom: int) -> int:
    while parents[atom] != atom:
        following = parents[parents[atom]]
        parents[atom] = following
        atom = int(following)  # pyrefly: ignore[unnecessary-type-conversion] -- ty or basedpyright types this value more loosely than pyrefly; the conversion narrows it for them.
    return atom


def _removals(count: int, *, depth: int) -> Iterator[tuple[int, ...]]:
    for first in range(count):
        yield (first,)
        if depth >= 2:
            for second in range(first + 1, count):
                yield first, second
                if depth >= 3:
                    for third in range(second + 1, count):
                        yield first, second, third


class GapGraphTable:
    """Read every prepared graph, preserving invalid rows and first duplicates."""

    class Config(Fig["GapGraphTable"]):
        base_dir: Path | str | None = None
        """Parent-supplied dataset root."""

        working_dir: Path | str = "native-chemistry/fragment-gap"
        """Complete independently prepared fallback graphs."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        root = Path(config.working_dir)
        header = json_loads(
            (root / "manifest.json").read_text(),
            PreparationHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(f"Gap graphs require format 1 and RDKit {IDENTITY_RDKIT}.")
        rows = from_plain(
            parquet.read_table(root / "graphs.parquet").to_pylist(),
            list[GapGraphRow],
        )
        self.graphs = tuple(_graph(row) for row in rows)
        self._positions: dict[str, int] = {}
        for index, row in enumerate(rows):
            self._positions.setdefault(row["smiles"], index)

    def row(self, index: int) -> GapGraph | None:
        """Return the complete nonnegative row, including explicit invalidity."""
        if index < 0 or index >= len(self.graphs):
            raise IndexError(index)
        return self.graphs[index]

    def for_smiles(self, smiles: str) -> GapGraph | None:
        """Require an explicitly prepared input and return its first original row."""
        return self.graphs[self._positions[smiles]]


def _graph(row: GapGraphRow) -> GapGraph | None:
    if not row["valid"]:
        return None
    weights = np.asarray(row["weights"], dtype=np.float64)
    bonds = np.asarray(row["bonds"], dtype=np.int64).reshape(-1, 2)
    if (
        len(weights) == 0
        or row["pieces"] < 1
        or row["pieces"] > len(weights)
        or np.any(bonds < 0)
        or np.any(bonds >= len(weights))
    ):
        raise ValueError("Inconsistent prepared gap graph.")
    weights.setflags(write=False)
    bonds.setflags(write=False)
    return GapGraph(weights=weights, bonds=bonds, pieces=row["pieces"])


def _column_score(
    column: ScoreColumn,
    candidates: Sequence[str],
    keys: Sequence[str] | None,
    index: int,
) -> float | None:
    if isinstance(column, Mapping):
        value = column.get(candidates[index])
        if value is None and keys is not None:
            value = column.get(keys[index])
    else:
        value = column[index] if column is not None and index < len(column) else None
    return finite_score(value)


def _finite_coverage(
    scores: Mapping[str, float | None],
    candidates: Sequence[str],
    keys: Sequence[str],
) -> int:
    count = 0
    for candidate, candidate_key in zip(candidates, keys, strict=True):
        value = scores.get(candidate)
        if value is None:
            value = scores.get(candidate_key)
        count += value is not None and math.isfinite(value)
    return count


def _prepare(
    spectrum: GapSpectrum,
    *,
    config: FragmentGap.Config,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int] | None:
    masses = np.asarray(spectrum["mz"], dtype=np.float64).ravel()
    intensity = np.asarray(spectrum["it"], dtype=np.float64).ravel()
    size = min(len(masses), len(intensity))
    masses, intensity = masses[:size], intensity[:size]
    admitted = np.isfinite(masses) & np.isfinite(intensity) & (intensity > 0)
    masses, intensity = masses[admitted], intensity[admitted]
    if len(masses) == 0:
        return None
    weighted = intensity**config.intensity_power
    if config.mz_power:
        precursor = float(spectrum["prec"] or masses.max())
        weighted = weighted * (masses / precursor) ** config.mz_power
    total = float(weighted.sum())
    if total <= 0 or np.isnan(total):
        return None
    shifts = np.array(
        [
            carrier + hydrogen * HYDROGEN_MASS
            for carrier in _carriers(spectrum["adduct"], mode=spectrum["mode"])
            for hydrogen in config.h_shifts
        ],
    )
    neutral = (masses[None, :] - shifts[:, None]).ravel()
    return weighted / total, neutral, len(shifts), len(masses)


def _carriers(adduct: str | None, *, mode: int | None) -> tuple[float, ...]:
    hydrogen, electron = HYDROGEN_MASS, ELECTRON_MASS
    proton = hydrogen - electron
    values = {
        "[M+H]+": (proton,),
        "[M-H2O+H]+": (proton,),
        "[M-2H2O+H]+": (proton,),
        "[M+NH4]+": (proton, ELEMENT_MASSES["N"] + 4 * hydrogen - electron),
        "[M+Na]+": (proton, ELEMENT_MASSES["Na"] - electron),
        "[M+K]+": (proton, ELEMENT_MASSES["K"] - electron),
        "[M-H]-": (-proton,),
        "[M-H2O-H]-": (-proton,),
        "[M+CH2O2-H]-": (
            -proton,
            ELEMENT_MASSES["C"] + 2 * hydrogen + 2 * ELEMENT_MASSES["O"] - proton,
        ),
        "[M+Cl]-": (-proton, ELEMENT_MASSES["Cl"] + electron),
    }
    if adduct in values:
        return values[adduct]
    if adduct is not None and adduct.endswith("-"):
        return (-proton,)
    if adduct is not None and adduct.endswith("+"):
        return (proton,)
    return (proton,) if (mode or 1) > 0 else (-proton,)


def _credited(
    levels: tuple[NDArray[np.float64], ...],
    *,
    weights: tuple[float, ...],
    prep: tuple[NDArray[np.float64], NDArray[np.float64], int, int],
    tol: float,
) -> float:
    intensity, neutral, shifts, peaks = prep
    best = np.zeros(peaks)
    for masses, weight in zip(levels, weights, strict=True):
        if len(masses) == 0 or weight <= 0:
            continue
        lower = np.searchsorted(masses, neutral - tol, side="left")
        upper = np.searchsorted(masses, neutral + tol, side="right")
        explained = (upper > lower).reshape(shifts, peaks).any(axis=0)
        best = np.where(explained & (weight > best), weight, best)
    return float(intensity @ best)
