"""Native complete first-engine evidence, features and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import io
import itertools
import math
import pickle
import re

from configgle import Fig
from numpy.typing import NDArray
from treekle import from_plain, json_loads

import numpy as np

from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    MainSearch,
    PeakStore,
    clean_main,
    merge_main,
)
from priml.baselines.casmi2026.custom_types import (
    Analog,
    CandidateSet,
    EngineDiagnostics,
    EngineResult,
    FloatArray,
    FragnetRecord,
    GenerationFlags,
    MolecularIdentity,
    PoolIndices,
    PreparationHeader,
    ProposalHeader,
    ProposalRow,
    ReactionProduct,
    ReactionRequest,
)
from priml.baselines.casmi2026.fragnet import FragnetTable, features
from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    FORMULA_ELEMENTS,
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    AnalogRecord,
    AnalogTable,
    ElementGraph,
    IntrinsicTable,
    Reaction,
    adduct_index,
    formula_vector as family_data_formula_vector,
    main_ion_deltas,
    polarity_sign,
    reaction_rules,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.scores import average_ranks, population_z, tanimoto
from priml.data.arrays import load_array
from priml.math import stats
from priml.memory import convert_to_tensor
from priml.model.ae.checkpoint import verify_sha256


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pyarrow import parquet

    from priml.baselines.casmi2026.chemistry import SpectralLibrary
    from priml.baselines.casmi2026.custom_types import (
        ProposalPath,
        SpectrumRequest,
    )
    from priml.baselines.casmi2026.dreams import MainBank, ModelViews
    from priml.baselines.casmi2026.fragnet import StageA
    from priml.baselines.casmi2026.isolated import ElementGraph
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")


FINGERPRINT_BLOCKS: Final = (
    ("ecfp4", 4096),
    ("ecfp6", 4096),
    ("fcfp4", 2048),
    ("rdk6", 2048),
    ("ap", 2048),
    ("tt", 2048),
    ("maccs", 167),
)


def sorted_z(values: FloatArray) -> NDArray[np.float32]:
    """Use sorted float64 population moments before returning float32 scores.

    Args:
      values: Candidate scores; sorting makes the reductions order-invariant.

    Returns:
      scores: Standardized values or zeros below the fixed variance cutoff.

    """
    values = np.asarray(values, np.float64)
    ordered = np.sort(values)
    deviation = ordered.std()
    if deviation > 1e-9:
        return ((values - ordered.mean()) / deviation).astype(np.float32)
    return np.zeros(len(values), np.float32)


def view_features(
    fingerprints: NDArray[np.float32],
    single: NDArray[np.float32],
    merged: NDArray[np.float32],
    modes: NDArray[np.int64],
    selected_bits: NDArray[np.int64],
    *,
    pairwise_rank_limit: int,
) -> dict[str, NDArray[np.float32]]:
    """Calculate the full model-view feature family for main-engine candidates.

    Args:
      fingerprints: Candidate selected binary fingerprints.
      single: A/B/D single-spectrum logits, indexed by model, spectrum and bit.
      merged: Per-polarity merged logits, indexed by model, item and bit.
      modes: Ionization signs of merged items, retaining their encounter order.
      selected_bits: Selected bit positions in the full seven-block fingerprint.
      pairwise_rank_limit: Candidate-count geometry cutoff for quadratic top-three ranks.

    Returns:
      features: Every named model-view feature, including both polarity defaults.

    """
    candidates = fingerprints.astype(np.float64)
    model_scores = [
        _model_score(candidates, single, merged, model)
        for model in range(min(3, len(single)))
    ]
    features: dict[str, NDArray[np.float32]] = {}
    ranks: list[NDArray[np.float64]] = []
    for name, scores in zip(("A", "B", "D"), model_scores, strict=False):
        values = scores.astype(np.float64)
        features[f"fz{name}_gap"] = ((values - values.max()) / 100.0).astype(np.float32)
        rank = average_ranks(values)
        features[f"fz{name}_rank"] = rank.astype(np.float32)
        features[f"fz{name}_z"] = sorted_z(values)
        ranks.append(rank)
    stacked_ranks = np.stack(ranks)
    features["rank_spread"] = (stacked_ranks.max(0) - stacked_ranks.min(0)).astype(
        np.float32,
    )
    features["rank_worst"] = stacked_ranks.max(0).astype(np.float32)
    ensemble = np.mean([scores.astype(np.float64) for scores in model_scores], 0)
    features["fzABD_gap"] = ((ensemble - ensemble.max()) / 100.0).astype(np.float32)
    features["fzABD_rank"] = average_ranks(ensemble).astype(np.float32)
    features.update(_spectrum_consensus(candidates, single, pairwise_rank_limit))
    features.update(_polarity_views(candidates, merged, modes))
    features.update(_block_views(candidates, single, merged, selected_bits))
    return features


type BinaryFeatures = NDArray[np.bool_] | NDArray[np.uint8] | NDArray[np.float32]


def analog_structure_features(
    candidates: Sequence[AnalogRecord],
    *,
    analogs: Sequence[AnalogRecord],
    similarities: NDArray[np.float64],
    analog_sid: NDArray[np.int64],
    parent_sid: NDArray[np.int64],
    fingerprints: BinaryFeatures,
    analog_fingerprints: BinaryFeatures,
    bits: NDArray[np.int64],
) -> dict[str, NDArray[np.float32]]:
    """Compute all36 structural, formula, graph and fingerprint-block relations.

    Args:
      candidates: Complete prepared positional candidate records.
      analogs: Prepared analog records in descending spectral-similarity order.
      similarities: Corresponding double spectral similarities.
      analog_sid: Original training-structure IDs for donor exclusion.
      parent_sid: Candidate donor IDs, with negative values for ordinary candidates.
      fingerprints: Expanded selected candidate fingerprint bits.
      analog_fingerprints: Expanded selected analog fingerprint bits.
      bits: Original fingerprint positions of every selected column.

    Returns:
      values: All36 source relation columns in declared order.

    """
    blocks = ("maccs", "rdk6", "fcfp4", "ap", "tt")
    names = (
        [
            "sc_el_max",
            "sc_el_wsum",
            "sc_gen_max",
            "sc_gen_wsum",
            "sc_el_top",
            "rs_jac_max",
            "rs_jac_wmean",
            "rs_cov_max",
            "rs_big_max",
            "sub_ca_max",
            "sub_ac_max",
            "fd_wmean",
            "fd_top",
            "fd_min",
            "fx_max",
            "fx_top",
            "dr_ring_top",
            "dr_rs_top",
            "dr_core_top",
            "dr_rhet_top",
            "dr_ring_wabs",
        ]
        + ["bt_" + block for block in blocks]
        + [
            "da_ring_top",
            "da_rs_top",
            "da_core_top",
            "da_rhet_top",
            "subf_ca_max",
            "subf_ac_max",
            "st_ap",
            "st_rank",
            "st_gap",
            "st_fgap",
        ]
    )
    count = len(candidates)
    width = min(15, len(analogs))
    output = {name: np.zeros(count) for name in names}
    if count == 0 or width == 0:
        for name in ("fd_wmean", "fd_top", "fd_min"):
            output[name][:] = -1.0
        output["st_rank"][:] = 0.5
        return {name: array.astype(np.float32) for name, array in output.items()}
    analogs = analogs[:width]
    ch = np.stack([row.hashes for row in candidates])
    ah = np.stack([row.hashes for row in analogs])
    cn = np.stack([row.numbers for row in candidates]).astype(np.float64)
    an = np.stack([row.numbers for row in analogs]).astype(np.float64)
    cf = np.stack([row.formula for row in candidates]).astype(np.int64)
    af = np.stack([row.formula for row in analogs]).astype(np.int64)
    use = (parent_sid[:, None] < 0) | np.not_equal(
        parent_sid[:, None],
        analog_sid[None, :width],
    )
    weight = np.where(use, similarities[None, :width] ** 4, 0.0)
    usable = use.any(1)
    top = np.argmax(use, 1)
    rows = np.arange(count)
    equality: list[NDArray[np.bool_]] = []
    for column, name in ((0, "el"), (1, "gen")):
        equal = np.equal(ch[:, column, None], ah[None, :, column]) & np.not_equal(
            ch[:, column, None],
            0,
        )
        equality.append(equal)
        output[f"sc_{name}_max"] = _maximum(equal, use=use, weights=weight)
        output[f"sc_{name}_wsum"] = _mean(equal, weights=weight)
        if column == 0:
            output["sc_el_top"] = np.where(usable, equal[rows, top], False).astype(
                np.float64,
            )
    cr, ar = ch[:, 2:], ah[:, 2:]
    intersection = np.zeros((count, width))
    for column in range(6):
        value = cr[:, column]
        intersection += np.equal(value[:, None, None], ar[None]).any(2) & np.not_equal(
            value[:, None],
            0,
        )
    candidate_count = np.not_equal(cr, 0).sum(1).astype(np.float64)
    analog_count = np.not_equal(ar, 0).sum(1).astype(np.float64)
    union = candidate_count[:, None] + analog_count[None] - intersection
    jaccard = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
    coverage = np.where(
        analog_count[None] > 0,
        intersection / np.maximum(analog_count[None], 1),
        0.0,
    )
    largest = np.equal(cr[:, 0, None], ar[None, :, 0]) & np.not_equal(cr[:, 0, None], 0)
    output.update(
        rs_jac_max=_maximum(jaccard, use=use, weights=weight),
        rs_jac_wmean=_mean(jaccard, weights=weight),
        rs_cov_max=_maximum(coverage, use=use, weights=weight),
        rs_big_max=_maximum(largest, use=use, weights=weight),
    )
    subset = min(5, width)
    core_ca, core_ac, full_ca, full_ac = (np.zeros((count, subset)) for _ in range(4))
    for candidate, molecule in enumerate(candidates):
        for index, analog in enumerate(analogs[:subset]):
            if use.item((candidate, index)):
                core_ca[candidate, index] = contains(molecule.graph, small=analog.core)
                core_ac[candidate, index] = contains(analog.graph, small=molecule.core)
                full_ca[candidate, index] = contains(molecule.graph, small=analog.graph)
                full_ac[candidate, index] = contains(analog.graph, small=molecule.graph)
    sub_weight = np.where(use[:, :subset], similarities[None, :subset], 0.0)
    for name, value in (
        ("sub_ca_max", core_ca),
        ("sub_ac_max", core_ac),
        ("subf_ca_max", full_ca),
        ("subf_ac_max", full_ac),
    ):
        output[name] = (value * sub_weight).max(1)
    candidate_valid, analog_valid = (cf >= 0).all(1), (af >= 0).all(1)
    difference = cf[:, None] - af[None]
    distance = np.where(
        candidate_valid[:, None] & analog_valid[None],
        np.abs(difference).sum(2).astype(np.float64),
        50.0,
    )
    output.update(
        fd_wmean=np.where(usable, _mean(distance, weights=weight), -1.0),
        fd_top=np.where(usable, distance[rows, top], -1.0),
        fd_min=np.where(usable, np.where(use, distance, 1e9).min(1), -1.0),
    )
    reachable = (
        np.isin(delta_codes(difference), _transform_codes())
        & candidate_valid[:, None]
        & analog_valid[None]
        & use
    )
    output["fx_max"] = _maximum(reachable, use=use, weights=weight)
    output["fx_top"] = np.where(usable, reachable[rows, top], 0.0).astype(np.float64)
    nearest = an[top]
    for name, column in (("ring", 0), ("rs", 1), ("core", 2)):
        output[f"dr_{name}_top"] = np.where(
            usable,
            cn[:, column] - nearest[:, column],
            0.0,
        )
    output["dr_rhet_top"] = np.where(
        usable,
        cn[:, 3]
        + cn[:, 4]
        + cn[:, 5]
        - (nearest[:, 3] + nearest[:, 4] + nearest[:, 5]),
        0.0,
    )
    output["dr_ring_wabs"] = _mean(
        np.abs(cn[:, 0, None] - an[None, :, 0]),
        weights=weight,
    )
    for name in ("ring", "rs", "core", "rhet"):
        output[f"da_{name}_top"] = np.abs(output[f"dr_{name}_top"])
    relation = np.zeros((count, width))
    relation += equality[0]
    relation += equality[1]
    relation += jaccard + coverage
    relation[:, :subset] += 0.5 * (core_ca + core_ac) + 0.5 * (full_ca + full_ac)
    relation /= 6.0
    strength = _maximum(relation, use=use, weights=weight)
    output.update(
        st_ap=strength,
        st_rank=average_ranks(strength),
        st_gap=strength - strength.max(),
    )
    _, groups = np.unique(
        np.where(candidate_valid[:, None], cf, -1),
        axis=0,
        return_inverse=True,
    )
    groups = np.asarray(groups).reshape(-1)
    maximum = np.full(int(groups.max()) + 1, -np.inf)
    np.maximum.at(maximum, groups, strength)
    output["st_fgap"] = strength - maximum[groups]
    names = [name for name, _ in FINGERPRINT_BLOCKS]
    block_ids = (
        np.searchsorted(
            np.cumsum([0, *[width for _, width in FINGERPRINT_BLOCKS]]),
            bits,
            side="right",
        )
        - 1
    )
    for name in ("maccs", "rdk6", "fcfp4", "ap", "tt"):
        index = names.index(name)
        columns = np.flatnonzero(np.equal(block_ids, index))
        a, b = (
            fingerprints[:, columns].astype(np.float64),
            analog_fingerprints[:width, columns].astype(np.float64),
        )
        common = a @ b.T
        tanimoto = common / np.maximum(
            a.sum(1)[:, None] + b.sum(1)[None] - common,
            1e-9,
        )
        output["bt_" + name] = _maximum(tanimoto, use=use, weights=weight)
    return {
        name: np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for name, array in output.items()
    }


def delta_codes(differences: NDArray[np.int64]) -> NDArray[np.int64]:
    """Encode exact ten-element signed base-81 deltas with the source sentinel.

    Args:
      differences: Complete final-axis C/H/N/O/P/S/F/Cl/Br/I difference vectors.

    Returns:
      codes: Exact integer codes, or maximum int64 for any component beyond40.

    """
    codes = (
        np.maximum(differences, -40) * np.array([81**power for power in range(10)])
    ).sum(-1)
    return np.where((np.abs(differences) > 40).any(-1), np.iinfo(np.int64).max, codes)


def contains(big: ElementGraph | None, *, small: ElementGraph | None) -> bool:
    """Find an element-preserving injective subgraph embedding in either prepared graph.

    Args:
      big: Complete prepared candidate element graph.
      small: Complete prepared query graph; an absent or empty query never matches.

    Returns:
      matched: Whether every query element and bond has an injective embedding.

    """
    if (
        big is None
        or small is None
        or len(small.elements) == 0
        or len(small.bonds) > len(big.bonds)
    ):
        return False
    big_adj, small_adj = _adjacency(big), _adjacency(small)
    options = [
        [
            j
            for j in range(len(big.elements))
            if small.elements.item(i) == big.elements.item(j)
            and len(small_adj[i]) <= len(big_adj[j])
        ]
        for i in range(len(small.elements))
    ]
    if any(not candidates for candidates in options):
        return False
    order = sorted(
        range(len(options)),
        key=lambda index: (len(options[index]), -len(small_adj[index]), index),
    )
    return _embedding(
        order,
        options=options,
        big=big_adj,
        small=small_adj,
        mapping={},
        used=set(),
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainLibraryEvidence:
    """Retain every source main-engine library channel before feature normalization."""

    maximum: NDArray[np.float64]
    mean: NDArray[np.float64]
    top3: NDArray[np.float64]
    count: NDArray[np.float64]
    cosine: NDArray[np.float64]
    same_adduct: NDArray[np.float64]


def frag2_features(
    spectra: Sequence[SpectrumRequest],
    *,
    search: MainSearch,
    formulas: Sequence[str],
    model_score: NDArray[np.float32],
    depth1: Sequence[NDArray[np.float64]],
    depth2: Sequence[NDArray[np.float64]],
    capabilities: NDArray[np.uint8],
    max_candidates: int,
) -> dict[str, NDArray[np.float32]]:
    """Compute every depth-aware explanation, coverage and diagnostic-loss column.

    Args:
      spectra: Complete family-prepared query spectra.
      search: Query peak preparation shared with the fragment channels.
      formulas: Candidate formulas in their original positional order.
      model_score: Base Full1 fz values used for the tie-inclusive fragment cap.
      depth1: Complete rounded depth-one fragment rows.
      depth2: Complete rounded depth-two fragment rows.
      capabilities: All22 prepared loss-capability counts per candidate.
      max_candidates: Source fragmentation geometry, including every cutoff tie.

    Returns:
      features: Every one of the31 source family outputs.

    """
    names = [
        "disc_max",
        "disc_mean",
        "disc1_mean",
        "idf_frac",
        "idf_rank",
        "uniq_int",
        "top5_expl",
        "top5_unexpl",
        "grp_disc",
        "grp_miss",
        "grp_rank",
        "grp_gap",
        "d1_int",
        "mix_disc",
        "mix_rank",
        "mix_grp_gap",
        "llr1",
        "cov1",
        "covw1",
        "cov_rank",
        "cov_grp_gap",
        "excl_gain",
        "excl_loss",
        "mix_vs_lead",
        "cov_vs_lead",
        "llr_vs_lead",
        "dl_sup_spec",
        "dl_unsup_spec",
        "dl_sup_gen",
        "dl_unsup_gen",
        "dl_sup_gap",
    ]
    count = len(model_score)
    output = {name: np.zeros(count, np.float64) for name in names}
    peaks = _family_peaks(spectra, search=search) if count else []
    if not peaks:
        return {name: values.astype(np.float32) for name, values in output.items()}
    selected = np.arange(count)
    if count > max_candidates:
        cutoff = np.sort(model_score)[::-1].item(max_candidates - 1)
        selected = np.flatnonzero(model_score >= cutoff)
    masses = np.concatenate([peak[0] for peak in peaks])
    weights = np.sqrt(np.concatenate([peak[1] for peak in peaks]))
    spec = np.asarray(
        np.repeat(np.arange(len(peaks)), [len(peak[0]) for peak in peaks]),
        np.int64,
    )
    deltas = [main_ion_deltas(peak[2])[:4] for peak in peaks]
    shifts2 = _fragment_shifts(
        [depth2[selected.item(i)] for i in range(len(selected))],
        masses=masses,
        spec=spec,
        deltas=deltas,
    )
    shifts1 = _fragment_shifts(
        [depth1[selected.item(i)] for i in range(len(selected))],
        masses=masses,
        spec=spec,
        deltas=deltas,
    )
    explained = (shifts2 <= 2).astype(np.float64)
    strict = (shifts2 <= 1).astype(np.float64)
    first = (shifts1 <= 2).astype(np.float64)
    mixed = np.maximum(first, 0.5 * explained)
    size = len(selected)
    counts = explained.sum(0)
    idf = np.log((size + 1.0) / (counts + 1.0)) / math.log(size + 1.0)
    unique = ((counts > 0) & (counts <= max(1, math.floor(0.1 * size)))).astype(
        np.float64,
    )
    rate1 = (first.sum(0) + 0.5) / (size + 1.0)
    baseline = np.log((1.0 - 0.9) / (1.0 - rate1))
    positive = np.log(0.9 / rate1) - baseline
    total = np.array([np.sqrt(peak[1]).sum() for peak in peaks])
    top = np.zeros(len(masses))
    top_count = np.zeros(len(peaks))
    top_weight = np.zeros(len(peaks))
    for index in range(len(peaks)):
        positions = np.flatnonzero(np.equal(spec, index))
        chosen = positions[np.argsort(-weights[positions], kind="mergesort")[:5]]
        top[chosen] = 1.0
        top_count[index], top_weight[index] = len(chosen), weights[chosen].sum()
    accum = _peak_accumulate(
        explained,
        np.stack(
            [
                weights,
                weights * (1.0 - counts / size),
                weights * idf,
                weights * unique,
                weights * top,
                top,
            ],
            1,
        ),
        spec,
        len(peaks),
    )
    accum_strict = _peak_accumulate(
        strict,
        (weights * (1.0 - strict.sum(0) / size))[:, None],
        spec,
        len(peaks),
    )
    accum_first = _peak_accumulate(
        first,
        np.stack([weights, weights * positive], 1),
        spec,
        len(peaks),
    )
    accum_mix = _peak_accumulate(
        mixed,
        (weights * (1.0 - mixed.sum(0) / size))[:, None],
        spec,
        len(peaks),
    )
    background = np.bincount(spec, weights=weights * baseline, minlength=len(peaks))
    idf_total = np.bincount(spec, weights=weights * idf, minlength=len(peaks))
    group_explained = np.zeros((size, len(peaks)))
    group_missed = np.zeros_like(group_explained)
    subset_formulas = [formulas[selected.item(i)] for i in range(len(selected))]
    for formula in sorted(set(subset_formulas)):
        rows = np.array(
            [i for i, value in enumerate(subset_formulas) if value == formula],
        )
        rates = explained[rows].sum(0) / len(rows)
        group_explained[rows] = _peak_accumulate(
            explained[rows],
            (weights * (1.0 - rates))[:, None],
            spec,
            len(peaks),
        )[:, :, 0]
        group_missed[rows] = _peak_accumulate(
            1.0 - explained[rows],
            (weights * rates)[:, None],
            spec,
            len(peaks),
        )[:, :, 0]
    coverage, coverage_weight = _cleavage_coverage(
        [depth1[selected.item(i)] for i in range(len(selected))],
        peaks=peaks,
        deltas=deltas,
    )
    reductions = {
        "disc_max": (accum[:, :, 1] / total).max(1),
        "disc_mean": _spectrum_mean(accum[:, :, 1] / total),
        "disc1_mean": _spectrum_mean(accum_strict[:, :, 0] / total),
        "idf_frac": _spectrum_mean(
            np.where(
                idf_total[None] > 1e-12,
                accum[:, :, 2] / np.maximum(idf_total, 1e-12)[None],
                0.0,
            ),
        ),
        "uniq_int": (accum[:, :, 3] / total).max(1),
        "top5_expl": _spectrum_mean(accum[:, :, 5] / top_count),
        "top5_unexpl": ((top_weight - accum[:, :, 4]) / total).max(1),
        "grp_disc": _spectrum_mean(group_explained / total),
        "grp_miss": _spectrum_mean(group_missed / total),
        "d1_int": _spectrum_mean(accum_first[:, :, 0] / total),
        "mix_disc": _spectrum_mean(accum_mix[:, :, 0] / total),
        "llr1": _spectrum_mean((background + accum_first[:, :, 1]) / total),
        "cov1": _spectrum_mean(coverage),
        "covw1": _spectrum_mean(coverage_weight),
    }
    leader = np.flatnonzero(
        np.equal(model_score[selected], model_score[selected].max()),
    )
    profile = mixed[leader].mean(0) if len(leader) > 1 else mixed[leader[0]].copy()
    reductions["excl_gain"] = _spectrum_mean(
        _peak_accumulate(
            np.maximum(mixed - profile, 0.0),
            weights[:, None],
            spec,
            len(peaks),
        )[:, :, 0]
        / total,
    )
    reductions["excl_loss"] = _spectrum_mean(
        _peak_accumulate(
            np.maximum(profile - mixed, 0.0),
            weights[:, None],
            spec,
            len(peaks),
        )[:, :, 0]
        / total,
    )
    for name, values in reductions.items():
        output[name][selected] = values
    for destination, source in (
        ("mix_vs_lead", "mix_disc"),
        ("cov_vs_lead", "covw1"),
        ("llr_vs_lead", "llr1"),
    ):
        values = output[source][selected]
        output[destination][selected] = values - float(np.sort(values[leader]).mean())
    if size < count:
        others = np.setdiff1d(np.arange(count), selected)
        output["top5_unexpl"][others] = (top_weight / total).max()
        output["llr1"][others] = _spectrum_mean((background / total)[None]).item(0)
    for rank, gap, source in (
        ("idf_rank", "grp_gap", "idf_frac"),
        ("mix_rank", "mix_grp_gap", "mix_disc"),
        ("cov_rank", "cov_grp_gap", "covw1"),
    ):
        output[rank] = average_ranks(output[source])
        for formula in sorted(set(formulas)):
            group = np.array(
                [i for i, value in enumerate(formulas) if value == formula],
            )
            output[gap][group] = output[source][group] - output[source][group].max()
            if source == "idf_frac":
                output["grp_rank"][group] = average_ranks(output[source][group])
    loss_mass, loss_specific, support = diagnostic_losses(capabilities)
    precursor = np.array([peak[3] for peak in peaks])[spec]
    loss = precursor - masses
    eligible = (
        np.abs(loss[:, None] - loss_mass[None])
        <= np.maximum(0.005, precursor * 10.0 * 1e-6)[:, None]
    ) & (loss[:, None] > 10.0)
    diagnostic = np.zeros((count, len(peaks), 4))
    hits = np.flatnonzero(eligible.any(1))
    for position in range(len(hits)):
        peak = hits.item(position)
        for flag, offset in ((True, 0), (False, 2)):
            matches = np.flatnonzero(
                np.logical_and(
                    eligible.take(peak, axis=0),
                    np.equal(loss_specific, flag),
                ),
            )
            if len(matches):
                present = support[:, matches].any(1)
                diagnostic[:, spec.item(peak), offset] += weights.item(peak) * present
                diagnostic[:, spec.item(peak), offset + 1] += (
                    weights.item(peak) * ~present
                )
    for index, name in enumerate(
        ("dl_sup_spec", "dl_unsup_spec", "dl_sup_gen", "dl_unsup_gen"),
    ):
        output[name] = (diagnostic[:, :, index] / total).max(1)
    supported = output["dl_sup_spec"] + output["dl_sup_gen"]
    output["dl_sup_gap"] = supported - supported.max()
    return {
        name: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for name, value in output.items()
    }


def diagnostic_losses(
    capabilities: NDArray[np.uint8],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Bind all27 individual and14 paired diagnostic losses to prepared counts.

    Args:
      capabilities: Every22-column substructure count vector in source order.

    Returns:
      masses: Complete ordered individual and paired loss masses.
      specific: Specific-versus-generic flags for every loss.
      support: Candidate-by-loss capability predicates, retaining multiplicities.

    """
    entries = (
        (18.010565, (0, 1)),
        (27.994915, (4, 2)),
        (43.989829, (1, 5)),
        (46.005479, (1,)),
        (15.023475, (8, 9)),
        (32.026215, (7,)),
        (17.026549, (3,)),
        (31.042199, (9, 3)),
        (45.057849, (9,)),
        (59.073499, (9,)),
        (42.010565, (6,)),
        (60.021129, (6, 14, 10, 12)),
        (90.031694, (14,)),
        (120.042259, (14,)),
        (162.052824, (10,)),
        (146.057909, (11,)),
        (132.042259, (12,)),
        (176.032088, (13,)),
        (86.000394, (15,)),
        (146.036779, (16,)),
        (162.031694, (17,)),
        (176.047344, (18,)),
        (152.010959, (19,)),
        (79.956815, (20,)),
        (97.967380, (20,)),
        (79.966331, (21,)),
        (97.976896, (21,)),
    )
    pairs = (
        (14, 14),
        (14, 15),
        (14, 16),
        (14, 17),
        (15, 15),
        (15, 16),
        (16, 16),
        (17, 17),
        (14, 18),
        (14, 19),
        (14, 20),
        (14, 21),
        (14, 22),
        (14, 10),
    )
    masses = np.array(
        [entry[0] for entry in entries]
        + [entries[left][0] + entries[right][0] for left, right in pairs],
    )
    specific = np.arange(len(masses)) >= 6
    support = np.zeros((len(capabilities), len(masses)), np.bool_)
    for index, (_, required) in enumerate(entries):
        support[:, index] = capabilities[:, required].sum(1) > 0
    for index, (left, right) in enumerate(pairs, start=len(entries)):
        a, b = entries[left][1], entries[right][1]
        ca = capabilities[:, a].sum(1)
        support[:, index] = (
            ca >= 2 if a == b else (ca > 0) & (capabilities[:, b].sum(1) > 0)
        )
    return masses, specific, support


def main_features(
    rows: CandidateRows,
    *,
    target: float,
    spectra: Sequence[SpectrumRequest],
    library: MainLibraryEvidence,
    analogs: Sequence[Analog],
    train_fingerprints: NDArray[np.uint8],
    views: ModelViews | None,
    fragments: NDArray[np.float32] | None,
    fragment_counts: NDArray[np.float32],
    subformulas: NDArray[np.float32] | None,
    similarity_power: float,
) -> NDArray[np.float32]:
    """Construct all94 source main-engine columns in their released positional order.

    Args:
      rows: Complete positional pool/generated candidate data.
      target: Query neutral mass.
      spectra: Source-capped complete query spectra.
      library: Complete library evidence aligned to candidates.
      analogs: Strict-best source analogs in descending encounter order.
      train_fingerprints: Packed selected-bit training fingerprints.
      views: Full1 predictions for both single and merged spectrum views.
      fragments: All five fragmentation channels, or None when disabled.
      fragment_counts: Complete fragment counts after the source candidate cap.
      subformulas: All four subformula channels, or None when disabled.
      similarity_power: Analog propagation exponent.

    Returns:
      matrix: Every source column, retaining duplicate candidates and empty groups.

    """
    count = len(rows.mass)
    names = main_feature_names()
    result = np.zeros((count, len(names)), np.float32)
    if not count:
        return result
    values = _main_library_features(library)
    values.update(
        _main_analog_features(
            rows.fingerprints,
            analogs,
            train_fingerprints,
            similarity_power,
        ),
    )
    model, scores = _main_model_features(rows, views=views)
    values.update(model)
    formula_counts = {
        formula: rows.candidates.formulas.count(formula)
        for formula in rows.candidates.formulas
    }
    values["n_formula"] = float(len(formula_counts))
    values["formula_share"] = np.array(
        [formula_counts[formula] / count for formula in rows.candidates.formulas],
        np.float32,
    )
    if fragments is not None:
        for index, name in enumerate(
            ("fr_int", "fr_int_mean", "fr_cnt", "fr_top10", "fr_strict"),
        ):
            values[name] = fragments[:, index]
        values.update(
            fr_rank=_stable_ranks(fragments[:, 0]),
            fr_gap=fragments[:, 0] - fragments[:, 0].max(),
            fr_z=population_z(fragments[:, 0]),
            fr_n=np.log1p(fragment_counts.astype(np.float64)).astype(np.float32),
        )
    if subformulas is not None:
        values.update(
            sf_int=subformulas[:, 0],
            sf_cnt=subformulas[:, 2],
            sf_top10=subformulas[:, 3],
            sf_rank=_stable_ranks(subformulas[:, 0]),
            sf_gap=subformulas[:, 0] - subformulas[:, 0].max(),
            formula_is_top=(subformulas[:, 0] >= subformulas[:, 0].max() - 1e-6).astype(
                np.float32,
            ),
        )
    ppm = (rows.mass - target) / target * 1e6
    ranks = _stable_ranks(scores)
    maximum = library.maximum.max().item()
    analog = np.asarray(values["ap"], np.float32)
    positives = sum(
        item["mode"] == "positive"
        if isinstance(item["mode"], str)
        else item["mode"] > 0
        for item in spectra
    )
    values.update(
        mass_ppm=ppm,
        mass_ppm_rank=_stable_ranks(-np.abs(ppm)),
        n_heavy=rows.heavy_atoms.astype(np.float64),
        fp_bits=rows.fingerprints.sum(1) / 100.0,
        n_cand=math.log(count),
        n_query=float(len(spectra)),
        n_pos=float(positives),
        n_neg=float(len(spectra) - positives),
        target_mass=target / 1000.0,
        q_npeaks=math.log1p(float(np.mean([len(item["mz"]) for item in spectra])))
        if spectra
        else 0.0,
        lib_x_fz=library.maximum * (1.0 - ranks),
        ap_x_fz=analog * (1.0 - ranks),
        agree_lib=1.0 - ranks.item(int(np.argmax(library.maximum)))
        if maximum > 0
        else 0.0,
        agree_ap=1.0 - ranks.item(int(np.argmax(analog))) if analog.max() > 0 else 0.0,
        fr_x_fz=(
            fragments[:, 0] if fragments is not None else np.zeros(count, np.float32)
        )
        * (1.0 - ranks),
        is_gen=rows.candidates.generated.astype(np.float32),
        gen_parent_sim=rows.parent_similarity,
        gen_steps=rows.steps,
        gen_parent_tan=rows.parent_tanimoto,
        n_gen=float(rows.candidates.generated.sum()),
    )
    generated_rank = np.zeros(count, np.float32)
    generated_rank[rows.candidates.generated] = _stable_ranks(
        scores[rows.candidates.generated],
    )
    values["gen_rank_fz"] = generated_rank
    for name, value in values.items():
        result[:, names.index(name)] = value
    return result


def main_feature_names() -> tuple[str, ...]:
    """Return every source Full1 engine feature in released column order.

    Returns:
      names: All94 feature names, including the unused source cross-encoder columns.

    """
    return (
        "lib_max",
        "lib_mean",
        "lib_top3",
        "lib_nspec",
        "lib_cos",
        "lib_max_rank",
        "lib_gap",
        "lib_grp_max",
        "lib_hit",
        "lib_same_adduct",
        "ap",
        "a1",
        "best_tan",
        "top_tan",
        "mean_tan",
        "ap_rank",
        "ap_gap",
        "ap_grp_max",
        "top_sim",
        "ap_np",
        "best_tan_np",
        "top_sim_np",
        "ap_np_gap",
        "ap_sum",
        "n_analog_hi",
        "tan_zero_shift",
        "fz",
        "fz_z",
        "fz_rank",
        "fz_gap",
        "fz_norm",
        "fz_top",
        "fz_softmax",
        "fz_grp_ent",
        "fz_grp_gap",
        "fz_single",
        "fz_single_gap",
        "fz_merged",
        "fz_merged_gap",
        "fp_cos",
        "fp_tan",
        "fp_ll",
        "fp_ll_gap",
        "el_c",
        "el_n",
        "el_o",
        "el_sum",
        "el_rank",
        "fr_int",
        "fr_int_mean",
        "fr_cnt",
        "fr_top10",
        "fr_strict",
        "fr_rank",
        "fr_gap",
        "fr_z",
        "fr_n",
        "sf_int",
        "sf_cnt",
        "sf_top10",
        "sf_rank",
        "sf_gap",
        "formula_share",
        "n_formula",
        "formula_is_top",
        "mass_ppm",
        "mass_ppm_rank",
        "n_heavy",
        "fp_bits",
        "n_cand",
        "n_query",
        "n_pos",
        "n_neg",
        "target_mass",
        "q_npeaks",
        "lib_x_fz",
        "ap_x_fz",
        "agree_lib",
        "agree_ap",
        "fr_x_fz",
        "is_gen",
        "gen_parent_sim",
        "gen_steps",
        "gen_parent_tan",
        "n_gen",
        "gen_rank_fz",
        "xs_max",
        "xs_mean",
        "xs_merged",
        "xs_gap",
        "xs_rank",
        "xs_z",
        "xs_top",
        "xs_grp_gap",
    )


def formula_vector(formula: str) -> NDArray[np.int64] | None:
    """Parse the source ten-element formula vocabulary, retaining charge stripping.

    Args:
      formula: Complete molecular formula string.

    Returns:
      counts: C/H/N/O/P/S/F/Cl/Br/I counts, or None for a known unsupported formula.

    """
    elements = FORMULA_ELEMENTS
    text = formula.replace("+", "").replace("-", "")
    counts = np.zeros(10, np.int64)
    consumed = 0
    for token in re.finditer(r"([A-Z][a-z]?)(\d*)", text):
        symbol, number = token.group(1, 2)
        if token.start() != consumed or symbol not in elements:
            return None
        counts[elements.index(symbol)] += int(number) if number else 1
        consumed = token.end()
    return counts if formula and consumed == len(text) else None


def main_fragment_features(
    masses: NDArray[np.float64],
    *,
    peaks: Sequence[tuple[NDArray[np.float64], NDArray[np.float64], str]],
) -> NDArray[np.float32]:
    """Aggregate all five original main-engine fragment explanation channels.

    Args:
      masses: Complete sorted neutral fragments for one candidate.
      peaks: Complete prepared masses, base-peak intensities and query adducts.

    Returns:
      features: Maximum/mean intensity, count, top10 and strict explained fractions.

    """
    return _main_explanation_features(
        masses,
        peaks=peaks,
        hshift=2,
        tolerance=0.01,
        ppm=15.0,
    )


def main_subformula_features(
    formula: str,
    *,
    peaks: Sequence[tuple[NDArray[np.float64], NDArray[np.float64], str]],
    max_out: int,
) -> NDArray[np.float32]:
    """Enumerate every source valence-bounded subformula and all four score channels.

    Args:
      formula: Complete candidate formula in the source ten-element vocabulary.
      peaks: Every prepared spectrum and adduct.
      max_out: Source enumeration safety cap, retaining reached partial-result order.

    Returns:
      features: Maximum/mean explained intensity, maximum count and top10 fractions.

    """
    counts = formula_vector(formula)
    if counts is None or not peaks:
        return np.zeros(4, np.float32)
    masses = main_subformula_masses(counts, max_out=max_out)
    return _main_explanation_features(
        masses,
        peaks=peaks,
        hshift=0,
        tolerance=0.005,
        ppm=10.0,
    )[:4]


def main_subformula_masses(
    counts: NDArray[np.int64],
    *,
    max_out: int,
) -> NDArray[np.float64]:
    """Enumerate the complete bounded ten-element subformula mass sequence.

    Args:
      counts: C/H/N/O/P/S/F/Cl/Br/I counts.
      max_out: Maximum emitted masses before the source unsorted early return.

    Returns:
      masses: Sorted complete masses, or source-order partial masses at the cap.

    """
    cmax, hmax, nmax, omax, pmax, smax, fmax, clmax, brmax, imax = map(int, counts)
    m_c, m_h, m_n, m_o, m_p, m_s, m_f, m_cl, m_br, m_i = (
        ELEMENT_MASSES[element] for element in FORMULA_ELEMENTS
    )
    output: list[float] = []
    for c, n, o, p, s, hal in itertools.product(
        range(cmax + 1),
        range(nmax + 1),
        range(omax + 1),
        range(pmax + 1),
        range(smax + 1),
        range(fmax + clmax + brmax + imax + 1),
    ):
        base = c * m_c + n * m_n + o * m_o + p * m_p + s * m_s
        for f in range(min(fmax, hal) + 1):
            for cl in range(min(clmax, hal - f) + 1):
                for br in range(min(brmax, hal - f - cl) + 1):
                    iodine = hal - f - cl - br
                    if iodine > imax:
                        continue
                    for h in range(min(hmax, 2 * c + n + 2 + p * 2) + 1):
                        if (
                            c + n + o + p + s + hal + h == 0
                            or 2 * c - (h + hal) + n + p + 2 < -2
                        ):
                            continue
                        if len(output) >= max_out:
                            return np.asarray(output, np.float64)
                        output.append(
                            base
                            + h * m_h
                            + f * m_f
                            + cl * m_cl
                            + br * m_br
                            + iodine * m_i,
                        )
    return np.sort(np.asarray(output, np.float64))


def _main_explanation_features(
    masses: NDArray[np.float64],
    *,
    peaks: Sequence[tuple[NDArray[np.float64], NDArray[np.float64], str]],
    hshift: int,
    tolerance: float,
    ppm: float,
) -> NDArray[np.float32]:
    fractions: list[float] = []
    counts: list[float] = []
    tops: list[float] = []
    strict: list[float] = []
    if len(masses):
        for mass, intensity, adduct in peaks:
            weights = np.asarray(np.sqrt(intensity), np.float64)
            total = weights.sum(dtype=np.float64).item()
            if not len(mass) or total <= 0:
                continue
            matched = np.zeros(len(mass), np.bool_)
            matched_strict = np.zeros_like(matched)
            for delta in main_ion_deltas(adduct):
                matched |= _main_matched_peaks(
                    masses,
                    mass=mass,
                    delta=delta,
                    hshift=hshift,
                    tolerance=tolerance,
                    ppm=ppm,
                )
                matched_strict |= _main_matched_peaks(
                    masses,
                    mass=mass,
                    delta=delta,
                    hshift=1,
                    tolerance=tolerance,
                    ppm=ppm,
                )
            fractions.append(weights[matched].sum().item() / total)
            counts.append(matched.mean(dtype=np.float64).item())
            tops.append(
                matched[np.argsort(-weights, kind="stable")[:10]]
                .mean(dtype=np.float64)
                .item(),
            )
            strict.append(weights[matched_strict].sum().item() / total)
    if not fractions:
        return np.zeros(5, np.float32)
    return np.asarray(
        [max(fractions), np.mean(fractions), max(counts), max(tops), max(strict)],
        np.float32,
    )


def _main_matched_peaks(
    fragments: NDArray[np.float64],
    *,
    mass: NDArray[np.float64],
    delta: float,
    hshift: int,
    tolerance: float,
    ppm: float,
) -> NDArray[np.bool_]:
    matched = np.zeros(len(mass), np.bool_)
    limits = np.maximum(tolerance, mass * ppm * 1e-6)
    for shift in range(-hshift, hshift + 1):
        target = mass - delta - shift * HYDROGEN_MASS
        insertion = np.searchsorted(fragments, target)
        for offset in (0, -1):
            indices = insertion + offset
            valid = (indices >= 0) & (indices < len(fragments))
            matched[valid] |= (
                np.abs(fragments[indices[valid]] - target[valid]) <= limits[valid]
            )
    return matched


class PlainDataUnpickler(pickle.Unpickler):
    """Allow the released primitive containers and reject object reconstruction."""

    @override
    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"Model assets require plain data: {module}.{name}",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TreeBank:
    """Released named feature schema and its complete ordered booster bank."""

    features: tuple[str, ...]
    models: tuple[LightGBMModel, ...]

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> Self:
        """Read a SHA-pinned plain-data envelope without importing a booster runtime.

        Args:
          path: Released ranker pickle containing feature names and model text.
          expected_sha256: Complete public asset digest.

        Returns:
          bank: Complete checked booster bank and feature schema.

        """
        raw = checked_bytes(path, expected_sha256)
        data = cast(object, PlainDataUnpickler(io.BytesIO(raw)).load())
        envelope = from_plain(data, dict[str, object])
        features = tuple(from_plain(envelope.get("features"), list[str]))
        models = tuple(
            LightGBMModel.parse(text)
            for text in from_plain(envelope.get("boosters"), list[str])
        )
        if not models or any(model.features != features for model in models):
            raise ValueError(
                "Every booster must declare the ranker's complete feature order.",
            )
        return cls(features=features, models=models)

    def score(
        self,
        values: NDArray[np.float32],
        names: Sequence[str],
    ) -> NDArray[np.float64]:
        """Select exact source feature names and average the complete booster outputs.

        Args:
          values: Complete float32 feature-family output matrix.
          names: Corresponding feature names in matrix column order.

        Returns:
          scores: Ordered double predictions averaged across the booster bank.

        """
        positions = {name: index for index, name in enumerate(names)}
        missing = [name for name in self.features if name not in positions]
        if missing:
            raise ValueError(f"Ranker features are missing: {missing}")
        selected = np.ascontiguousarray(
            values[:, [positions[name] for name in self.features]],
            dtype=np.float32,
        )
        numeric = selected.astype(np.float64)
        return np.mean([model.predict(numeric) for model in self.models], axis=0)


def load_tree(path: Path, expected_sha256: str) -> LightGBMModel:
    """Read a complete SHA-pinned StageB text model without an external runtime."""
    return LightGBMModel.parse(checked_bytes(path, expected_sha256).decode())


def checked_bytes(path: Path, expected_sha256: str) -> bytes:
    """Read a plain-data asset after requiring and checking its pinned digest."""
    if not expected_sha256:
        raise ValueError(f"A pinned asset requires its SHA256: {path}")
    verify_sha256(path, expected_sha256)
    return path.read_bytes()


type Combo = tuple[str, ...]
type Support = tuple[int, float, Combo]


POOL_COLUMNS: Final = (
    "n_par",
    "best_sim",
    "sum_sim4",
    "min_steps",
    "par_rank",
    "tf_logfr",
    "site_logfr",
    "sib_frac",
    "prior_miss",
    "best_sim_nz",
    "best_sim_zd",
    "zd_only",
)
GEN_COLUMNS: Final = (
    "n_par",
    "sum_sim4",
    "min_steps",
    "par_rank",
    "n_combo",
    "tf_logfr",
    "site_logfr",
    "sib_rank",
    "n_sib",
    "sib_frac",
    "prior_miss",
    "zd_only",
)


class ProposalStore(Protocol):
    """Require complete ordered paths and explicitly prepared identities."""

    def paths(self, parent_smiles: str, combo: Combo) -> tuple[ProposalPath, ...]:
        """Return complete staged terminal paths or raise for a missing pair."""
        ...

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        """Return declared nullable key/skeleton identities or raise if unstaged."""
        ...


class PriorTables(TypedDict):
    """Retain all numerical transform and radius-backoff counts."""

    glob: float
    tf: dict[str, tuple[float, float]]
    env: dict[int, dict[tuple[str, int], tuple[float, float]]]
    meta: NotRequired[object]


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorFeatures:
    """Keep the complete strict-best sibling record for one product."""

    tf_logfr: float
    site_logfr: float
    sibling_rank: int
    siblings: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivationLog:
    """Retain one rejected-product support without losing multiplicity."""

    parent: int
    similarity: float
    combo: Combo | None
    kind: Literal["pool", "gen"]
    how: Literal["smi", "key"]
    identity: str | None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class DerivationContext:
    """Bind every candidate position to its complete derivation evidence."""

    smiles: tuple[str, ...]
    keys: tuple[str | None, ...]
    formulas: tuple[str, ...]
    generated: GenerationFlags
    parent_sid: PoolIndices
    similarities: FloatArray
    combos: tuple[Combo | None, ...]
    analog_sid: PoolIndices
    model_score: FloatArray
    parent_smiles: tuple[str, ...]
    logs: tuple[DerivationLog, ...]


class DerivationPriors:
    """Apply complete source transform and site backoff to prepared proposals."""

    def __init__(self, tables: PriorTables, *, proposals: ProposalStore) -> None:
        self.tables = tables
        self.proposals = proposals
        self.probabilities = {
            name: (success + 2.0 * tables["glob"]) / (count + 2.0)
            for name, (success, count) in tables["tf"].items()
        }
        self._groups: dict[tuple[str, Combo], dict[str, PriorFeatures]] = {}

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        proposals: ProposalStore,
    ) -> DerivationPriors:
        """Load the complete pinned plain-data prior envelope.

        Args:
          path: Released transform/environment count pickle.
          expected_sha256: Complete public asset digest checked before decoding.
          proposals: Complete independent prepared paths and molecular identities.

        Returns:
          priors: Every count table and metadata entry with its numerical backoff.

        """
        raw = checked_bytes(path, expected_sha256)
        tables = from_plain(
            cast(object, PlainDataUnpickler(io.BytesIO(raw)).load()),
            PriorTables,
        )
        return cls(tables, proposals=proposals)

    def p_transform(self, name: str) -> float:
        """Return the smoothed transform rate or the complete global backoff."""
        return self.probabilities.get(name, self.tables["glob"])

    def p_site(self, name: str, environments: tuple[int, int, int]) -> float:
        """Back off at the first missing or under-supported radius.

        Args:
          name: Transform whose found-rate supplies the initial prior.
          environments: Complete radius-one/two/three site environment IDs.

        Returns:
          probability: The last supported hierarchical smoothed found-rate.

        """
        probability = self.p_transform(name)
        for radius in (1, 2, 3):
            counts = self.tables["env"][radius].get((name, environments[radius - 1]))
            if counts is None or counts[1] < 20:
                break
            probability = (counts[0] + 2.0 * probability) / (counts[1] + 2.0)
        return probability

    def group(self, parent_smiles: str, combo: Combo) -> dict[str, PriorFeatures]:
        """Return strict-best site priors and every sibling's rank/count.

        Args:
          parent_smiles: Original parent string used as a preparation lookup key.
          combo: Every transformation in its reached encounter order.

        Returns:
          group: All valid standardized products in first-encounter order.

        """
        key = parent_smiles, combo
        if key in self._groups:
            return self._groups[key]
        best: dict[str, tuple[float, float]] = {}
        for path in self.proposals.paths(parent_smiles, combo):
            product = path["standardized"]
            if path["raw"] is None or product is None:
                continue
            site = transform = 0.0
            for name, environment in zip(combo, path["environments"], strict=True):
                site += math.log(self.p_site(name, environment))
                transform += math.log(self.p_transform(name))
            if product not in best or site > best[product][0]:
                best[product] = site, transform
        values = np.array([value[0] for value in best.values()])
        result = {
            product: PriorFeatures(
                tf_logfr=transform,
                site_logfr=site,
                sibling_rank=1 + int((values > site + 1e-12).sum()),
                siblings=len(values),
            )
            for product, (site, transform) in best.items()
        }
        if len(self._groups) > 5_000:
            self._groups.clear()
        self._groups[key] = result
        return result


def combo_mass(combo: Combo) -> float:
    """Accumulate the exact ordered elemental difference of any supported combo.

    Args:
      combo: Complete reached transform names, including the v1-only fallback.

    Returns:
      mass: Exact source ordered elemental sum, with cancelled entries retained.

    """
    rules = {rule.name: rule for rule in v6_reaction_rules()}
    rules.update(
        (rule.name, rule) for rule in reaction_rules() if rule.name == "dehexose2"
    )
    atoms: dict[str, int] = {}
    for name in combo:
        for element, value in rules[name].delta:
            atoms[element] = atoms.get(element, 0) + value
    return Reaction(name="", smarts="", delta=tuple(atoms.items())).mass


def supports(
    context: DerivationContext,
    *,
    proposals: ProposalStore,
) -> list[list[Support]]:
    """Resolve every own/log support while preserving candidate duplicates.

    Args:
      context: Complete positional candidate and derivation-log evidence.
      proposals: Explicit staged molecular identities for rejected products.

    Returns:
      evidence: Every support in original encounter order, including duplicates.

    """
    maps: dict[tuple[bool, str], dict[str | None, list[int]]] = {
        (generated, kind): {} for generated in (False, True) for kind in ("smi", "key")
    }
    evidence: list[list[Support]] = [[] for _ in context.smiles]
    for position, (smiles, key, combo) in enumerate(
        zip(context.smiles, context.keys, context.combos, strict=True),
    ):
        generated = context.generated.item(position)
        maps[generated, "smi"].setdefault(smiles, []).append(position)
        maps[generated, "key"].setdefault(key, []).append(position)
        if generated and context.parent_sid.item(position) >= 0 and combo is not None:
            evidence[position].append(
                (
                    context.parent_sid.item(position),
                    context.similarities.item(position),
                    combo,
                ),
            )
    for log in context.logs:
        if log.combo is None:
            continue
        selected = maps[log.kind == "gen", log.how].get(log.identity)
        if selected is None and log.how == "smi" and log.kind == "gen":
            if log.identity is None:
                raise ValueError("Expected log.identity is not None.")
            key = proposals.identity(log.identity)[0]
            selected = (
                maps[False, "key"].get(key, []) + maps[True, "key"].get(key, [])
                if key is not None
                else None
            )
        for position in selected or ():
            evidence[position].append((log.parent, log.similarity, log.combo))
    return evidence


def aggregate(
    context: DerivationContext,
    *,
    priors: DerivationPriors,
    evidence: list[list[Support]],
) -> dict[str, NDArray[np.float64]]:
    """Calculate all fifteen support/prior aggregates with source sentinels.

    Args:
      context: Complete candidate identities, parents and analog encounter order.
      priors: Every transform/site table and explicit proposal group.
      evidence: Complete positional support lists, including repeated entries.

    Returns:
      values: Every double-precision support aggregate in declared source order.

    """
    output = {
        name: np.full(len(context.smiles), value) for name, value in _defaults().items()
    }
    ranks: dict[int, int] = {}
    for index in range(len(context.analog_sid)):
        ranks.setdefault(context.analog_sid.item(index), index)
    for position, items in enumerate(evidence):
        if not items:
            continue
        unique = sorted({(parent, combo) for parent, _, combo in items})
        similarities: dict[int, float] = {}
        zero = nonzero = 0.0
        all_zero = True
        for parent, similarity, combo in items:
            similarities[parent] = max(similarities.get(parent, 0.0), similarity)
            if abs(combo_mass(combo)) < 0.01:
                zero = max(zero, similarity)
            else:
                nonzero = max(nonzero, similarity)
                all_zero = False
        values = {
            "n_par": len(similarities),
            "best_sim": max(similarities.values()),
            "sum_sim4": math.fsum(
                similarity**4 for similarity in sorted(similarities.values())
            ),
            "min_steps": min(len(combo) for _, combo in unique),
            "par_rank": min(ranks.get(parent, 10) for parent in similarities),
            "n_combo": len({combo for _, combo in unique}),
            "best_sim_nz": nonzero,
            "best_sim_zd": zero,
            "zd_only": float(all_zero),
        }
        best: tuple[float, float, float, float] | None = None
        transform = -25.0
        for parent, combo in unique:
            transform = max(
                transform,
                sum(math.log(priors.p_transform(name)) for name in combo),
            )
            row = priors.group(context.parent_smiles[parent], combo).get(
                context.smiles[position],
            )
            if row is not None:
                value = _prior_tuple(row)
                if best is None or value > best:
                    best = value
        key = context.keys[position]
        if best is None and key is not None:
            skeleton = priors.proposals.identity(context.smiles[position])[1]
            if skeleton is not None:
                for parent, combo in unique:
                    for product, row in priors.group(
                        context.parent_smiles[parent],
                        combo,
                    ).items():
                        product_key, product_skeleton = priors.proposals.identity(
                            product,
                        )
                        if product_skeleton == skeleton and product_key == key:
                            value = _prior_tuple(row)
                            if best is None or value > best:
                                best = value
        values["tf_logfr"] = transform
        if best is None:
            values.update(site_logfr=transform, prior_miss=1.0)
        else:
            sibling_rank, siblings = -best[2], -best[3]
            values.update(
                site_logfr=best[0],
                sib_rank=sibling_rank,
                n_sib=siblings,
                sib_frac=(sibling_rank - 1.0) / max(siblings - 1.0, 1.0),
            )
        for name, value in values.items():
            output[name][position] = value
    return output


def derivation_features(
    context: DerivationContext,
    *,
    priors: DerivationPriors,
) -> dict[str, NDArray[np.float32]]:
    """Build all41 pool/generated/joint/sibling family columns in source order.

    Args:
      context: Complete positional candidates and every supporting derivation.
      priors: Complete numerical tables and explicitly staged chemistry rows.

    Returns:
      values: All41 finite float32 columns in the pinned family's declared order.

    """
    evidence = supports(context, proposals=priors.proposals)
    values = aggregate(context, priors=priors, evidence=evidence)
    defaults = _defaults()
    pool = ~context.generated
    derived = values["n_par"] > 0
    hit = derived & pool
    generated = derived & context.generated
    score = context.model_score.astype(np.float64)
    output = {"dp_hit": hit.astype(np.float64)}
    output.update(
        ("dp_" + name, np.where(pool, values[name], defaults[name]))
        for name in POOL_COLUMNS
    )
    output.update(
        dp_rank_sim=_subset_rank(values["best_sim"], pool),
        dp_rank_fz_hit=_subset_rank(score, hit),
    )
    count = len(context.smiles)
    iso_count, gap, nonzero_gap, formula_rank = (
        np.zeros(count),
        np.zeros(count),
        np.zeros(count),
        np.full(count, -1.0),
    )
    groups: dict[str, list[int]] = {}
    for position, formula in enumerate(context.formulas):
        groups.setdefault(formula, []).append(position)
    for positions in groups.values():
        rows = np.array(positions)
        iso_count[rows] = math.log1p(int(hit[rows].sum()))
        ordinary = rows[pool[rows]]
        if len(ordinary):
            for name, target in (("best_sim", gap), ("best_sim_nz", nonzero_gap)):
                scoped = np.where(pool, values[name], defaults[name])[ordinary]
                target[ordinary] = scoped - scoped.max()
        selected = rows[derived[rows]]
        if len(selected):
            formula_rank[selected] = average_ranks(values["site_logfr"][selected])
    output.update(
        dp_iso_n_hit=iso_count,
        dp_iso_sim_gap=gap,
        dp_iso_nz_gap=nonzero_gap,
        dp_q_n_hit=np.full(count, math.log1p(int(hit.sum()))),
    )
    output.update(
        ("dg_" + name, np.where(context.generated, values[name], defaults[name]))
        for name in GEN_COLUMNS
    )
    output.update(
        dg_rank_site=_subset_rank(values["site_logfr"], generated),
        dg_rank_sup=_subset_rank(values["sum_sim4"], generated),
        dj_rank_site_der=_subset_rank(values["site_logfr"], derived),
        dj_rank_sim=average_ranks(values["best_sim"]),
        dj_rank_fz_der=_subset_rank(score, derived),
    )
    siblings: dict[tuple[int, Combo], list[int]] = {}
    for position in np.flatnonzero(derived):
        for support in {(parent, combo) for parent, _, combo in evidence[position]}:
            siblings.setdefault(support, []).append(int(position))
    sibling_count, score_rank, score_gap, site_rank = (
        np.zeros(count),
        np.full(count, 2.0),
        np.full(count, -10.0),
        np.full(count, 2.0),
    )
    for positions in siblings.values():
        rows = np.array(sorted(set(positions)))
        sibling_count[rows] = np.maximum(sibling_count[rows], math.log1p(len(rows)))
        score_rank[rows] = np.minimum(score_rank[rows], average_ranks(score[rows]))
        score_gap[rows] = np.maximum(
            score_gap[rows],
            np.clip((score[rows] - score[rows].max()) / 100.0, -10.0, 0.0),
        )
        site_rank[rows] = np.minimum(
            site_rank[rows],
            average_ranks(values["site_logfr"][rows]),
        )
    score_rank[~derived] = site_rank[~derived] = -1.0
    output.update(
        ds_n=sibling_count,
        ds_fz_rank=score_rank,
        ds_fz_gap=score_gap,
        ds_site_rank=site_rank,
        ds_fm_site_rank=formula_rank,
    )
    return {
        name: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for name, value in output.items()
    }


class DerivationTable:
    """Require explicit prepared pairs and molecular identities."""

    def __init__(self, directory: Path) -> None:
        header = json_loads(
            (directory / "manifest.json").read_text(),
            PreparationHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Derivation data requires format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.requests = tuple(
            from_plain(
                parquet.read_table(directory / "requests.parquet").to_pylist(),
                list[ProposalHeader],
            ),
        )
        grouped: dict[int, list[ProposalRow]] = {}
        for row in from_plain(
            parquet.read_table(directory / "paths.parquet").to_pylist(),
            list[ProposalRow],
        ):
            grouped.setdefault(row["request"], []).append(row)
        self._paths: dict[tuple[str, tuple[str, ...]], tuple[ProposalPath, ...]] = {}
        for index, request in enumerate(self.requests):
            paths = tuple(grouped.pop(index, ()))
            if len(paths) != request["terminal_count"] or [
                row["position"] for row in paths
            ] != list(range(len(paths))):
                raise ValueError(
                    "Derivation terminal path layout differs from its headers.",
                )
            self._paths.setdefault((request["parent_smiles"], request["combo"]), paths)
        if grouped:
            raise ValueError(
                "Derivation terminal path layout differs from its headers.",
            )
        self._identities: dict[str, tuple[str | None, str | None]] = {}
        for identity in from_plain(
            parquet.read_table(directory / "identities.parquet").to_pylist(),
            list[MolecularIdentity],
        ):
            self._identities.setdefault(
                identity["smiles"],
                (identity["key"], identity["skeleton"]),
            )

    def paths(
        self,
        parent_smiles: str,
        combo: tuple[str, ...],
    ) -> tuple[ProposalPath, ...]:
        """Return every terminal path, including known-invalid products."""
        return self._paths[(parent_smiles, combo)]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        """Return declared identities; unstaged strings raise KeyError."""
        return self._identities[smiles]


type FragmentRows = tuple[NDArray[np.float64], ...]
type CapabilityMatrix = NDArray[np.uint8]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainFamilyRows:
    """Retain every required prepared row in candidate or analog encounter order."""

    candidate_structures: tuple[AnalogRecord, ...]
    analog_structures: tuple[AnalogRecord, ...]
    train_smiles: tuple[str, ...]
    depth1: FragmentRows
    depth2: FragmentRows
    capabilities: CapabilityMatrix
    fragnet: tuple[FragnetRecord | None, ...]
    priors: DerivationPriors


class MainFamilyProvider(Protocol):
    """Require explicit positional joins and raise for every missing prepared row."""

    def select(
        self,
        rows: CandidateRows,
        analogs: Sequence[Analog],
    ) -> MainFamilyRows:
        """Resolve all candidate and analog positions without SMILES deduplication."""
        ...


class MainEvidence(Protocol):
    """Read complete base evidence without depending on engine orchestration."""

    @property
    def rows(self) -> CandidateRows:
        """Return every positional candidate and generation field."""
        ...

    @property
    def features(self) -> NDArray[np.float32]:
        """Return every base94 source column."""
        ...

    @property
    def analogs(self) -> tuple[Analog, ...]:
        """Return complete strict-best analogs in source order."""
        ...


def family_features(
    base: MainEvidence,
    spectra: Sequence[SpectrumRequest],
    *,
    provider: MainFamilyProvider,
    views_bank: MainBank,
    train_fingerprints: NDArray[np.uint8],
    bits: NDArray[np.int64],
    stage_a: StageA,
    stage_b: LightGBMModel,
    logs: tuple[DerivationLog, ...] = (),
    search: MainSearch,
    query_floor: float,
    query_max_peaks: int,
    query_over: float,
    max_candidates: int,
    pairwise_rank_limit: int,
) -> tuple[dict[str, NDArray[np.float32]], ModelViews]:
    """Compute every output of all five pinned families from complete prepared rows.

    Args:
      base: Complete ordered main94 evidence and admitted generated products.
      spectra: Capped raw queries, retaining every conditioning field.
      provider: Strict positional prepared chemistry and fragment joins.
      views_bank: Complete independent A/B/D neural bank.
      train_fingerprints: Packed fingerprints in full training SID order.
      bits: Original positions of every selected fingerprint column.
      stage_a: Complete Fragnet categorical backoff bank.
      stage_b: Complete Fragnet31-column booster.
      logs: Every generation rejection/support in original encounter order.
      search: Query peak preparation of the depth-aware fragment family.
      query_floor: Relative intensity cutoff of the first-engine query cleaning.
      query_max_peaks: Intensity-ranked peak cap of that cleaning.
      query_over: Mass above the precursor that cleaning still admits.
      max_candidates: Top-f.z cap of that family, including every tie.
      pairwise_rank_limit: Candidate count above which quadratic spectrum ranks stop.

    Returns:
      values: All outputs in family order, sorted within each family.
      views: Both prediction heads with all model/spectrum axes retained.

    """
    prepared: list[SpectrumRequest] = []
    for spectrum in spectra:
        mass, intensity = clean_main(
            (
                np.asarray(spectrum["mz"], np.float64),
                np.asarray(spectrum["it"], np.float64),
            ),
            precursor=spectrum["prec"],
            floor=query_floor,
            max_peaks=query_max_peaks,
            over=query_over,
        )
        query = spectrum.copy()
        query["mz"] = [mass.item(i) for i in range(len(mass))]
        query["it"] = [intensity.item(i) for i in range(len(intensity))]
        prepared.append(query)
    views = views_bank.views(prepared)
    rows, analogs = base.rows, base.analogs
    inputs = provider.select(rows, analogs)
    fz = base.features[:, main_feature_names().index("fz")]
    analog_sid = np.array([analog.sid for analog in analogs], np.int64)
    analog_fp = np.unpackbits(train_fingerprints[analog_sid], axis=1)[:, : len(bits)]
    context = DerivationContext(
        smiles=rows.candidates.smiles,
        keys=rows.candidates.keys,
        formulas=rows.candidates.formulas,
        generated=rows.candidates.generated,
        parent_sid=rows.parent_sid,
        similarities=rows.parent_similarity,
        combos=rows.combos,
        analog_sid=analog_sid,
        model_score=fz * 100.0,
        parent_smiles=inputs.train_smiles,
        logs=logs,
    )
    families = (
        ("derivation", derivation_features(context, priors=inputs.priors)),
        (
            "frag2",
            frag2_features(
                prepared,
                search=search,
                formulas=rows.candidates.formulas,
                model_score=fz,
                depth1=inputs.depth1,
                depth2=inputs.depth2,
                capabilities=inputs.capabilities,
                max_candidates=max_candidates,
            ),
        ),
        (
            "analog_struct",
            analog_structure_features(
                inputs.candidate_structures,
                analogs=inputs.analog_structures,
                similarities=np.array(
                    [analog.similarity for analog in analogs],
                    np.float32,
                ).astype(np.float64),
                analog_sid=analog_sid,
                parent_sid=rows.parent_sid,
                fingerprints=rows.fingerprints,
                analog_fingerprints=analog_fp,
                bits=bits,
            ),
        ),
        (
            "model_views",
            view_features(
                rows.fingerprints,
                views.single,
                views.merged,
                views.modes.astype(np.int64),
                bits,
                pairwise_rank_limit=pairwise_rank_limit,
            ),
        ),
        (
            "fragnet",
            features(
                inputs.fragnet,
                prepared,
                formulas=rows.candidates.formulas,
                model_score=fz,
                stage_a=stage_a,
                stage_b=stage_b,
            ),
        ),
    )
    return {
        family + "__" + name: columns[name]
        for family, columns in families
        for name in sorted(columns)
    }, views


def select_features(
    values: dict[str, NDArray[np.float32]],
    *,
    names: Sequence[str],
    count: int,
) -> NDArray[np.float32]:
    """Select every released column in exact order and reject missing evidence.

    Args:
      values: Complete base and family output vectors keyed by public names.
      names: Exact ordered schema declared by the released ranker.
      count: Complete positional candidate count, including duplicate structures.

    Returns:
      matrix: Contiguous float32 candidate rows in the requested feature order.

    """
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Ranker features missing: {missing}")
    columns = [values[name] for name in names]
    if any(column.ndim != 1 for column in columns):
        raise ValueError("Every feature must be one-dimensional.")
    if any(len(column) != count for column in columns):
        raise ValueError("Feature vectors differ from the candidate positions.")
    return (
        np.ascontiguousarray(np.stack(columns, 1))
        if columns
        else np.zeros((count, 0), np.float32)
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class GeneratedCandidate:
    """Carry one admitted product's complete prepared evidence in source order."""

    smiles: str
    key: str
    formula: str
    mass: float
    heavy_atoms: int
    fingerprint: NDArray[np.float32]
    fragments: NDArray[np.float64]
    parent_similarity: float
    steps: int
    parent_tanimoto: float
    parent_sid: int = -1
    combo: tuple[str, ...] | None = None
    raw: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CandidateRows:
    """Bind positional identities to all first-engine numerical candidate inputs."""

    candidates: CandidateSet
    mass: NDArray[np.float64]
    heavy_atoms: NDArray[np.int32]
    train_sid: NDArray[np.int64]
    fingerprints: NDArray[np.float32]
    fragments: tuple[NDArray[np.float64] | None, ...]
    parent_similarity: NDArray[np.float32]
    steps: NDArray[np.float32]
    parent_tanimoto: NDArray[np.float32]
    parent_sid: NDArray[np.int64]
    combos: tuple[tuple[str, ...] | None, ...]
    raw: tuple[str | None, ...]


class PolarityMerge:
    """Merge the query spectra of one polarity into a single peak list."""

    class Config(Fig["PolarityMerge"]):
        tolerance: float = 0.005
        """Strict mass distance below which merged peaks count as duplicates."""

    def __init__(self, config: Config) -> None:
        self.tolerance = config.tolerance

    def __call__(
        self,
        group: Sequence[tuple[FloatArray, FloatArray]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return the merged masses and intensities of normalized spectra."""
        return merge_main(group, tolerance=self.tolerance, max_peaks=None)


class MainRetrieval:
    """Select library-aware representatives and strict-best mass-shifted analogs."""

    class Config(Fig["MainRetrieval"]):
        search: MainSearch.Config = field(default_factory=MainSearch.Config)
        """Complete first-engine matching and preparation arithmetic."""

        analog_window: float = 200.0
        """Inclusive neutral-mass half-window in daltons."""

        n_analog: int = 100
        """Final number of distinct analog structures."""

        merge: PolarityMerge.Config | None = field(default_factory=PolarityMerge.Config)
        """Merge each polarity before analog search; None searches every spectrum."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.search = config.search.make()
        self.merge = None if config.merge is None else config.merge.make()

    def representatives(self, library: SpectralLibrary) -> NDArray[np.int64]:
        """Select one richest spectrum per structure, polarity and source library.

        Args:
          library: Complete prepared first-engine spectral library.

        Returns:
          rows: Representatives in stable increasing labelled neutral-mass order.

        """
        positive = np.greater(library.mode, 0)
        ranked = np.lexsort(
            (-library.peak_count, library.library, positive, library.sid),
        )
        groups = np.column_stack([library.sid, positive, library.library])
        first = np.ones(len(ranked), dtype=np.bool_)
        first[1:] = np.any(
            np.not_equal(groups[ranked[1:]], groups[ranked[:-1]]),
            axis=1,
        )
        selected = ranked[first]
        return selected[np.argsort(library.mass[selected], stable=True)]

    def analogs(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        library: SpectralLibrary,
        target: float,
        excluded: NDArray[np.bool_] | None = None,
        excluded_sid: int = -1,
        excluded_library: int = -1,
    ) -> tuple[Analog, ...]:
        """Keep strict-best SID evidence among each query's top-scoring rows.

        Args:
          spectra: Complete query spectra before the caller's query-count cap.
          library: Prepared first-engine spectral library.
          target: Target neutral mass.
          excluded: Withheld spectrum flags, applied after representative selection.
          excluded_sid: Optional withheld training structure.
          excluded_library: Restrict structure withholding to one library if nonnegative.

        Returns:
          analogs: Descending similarity, retaining first-encounter score ties.

        """
        representatives = self.representatives(library)
        masses = library.mass[representatives]
        begin = int(np.searchsorted(masses, target - self.config.analog_window))
        end = int(
            np.searchsorted(masses, target + self.config.analog_window, side="right"),
        )
        rows = representatives[begin:end]
        keep = np.ones(len(rows), dtype=np.bool_)
        if excluded_sid >= 0:
            withheld = np.equal(library.sid[rows], excluded_sid)
            if excluded_library >= 0:
                withheld &= np.equal(library.library[rows], excluded_library)
            keep &= ~withheld
        if excluded is not None:
            keep &= ~excluded[rows]
        rows = rows[keep]
        store = PeakStore(
            offsets=library.offsets,
            mz=library.mz,
            intensity=library.intensity,
        )
        best: dict[int, Analog] = {}
        for mass, intensity, sign in self._queries(spectra):
            if not len(self.search.prepare((mass, intensity))[0]):
                continue
            selected = rows[library.mode[rows] == sign]
            if not len(selected):
                continue
            shifts = target - library.mass[selected]
            scores = self.search.score(
                (mass, intensity),
                store=store,
                candidates=selected,
                shifts=shifts,
            )
            positions = np.sort(
                np.argsort(-scores, kind="stable")[: 4 * self.config.n_analog],
            )
            for position in positions:
                row = selected.item(position)
                sid, similarity = library.sid.item(row), scores.item(position)
                if sid not in best or similarity > best[sid].similarity:
                    best[sid] = Analog(
                        sid=sid,
                        similarity=similarity,
                        shift=shifts.item(position),
                        natural_product=library.library.item(row)
                        in (2, 3, 4, 5, 6, 7, 10),
                    )
        return tuple(
            sorted(best.values(), key=lambda item: -item.similarity)[
                : self.config.n_analog
            ],
        )

    def _queries(
        self,
        spectra: Sequence[SpectrumRequest],
    ) -> list[tuple[FloatArray, FloatArray, int]]:
        queries: list[tuple[FloatArray, FloatArray, int]] = [
            (
                np.asarray(item["mz"], dtype=np.float64),
                np.asarray(item["it"], dtype=np.float64),
                polarity_sign(item["mode"]),
            )
            for item in spectra
        ]
        if self.merge is None:
            return queries
        merged: list[tuple[FloatArray, FloatArray, int]] = []
        for sign in (1, -1):
            group = [
                (mass, intensity) for mass, intensity, mode in queries if mode == sign
            ]
            if group:
                mass, intensity = self.merge(group)
                merged.append((mass, intensity, sign))
        return merged


def candidates(
    pool: CandidatePool,
    *,
    target: float,
    generated: Sequence[GeneratedCandidate] = (),
    drop_row: int = -1,
    ppm: float,
    fallback_ppm: float,
) -> CandidateRows:
    """Append admitted products to the source mass-window rows without deduplication.

    Args:
      pool: Complete prepared first-engine candidate pool.
      target: Query neutral mass.
      generated: Admitted products in reaction encounter order.
      drop_row: Optional withheld pool row, removed after fallback selection.
      ppm: Initial mass-window half-width.
      fallback_ppm: Half-width used only when the initial window is empty.

    Returns:
      rows: Complete aligned identities, fingerprints and generation metadata.

    """
    selected = pool.window(target, ppm=ppm)
    if not len(selected):
        selected = pool.window(target, ppm=fallback_ppm)
    if drop_row >= 0:
        selected = selected[selected != drop_row]
    count = len(selected)
    products = len(generated)
    fingerprints = np.concatenate(
        [
            pool.fingerprints(selected)
            .astype(np.float32)
            .reshape(count, len(pool.bits)),
            *[item.fingerprint[None] for item in generated],
        ],
    )
    return CandidateRows(
        candidates=CandidateSet(
            smiles=tuple(pool.smiles[selected.item(i)] for i in range(count))
            + tuple(item.smiles for item in generated),
            keys=tuple(pool.keys[selected.item(i)] for i in range(count))
            + tuple(item.key for item in generated),
            formulas=tuple(pool.formulas[selected.item(i)] for i in range(count))
            + tuple(item.formula for item in generated),
            pool_rows=np.concatenate([selected, np.full(products, -1, dtype=np.int64)]),
            generated=np.concatenate(
                [np.zeros(count, dtype=np.bool_), np.ones(products, dtype=np.bool_)],
            ),
            sources=("pool",) * count + ("generated",) * products,
        ),
        mass=np.concatenate(
            [pool.mass[selected], [item.mass for item in generated]],
        ).astype(np.float64),
        heavy_atoms=np.concatenate(
            [pool.heavy_atoms[selected], [item.heavy_atoms for item in generated]],
        ).astype(np.int32),
        train_sid=np.concatenate(
            [pool.train_sid[selected], np.full(products, -1, dtype=np.int64)],
        ),
        fingerprints=fingerprints,
        fragments=(None,) * count + tuple(item.fragments for item in generated),
        parent_similarity=np.concatenate(
            [np.zeros(count), [item.parent_similarity for item in generated]],
        ).astype(np.float32),
        steps=np.concatenate(
            [np.zeros(count), [item.steps for item in generated]],
        ).astype(np.float32),
        parent_tanimoto=np.concatenate(
            [np.zeros(count), [item.parent_tanimoto for item in generated]],
        ).astype(np.float32),
        parent_sid=np.concatenate(
            [np.full(count, -1, np.int64), [item.parent_sid for item in generated]],
        ).astype(np.int64),
        combos=(None,) * count + tuple(item.combo for item in generated),
        raw=(None,) * count + tuple(item.raw for item in generated),
    )


type PreparedPeaks = tuple[NDArray[np.float64], NDArray[np.float64], str]


class FragmentChannels:
    """Score the five original fragment channels of the top-scoring candidates."""

    class Config(Fig["FragmentChannels"]):
        max_candidates: int = 600
        """Ordinary top-f.z cap, without family tie inclusion."""

    def __init__(self, config: Config) -> None:
        self.max_candidates = config.max_candidates

    def __call__(
        self,
        rows: CandidateRows,
        *,
        pool: CandidatePool,
        peaks: Sequence[PreparedPeaks],
        model_score: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute the channels and fragment counts, leaving unscored rows at zero.

        Args:
          rows: Complete positional candidates.
          pool: Complete main candidate and fragment pool.
          peaks: Prepared query masses, base-peak intensities and adducts.
          model_score: Base Full1 fz values ranking the candidates.

        Returns:
          fragment: Five channels per candidate.
          counts: Fragment count of each scored candidate.

        """
        selected = np.arange(len(rows.mass))
        if len(selected) > self.max_candidates:
            selected = np.argsort(-model_score, kind="stable")[: self.max_candidates]
        fragment = np.zeros((len(rows.mass), 5), np.float32)
        counts = np.zeros(len(rows.mass), np.float32)
        for position in selected:
            masses = rows.fragments[position]
            if masses is None:
                masses = pool.fragments(rows.candidates.pool_rows.item(position))
            fragment[position] = main_fragment_features(masses, peaks=peaks)
            counts[position] = len(masses)
        return fragment, counts


class SubformulaChannels:
    """Score the four formula-only fragment channels of every candidate formula."""

    class Config(Fig["SubformulaChannels"]):
        max_out: int = 2_000_000
        """Enumeration safety cap, retaining the partial result order once reached."""

    def __init__(self, config: Config) -> None:
        self.max_out = config.max_out

    def __call__(
        self,
        rows: CandidateRows,
        *,
        peaks: Sequence[PreparedPeaks],
    ) -> NDArray[np.float32]:
        """Compute the channels of each distinct formula and repeat them per candidate.

        Args:
          rows: Complete positional candidates.
          peaks: Prepared query masses, base-peak intensities and adducts.

        Returns:
          subformulas: Four channels per candidate.

        """
        formulas = {
            formula: main_subformula_features(
                formula,
                peaks=peaks,
                max_out=self.max_out,
            )
            for formula in rows.candidates.formulas
        }
        return np.stack([formulas[formula] for formula in rows.candidates.formulas])


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainBase:
    """Retain the complete Full1 evidence before additional feature families."""

    rows: CandidateRows
    features: NDArray[np.float32]
    analogs: tuple[Analog, ...]
    diagnostics: EngineDiagnostics


class MainEngine:
    """Assemble complete Full1 evidence with the pinned source caps and branches."""

    class Config(Fig["MainEngine"]):
        retrieval: MainRetrieval.Config = field(default_factory=MainRetrieval.Config)
        """Complete representative and spectral matching recipe."""

        max_query_spectra: int = 16
        """Encounter-order query cap before every first-engine channel."""

        ppm: float = 10.0
        """Initial candidate and direct-library neutral-mass half-window."""

        fallback_ppm: float = 30.0
        """Candidate half-window used only when the initial window is empty."""

        similarity_power: float = 4.0
        """Analog propagation exponent."""

        fragment_search: MainSearch.Config = field(
            default_factory=lambda: MainSearch.Config(
                max_peaks=128,
                entropy_weight=False,
            ),
        )
        """Query peak preparation shared by every fragment channel and family."""

        fragments: FragmentChannels.Config | None = field(
            default_factory=FragmentChannels.Config,
        )
        """The five original fragment channels; None skips them."""

        subformulas: SubformulaChannels.Config | None = field(
            default_factory=SubformulaChannels.Config,
        )
        """The four formula-only fragment channels; None skips them."""

        query_floor: float = 0.001
        """Relative intensity cutoff when cleaning queries for the feature families."""

        query_max_peaks: int = 512
        """Intensity-ranked peak cap of that cleaning."""

        query_over: float = 2.0
        """Mass above the precursor that cleaning still admits."""

        family_max_candidates: int = 600
        """Top-f.z cap of the depth-aware fragment family, including every tie."""

        pairwise_rank_limit: int = 3_000
        """Candidate count above which the quadratic top-three spectrum ranks are skipped."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.retrieval = config.retrieval.make()
        self.fragment_search = config.fragment_search.make()
        self.fragments = None if config.fragments is None else config.fragments.make()
        self.subformulas = (
            None if config.subformulas is None else config.subformulas.make()
        )

    def base(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        target: float,
        library: SpectralLibrary,
        pool: CandidatePool,
        train_fingerprints: NDArray[np.uint8],
        bank: MainBank | None = None,
        generated: Sequence[GeneratedCandidate] = (),
        excluded: NDArray[np.bool_] | None = None,
        excluded_sid: int = -1,
        excluded_library: int = -1,
        drop_row: int = -1,
        merged_counts: Sequence[int] = (),
    ) -> MainBase:
        """Compute all94 source columns and complete ordered candidate evidence.

        Args:
          spectra: Complete original query spectra.
          target: Target neutral mass.
          library: Complete prepared labelled spectral cache.
          pool: Complete main candidate and fragment pool.
          train_fingerprints: Packed selected-bit training fingerprints.
          bank: The released Full1 bank, or None for the source disabled branch.
          generated: Admitted reaction products in their source encounter order.
          excluded: Withheld library-spectrum flags.
          excluded_sid: Withheld training structure for analog retrieval.
          excluded_library: Optional library restriction on structure withholding.
          drop_row: Withheld pool candidate removed after fallback selection.
          merged_counts: Source ce_n values for all supplied spectra.

        Returns:
          base: Candidates, every base column, analogs and source diagnostics.

        """
        spectra = spectra[: self.config.max_query_spectra]
        analogs = self.retrieval.analogs(
            spectra,
            library=library,
            target=target,
            excluded=excluded,
            excluded_sid=excluded_sid,
            excluded_library=excluded_library,
        )
        rows = candidates(
            pool,
            target=target,
            generated=generated,
            drop_row=drop_row,
            ppm=self.config.ppm,
            fallback_ppm=self.config.fallback_ppm,
        )
        count = len(rows.mass)
        if not count:
            return MainBase(
                rows=rows,
                features=np.zeros((0, len(main_feature_names())), np.float32),
                analogs=analogs,
                diagnostics=EngineDiagnostics(
                    n_cand=0,
                    lib_max=0.0,
                    top_sim=0.0,
                    fzmax=0.0,
                ),
            )
        evidence = _main_library(
            spectra,
            rows=rows,
            library=library,
            target=target,
            search=self.retrieval.search,
            ppm=self.config.ppm,
            excluded=excluded,
        )
        views = (
            bank.views(spectra, merged_counts=merged_counts[: len(spectra)])
            if bank is not None
            else None
        )
        model_score = (
            (
                rows.fingerprints.astype(np.float64)
                @ (
                    0.5
                    * (
                        views.single.mean(0, dtype=np.float32).mean(0, dtype=np.float32)
                        + views.merged.mean(0, dtype=np.float32).mean(
                            0,
                            dtype=np.float32,
                        )
                    )
                ).astype(np.float64)
            ).astype(np.float32)
            if views is not None
            else np.zeros(count, np.float32)
        )
        fragment, fragment_counts, subformulas = self._fragment_evidence(
            spectra,
            rows=rows,
            pool=pool,
            model_score=model_score,
        )
        matrix = main_features(
            rows,
            target=target,
            spectra=spectra,
            library=evidence,
            analogs=analogs,
            train_fingerprints=train_fingerprints,
            views=views,
            fragments=fragment,
            fragment_counts=fragment_counts,
            subformulas=subformulas,
            similarity_power=self.config.similarity_power,
        )
        return MainBase(
            rows=rows,
            features=matrix,
            analogs=analogs,
            diagnostics=EngineDiagnostics(
                n_cand=count,
                lib_max=evidence.maximum.max().item(),
                top_sim=float(np.float32(analogs[0].similarity)) if analogs else 0.0,
                fzmax=model_score.max().item(),
                n_gen=int(rows.candidates.generated.sum()),
            ),
        )

    def run(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        target: float,
        library: SpectralLibrary,
        pool: CandidatePool,
        train_fingerprints: NDArray[np.uint8],
        bank: MainBank,
        views_bank: MainBank,
        provider: MainFamilyProvider,
        stage_a: StageA,
        stage_b: LightGBMModel,
        rankers: TreeBank,
        generated: Sequence[GeneratedCandidate] = (),
        logs: tuple[DerivationLog, ...] = (),
        excluded: NDArray[np.bool_] | None = None,
        excluded_sid: int = -1,
        excluded_library: int = -1,
        drop_row: int = -1,
        merged_counts: Sequence[int] = (),
    ) -> EngineResult:
        """Compute main94, all five families, exact main160 and the four-model mean.

        Args:
          spectra: Complete raw query spectra and their conditioning fields.
          target: Query neutral mass.
          library: Complete prepared labelled spectral cache.
          pool: Complete main candidate and provided fragment pool.
          train_fingerprints: Packed selected-bit rows in full training SID order.
          bank: Complete released Full1 neural bank.
          views_bank: Complete independent A/B/D neural bank.
          provider: Strict positional prepared family joins.
          stage_a: Complete Fragnet categorical backoff bank.
          stage_b: Complete Fragnet31-column booster.
          rankers: Every main booster and the exact ordered160-column schema.
          generated: Admitted products with complete numerical inputs/provenance.
          logs: All generation rejection/support records in encounter order.
          excluded: Withheld library-spectrum flags.
          excluded_sid: Withheld training structure for analog retrieval.
          excluded_library: Optional library restriction on structure withholding.
          drop_row: Withheld pool candidate removed after fallback selection.
          merged_counts: Explicit source ce_n overrides, or empty to read requests.

        Returns:
          result: Complete positional candidates, selected160 features and mean scores.

        """
        spectra = spectra[: self.config.max_query_spectra]
        base = self.base(
            spectra,
            target=target,
            library=library,
            pool=pool,
            train_fingerprints=train_fingerprints,
            bank=bank,
            generated=generated,
            excluded=excluded,
            excluded_sid=excluded_sid,
            excluded_library=excluded_library,
            drop_row=drop_row,
            merged_counts=merged_counts,
        )
        count = len(base.rows.mass)
        if not count:
            return EngineResult(
                candidates=base.rows.candidates,
                features=np.zeros((0, len(rankers.features)), np.float32),
                feature_names=rankers.features,
                scores=np.zeros(0, np.float64),
                diagnostics=base.diagnostics,
            )
        if merged_counts:
            spectra = [spectrum.copy() for spectrum in spectra]
            for spectrum, value in zip(
                spectra,
                merged_counts[: len(spectra)],
                strict=True,
            ):
                spectrum["ce_n"] = value
        families, _ = family_features(
            base,
            spectra,
            provider=provider,
            views_bank=views_bank,
            train_fingerprints=train_fingerprints,
            bits=pool.bits,
            stage_a=stage_a,
            stage_b=stage_b,
            logs=logs,
            search=self.fragment_search,
            query_floor=self.config.query_floor,
            query_max_peaks=self.config.query_max_peaks,
            query_over=self.config.query_over,
            max_candidates=self.config.family_max_candidates,
            pairwise_rank_limit=self.config.pairwise_rank_limit,
        )
        values = {
            name: base.features[:, i] for i, name in enumerate(main_feature_names())
        } | families
        features = select_features(values, names=rankers.features, count=count)
        return EngineResult(
            candidates=base.rows.candidates,
            features=features,
            feature_names=rankers.features,
            scores=rankers.score(features, rankers.features),
            diagnostics=base.diagnostics,
        )

    def _fragment_evidence(
        self,
        spectra: Sequence[SpectrumRequest],
        *,
        rows: CandidateRows,
        pool: CandidatePool,
        model_score: NDArray[np.float32],
    ) -> tuple[
        NDArray[np.float32] | None,
        NDArray[np.float32],
        NDArray[np.float32] | None,
    ]:
        peaks: list[PreparedPeaks] = []
        for spectrum in spectra:
            mass, values = self.fragment_search.prepare(
                (
                    np.asarray(spectrum["mz"], np.float64),
                    np.asarray(spectrum["it"], np.float64),
                ),
            )
            if len(mass):
                peaks.append((mass, values / values.max(), spectrum["adduct"]))
        fragment_counts = np.zeros(len(rows.mass), np.float32)
        fragment = subformulas = None
        if self.fragments is not None and peaks:
            fragment, fragment_counts = self.fragments(
                rows,
                pool=pool,
                peaks=peaks,
                model_score=model_score,
            )
        if self.subformulas is not None and peaks:
            subformulas = self.subformulas(rows, peaks=peaks)
        return fragment, fragment_counts, subformulas


class MainFragmentTable:
    """Require an explicit prepared pair for every generated molecular string."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        header = json_loads(
            (directory / "manifest.json").read_text(),
            PreparationHeader,
        )
        if header["format_version"] != 1 or header["rdkit"] != IDENTITY_RDKIT:
            raise ValueError(
                f"Main fragments require format 1 and RDKit {IDENTITY_RDKIT}.",
            )
        self.smiles = tuple(
            from_plain(
                parquet.read_table(directory / "molecules.parquet", columns=["smiles"])
                .column("smiles")
                .to_pylist(),
                list[str],
            ),
        )
        self.depth_two = _array(directory / "frag_mass.npy", dtype=np.float32)
        self.depth_one = _array(directory / "f1_mass.npy", dtype=np.float32)
        self.depth_two_offsets = _offsets(
            directory / "frag_off.npy",
            rows=len(self.smiles),
            length=len(self.depth_two),
        )
        self.depth_one_offsets = _offsets(
            directory / "f1_off.npy",
            rows=len(self.smiles),
            length=len(self.depth_one),
        )
        self._positions: dict[str, int] = {}
        for index, value in enumerate(self.smiles):
            self._positions.setdefault(value, index)

    def row(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return complete readonly depth-two and depth-one vectors in that order.

        Args:
          index: Nonnegative original preparation row.

        Returns:
          depth_two: Complete readonly rounded depth-two mass vector.
          depth_one: Complete readonly rounded depth-one mass vector.

        """
        if index < 0 or index >= len(self.smiles):
            raise IndexError(index)
        return (
            self.depth_two[
                self.depth_two_offsets.item(index) : self.depth_two_offsets.item(
                    index + 1,
                )
            ],
            self.depth_one[
                self.depth_one_offsets.item(index) : self.depth_one_offsets.item(
                    index + 1,
                )
            ],
        )

    def for_smiles(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return the first explicit prepared pair or raise for absent structures."""
        return self.row(self._positions[smiles])

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        """Return the first complete pre-cast MAIN94 row or raise for absent data."""
        index = self._positions[smiles]
        return self._base_masses[
            self.depth_two_offsets.item(index) : self.depth_two_offsets.item(index + 1)
        ]

    @cached_property
    def _base_masses(self) -> NDArray[np.float64]:
        values = _array(self.directory / "base_mass.npy", dtype=np.float64)
        if len(values) != len(self.depth_two):
            raise ValueError(
                "Main fragment base masses differ from the depth-two layout.",
            )
        return values


def _array[S: np.generic](path: Path, *, dtype: type[S]) -> NDArray[S]:
    values = load_array(path, dtype, mmap=True)
    if values.ndim != 1:
        raise ValueError(f"Main fragment array {path.name} has an invalid rank.")
    return values


def _offsets(path: Path, *, rows: int, length: int) -> NDArray[np.int64]:
    values = _array(path, dtype=np.int64)
    if (
        values.shape != (rows + 1,)
        or values.item(0) != 0
        or values.item(-1) != length
        or np.any(np.diff(values) < 0)
    ):
        raise ValueError("Main fragment offsets differ from the complete row layout.")
    return values


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainPoolFamilies:
    """Retain the provided PID-aligned depth-one fragments and capabilities."""

    offsets: NDArray[np.int64]
    mass: NDArray[np.float32]
    capabilities: NDArray[np.uint8]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) != 22:
            raise ValueError("Provided pool family capability vocabulary differs.")
        if (
            self.offsets.ndim != 1
            or self.mass.ndim != 1
            or self.capabilities.ndim != 2
            or self.capabilities.shape[1] != len(self.names)
        ):
            raise ValueError("Provided pool family array rank differs.")
        if (
            self.offsets.shape != (len(self.capabilities) + 1,)
            or self.offsets.item(0) != 0
            or self.offsets.item(-1) != len(self.mass)
            or np.any(np.diff(self.offsets) < 0)
        ):
            raise ValueError("Provided pool family offsets differ from complete rows.")

    @classmethod
    def load(cls, directory: Path, *, count: int) -> MainPoolFamilies:
        """Read the complete declared public frag2 data in original PID order.

        Args:
          directory: Provided frag2 directory containing both fragment and cap arrays.
          count: Complete number of positions in the associated main candidate pool.

        Returns:
          families: Readonly complete arrays retaining the provided vocabulary.

        """
        result = cls(
            offsets=load_array(directory / "pool_f1_off.npy", np.int64, mmap=True),
            mass=load_array(directory / "pool_f1_mass.npy", np.float32, mmap=True),
            capabilities=load_array(directory / "pool_cap.npy", np.uint8, mmap=True),
            names=tuple((directory / "caps.txt").read_text().splitlines()),
        )
        if len(result.capabilities) != count:
            raise ValueError("Provided pool family count differs from pool positions.")
        return result


def training_smiles(path: Path) -> tuple[str, ...]:
    """Read one explicit structure per contiguous training SID, preserving aliases.

    Args:
      path: Provided training structure parquet with explicit sid and smiles columns.

    Returns:
      smiles: Every structure in contiguous SID order, including duplicated strings.

    """
    records = sorted(
        from_plain(
            parquet.read_table(path, columns=["sid", "smiles"]).to_pylist(),
            list[_TrainingRow],
        ),
        key=lambda row: row["sid"],
    )
    if [row["sid"] for row in records] != list(range(len(records))):
        raise ValueError("Expected one structure per contiguous training SID.")
    return tuple(row["smiles"] for row in records)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MainFamilyData:
    """Require complete declared pool, training and generated molecular evidence."""

    pool: CandidatePool
    train_smiles: tuple[str, ...]
    pool_structures: AnalogTable
    train_structures: AnalogTable
    generated_structures: AnalogTable
    pool_families: MainPoolFamilies
    generated_fragments: MainFragmentTable
    generated_intrinsic: IntrinsicTable
    fragnet: FragnetTable
    priors: DerivationPriors

    def __post_init__(self) -> None:
        if (
            tuple(row["smiles"] for row in self.pool_structures.metadata)
            != self.pool.smiles
            or tuple(row["formula_input"] for row in self.pool_structures.metadata)
            != self.pool.formulas
        ):
            raise ValueError("Prepared structures differ from pool positions.")
        if (
            tuple(row["smiles"] for row in self.train_structures.metadata)
            != self.train_smiles
        ):
            raise ValueError("Prepared structures differ from training SID positions.")
        if self.pool_families.names != self.generated_intrinsic.capability_names:
            raise ValueError("Provided and generated capability vocabulary differs.")
        if len(self.pool_families.capabilities) != len(self.pool.mass):
            raise ValueError("Provided family rows differ from pool positions.")

    def select(
        self,
        rows: CandidateRows,
        analogs: Sequence[Analog],
    ) -> MainFamilyRows:
        """Resolve every position; absent prepared rows raise rather than default.

        Args:
          rows: Complete candidate positions and their original generation provenance.
          analogs: Complete encountered analog sequence with training structure SIDs.

        Returns:
          families: Every aligned numeric family input, preserving all duplicates.

        """
        identities = rows.candidates
        count = len(identities.smiles)
        if any(
            len(values) != count
            for values in (
                identities.pool_rows,
                identities.generated,
                identities.formulas,
                rows.parent_sid,
                rows.combos,
                rows.raw,
            )
        ):
            raise ValueError("Family evidence differs from candidate positions.")
        if not np.array_equal(identities.generated, identities.pool_rows < 0):
            raise ValueError("Pool and generated positions differ.")
        structures: list[AnalogRecord] = []
        depth_one: list[NDArray[np.float64]] = []
        depth_two: list[NDArray[np.float64]] = []
        capabilities = np.empty((count, len(self.pool_families.names)), np.uint8)
        fragnet: list[FragnetRecord | None] = []
        for position, smiles in enumerate(identities.smiles):
            pid = identities.pool_rows.item(position)
            formula = identities.formulas[position]
            if pid >= 0:
                if pid >= len(self.pool.mass):
                    raise IndexError(pid)
                if (
                    smiles != self.pool.smiles[pid]
                    or formula != self.pool.formulas[pid]
                ):
                    raise ValueError("Candidate pool identity differs from its PID.")
                structures.append(self.pool_structures.row(pid))
                begin, end = (
                    self.pool_families.offsets.item(pid),
                    self.pool_families.offsets.item(pid + 1),
                )
                depth_one.append(self.pool_families.mass[begin:end].astype(np.float64))
                depth_two.append(self.pool.fragments(pid))
                capabilities[position] = self.pool_families.capabilities[pid]
            else:
                structures.append(
                    replace(
                        self.generated_structures.for_smiles(smiles),
                        formula_input=formula,
                        formula=family_data_formula_vector(formula),
                    ),
                )
                second, first = self.generated_fragments.for_smiles(smiles)
                depth_one.append(first.astype(np.float64))
                depth_two.append(second.astype(np.float64))
                capabilities[position] = self.generated_intrinsic.select((smiles,))[1][
                    0
                ]
            fragnet.append(self.fragnet.for_smiles(smiles))
            _ = self.priors.proposals.identity(smiles)
            raw = rows.raw[position]
            if raw is not None:
                _ = self.priors.proposals.identity(raw)
            parent, combo = rows.parent_sid.item(position), rows.combos[position]
            if parent >= 0:
                if parent >= len(self.train_smiles):
                    raise IndexError(parent)
                if combo is not None:
                    _ = self.priors.proposals.paths(self.train_smiles[parent], combo)
        analog_structures: list[AnalogRecord] = []
        for analog in analogs:
            if analog.sid < 0 or analog.sid >= len(self.train_smiles):
                raise IndexError(analog.sid)
            analog_structures.append(self.train_structures.row(analog.sid))
        return MainFamilyRows(
            candidate_structures=tuple(structures),
            analog_structures=tuple(analog_structures),
            train_smiles=self.train_smiles,
            depth1=tuple(depth_one),
            depth2=tuple(depth_two),
            capabilities=capabilities,
            fragnet=tuple(fragnet),
            priors=self.priors,
        )


class ReactionProducts(Protocol):
    """Complete ordered source-compatible proposals for each explicit request."""

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        """Return all ordered proposals, raising for an unstaged request."""
        ...


class ProductChemistry(Protocol):
    """Independent prepared outcomes, with missing data remaining errors."""

    def standardize(self, smiles: str) -> str | None:
        """Return the independently prepared standardized string or rejection."""
        ...

    def key(self, smiles: str) -> str | None:
        """Return the independently prepared legacy score key or rejection."""
        ...

    def properties(self, smiles: str, /) -> tuple[str, float, int] | None:
        """Return standardized formula, exact mass and heavy-atom count."""
        ...

    def fingerprint(self, smiles: str, /) -> NDArray[np.uint8] | None:
        """Return every packed raw bit, independently of score-key validity."""
        ...


class ProductFragments(Protocol):
    """Complete pre-cast F64 MAIN94 fragment rows, distinct from family F32 rows."""

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        """Return the complete pre-cast base fragment masses for a staged product."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationResult:
    """Retain every admitted product and duplicate-key support in source order."""

    candidates: tuple[GeneratedCandidate, ...]
    logs: tuple[DerivationLog, ...]


class ReactionScreen:
    """Preserve complete source parent, property, fingerprint and key gates."""

    class Config(Fig["ReactionScreen"]):
        parents: int = 6
        """Maximum number of similar training parents to transform."""

        min_similarity: float = 0.3
        """Inclusive parent similarity threshold."""

        per_parent: int = 60
        """Maximum raw proposals retained from one parent."""

        total: int = 150
        """Stop immediately after admitting this many products."""

        tolerance: float = 0.004
        """Absolute mass-window floor in daltons."""

        ppm: float = 10.0
        """Relative mass-window half-width."""

        max_steps: int = 2
        """Maximum number of sequential transformations."""

        application_budget: int = 800
        """Complete reaction-application budget per parent."""

        stage_parents: int = 24
        """Maximum intermediate molecules visited at each depth."""

        first_products: int = 32
        """Maximum products from a first-depth application."""

        later_products: int = 16
        """Maximum products from every later-depth application."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def plan(
        self,
        analogs: Sequence[Analog],
        *,
        target: float,
        train_smiles: Sequence[str],
        train_mass: NDArray[np.float64],
    ) -> tuple[tuple[Analog, ReactionRequest], ...]:
        """Declare every potential parent request before product admission.

        Args:
          analogs: Complete ordered parent retrieval evidence.
          target: Molecule-level neutral mass.
          train_smiles: Complete training structures in SID order.
          train_mass: Complete training neutral masses in SID order.

        Returns:
          requests: Every permitted parent and complete enumeration budget.

        """
        tolerance = max(self.config.tolerance, target * self.config.ppm * 1e-6)
        parents = [
            analog
            for analog in analogs
            if analog.similarity >= self.config.min_similarity
        ][: self.config.parents]
        requests: list[tuple[Analog, ReactionRequest]] = []
        for parent in parents:
            request: ReactionRequest = {
                "smiles": train_smiles[parent.sid],
                "delta": target - train_mass.item(parent.sid),
                "tolerance": tolerance,
                "max_steps": self.config.max_steps,
                "max_products": self.config.per_parent,
                "allow_isomerization": True,
                "application_budget": self.config.application_budget,
                "stage_parents": self.config.stage_parents,
                "first_products": self.config.first_products,
                "later_products": self.config.later_products,
            }
            requests.append((parent, request))
        return tuple(requests)

    def run(
        self,
        analogs: Sequence[Analog],
        *,
        target: float,
        excluded: set[str],
        train_smiles: Sequence[str],
        train_mass: NDArray[np.float64],
        train_fingerprints: NDArray[np.uint8],
        bits: NDArray[np.int64],
        proposals: ReactionProducts,
        chemistry: ProductChemistry,
        fragments: ProductFragments,
    ) -> GenerationResult:
        """Admit every reached product with exact source limits and arithmetic.

        Args:
          analogs: Ordered parent-SID retrieval evidence.
          target: Molecule-level neutral mass.
          excluded: Pool keys already inside the mass window.
          train_smiles: Complete training structures in SID order.
          train_mass: Complete training neutral masses in SID order.
          train_fingerprints: Packed selected-bit fingerprints in SID order.
          bits: Complete selected-bit positions in the raw fingerprint.
          proposals: Explicitly staged ordered reaction requests.
          chemistry: Independent prepared molecular decisions.
          fragments: Complete pre-cast F64 base fragments.

        Returns:
          result: All admitted products and duplicate-key supports in encounter order.

        """
        output: list[GeneratedCandidate] = []
        logs: list[DerivationLog] = []
        seen = set(excluded)
        tolerance = max(self.config.tolerance, target * self.config.ppm * 1e-6)
        for parent, request in self.plan(
            analogs,
            target=target,
            train_smiles=train_smiles,
            train_mass=train_mass,
        ):
            parent_fp = np.unpackbits(
                cast("NDArray[np.uint8]", train_fingerprints[parent.sid]),
            )[: len(bits)].astype(np.float32)
            for product in proposals.products(request):
                standardized = chemistry.standardize(product["raw"])
                if standardized is None:
                    continue
                key = chemistry.key(standardized)
                if key is None:
                    continue
                if key in seen:
                    logs.append(
                        DerivationLog(
                            parent=parent.sid,
                            similarity=float(parent.similarity),
                            combo=product["combo"],
                            kind="pool" if key in excluded else "gen",
                            how="key",
                            identity=key,
                        ),
                    )
                    continue
                properties = chemistry.properties(standardized)
                if properties is None or abs(properties[1] - target) > tolerance:
                    continue
                packed = chemistry.fingerprint(standardized)
                if packed is None:
                    continue
                fingerprint = np.unpackbits(packed)[bits].astype(np.float32)
                similarity = tanimoto(
                    (fingerprint * parent_fp).sum(),
                    fingerprint.sum(),
                    parent_fp.sum(),
                )
                seen.add(key)
                output.append(
                    GeneratedCandidate(
                        smiles=standardized,
                        key=key,
                        formula=properties[0],
                        mass=properties[1],
                        heavy_atoms=properties[2],
                        fingerprint=fingerprint,
                        fragments=fragments.base_for_smiles(standardized),
                        parent_similarity=parent.similarity,
                        steps=1,
                        parent_tanimoto=float(similarity),
                        parent_sid=parent.sid,
                        combo=product["combo"],
                        raw=product["raw"],
                    ),
                )
                if len(output) >= self.config.total:
                    return GenerationResult(candidates=tuple(output), logs=tuple(logs))
        return GenerationResult(candidates=tuple(output), logs=tuple(logs))


def _model_score(
    candidates: NDArray[np.float64],
    single: NDArray[np.float32],
    merged: NDArray[np.float32],
    model: int,
) -> NDArray[np.float32]:
    mean_single = (
        single.take(model, axis=0).astype(np.float64).mean(0)
        if single.shape[1]
        else np.zeros(len(candidates.T))
    )
    mean_merged = (
        merged.take(model, axis=0).astype(np.float64).mean(0)
        if merged.shape[1]
        else np.zeros(len(candidates.T))
    )
    return (candidates @ (0.5 * (mean_single + mean_merged))).astype(np.float32)


def _spectrum_consensus(
    candidates: NDArray[np.float64],
    single: NDArray[np.float32],
    pairwise_rank_limit: int,
) -> dict[str, NDArray[np.float32]]:
    logits = single[:2].astype(np.float64).mean(0)
    if not len(logits):
        return {
            name: np.zeros(len(candidates), np.float32)
            for name in (
                "sp_rank_mean",
                "sp_rank_min",
                "sp_rank_max",
                "sp_top3_frac",
                "sp_gap_mean",
            )
        }
    scores = candidates @ logits.T
    ranks = np.stack(
        [average_ranks(scores[:, index]) for index in range(len(scores.T))],
        1,
    )
    top_three = np.zeros(len(candidates))
    if len(candidates) <= pairwise_rank_limit:
        positions = np.stack(
            [
                1 + (scores[:, index][None, :] > scores[:, index][:, None]).sum(1)
                for index in range(len(scores.T))
            ],
            1,
        )
        top_three = (positions <= 3).mean(1)
    gap = scores - scores.max(0, keepdims=True)
    return {
        "sp_rank_mean": ranks.mean(1).astype(np.float32),
        "sp_rank_min": ranks.min(1).astype(np.float32),
        "sp_rank_max": ranks.max(1).astype(np.float32),
        "sp_top3_frac": top_three.astype(np.float32),
        "sp_gap_mean": (gap.mean(1) / 100.0).astype(np.float32),
    }


def _polarity_views(
    candidates: NDArray[np.float64],
    merged: NDArray[np.float32],
    modes: NDArray[np.int64],
) -> dict[str, NDArray[np.float32]]:
    features: dict[str, NDArray[np.float32]] = {}
    for mode, name in ((1, "pos"), (-1, "neg")):
        selected = np.flatnonzero(np.equal(modes, mode))
        if len(selected):
            logits = merged[:2, selected[0]].astype(np.float64).mean(0)
            scores = candidates @ logits
            features[f"pol_{name}_rank"] = average_ranks(scores).astype(np.float32)
            features[f"pol_{name}_gap"] = ((scores - scores.max()) / 100.0).astype(
                np.float32,
            )
        else:
            features[f"pol_{name}_rank"] = np.full(len(candidates), 0.5, np.float32)
            features[f"pol_{name}_gap"] = np.zeros(len(candidates), np.float32)
    return features


def _block_views(
    candidates: NDArray[np.float64],
    single: NDArray[np.float32],
    merged: NDArray[np.float32],
    selected_bits: NDArray[np.int64],
) -> dict[str, NDArray[np.float32]]:
    single_mean = (
        single[:2].astype(np.float64).mean(0).mean(0)
        if single.shape[1]
        else np.zeros(len(candidates.T))
    )
    merged_mean = (
        merged[:2].astype(np.float64).mean(0).mean(0)
        if merged.shape[1]
        else np.zeros(len(candidates.T))
    )
    logits = 0.5 * (single_mean + merged_mean)
    edges = np.cumsum([0, *[width for _, width in FINGERPRINT_BLOCKS]])
    block_ids = np.searchsorted(edges, selected_bits, side="right") - 1
    features: dict[str, NDArray[np.float32]] = {}
    for index, (name, _) in enumerate(FINGERPRINT_BLOCKS):
        selected = np.equal(block_ids, index)
        scores = candidates[:, selected] @ logits[selected]
        features[f"blk_{name}_gap"] = ((scores - scores.max()) / 100.0).astype(
            np.float32,
        )
        features[f"blk_{name}_rank"] = average_ranks(scores).astype(np.float32)
    return features


def _maximum(
    values: NDArray[np.bool_] | NDArray[np.float64],
    *,
    use: NDArray[np.bool_],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(use, values * weights, 0.0).max(1)


def _mean(
    values: NDArray[np.bool_] | NDArray[np.float64],
    *,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    total = weights.sum(1)
    return np.where(
        total > 0,
        (values * weights).sum(1) / np.maximum(total, 1e-12),
        0.0,
    )


def _transform_codes() -> NDArray[np.int64]:
    elements = FORMULA_ELEMENTS
    vectors = [
        np.array([dict(rule.delta).get(element, 0) for element in elements], np.int64)
        for rule in v6_reaction_rules()
    ]
    differences = [np.zeros(10, np.int64), *vectors]
    differences.extend(
        first + second
        for first, second in itertools.combinations_with_replacement(vectors, 2)
    )
    return np.unique(delta_codes(np.stack(differences)))


def _adjacency(graph: ElementGraph) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(len(graph.elements))]
    for edge in range(len(graph.bonds)):
        left, right = (graph.bonds.item((edge, side)) for side in range(2))
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _embedding(
    order: list[int],
    *,
    options: list[list[int]],
    big: list[set[int]],
    small: list[set[int]],
    mapping: dict[int, int],
    used: set[int],
) -> bool:
    if len(mapping) == len(order):
        return True
    atom = order[len(mapping)]
    for target in options[atom]:
        if target in used or any(
            mapping[neighbor] not in big[target]
            for neighbor in small[atom]
            if neighbor in mapping
        ):
            continue
        mapping[atom] = target
        used.add(target)
        if _embedding(
            order,
            options=options,
            big=big,
            small=small,
            mapping=mapping,
            used=used,
        ):
            return True
        used.remove(target)
        del mapping[atom]
    return False


def _stable_ranks(values: FloatArray) -> NDArray[np.float64]:
    order = np.argsort(-values, stable=True)
    ranks = np.asarray(
        stats.ordinal_ranks(values.copy(), order=convert_to_tensor(order)).numpy(),
        np.float64,
    )
    return ranks / max(1, len(values) - 1)


def _main_library_features(
    evidence: MainLibraryEvidence,
) -> dict[str, FloatArray | float]:
    maximum = evidence.maximum.max().item()
    return {
        "lib_max": evidence.maximum,
        "lib_mean": evidence.mean,
        "lib_top3": evidence.top3,
        "lib_nspec": np.log1p(evidence.count),
        "lib_cos": evidence.cosine,
        "lib_max_rank": _stable_ranks(evidence.maximum),
        "lib_gap": evidence.maximum - maximum,
        "lib_grp_max": maximum,
        "lib_hit": (evidence.maximum > 0).astype(np.float32),
        "lib_same_adduct": evidence.same_adduct,
    }


def _main_analog_features(
    fingerprints: NDArray[np.float32],
    analogs: Sequence[Analog],
    packed: NDArray[np.uint8],
    power: float,
) -> dict[str, FloatArray | float]:
    count = len(fingerprints)
    empty = np.zeros(count, np.float32)
    propagated = linear = best = top = mean = summed = unshifted = empty
    natural = natural_best = empty
    similarity_top = similarity_natural = 0.0
    high = 0
    if analogs:
        ids = np.fromiter((item.sid for item in analogs), np.int64)
        similarity = np.asarray([item.similarity for item in analogs], np.float32)
        shifts = np.asarray([item.shift for item in analogs], np.float64)
        natural_mask = np.asarray([item.natural_product for item in analogs], np.bool_)
        source = np.unpackbits(packed[ids], axis=1)[:, : fingerprints.shape[1]].astype(
            np.float32,
        )
        similar = tanimoto(
            fingerprints @ source.T,
            fingerprints.sum(1)[:, None],
            source.sum(1)[None, :],
        )
        weights = (similarity.astype(np.float64) ** power).astype(np.float32)
        weighted = similar * weights[None, :]
        propagated, linear, best, top = (
            weighted.max(1),
            (similar * similarity[None, :]).max(1),
            similar.max(1),
            similar[:, 0],
        )
        mean = weighted.sum(1) / (weights.sum() + 1e-9)
        summed = np.log1p(weighted.sum(1).astype(np.float64)).astype(np.float32)
        high = int((similarity > 0.5).sum())
        similarity_top = similarity.item(0)
        zero = np.abs(shifts) < 0.01
        if zero.any():
            unshifted = (similar[:, zero] * similarity[None, zero]).max(1)
        if natural_mask.any():
            natural = weighted[:, natural_mask].max(1)
            natural_best = similar[:, natural_mask].max(1)
            similarity_natural = similarity[natural_mask].item(0)
    return {
        "ap": propagated,
        "a1": linear,
        "best_tan": best,
        "top_tan": top,
        "mean_tan": mean,
        "ap_rank": _stable_ranks(propagated),
        "ap_gap": propagated - propagated.max(),
        "ap_grp_max": propagated.max().item(),
        "top_sim": similarity_top,
        "ap_np": natural,
        "best_tan_np": natural_best,
        "top_sim_np": similarity_natural,
        "ap_np_gap": natural - natural.max(),
        "ap_sum": summed,
        "n_analog_hi": float(high),
        "tan_zero_shift": unshifted,
    }


def _main_model_features(
    rows: CandidateRows,
    *,
    views: ModelViews | None,
) -> tuple[dict[str, FloatArray | float], NDArray[np.float32]]:
    fingerprints = rows.fingerprints
    count, width = len(fingerprints), len(fingerprints.T)
    cardinality = fingerprints.sum(1)
    single = merged = scores = np.zeros(count, np.float32)
    features: dict[str, FloatArray | float] = {}
    if views is not None:
        zs = views.single.mean(0, dtype=np.float32).mean(0, dtype=np.float32)
        zm = views.merged.mean(0, dtype=np.float32).mean(0, dtype=np.float32)
        wide = fingerprints.astype(np.float64)
        single = (wide @ zs.astype(np.float64)).astype(np.float32)
        merged = (wide @ zm.astype(np.float64)).astype(np.float32)
        logits = 0.5 * (zs + zm)
        scores = (wide @ logits.astype(np.float64)).astype(np.float32)
        decay = np.exp(-np.clip(logits, -30, 30).astype(np.float64)).astype(np.float32)
        probability = np.clip(1.0 / (1.0 + decay), 1e-6, 1 - 1e-6)
        likelihood = (
            fingerprints * np.log(probability.astype(np.float64)).astype(np.float32)
        ).sum(1) + (
            (1 - fingerprints)
            * np.log((1 - probability).astype(np.float64)).astype(np.float32)
        ).sum(1)
        hard = (probability > 0.5).astype(np.float32)
        intersection = fingerprints @ hard
        agreement = (wide @ probability.astype(np.float64)).astype(np.float32)
        norm = np.float32(np.linalg.norm(probability.astype(np.float64)))
        features.update(
            fp_ll=likelihood / max(1.0, width) * 100,
            fp_ll_gap=(likelihood - likelihood.max()) / 100.0,
            fp_cos=agreement / (np.sqrt(cardinality) * norm + 1e-9),
            fp_tan=tanimoto(intersection, cardinality, hard.sum()),
        )
        element_prediction = 0.5 * (
            views.single_elements.mean(0, dtype=np.float32).mean(0, dtype=np.float32)
            + views.merged_elements.mean(0, dtype=np.float32).mean(0, dtype=np.float32)
        )
        elements = np.zeros((count, 10), np.float32)
        for position, formula in enumerate(rows.candidates.formulas):
            vector = formula_vector(formula)
            if vector is not None:
                elements[position] = np.log1p(vector)
        difference = np.abs(elements - element_prediction[None, :])
        total = difference.sum(1)
        features.update(
            el_c=difference[:, 0],
            el_n=difference[:, 2],
            el_o=difference[:, 3],
            el_sum=total,
            el_rank=_stable_ranks(-total),
        )
    maximum = scores.max().item()
    soft = np.exp(
        ((scores - maximum) / width**0.5 * 4.0).astype(np.float64),
    ).astype(np.float32)
    soft = soft / soft.sum()
    ordered = np.sort(scores)[::-1]
    features.update(
        fz=scores / 100.0,
        fz_z=population_z(scores),
        fz_rank=_stable_ranks(scores),
        fz_gap=(scores - maximum) / 100.0,
        fz_norm=scores / np.sqrt(np.maximum(cardinality, 1)) / 10.0,
        fz_top=np.equal(scores, maximum).astype(np.float32),
        fz_softmax=soft,
        fz_grp_ent=-(
            soft * np.log((soft + 1e-12).astype(np.float64)).astype(np.float32)
        )
        .sum()
        .item(),
        fz_grp_gap=((ordered[:1] - ordered[1:2]) / 100.0).item(0) if count > 1 else 0.0,
        fz_single=single / 100.0,
        fz_single_gap=(single - single.max()) / 100.0,
        fz_merged=merged / 100.0,
        fz_merged_gap=(merged - merged.max()) / 100.0,
    )
    return features, scores


def _family_peaks(
    spectra: Sequence[SpectrumRequest],
    *,
    search: MainSearch,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], str, float]]:
    peaks: list[tuple[NDArray[np.float64], NDArray[np.float64], str, float]] = []
    for spectrum in spectra:
        mass, values = search.prepare(
            (
                np.asarray(spectrum["mz"], np.float64),
                np.asarray(spectrum["it"], np.float64),
            ),
        )
        if len(mass):
            peaks.append(
                (mass, values / values.max(), spectrum["adduct"], spectrum["prec"]),
            )
    return peaks


def _fragment_shifts(
    rows: Sequence[NDArray[np.float64]],
    *,
    masses: NDArray[np.float64],
    spec: NDArray[np.int64],
    deltas: Sequence[tuple[float, ...]],
) -> NDArray[np.uint8]:
    output = np.full((len(rows), len(masses)), 255, np.uint8)
    tolerance = np.maximum(0.01, masses * 15.0 * 1e-6)
    for row, fragments in enumerate(rows):
        if not len(fragments):
            continue
        for sign in range(-2, 3):
            for spectrum, carriers in enumerate(deltas):
                selected = np.flatnonzero(np.equal(spec, spectrum))
                for carrier in carriers:
                    base = masses[selected] - carrier
                    positions = np.searchsorted(fragments, base - sign * HYDROGEN_MASS)
                    for near in (positions - 1, positions):
                        valid = (near >= 0) & (near < len(fragments))
                        chosen = selected[valid]
                        matched = (
                            np.abs(
                                fragments[near[valid]]
                                + sign * HYDROGEN_MASS
                                - base[valid],
                            )
                            <= tolerance[chosen]
                        )
                        row_values = output.take(row, axis=0)
                        output[row, chosen[matched]] = np.minimum(
                            row_values.take(chosen[matched]),
                            abs(sign),
                        )
    return output


def _peak_accumulate(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    spec: NDArray[np.int64],
    count: int,
) -> NDArray[np.float64]:
    result = np.zeros((len(values), count, weights.shape[1]))
    for index in range(len(spec)):
        result[:, spec.item(index)] += values[:, index, None] * weights[index]
    return result


def _spectrum_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    total = values[:, 0].copy()
    for index in range(1, len(values.T)):
        total = total + values[:, index]
    return total / len(values.T)


def _cleavage_coverage(
    rows: Sequence[NDArray[np.float64]],
    *,
    peaks: Sequence[tuple[NDArray[np.float64], NDArray[np.float64], str, float]],
    deltas: Sequence[tuple[float, ...]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coverage = np.zeros((len(rows), len(peaks)))
    weighted = np.zeros_like(coverage)
    for row, fragments in enumerate(rows):
        if len(fragments) < 2:
            continue
        parent = fragments.item(-1)
        for spectrum, (raw_mass, values, _, precursor) in enumerate(peaks):
            order = np.argsort(raw_mass, kind="mergesort")
            mass = raw_mass.take(order)
            weight = np.asarray(np.sqrt(values.take(order)), np.float64)
            low, high = max(50.0, mass.min().item() - 0.5), precursor - 2.0
            observed = matched = 0
            best_sum = 0.0
            for position in range(len(fragments) - 1):
                a = fragments.item(position)
                b = parent - a
                if a > b + 1e-6:
                    break
                allowed = [
                    side for side in (a, b) if low <= side + deltas[spectrum][0] <= high
                ]
                if not allowed:
                    continue
                observed += 1
                hit, best = False, 0.0
                for side in allowed:
                    for carrier in deltas[spectrum]:
                        for shift in (-1, 0, 1):
                            ion = side + carrier + shift * HYDROGEN_MASS
                            tolerance = max(0.01, ion * 15.0 * 1e-6)
                            start = int(np.searchsorted(mass, ion - tolerance))
                            stop = int(
                                np.searchsorted(mass, ion + tolerance, side="right"),
                            )
                            if stop > start:
                                hit = True
                                best = max(best, weight[start:stop].max().item())
                if hit:
                    matched += 1
                    best_sum += best
            if observed:
                coverage[row, spectrum], weighted[row, spectrum] = (
                    matched / observed,
                    best_sum / observed,
                )
    return coverage, weighted


def _defaults() -> dict[str, float]:
    return {
        "n_par": 0.0,
        "best_sim": 0.0,
        "sum_sim4": 0.0,
        "min_steps": 0.0,
        "par_rank": 10.0,
        "n_combo": 0.0,
        "tf_logfr": -25.0,
        "site_logfr": -25.0,
        "sib_rank": 0.0,
        "n_sib": 0.0,
        "sib_frac": -1.0,
        "prior_miss": 0.0,
        "best_sim_nz": 0.0,
        "best_sim_zd": 0.0,
        "zd_only": 0.0,
    }


def _prior_tuple(row: PriorFeatures) -> tuple[float, float, float, float]:
    return row.site_logfr, row.tf_logfr, -float(row.sibling_rank), -float(row.siblings)


def _subset_rank(
    values: NDArray[np.float64],
    selected: NDArray[np.bool_],
) -> NDArray[np.float64]:
    output = np.full(len(values), -1.0)
    rows = np.flatnonzero(selected)
    if len(rows):
        output[rows] = average_ranks(values[rows])
    return output


def _main_library(
    spectra: Sequence[SpectrumRequest],
    *,
    rows: CandidateRows,
    library: SpectralLibrary,
    target: float,
    search: MainSearch,
    ppm: float,
    excluded: NDArray[np.bool_] | None,
) -> MainLibraryEvidence:
    count = len(rows.mass)
    maximum, mean, top3, counts, cosine, same_adduct = [
        np.zeros(count) for _ in range(6)
    ]
    positions = {
        rows.train_sid.item(i): i for i in range(count) if rows.train_sid.item(i) >= 0
    }
    encountered: dict[int, list[float]] = {}
    store = PeakStore(
        offsets=library.offsets,
        mz=library.mz,
        intensity=library.intensity,
    )
    for spectrum in spectra:
        query = (
            np.asarray(spectrum["mz"]),
            np.asarray(spectrum["it"]),
        )
        if not len(search.prepare(query)[0]):
            continue
        mode = 1 if spectrum["mode"] in (1, "positive") else -1
        selected = library.window(target, ppm=ppm, excluded=excluded, mode=mode)
        if not len(selected):
            continue
        scores = search.score(query, store=store, candidates=selected)
        cosines = search.score(query, store=store, candidates=selected, cosine=True)
        best: dict[int, float] = {}
        for position in range(len(selected)):
            row = selected.item(position)
            candidate = positions.get(library.sid.item(row))
            if candidate is None:
                continue
            value = scores.item(position)
            if value > best.get(candidate, 0.0):
                best[candidate] = value
            maximum[candidate] = max(maximum.item(candidate), value)
            cosine[candidate] = max(cosine.item(candidate), cosines.item(position))
            if library.adduct.item(row) == adduct_index(spectrum["adduct"]):
                same_adduct[candidate] = max(same_adduct.item(candidate), value)
            counts[candidate] += 1
        for candidate, score in best.items():
            encountered.setdefault(candidate, []).append(score)
    divisor = max(1, len(spectra))
    for candidate, scores in encountered.items():
        mean[candidate] = sum(scores) / divisor
        top3[candidate] = np.mean(sorted(scores)[-3:])
    return MainLibraryEvidence(
        maximum=maximum,
        mean=mean,
        top3=top3,
        count=counts / divisor,
        cosine=cosine,
        same_adduct=same_adduct,
    )


class _TrainingRow(TypedDict):
    sid: int
    smiles: str
