"""Checks for the main engine module."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import hashlib
import json
import math
import pickle
import sys

from numpy.typing import NDArray
from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, from_plain, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.boosting_test import model_text, stump
from priml.baselines.casmi2026.chemistry import (
    CandidatePool,
    MainSearch,
    PeakStore,
    SpectralLibrary,
)
from priml.baselines.casmi2026.custom_types import (
    Analog,
    ReactionRequest,
    SpectrumRequest,
)
from priml.baselines.casmi2026.dreams import (
    MainBank,
    ModelViews,
    keep_count,
    modal_adduct,
)
from priml.baselines.casmi2026.dreams_test import (
    checkpoint_config,
    save_bank,
    source_checkpoint,
)
from priml.baselines.casmi2026.fragnet import (
    FragnetTable,
    StageA,
    stage_b_names,
)
from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    AnalogRecord,
    AnalogTable,
    ElementGraph,
    IntrinsicTable,
    adduct_parameters,
    formula_vector as family_data_formula_vector,
    main_ion_deltas,
    record_fields,
)
from priml.baselines.casmi2026.main_engine import (
    CandidateRows,
    Combo,
    DerivationContext,
    DerivationLog,
    DerivationPriors,
    DerivationTable,
    GeneratedCandidate,
    GenerationResult,
    MainBase,
    MainEngine,
    MainFamilyData,
    MainFamilyRows,
    MainFragmentTable,
    MainLibraryEvidence,
    MainPoolFamilies,
    MainRetrieval,
    PolarityMerge,
    PriorFeatures,
    PriorTables,
    ReactionScreen,
    TreeBank,
    _main_analog_features,
    _main_library,
    _main_model_features,
    _stable_ranks,
    aggregate,
    analog_structure_features,
    candidates,
    combo_mass,
    delta_codes,
    derivation_features,
    diagnostic_losses,
    family_features,
    formula_vector as engine_formula_vector,
    frag2_features,
    load_tree,
    main_feature_names,
    main_features,
    main_fragment_features,
    main_subformula_features,
    main_subformula_masses,
    select_features,
    supports,
    training_smiles,
    view_features,
)
from priml.baselines.casmi2026.scripts.main_fragments import fragment_masses
from priml.baselines.casmi2026.testing import retrieval_inputs
from priml.math import stats


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.baselines.casmi2026.custom_types import (
        EngineResult,
        FragnetRecord,
        ProposalPath,
        ReactionProduct,
    )
    from priml.baselines.casmi2026.main_engine import MainFamilyProvider
    from priml.math.custom_types import Tensorable


def structure(
    elements: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    *,
    hashes: Sequence[int],
    numbers: Sequence[int],
    formula: Sequence[int],
    core: ElementGraph | None = None,
) -> AnalogRecord:
    """Build a prepared structure from literal hashes, counts and element graphs."""
    return AnalogRecord(
        smiles="",
        formula_input=None,
        hashes=np.array(hashes, np.int64),
        numbers=np.array(numbers, np.int16),
        formula=np.array([*formula, *[0] * (10 - len(formula))], np.int16),
        core_smiles="",
        element_smiles="",
        query_smiles="",
        graph=element_graph(elements, bonds),
        core=core,
        core_mask=np.zeros(len(elements), np.bool_),
    )


def element_graph(
    elements: Sequence[int],
    bonds: Sequence[tuple[int, int]],
) -> ElementGraph:
    """Pair atomic numbers with undirected bonds."""
    return ElementGraph(
        elements=np.array(elements, np.int16),
        bonds=np.array(bonds, np.int32).reshape(-1, 2),
    )


def ethanol() -> AnalogRecord:
    """Hash, count and graph C-C-O without a core."""
    return structure(
        (6, 6, 8),
        ((0, 1), (1, 2)),
        hashes=(5, 7, 11, 13, 0, 0, 0, 0),
        numbers=(0, 0, 0, 0, 0, 0, 0, 3),
        formula=(2, 6, 0, 1),
    )


def propanol() -> AnalogRecord:
    """Share ethanol's generic hash and first ring hash; keep a C-O core."""
    return structure(
        (6, 6, 6, 8),
        ((0, 1), (1, 2), (2, 3)),
        hashes=(9, 7, 11, 17, 0, 0, 0, 0),
        numbers=(1, 0, 0, 0, 0, 0, 0, 4),
        formula=(3, 8, 0, 1),
        core=element_graph((6, 8), ((0, 1),)),
    )


# One selected bit in each of ecfp4, fcfp4, rdk6, ap, tt and maccs.
STRUCTURE_BITS: Final = np.array([10, 8200, 10_300, 12_300, 14_400, 16_400], np.int64)


def test_identical_structures_relate_on_every_channel() -> None:
    values = analog_structure_features(
        [ethanol(), ethanol()],
        analogs=[ethanol(), ethanol(), ethanol()],
        # Maxima take the top analog's unit weight; weighted means divide by the sum.
        similarities=np.array([1.0, 0.5, 0.25]),
        analog_sid=np.array([4, 9, 12], np.int64),
        parent_sid=np.array([-1, -1], np.int64),
        fingerprints=np.ones((2, 6), np.float32),
        analog_fingerprints=np.ones((3, 6), np.float32),
        bits=STRUCTURE_BITS,
    )
    assert len(values) == 36
    assert all(array.dtype == np.float32 for array in values.values())
    ones = (
        "sc_el_max",
        "sc_el_wsum",
        "sc_gen_max",
        "sc_gen_wsum",
        "sc_el_top",
        "rs_jac_max",
        "rs_jac_wmean",
        "rs_cov_max",
        "rs_big_max",
        "subf_ca_max",
        "subf_ac_max",
        "fx_max",
        "fx_top",
        *("bt_" + block for block in ("maccs", "rdk6", "fcfp4", "ap", "tt")),
    )
    for name, array in values.items():
        expected = 1.0 if name in ones else 0.0
        if name == "st_ap":
            # Four hash relations and both full containments, out of six.
            expected = float(np.float32(5 / 6))
        elif name == "st_rank":
            # Two tied candidates share the average of ranks 0 and 1.
            expected = (0 + 1) / 2
        assert array.tolist() == [expected, expected], name


def test_structures_exclude_donors_and_cap_invalid_formula_distance() -> None:
    blank = structure(
        (),
        (),
        hashes=(0,) * 8,
        numbers=(0,) * 8,
        formula=(-1,),
    )
    values = analog_structure_features(
        [propanol(), propanol()],
        analogs=[ethanol(), replace(blank, graph=None)],
        similarities=np.array([1.0, 0.5]),
        analog_sid=np.array([4, 9], np.int64),
        # The second candidate comes from analog 4, so only analog 9 counts.
        parent_sid=np.array([-1, 4], np.int64),
        fingerprints=np.ones((2, 6), np.float32),
        analog_fingerprints=np.vstack([np.ones(6), np.zeros(6)]).astype(np.float32),
        bits=STRUCTURE_BITS,
    )
    weights = 1.0 + 0.5**4
    expected = {
        "sc_el_max": [0.0, 0.0],
        "sc_gen_max": [1.0, 0.0],
        "sc_gen_wsum": [1.0 / weights, 0.0],
        "sc_el_top": [0.0, 0.0],
        # Ring hashes {11, 17} and {11, 13} share one of three.
        "rs_jac_max": [1 / 3, 0.0],
        "rs_jac_wmean": [1 / 3 / weights, 0.0],
        "rs_cov_max": [0.5, 0.0],
        "rs_big_max": [1.0, 0.0],
        # Ethanol holds the C-O core and lies inside C-C-C-O.
        "sub_ca_max": [0.0, 0.0],
        "sub_ac_max": [1.0, 0.0],
        "subf_ca_max": [1.0, 0.0],
        "subf_ac_max": [0.0, 0.0],
        # CH2 separates the formulas; an invalid formula sits 50 away.
        "fd_wmean": [(3.0 + 50.0 * 0.5**4) / weights, 50.0],
        "fd_top": [3.0, 50.0],
        "fd_min": [3.0, 50.0],
        "fx_max": [1.0, 0.0],
        "fx_top": [1.0, 0.0],
        "dr_ring_top": [1.0, 1.0],
        "dr_ring_wabs": [1.0, 1.0],
        "da_ring_top": [1.0, 1.0],
        "bt_maccs": [1.0, 0.0],
        "st_ap": [(1 + 1 / 3 + 1 / 2 + 0.5 + 0.5) / 6, 0.0],
        "st_rank": [0.0, 1.0],
        "st_gap": [0.0, -(1 + 1 / 3 + 1 / 2 + 0.5 + 0.5) / 6],
        "st_fgap": [0.0, -(1 + 1 / 3 + 1 / 2 + 0.5 + 0.5) / 6],
    }
    for name, wanted in expected.items():
        assert values[name].tolist() == pytest.approx(wanted, abs=1e-6), name
    for name in ("dr_rs_top", "dr_core_top", "dr_rhet_top"):
        assert values[name].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(("size", "count"), [(1, 0), (0, 1)])
def test_structures_without_candidates_or_analogs_keep_sentinels(
    size: int,
    count: int,
) -> None:
    values = analog_structure_features(
        [ethanol()][:size],
        analogs=[ethanol()][:count],
        similarities=np.ones(count),
        analog_sid=np.zeros(count, np.int64),
        parent_sid=np.full(size, -1, np.int64),
        fingerprints=np.ones((size, 6), np.float32),
        analog_fingerprints=np.ones((count, 6), np.float32),
        bits=STRUCTURE_BITS,
    )
    assert len(values) == 36
    for name, array in values.items():
        sentinel = -1.0 if name.startswith("fd_") else 0.5 if name == "st_rank" else 0
        assert array.tolist() == [sentinel] * size, name


def test_signed_codes_keep_vector_axes_and_exact_sentinel_boundaries() -> None:
    differences = np.zeros((2, 3, 10), np.int64)
    differences[0, 0, 0] = 40
    differences[0, 1, 1] = -40
    differences[0, 2, 0] = 41
    differences[1, 0, 1] = -41
    differences[1, 1, 9] = 40
    differences[1, 2, 9] = -40
    result = delta_codes(differences)
    assert np.array_equal(
        result,
        np.array(
            [
                [40, -40 * 81, np.iinfo(np.int64).max],
                [np.iinfo(np.int64).max, 40 * 81**9, -40 * 81**9],
            ],
            np.int64,
        ),
    )
    assert result.dtype == np.int64


def test_codes_keep_the_int64_absolute_value_wraparound() -> None:
    # np.abs maps the lowest int64 to itself, so it never exceeds the sentinel bound.
    differences = np.zeros((2, 10), np.int64)
    differences[0, 0] = differences[1, 9] = np.iinfo(np.int64).min
    np.testing.assert_array_equal(
        delta_codes(differences),
        np.array([-40, -40 * 81**9], np.int64),
        strict=True,
    )


def test_stable_ranks_reuses_shared_inversion_with_original_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = (np.arange(129) % 5).astype(np.float32)[::-1]
    order = np.argsort(-values, stable=True)
    expected = np.empty(len(values), np.float64)
    expected[order] = np.arange(len(values))
    observed: list[Tensor | None] = []
    original = stats.ordinal_ranks

    def capture(x: Tensorable, *, order: Tensor | None = None) -> Tensor:
        observed.append(order)
        return original(x, order=order)

    monkeypatch.setattr(stats, "ordinal_ranks", capture)
    actual = _stable_ranks(values)
    np.testing.assert_array_equal(
        actual.view(np.uint64),
        (expected / 128).view(np.uint64),
    )
    assert len(observed) == 1
    assert observed[0] is not None
    assert torch.equal(observed[0], torch.from_numpy(order))


def test_main_engine_empty_group_keeps_all94_columns() -> None:
    _, pool, _ = retrieval_inputs()
    rows = candidates(pool, target=900.0, ppm=10.0, fallback_ppm=30.0)
    empty = np.zeros(0, np.float64)
    result = main_features(
        rows,
        target=900.0,
        spectra=[],
        library=MainLibraryEvidence(
            maximum=empty,
            mean=empty,
            top3=empty,
            count=empty,
            cosine=empty,
            same_adduct=empty,
        ),
        analogs=(),
        # main_features reads the retrieval fixture's seven bits in one byte.
        train_fingerprints=np.zeros((9, 1), np.uint8),
        views=None,
        fragments=None,
        fragment_counts=np.zeros(0, np.float32),
        subformulas=None,
        similarity_power=4.0,
    )
    assert result.shape == (0, 94)


def frag2_spectrum() -> SpectrumRequest:
    """Place protonated peaks at 50 and 80 Da, with water lost from the first."""
    proton = main_ion_deltas("[M+H]+")[0]
    return {
        "mz": [50.0 + proton, 80.0 + proton],
        "it": [1.0, 0.25],
        "prec": 50.0 + proton + 18.010565,
        "adduct": "[M+H]+",
        "mode": 1,
    }


def _frag2(
    spectra: Sequence[SpectrumRequest],
    *,
    scores: Sequence[float],
    limit: int = 600,
) -> dict[str, NDArray[np.float32]]:
    capabilities = np.zeros((len(scores), 22), np.uint8)
    capabilities[:1, 0] = 1
    return frag2_features(
        spectra,
        search=MainEngine.Config().fragment_search.make(),
        formulas=("C3H8O",) * len(scores),
        model_score=np.array(scores, np.float32),
        depth1=[np.array([50.0, 130.0]), np.zeros(0)][: len(scores)],
        depth2=[np.array([50.0, 80.0]), np.array([50.0 + HYDROGEN_MASS])][
            : len(scores)
        ],
        capabilities=capabilities,
        max_candidates=limit,
    )


def test_frag2_explains_shared_peaks_coverage_and_water_loss_by_hand() -> None:
    values = _frag2([frag2_spectrum()], scores=(0.5, 0.2))
    assert len(values) == 31
    assert all(array.dtype == np.float32 for array in values.values())
    # Square-root weights are 1 and 0.5; the leader explains both peaks, the other
    # explains the first with one hydrogen shift and nothing at depth one.
    base, heavy = math.log(0.1 / 0.5), math.log(0.1 / (5 / 6))
    background = (base + 0.5 * heavy) / 1.5
    mixed = (0.25 + 0.5 * 0.375, 0.5 * 0.25)
    expected = {
        "disc_max": [0.25 / 1.5, 0.0],
        "disc_mean": [0.25 / 1.5, 0.0],
        "disc1_mean": [0.25 / 1.5, 0.0],
        "idf_frac": [1.0, 0.0],
        "idf_rank": [0.0, 1.0],
        "uniq_int": [0.5 / 1.5, 0.0],
        "top5_expl": [1.0, 0.5],
        "top5_unexpl": [0.0, 0.5 / 1.5],
        "grp_disc": [0.25 / 1.5, 0.0],
        "grp_miss": [0.0, 0.25 / 1.5],
        "grp_rank": [0.0, 1.0],
        "grp_gap": [0.0, -1.0],
        "d1_int": [1 / 1.5, 0.0],
        "mix_disc": [mixed[0] / 1.5, mixed[1] / 1.5],
        "mix_rank": [0.0, 1.0],
        "mix_grp_gap": [0.0, (mixed[1] - mixed[0]) / 1.5],
        "llr1": [background + math.log(9) / 1.5, background],
        "cov1": [1.0, 0.0],
        "covw1": [1.0, 0.0],
        "cov_rank": [0.0, 1.0],
        "cov_grp_gap": [0.0, -1.0],
        "excl_gain": [0.0, 0.0],
        "excl_loss": [0.0, 0.75 / 1.5],
        "mix_vs_lead": [0.0, (mixed[1] - mixed[0]) / 1.5],
        "cov_vs_lead": [0.0, -1.0],
        "llr_vs_lead": [0.0, -math.log(9) / 1.5],
        # Only the first candidate can lose water from the first peak.
        "dl_sup_spec": [0.0, 0.0],
        "dl_unsup_spec": [0.0, 0.0],
        "dl_sup_gen": [1 / 1.5, 0.0],
        "dl_unsup_gen": [0.0, 1 / 1.5],
        "dl_sup_gap": [0.0, -1 / 1.5],
    }
    assert set(values) == set(expected)
    for name, wanted in expected.items():
        assert values[name].tolist() == pytest.approx(wanted, abs=1e-6), name


def test_frag2_caps_candidates_with_ties_and_skips_empty_inputs() -> None:
    capped = _frag2([frag2_spectrum()], scores=(0.5, 0.2), limit=1)
    # The unscored candidate keeps every top-five peak unexplained.
    assert capped["top5_unexpl"].tolist() == [0.0, 1.0]
    assert capped["disc_max"].tolist() == [0.0, 0.0]
    assert capped["d1_int"].tolist() == pytest.approx([1 / 1.5, 0.0], abs=1e-6)
    tied = _frag2([frag2_spectrum()], scores=(0.5, 0.5), limit=1)
    assert tied["top5_unexpl"].tolist() == pytest.approx([0.0, 0.5 / 1.5], abs=1e-6)
    for values in (
        _frag2([], scores=(0.5, 0.2)),
        _frag2([{**frag2_spectrum(), "mz": [], "it": []}], scores=(0.5, 0.2)),
    ):
        assert all(array.tolist() == [0.0, 0.0] for array in values.values())
    assert all(
        array.shape == (0,) for array in _frag2([frag2_spectrum()], scores=()).values()
    )


def test_diagnostic_losses_bind_singles_and_pairs_to_capabilities() -> None:
    capabilities = np.zeros((3, 22), np.uint8)
    capabilities[0, 0] = 1
    capabilities[1, 10] = 2
    capabilities[2, 10] = 1
    masses, specific, support = diagnostic_losses(capabilities)
    assert masses.shape == specific.shape == (41,)
    assert masses[0] == 18.010565
    # The first pair doubles the hexose loss, which needs two hexose sites.
    assert masses[27] == 162.052824 + 162.052824
    assert specific.tolist() == [False] * 6 + [True] * 35
    assert support.dtype == np.bool_
    assert support[:, 0].tolist() == [True, False, False]
    assert support[:, 14].tolist() == [False, True, True]
    assert support[:, 27].tolist() == [False, True, False]


def _views(
    fingerprints: list[list[int]],
    *,
    spectra: int = 1,
    modes: Sequence[int] = (1,),
) -> dict[str, NDArray[np.float32]]:
    logits = np.tile(np.array([1.0, 3.0], np.float32), (3, 1, 1))
    return view_features(
        np.array(fingerprints, np.float32),
        logits[:, :spectra],
        np.tile(logits, (1, len(modes), 1)),
        np.array(modes, np.int64),
        # Selected bits fall in the ecfp4 and fcfp4 blocks.
        np.array([10, 8200], np.int64),
        pairwise_rank_limit=7,
    )


def test_model_views_rank_scores_spectra_polarities_and_blocks_by_hand() -> None:
    # All three models score bit logits 1 and 3, so the candidates score 1, 3 and 1.
    values = _views([[1, 0], [0, 1], [1, 0]])
    assert len(values) == 36
    assert all(array.dtype == np.float32 for array in values.values())
    ranks, gaps = [0.75, 0.0, 0.75], [-0.02, 0.0, -0.02]
    expected = {
        **{f"fz{name}_gap": gaps for name in ("A", "B", "D")},
        **{f"fz{name}_rank": ranks for name in ("A", "B", "D")},
        **{
            f"fz{name}_z": [-(0.5**0.5), 2**0.5, -(0.5**0.5)]
            for name in ("A", "B", "D")
        },
        "rank_spread": [0.0] * 3,
        "rank_worst": ranks,
        "fzABD_gap": gaps,
        "fzABD_rank": ranks,
        "sp_rank_mean": ranks,
        "sp_rank_min": ranks,
        "sp_rank_max": ranks,
        "sp_top3_frac": [1.0] * 3,
        "sp_gap_mean": gaps,
        "pol_pos_rank": ranks,
        "pol_pos_gap": gaps,
        # No negative item: the polarity defaults to the middle rank.
        "pol_neg_rank": [0.5] * 3,
        "pol_neg_gap": [0.0] * 3,
        "blk_ecfp4_gap": [0.0, -0.01, 0.0],
        "blk_ecfp4_rank": [0.25, 1.0, 0.25],
        "blk_fcfp4_gap": [-0.03, 0.0, -0.03],
        "blk_fcfp4_rank": ranks,
        **{
            f"blk_{name}_{field}": [0.0] * 3 if field == "gap" else [0.5] * 3
            for name in ("ecfp6", "rdk6", "ap", "tt", "maccs")
            for field in ("gap", "rank")
        },
    }
    assert set(values) == set(expected)
    for name, wanted in expected.items():
        assert values[name].tolist() == pytest.approx(wanted, abs=1e-6), name


def test_model_views_without_spectra_or_spread_fall_back_to_zero() -> None:
    silent = _views([[1, 0], [0, 1], [1, 0]], spectra=0, modes=(1, -1))
    for name in (
        "sp_rank_mean",
        "sp_rank_min",
        "sp_rank_max",
        "sp_top3_frac",
        "sp_gap_mean",
    ):
        assert silent[name].tolist() == [0.0] * 3, name
    # The merged view alone halves the scores; the negative item ranks like the positive.
    assert silent["fzA_gap"].tolist() == pytest.approx([-0.01, 0.0, -0.01], abs=1e-6)
    assert silent["pol_neg_rank"].tolist() == silent["pol_pos_rank"].tolist()
    constant = _views([[1, 0]] * 3)
    assert constant["fzA_z"].tolist() == [0.0] * 3
    assert constant["fzA_rank"].tolist() == [0.5] * 3


def test_spectrum_and_block_views_average_only_the_first_two_models() -> None:
    # Models A and B score the bits 1 and 3; model D scores them 5 and -3.
    logits = np.array([[[1.0, 3.0]], [[1.0, 3.0]], [[5.0, -3.0]]], np.float32)
    values = view_features(
        np.array([[1, 0], [0, 1], [1, 0]], np.float32),
        logits,
        logits,
        np.array([1], np.int64),
        np.array([10, 8200], np.int64),
        pairwise_rank_limit=7,
    )
    ranks, gaps = [0.75, 0.0, 0.75], [-0.02, 0.0, -0.02]
    reversed_ranks = [0.25, 1.0, 0.25]
    expected = {
        "fzD_gap": [0.0, -0.08, 0.0],
        "fzD_rank": reversed_ranks,
        "rank_spread": [0.5, 1.0, 0.5],
        # Each candidate's worst rank across the three models.
        "rank_worst": [0.75, 1.0, 0.75],
        # All three models average to the scores 7/3, 1 and 7/3.
        "fzABD_gap": [0.0, (1 - 7 / 3) / 100, 0.0],
        "fzABD_rank": reversed_ranks,
        "sp_rank_mean": ranks,
        "sp_gap_mean": gaps,
        "blk_ecfp4_gap": [0.0, -0.01, 0.0],
        "blk_fcfp4_gap": [-0.03, 0.0, -0.03],
        "blk_fcfp4_rank": ranks,
    }
    for name, wanted in expected.items():
        assert values[name].tolist() == pytest.approx(wanted, abs=1e-6), name


def test_formula_vocabulary_retains_all_elements_and_complete_grammar() -> None:
    vector = engine_formula_vector("CH2N3O4P5S6F7Cl8Br9I10")
    assert vector is not None
    assert vector.dtype == np.int64
    assert np.array_equal(vector, np.arange(1, 11, dtype=np.int64))
    repeated = np.zeros(10, np.int64)
    repeated[:2] = [5, 1]
    vector = engine_formula_vector("C2C3H+-")
    assert vector is not None
    assert np.array_equal(vector, repeated)
    vector = engine_formula_vector("+")
    assert vector is not None
    assert np.array_equal(vector, np.zeros(10, np.int64))
    for formula in ("", "NaCl", "C.2H", "xCH", "C1xH", "[C]"):
        assert engine_formula_vector(formula) is None


def test_peaks_without_masses_or_intensity_are_skipped_by_the_explanation() -> None:
    masses = np.array([10.0, 20.0, 30.0])
    valid = (np.array([10.0, 20.0]), np.array([1.0, 0.5]), "[M+H]+")
    blank = [
        (np.zeros(0), np.zeros(0), "[M+H]+"),
        (np.array([10.0]), np.zeros(1), "[M+H]+"),
    ]
    assert np.array_equal(
        main_fragment_features(masses, peaks=[*blank, valid]),
        main_fragment_features(masses, peaks=[valid]),
    )
    assert not main_fragment_features(masses, peaks=blank).any()


def test_main_ranker_top_gap_keeps_float32_subtraction() -> None:
    _, pool, _ = retrieval_inputs()
    rows = candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0, drop_row=0)
    logits = np.array([0.0, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0], np.float32)
    # The pinned Full1 bank loads exactly one checkpoint for these base columns.
    views = ModelViews(
        single=np.tile(logits[None, None], (1, 3, 1)),
        merged=np.tile(logits[None, None], (1, 2, 1)),
        single_elements=np.zeros((1, 3, 10), np.float32),
        merged_elements=np.zeros((1, 2, 10), np.float32),
        modes=np.array([1, -1], np.float32),
    )
    values, scores = _main_model_features(rows, views=views)
    assert np.array_equal(scores, np.array([-0.2, 0.3], np.float32))
    gap = values["fz_grp_gap"]
    assert isinstance(gap, float)
    assert np.float32(gap) == (np.float32(0.3) - np.float32(-0.2)) / 100.0


def test_analog_sum_is_correctly_rounded_float32_log1p() -> None:
    similarity = 0.46771079301834106
    values = _main_analog_features(
        # _main_analog_features isolates one candidate's single similarity term.
        np.ones((1, 1), np.float32),
        (Analog(sid=0, similarity=similarity),),
        np.array([[128]], np.uint8),
        1.0,
    )
    expected = np.array([math.log1p(np.float32(similarity))], np.float32)
    assert np.array_equal(values["ap_sum"], expected)


def test_main_likelihood_is_evaluated_in_float64_and_rounded_once() -> None:
    _, pool, _ = retrieval_inputs()
    rows = replace(
        candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0),
        fingerprints=np.array(
            [[1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
            np.float32,
        ),
    )
    logits = np.array(
        [
            -0.32868582010269165,
            -0.17201346158981323,
            0.22252917289733887,
            -0.4066565930843353,
            -0.29706329107284546,
            0.056098613888025284,
            -0.24937939643859863,
        ],
        np.float32,
    )
    views = ModelViews(
        single=logits[None, None],
        merged=logits[None, None],
        # ModelViews holds one Full1 checkpoint and one source query spectrum.
        single_elements=np.zeros((1, 1, 10), np.float32),
        merged_elements=np.zeros((1, 1, 10), np.float32),
        modes=np.ones(1, np.float32),
    )
    values, _ = _main_model_features(rows, views=views)
    one = np.float32(1)
    probability = [one / (one + np.float32(math.exp(-z))) for z in logits.tolist()]
    likelihood = np.zeros(3, np.float32)
    for row, bits in enumerate(rows.fingerprints.tolist()):
        present = absent = np.float32(0)
        for bit, p in zip(bits, probability, strict=True):
            present += np.float32(bit) * np.float32(math.log(p))
            absent += np.float32(1 - bit) * np.float32(math.log(one - p))
        likelihood[row] = present + absent
    assert np.array_equal(values["fp_ll"], likelihood / np.float32(7) * 100)


def ranker_text(feature: int, *, scale: float = 1.0) -> str:
    """Serialize three stumps over the first three columns, led by one feature."""
    return model_text(
        stump(feature, left=-scale, right=scale),
        stump((feature + 1) % 3, left=0.5 * scale, right=-0.5 * scale),
        stump((feature + 2) % 3, left=0.25, right=0.75),
    )


def test_complete_tree_asset_bank_binds_features_and_averages_its_models(
    tmp_path: Path,
) -> None:
    models = [ranker_text(feature, scale=feature + 1.0) for feature in (0, 1, 2, 0)]
    payload = pickle.dumps(
        {
            "features": ["Column_0", "Column_1", "Column_2"],
            "boosters": models,
            "params": {},
            "rounds": 3,
        },
    )
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    bank = TreeBank.load(path, hashlib.sha256(payload).hexdigest())
    query = np.array([[0.0, 1.0, -1.0], [2.0, 0.0, 3.0], [-1.0, -2.0, 0.0]], np.float32)
    actual = bank.score(query[:, ::-1], ["Column_2", "Column_1", "Column_0"])
    wanted = np.mean(
        [
            LightGBMModel.parse(text).predict(query.astype(np.float64))
            for text in models
        ],
        axis=0,
    )
    np.testing.assert_array_equal(actual, wanted, strict=True)
    # Row 0 sends column 0 left, column 1 right and column 2 left in every stump.
    first = LightGBMModel.parse(models[0]).predict(query.astype(np.float64))
    assert first.tolist()[0] == -1.0 - 0.5 + 0.25
    assert len(bank.models) == 4
    assert [len(model.trees) for model in bank.models] == [3, 3, 3, 3]
    with pytest.raises(ValueError, match="missing"):
        _ = bank.score(query, ["Column_0", "other", "Column_2"])


def test_stage_b_text_is_read_as_a_complete_numeric_tree_asset(tmp_path: Path) -> None:
    text = ranker_text(0)
    path = tmp_path / "stageB.txt"
    _ = path.write_text(text)
    model = load_tree(path, hashlib.sha256(text.encode()).hexdigest())
    assert len(model.trees) == 3
    assert model.features == ("Column_0", "Column_1", "Column_2")


def test_tree_asset_digest_is_checked_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(b"invalid model")
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = TreeBank.load(path, "0" * 64)
    with pytest.raises(ValueError, match="requires its SHA256"):
        _ = TreeBank.load(path, "")


def test_tree_asset_rejects_pickled_runtime_objects(tmp_path: Path) -> None:
    payload = pickle.dumps(Path("runtime"))
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    with pytest.raises(pickle.UnpicklingError, match="plain data"):
        _ = TreeBank.load(path, hashlib.sha256(payload).hexdigest())


def test_each_booster_must_declare_the_complete_ranker_feature_order(
    tmp_path: Path,
) -> None:
    payload = pickle.dumps(
        {
            "features": ["wrong"],
            "boosters": [ranker_text(0)],
        },
    )
    path = tmp_path / "ranker.pkl"
    _ = path.write_bytes(payload)
    with pytest.raises(ValueError, match="feature"):
        _ = TreeBank.load(path, hashlib.sha256(payload).hexdigest())


class PreparedPaths:
    """Serve staged terminal paths and nullable identities from plain tables."""

    def __init__(
        self,
        rows: dict[tuple[str, Combo], tuple[ProposalPath, ...]] | None = None,
        identities: dict[str, tuple[str | None, str | None]] | None = None,
    ) -> None:
        self.rows: dict[tuple[str, Combo], tuple[ProposalPath, ...]] = dict(rows or {})
        self.identities: dict[str, tuple[str | None, str | None]] = dict(
            identities or {},
        )

    def paths(
        self,
        parent_smiles: str,
        combo: Combo,
    ) -> tuple[ProposalPath, ...]:
        """Require a staged pair, including an explicitly empty tuple."""
        return self.rows[parent_smiles, combo]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        """Require an explicitly staged nullable molecular identity."""
        return self.identities[smiles]


def path(
    raw: str | None,
    standardized: str | None,
    *environments: tuple[int, int, int],
) -> ProposalPath:
    """Stage one terminal path whose identity the caller resolves."""
    return {
        "raw": raw,
        "standardized": standardized,
        "key": None,
        "skeleton": None,
        "environments": environments,
    }


def derivation_priors() -> DerivationPriors:
    """Back methyl ethers off to radius one and demethylation to its transform rate."""
    proposals = PreparedPaths(
        {
            ("CCO", ("methyl_O",)): (
                path("COCC", "CCOC", (11, 12, 13)),
                path("COC", "COC", (99, 99, 99)),
                path(None, "lost", (11, 12, 13)),
                path("unstandardized", None, (11, 12, 13)),
                path("CCOC", "CCOC", (99, 99, 99)),
            ),
            ("CCCCO", ("demethyl_O",)): (path("CCCO", "CCCO", (5, 5, 5)),),
        },
        {
            "CCCO": ("K1", "S1"),
            "CCOC": ("K2", "S2"),
            "COC": ("K3", "S3"),
            # Other spellings: one shares CCOC's key and skeleton, one only its skeleton.
            "C(C)OC": ("K2", "S2"),
            "COC(C)": ("K9", "S2"),
        },
    )
    return DerivationPriors(
        {
            "glob": 0.2,
            "tf": {"methyl_O": (3.0, 40.0), "demethyl_O": (4.0, 20.0)},
            "env": {
                1: {("methyl_O", 11): (10.0, 30.0)},
                2: {("methyl_O", 12): (1.0, 10.0)},
                3: {},
            },
        },
        proposals=proposals,
    )


# Smoothed rates (success + 2 * 0.2) / (count + 2) and the radius-one backoff.
METHYL: Final = (3.0 + 2 * 0.2) / (40.0 + 2)
DEMETHYL: Final = (4.0 + 2 * 0.2) / (20.0 + 2)
SITE: Final = (10.0 + 2 * METHYL) / (30.0 + 2)


def test_prior_groups_keep_each_products_best_site_and_sibling_rank() -> None:
    priors = derivation_priors()
    assert priors.p_transform("methyl_O") == METHYL
    assert priors.p_transform("unseen") == 0.2
    # Radius two holds only ten observations, so the backoff stops at radius one.
    assert priors.p_site("methyl_O", (11, 12, 13)) == SITE
    assert priors.p_site("methyl_O", (99, 99, 99)) == METHYL
    group = priors.group("CCO", ("methyl_O",))
    assert tuple(group) == ("CCOC", "COC")
    assert group["CCOC"] == PriorFeatures(
        tf_logfr=math.log(METHYL),
        site_logfr=math.log(SITE),
        sibling_rank=1,
        siblings=2,
    )
    assert group["COC"] == PriorFeatures(
        tf_logfr=math.log(METHYL),
        site_logfr=math.log(METHYL),
        sibling_rank=2,
        siblings=2,
    )
    assert priors.group("CCO", ("methyl_O",)) is group
    assert combo_mass(("methyl_O",)) == pytest.approx(12.0 + 2 * 1.00782503207)
    assert combo_mass(("methyl_O", "demethyl_O")) == 0.0
    assert combo_mass(("dehexose2",)) < 0


def derivation_context() -> DerivationContext:
    """Bind a supported pool alcohol, two generated ethers and an unsupported row."""
    return DerivationContext(
        smiles=("CCCO", "CCOC", "COC", "OCC"),
        keys=("K1", "K2", "K3", None),
        formulas=("C3H8O", "C3H8O", "C2H6O", "C2H6O"),
        generated=np.array([False, True, True, False]),
        parent_sid=np.array([-1, 0, 0, -1], np.int64),
        similarities=np.array([0.0, 0.9, 0.5, 0.0]),
        combos=(None, ("methyl_O",), ("methyl_O",), None),
        analog_sid=np.array([0, 1], np.int64),
        model_score=np.array([0.1, 0.3, 0.2, 0.0], np.float32),
        parent_smiles=("CCO", "CCCCO"),
        logs=(
            DerivationLog(
                parent=1,
                similarity=0.6,
                combo=("demethyl_O",),
                kind="pool",
                how="smi",
                identity="CCCO",
            ),
        ),
    )


def test_aggregates_take_each_supports_parent_combo_and_best_prior() -> None:
    priors = derivation_priors()
    context = derivation_context()
    evidence = supports(context, proposals=priors.proposals)
    assert evidence == [
        [(1, 0.6, ("demethyl_O",))],
        [(0, 0.9, ("methyl_O",))],
        [(0, 0.5, ("methyl_O",))],
        [],
    ]
    values = aggregate(context, priors=priors, evidence=evidence)
    expected = {
        "n_par": [1, 1, 1, 0],
        "best_sim": [0.6, 0.9, 0.5, 0.0],
        "sum_sim4": [0.6**4, 0.9**4, 0.5**4, 0.0],
        "min_steps": [1, 1, 1, 0],
        "par_rank": [1, 0, 0, 10],
        "n_combo": [1, 1, 1, 0],
        "tf_logfr": [math.log(DEMETHYL), math.log(METHYL), math.log(METHYL), -25],
        "site_logfr": [math.log(DEMETHYL), math.log(SITE), math.log(METHYL), -25],
        "sib_rank": [1, 1, 2, 0],
        "n_sib": [1, 2, 2, 0],
        "sib_frac": [0, 0, 1, -1],
        "prior_miss": [0, 0, 0, 0],
        "best_sim_nz": [0.6, 0.9, 0.5, 0.0],
        "best_sim_zd": [0, 0, 0, 0],
        "zd_only": [0, 0, 0, 0],
    }
    assert set(values) == set(expected)
    for name, wanted in expected.items():
        assert values[name].tolist() == pytest.approx(wanted), name
    features = derivation_features(context, priors=priors)
    assert len(features) == 41
    assert all(array.dtype == np.float32 for array in features.values())
    rows = {
        "dp_hit": [1, 0, 0, 0],
        "dp_n_par": [1, 0, 0, 0],
        "dp_best_sim": [0.6, 0, 0, 0],
        "dp_rank_sim": [0, -1, -1, 1],
        "dp_iso_n_hit": [math.log(2), math.log(2), 0, 0],
        "dp_q_n_hit": [math.log(2)] * 4,
        "dg_n_par": [0, 1, 1, 0],
        "dg_sib_rank": [0, 1, 2, 0],
        "dg_rank_site": [-1, 0, 1, -1],
        "dj_rank_sim": [1 / 3, 0, 2 / 3, 1],
        "ds_n": [math.log(2), math.log(3), math.log(3), 0],
        "ds_fz_rank": [0, 0, 1, -1],
        "ds_fz_gap": [0, 0, -0.1 / 100, -10],
        "ds_site_rank": [0, 0, 1, -1],
        "ds_fm_site_rank": [1, 0, 0, -1],
    }
    for name, wanted in rows.items():
        assert features[name].tolist() == pytest.approx(wanted, abs=1e-6), name


def test_support_identity_routes_keep_generated_duplicates_and_known_invalid_keys() -> (
    None
):
    proposals = PreparedPaths()
    proposals.identities.update(
        fallback=("shared", "skeleton"),
        known_invalid=(None, None),
    )
    routes: tuple[
        tuple[
            Combo | None,
            Literal["pool", "gen"],
            Literal["smi", "key"],
            str | None,
        ],
        ...,
    ] = (
        (("a",), "pool", "smi", "pool"),
        (None, "pool", "smi", "pool"),
        (("a",), "gen", "smi", "duplicate"),
        (("b",), "pool", "key", "shared"),
        (("b",), "gen", "key", "shared"),
        (("a",), "gen", "smi", "fallback"),
        (("a",), "gen", "smi", "known_invalid"),
        (("a",), "gen", "key", None),
    )
    context = DerivationContext(
        smiles=("pool", "duplicate", "duplicate", "earlier", "invalid"),
        keys=("shared", "shared", "shared", "shared", None),
        formulas=("",) * 5,
        generated=np.array([False, True, True, True, True]),
        parent_sid=np.array([-1, -1, 2, 3, 4], np.int64),
        similarities=np.array([0, 0.2, 0.3, 0.4, 0.5]),
        combos=(None, ("a",), None, ("a",), ("b",)),
        analog_sid=np.zeros(0, np.int64),
        model_score=np.zeros(5),
        parent_smiles=(),
        logs=tuple(
            DerivationLog(
                parent=7,
                similarity=0.75,
                combo=combo,
                kind=kind,
                how=how,
                identity=identity,
            )
            for combo, kind, how, identity in routes
        ),
    )
    a, b = (7, 0.75, ("a",)), (7, 0.75, ("b",))
    assert supports(context, proposals=proposals) == [
        [a, b, a],
        [a, b, a],
        [a, b, a],
        [(3, 0.4, ("a",)), b, a],
        [(4, 0.5, ("b",)), a],
    ]
    with pytest.raises(ValueError, match="zip"):
        _ = supports(replace(context, keys=()), proposals=proposals)
    invalid = DerivationLog(
        parent=0,
        similarity=1.0,
        combo=("a",),
        kind="gen",
        how="smi",
        identity=None,
    )
    with pytest.raises(ValueError, match=r"^Expected log.identity is not None\.$"):
        _ = supports(replace(context, logs=(invalid,)), proposals=proposals)


def test_group_rejects_invalid_paths_preserves_empty_products_and_evicts_only_after5000() -> (
    None
):
    proposals = PreparedPaths()
    row: ProposalPath = {
        "raw": "",
        "standardized": "",
        "key": None,
        "skeleton": "",
        "environments": ((-1, -1, -1),),
    }
    serialization_failure: ProposalPath = {**row, "raw": None}
    standardization_failure: ProposalPath = {**row, "standardized": None}
    proposals.rows["", ("a",)] = (serialization_failure, standardization_failure, row)
    priors = DerivationPriors(
        {"glob": 0.2, "tf": {}, "env": {1: {}, 2: {}, 3: {}}},
        proposals=proposals,
    )
    assert tuple(priors.group("", ("a",))) == ("",)
    first = priors.group("", ("a",))
    for index in range(5_000):
        proposals.rows[str(index), ("a",)] = ()
        assert priors.group(str(index), ("a",)) == {}
    assert priors.group("", ("a",)) is first
    proposals.rows["last", ("a",)] = ()
    assert priors.group("last", ("a",)) == {}
    assert priors.group("", ("a",)) == first
    assert priors.group("", ("a",)) is not first


def test_prior_asset_pin_preserves_all_tables_and_metadata_without_runtime_objects(
    tmp_path: Path,
) -> None:
    proposals = PreparedPaths()
    tables: PriorTables = {
        "glob": 0.2,
        "tf": {"a": (3.0, 30.0)},
        "env": {1: {("a", 4): (2.0, 20.0)}, 2: {}, 3: {}},
        "meta": {"description": "complete tiny table", "rows": 2},
    }
    path = tmp_path / "prior.pkl"
    path.write_bytes(pickle.dumps(tables))
    seal = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = DerivationPriors.load(
        path,
        expected_sha256=seal,
        proposals=proposals,
    )
    assert loaded.tables == tables
    assert loaded.proposals is proposals
    assert loaded.p_site("a", (4, 5, 6)) == (
        2.0 + 2.0 * ((3.0 + 2.0 * 0.2) / (30.0 + 2.0))
    ) / (20.0 + 2.0)
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = DerivationPriors.load(
            path,
            expected_sha256="invalid",
            proposals=proposals,
        )
    path.write_bytes(pickle.dumps(Path("forbidden-runtime-object")))
    with pytest.raises(pickle.UnpicklingError, match="plain data"):
        _ = DerivationPriors.load(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            proposals=proposals,
        )


def test_complete_paths_keep_invalids_multiplicity_zero_pairs_and_missing_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    header = {"format_version": 1, "rdkit": IDENTITY_RDKIT}
    _ = (tmp_path / "manifest.json").write_text(json.dumps(to_plain(header)))
    requests: list[dict[str, object]] = [
        {
            "parent_smiles": "parent",
            "combo": ["a", "b"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 2,
            "applications": "parent\ta\t32\t2",
        },
        {
            "parent_smiles": "empty",
            "combo": ["a"],
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 0,
            "applications": "empty\ta\t32\t0",
        },
    ]
    requests.append({**requests[0], "terminal_count": 0, "max_mols": 0})
    paths = [
        {
            "request": 0,
            "position": index,
            "raw": "raw" if index == 0 else None,
            "standardized": "product" if index == 0 else None,
            "key": "key" if index == 0 else None,
            "skeleton": "skeleton" if index == 0 else None,
            "environments": [[-1, 7, 7], [2, 3, 4]],
        }
        for index in range(2)
    ]
    identities = [
        {"smiles": "candidate", "key": "key", "skeleton": "skeleton"},
        {"smiles": "candidate", "key": "later", "skeleton": "later"},
        {"smiles": "invalid", "key": None, "skeleton": None},
    ]
    parquet.write_table(Table.from_pylist(requests), tmp_path / "requests.parquet")
    parquet.write_table(Table.from_pylist(paths), tmp_path / "paths.parquet")
    parquet.write_table(Table.from_pylist(identities), tmp_path / "identities.parquet")
    table = DerivationTable(tmp_path)
    assert table.paths("parent", ("a", "b")) == (
        {
            "request": 0,
            "position": 0,
            "raw": "raw",
            "standardized": "product",
            "key": "key",
            "skeleton": "skeleton",
            "environments": ((-1, 7, 7), (2, 3, 4)),
        },
        {
            "request": 0,
            "position": 1,
            "raw": None,
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": ((-1, 7, 7), (2, 3, 4)),
        },
    )
    assert table.paths("empty", ("a",)) == ()
    assert table.identity("candidate") == ("key", "skeleton")
    assert table.identity("invalid") == (None, None)
    for call in (
        lambda: table.paths("unstaged", ("a",)),
        lambda: table.identity("unstaged"),
    ):
        with pytest.raises(KeyError, match="unstaged"):
            _ = call()
    for altered in (
        [{**requests[0], "terminal_count": 1}, requests[1]],
        [requests[1]],
    ):
        parquet.write_table(Table.from_pylist(altered), tmp_path / "requests.parquet")
        with pytest.raises(
            ValueError,
            match=r"^Derivation terminal path layout differs from its headers\.$",
        ):
            _ = DerivationTable(tmp_path)
    parquet.write_table(Table.from_pylist(requests), tmp_path / "requests.parquet")
    parquet.write_table(
        Table.from_pylist([*paths, {**paths[0], "request": 99}]),
        tmp_path / "paths.parquet",
    )
    with pytest.raises(
        ValueError,
        match=r"^Derivation terminal path layout differs from its headers\.$",
    ):
        _ = DerivationTable(tmp_path)
    parquet.write_table(
        Table.from_pylist([{**row, "position": 0} for row in paths]),
        tmp_path / "paths.parquet",
    )
    with pytest.raises(
        ValueError,
        match=r"^Derivation terminal path layout differs from its headers\.$",
    ):
        _ = DerivationTable(tmp_path)
    for altered_header in (
        {"format_version": 2, "rdkit": IDENTITY_RDKIT},
        {"format_version": 1, "rdkit": SIMULATOR_RDKIT},
    ):
        _ = (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain(altered_header)),
        )
        with pytest.raises(
            ValueError,
            match=r"^Derivation data requires format 1 and RDKit 2026\.03\.3\.$",
        ):
            _ = DerivationTable(tmp_path)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": True, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ValueError, match="cannot read True as int"):
        _ = DerivationTable(tmp_path)


def test_complete_main_public_pipeline_all_families_and_four_rankers(
    tmp_path: Path,
) -> None:
    assert_main_pipeline(tmp_path)


def test_selected_features_preserve_duplicates_order_dtype_and_empty_rows() -> None:
    values = {
        "base": np.array([3, 7, 3], np.float32),
        "family__value": np.array([11, 13, 11], np.float32),
    }
    result = select_features(values, names=("family__value", "base"), count=3)
    np.testing.assert_array_equal(result, [[11, 3], [13, 7], [11, 3]])
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    empty = select_features({"x": np.zeros(0, np.float32)}, names=("x",), count=0)
    assert empty.shape == (0, 1)
    no_columns = select_features({}, names=(), count=2)
    assert no_columns.shape == (2, 0)
    assert no_columns.dtype == np.float32
    with pytest.raises(ValueError, match=r"missing.*absent"):
        select_features(values, names=("absent",), count=3)
    with pytest.raises(
        ValueError,
        match=r"^Feature vectors differ from the candidate positions\.$",
    ):
        select_features(values, names=("base",), count=2)
    with pytest.raises(ValueError, match=r"^Every feature must be one-dimensional\.$"):
        # select_features must reject a column matrix in place of a flat vector.
        select_features({"x": np.zeros((3, 1), np.float32)}, names=("x",), count=3)


def test_representatives_keep_the_richest_spectrum_per_structure_mode_and_library() -> (
    None
):
    library, _, _ = retrieval_inputs()
    # Rows 0 and 1 share structure, mode and library; row 1 has more peaks.
    expected = [8, 10, 2, 1, 12, 4, 3, 5, 6, 7, 9, 11]
    for merge in (PolarityMerge.Config(), None):
        config = MainRetrieval.Config()
        config.merge = merge
        assert config.make().representatives(library).tolist() == expected
    count = 37
    ties = replace(
        library,
        sid=np.arange(count, dtype=np.int64)[::-1],
        mode=np.array([-1, 0, 1], dtype=np.int8)[np.arange(count) % 3],
        library=(np.arange(count) % 11).astype(np.int8),
        peak_count=(np.arange(count) % 4 + 2).astype(np.int32),
        mass=np.full(count, 100.0, dtype=np.float64),
    )
    # Every row is its own group; equal masses keep structure order.
    representatives = MainRetrieval.Config().make().representatives(ties)
    assert representatives.tolist() == list(range(count))[::-1]
    empty = replace(
        ties,
        sid=ties.sid[:0],
        mode=ties.mode[:0],
        library=ties.library[:0],
        peak_count=ties.peak_count[:0],
        mass=ties.mass[:0],
    )
    assert MainRetrieval.Config().make().representatives(empty).shape == (0,)


def test_analogs_keep_strict_best_structures_and_every_exclusion() -> None:
    library, _, spectra = retrieval_inputs()
    for merge in (PolarityMerge.Config(), None):
        config = MainRetrieval.Config()
        config.n_analog = 2
        config.merge = merge
        search = config.make()
        analogs = search.analogs(spectra, library=library, target=100.0)
        assert 0 < len(analogs) <= 2
        assert len({analog.sid for analog in analogs}) == len(analogs)
        similarity = [analog.similarity for analog in analogs]
        assert similarity == sorted(similarity, reverse=True)
        # Row 9 (structure 5, library 5, 110 Da) copies the positive query.
        assert analogs[0] == Analog(
            sid=5,
            similarity=1.0,
            shift=-10.0,
            natural_product=True,
        )
        for analog in analogs:
            # Shifts move each structure to the target.
            assert analog.shift == 100.0 - library.structure_mass[analog.sid]
        withheld = search.analogs(
            spectra,
            library=library,
            target=100.0,
            excluded_sid=analogs[0].sid,
        )
        assert analogs[0].sid not in {analog.sid for analog in withheld}
        masked = search.analogs(
            spectra,
            library=library,
            target=100.0,
            excluded=np.ones(13, np.bool_),
        )
        assert masked == ()
        other_library = search.analogs(
            spectra,
            library=library,
            target=100.0,
            excluded_sid=analogs[0].sid,
            excluded_library=99,
        )
        assert other_library == analogs
        assert search.analogs(spectra, library=library, target=900.0) == ()
        assert search.analogs(spectra[2:], library=library, target=100.0) == ()
    partition_library = replace(library, library=library.library.copy())
    partition_library.library[0] = 1
    config = MainRetrieval.Config()
    config.n_analog = 2
    partition = config.make()
    representatives = partition.representatives(partition_library)
    assert (
        len(representatives[partition_library.mode[representatives] == 1])
        > 4 * config.n_analog
    )
    assert (
        0
        < len(partition.analogs(spectra, library=partition_library, target=100.0))
        <= 2
    )


def test_analogs_keep_the_first_equal_row_and_replace_on_a_better_one() -> None:
    # Structure 0 has two copies of the query; structure 1 a partial match, then one.
    library = SpectralLibrary(
        offsets=np.arange(0, 9, 2, dtype=np.int64),
        mz=np.array([20, 30, 20, 30, 20, 50, 20, 30], np.float32),
        intensity=np.tile(np.array([1, 2], np.float32), 4),
        structure_mass=np.array([100.0, 100.0]),
        keys=("first", "second"),
        smiles=("C", "CC"),
        sid=np.array([0, 0, 1, 1], np.int64),
        # Library 0 is synthetic; libraries 2 and 10 are natural products.
        library=np.array([0, 2, 0, 10], np.int8),
        mode=np.ones(4, np.int8),
        instrument=np.zeros(4, np.int8),
        adduct=np.zeros(4, np.int16),
        precursor=np.full(4, 101.0),
        peak_count=np.full(4, 2, np.int32),
        mass=np.full(4, 100.0),
        order=np.arange(4, dtype=np.int64),
        mass_sorted=np.full(4, 100.0),
    )
    query: SpectrumRequest = {
        "mz": [20.0, 30.0],
        "it": [1.0, 2.0],
        "mode": 1,
        "prec": 101.0,
        "adduct": "[M+H]+",
    }
    analogs = (
        MainRetrieval.Config().make().analogs([query], library=library, target=100.0)
    )
    assert [(analog.sid, analog.natural_product) for analog in analogs] == [
        (0, False),
        (1, True),
    ]
    assert analogs[0].similarity == analogs[1].similarity
    assert [analog.shift for analog in analogs] == [0.0, 0.0]


def generated_candidates() -> list[GeneratedCandidate]:
    """Return an alias of pool key a and a new structure, both near 100 Da."""
    return [
        GeneratedCandidate(
            smiles="C",
            key="a",
            formula="CH4",
            mass=100.0,
            heavy_atoms=1,
            fingerprint=np.arange(7, dtype=np.float32) % 2,
            fragments=np.array([12, 13, 14], dtype=np.float64),
            parent_similarity=0.75,
            steps=1,
            parent_tanimoto=0.5,
        ),
        GeneratedCandidate(
            smiles="CC",
            key="new",
            formula="C2H6",
            mass=100.001,
            heavy_atoms=2,
            fingerprint=np.equal(np.arange(7) % 3, 0).astype(np.float32),
            fragments=np.array([22, 23], dtype=np.float64),
            parent_similarity=0.5,
            steps=2,
            parent_tanimoto=0.25,
        ),
    ]


def test_candidates_append_products_to_the_window_or_its_fallback() -> None:
    _, pool, _ = retrieval_inputs()
    generated = generated_candidates()
    rows = candidates(
        pool,
        target=100.0,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=generated,
    )
    identities = rows.candidates
    assert identities.smiles == ("C", "CC", "C", "C", "CC")
    assert identities.keys == ("a", "b", "a", "a", "new")
    assert identities.formulas == ("CH4", "C2H6", "CH4", "CH4", "C2H6")
    assert identities.sources == ("pool", "pool", "pool", "generated", "generated")
    assert identities.pool_rows.tolist() == [0, 1, 2, -1, -1]
    assert identities.generated.tolist() == [False] * 3 + [True] * 2
    assert rows.mass.tolist() == [99.999, 100.0, 100.001, 100.0, 100.001]
    assert rows.heavy_atoms.tolist() == [1, 2, 1, 1, 2]
    assert rows.train_sid.tolist() == [0, 1, 2, -1, -1]
    # Packed rows 17 and 33 unpack to bits 3 and 2 of the first seven.
    assert rows.fingerprints.tolist() == [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
    ]
    assert rows.fingerprints.dtype == np.float32
    assert rows.fragments[:3] == (None, None, None)
    assert [row.tolist() for row in rows.fragments[3:] if row is not None] == [
        [12, 13, 14],
        [22, 23],
    ]
    assert rows.parent_similarity.tolist() == [0, 0, 0, 0.75, 0.5]
    assert rows.steps.tolist() == [0, 0, 0, 1, 2]
    assert rows.parent_tanimoto.tolist() == [0, 0, 0, 0.5, 0.25]
    # No pool mass lies within 10 ppm of 100.004 Da; 30 ppm reaches rows 2 and 3.
    fallback = candidates(pool, target=100.004, ppm=10.0, fallback_ppm=30.0)
    assert fallback.candidates.pool_rows.tolist() == [2, 3]
    dropped = candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0, drop_row=0)
    assert dropped.candidates.pool_rows.tolist() == [1, 2]
    distant = candidates(pool, target=900.0, ppm=10.0, fallback_ppm=30.0)
    assert distant.mass.shape == (0,)
    assert distant.fingerprints.shape == (0, 7)
    only = candidates(
        pool,
        target=900.0,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=generated,
    )
    assert only.candidates.sources == ("generated", "generated")


def test_candidate_metadata_retains_each_admitted_parent_and_path() -> None:
    _, pool, _ = retrieval_inputs()
    product = GeneratedCandidate(
        smiles="CCOC",
        key="generated",
        formula="C3H8O",
        mass=100.0,
        heavy_atoms=4,
        fingerprint=np.zeros(7, np.float32),
        fragments=np.array([15.0, 30.0]),
        parent_similarity=0.75,
        steps=1,
        parent_tanimoto=0.5,
        parent_sid=7,
        combo=("methyl_O",),
        raw="COCC",
    )
    rows = candidates(
        pool,
        target=100.0,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=(product, product),
    )
    np.testing.assert_array_equal(rows.parent_sid, [-1, -1, -1, 7, 7])
    assert rows.combos == (None, None, None, ("methyl_O",), ("methyl_O",))
    assert rows.raw == (None, None, None, "COCC", "COCC")
    empty = candidates(pool, target=900.0, ppm=10.0, fallback_ppm=30.0)
    assert empty.parent_sid.shape == (0,)
    assert empty.parent_sid.dtype == np.int64
    assert empty.combos == empty.raw == ()


def full_bank(directory: Path) -> MainBank:
    """Pin the seeded Full1 checkpoint with its released merged-spectrum rules."""
    path = directory / "full1.pt"
    torch.save(source_checkpoint(index=0), path)
    config = MainBank.Config()
    config.checkpoints = [checkpoint_config(path, index=0)]
    config.merged_adduct = modal_adduct
    config.single_count = keep_count
    with torch.random.fork_rng(devices=[]):
        return config.make()


def _base(
    spectra: Sequence[SpectrumRequest],
    *,
    target: float = 100.0,
    bank: MainBank | None = None,
    fragments: int | None = 600,
    excluded: NDArray[np.bool_] | None = None,
    generated: Sequence[GeneratedCandidate] = (),
) -> MainBase:
    library, pool, _ = retrieval_inputs()
    config = MainEngine.Config()
    if fragments is None:
        config.fragments = config.subformulas = None
    else:
        assert config.fragments is not None
        config.fragments.max_candidates = fragments
    return config.make().base(
        spectra,
        target=target,
        library=library,
        pool=pool,
        train_fingerprints=(np.arange(9)[:, None] * 17 % 251).astype(np.uint8),
        bank=bank,
        excluded=excluded,
        generated=generated,
        merged_counts=[0, 12, 2][: len(spectra)],
    )


def test_main_base_binds_library_analog_model_and_fragment_columns(
    tmp_path: Path,
) -> None:
    _, _, spectra = retrieval_inputs()
    names = main_feature_names()
    bank = full_bank(tmp_path)
    # A product holding both pool bits 2 and 3 scores their summed logits.
    product = replace(
        generated_candidates()[0],
        fingerprint=np.array([0, 0, 1, 1, 0, 0, 0], np.float32),
    )
    base = _base(spectra, bank=bank, generated=(product,))
    rows = base.rows
    assert base.features.shape == (4, 94)
    assert base.features.dtype == np.float32
    np.testing.assert_array_equal(rows.candidates.pool_rows, [0, 1, 2, -1])
    _, scores = _main_model_features(
        rows,
        views=bank.views(spectra, merged_counts=[0, 12, 2]),
    )
    assert scores[0] == scores[2]
    assert scores[3] == np.float32(np.float64(scores[0]) + np.float64(scores[1]))
    np.testing.assert_array_equal(base.features[:, names.index("fz")], scores / 100)
    diagnostics = base.diagnostics
    assert (diagnostics.n_cand, diagnostics.n_gen) == (4, 1)
    assert diagnostics.fzmax == scores.max()
    assert diagnostics.lib_max == base.features[:, names.index("lib_max")].max()
    assert base.analogs
    assert diagnostics.top_sim == float(np.float32(base.analogs[0].similarity))
    # Library evidence is a similarity in [0, 1]; its mean divides by every query.
    library = base.features[:, names.index("lib_max")]
    assert np.all((library >= 0) & (library <= 1))
    assert np.all(base.features[:, names.index("lib_mean")] <= library)
    assert library[3] == 0
    assert base.features[:, names.index("n_query")].tolist() == [3.0] * 4
    assert base.features[:, names.index("n_pos")].tolist() == [2.0] * 4
    assert base.features[:, names.index("is_gen")].tolist() == [0, 0, 0, 1]
    assert (
        base.features[:, names.index("target_mass")].tolist() == [np.float32(0.1)] * 4
    )
    # Pool rows 0 and 2 and the product share CH4; pool row 1 holds C2H6.
    assert base.features[:, names.index("n_formula")].tolist() == [2.0] * 4
    assert base.features[:, names.index("formula_share")].tolist() == [
        np.float32(3 / 4),
        np.float32(1 / 4),
        np.float32(3 / 4),
        np.float32(3 / 4),
    ]
    # Every pool row holds two fragments; the product brings three.
    assert base.features[:, names.index("fr_n")].tolist() == [
        np.float32(np.log1p(2.0)),
        np.float32(np.log1p(2.0)),
        np.float32(np.log1p(2.0)),
        np.float32(np.log1p(3.0)),
    ]
    unscored = _base(spectra)
    assert not unscored.features[:, names.index("fz")].any()
    assert unscored.diagnostics.fzmax == 0.0
    np.testing.assert_array_equal(
        unscored.features[:, names.index("lib_max")],
        library[:3],
    )
    capped = _base(spectra, bank=bank, fragments=2)
    # Only the two strongest f.z candidates count fragments; ties keep row order.
    counted = capped.features[:, names.index("fr_n")] > 0
    assert np.flatnonzero(~counted).tolist() == [1 if scores[0] > scores[1] else 2]
    disabled = _base(spectra, bank=bank, fragments=None)
    for name in ("fr_int", "fr_n", "sf_int", "sf_cnt", "formula_is_top"):
        assert not disabled.features[:, names.index(name)].any(), name
    silent = _base([], fragments=None)
    assert silent.features[:, names.index("n_query")].tolist() == [0.0] * 3
    # Library rows 0-2 are every spectrum of structure 0, the first pool row.
    withheld = _base(spectra, excluded=np.isin(np.arange(13), [0, 1, 2, 5]))
    assert 0 not in {analog.sid for analog in withheld.analogs}
    assert withheld.features[0, names.index("lib_max")] == 0
    empty = _base(spectra, target=900.0, bank=bank)
    assert empty.features.shape == (0, 94)
    assert empty.diagnostics.n_cand == 0


def test_ion_deltas_follow_the_adduct_polarity_and_named_carriers() -> None:
    proton = adduct_parameters("[M+H]+")[2]
    deprotonation = adduct_parameters("[M-H]-")[2]
    assert main_ion_deltas("[M+H]+") == (proton,)
    assert main_ion_deltas("unknown+") == (proton,)
    assert main_ion_deltas("unknown") == (deprotonation,)
    assert main_ion_deltas("[M-H]-") == (deprotonation,)
    for adduct in ("[M+Na]+", "[M+K]+", "[M+NH4]+"):
        assert main_ion_deltas(adduct) == (proton, adduct_parameters(adduct)[2])
    assert main_ion_deltas("[M+Cl]-") == (
        deprotonation,
        adduct_parameters("[M+Cl]-")[2],
    )
    carrier = 12.0 + 2 * 1.00782503207 + 2 * 15.9949146196 - proton
    for adduct in ("[M+CH2O2-H]-", "[M+HCOO]-"):
        assert main_ion_deltas(adduct) == (deprotonation, carrier)


def test_fragment_channels_count_shifted_and_strict_matches_by_hand() -> None:
    proton = main_ion_deltas("[M+H]+")[0]
    hydrogen = 1.00782503207
    masses = np.array([10.0, 20.0, 30.0 - 2 * hydrogen, 9000.0])
    peaks = [
        (np.array([10.0, 20.0, 30.0, 40.0]) + proton, np.ones(4), "[M+H]+"),
        (np.array([9000.0, 10.0]) + proton, np.array([1.0, 0.25]), "[M+H]+"),
    ]
    # Three of four equal peaks match, one only through a two-hydrogen shift.
    features = main_fragment_features(masses, peaks=peaks)
    assert features.dtype == np.float32
    assert features.tolist() == [1.0, 0.875, 1.0, 1.0, 1.0]
    first = main_fragment_features(masses, peaks=peaks[:1])
    assert first.tolist() == [0.75, 0.75, 0.75, 0.75, 0.5]
    for empty in (
        main_fragment_features(np.zeros(0), peaks=peaks),
        main_fragment_features(masses, peaks=[]),
    ):
        assert empty.tolist() == [0.0] * 5


def test_subformulas_enumerate_bounded_hydrogens_and_score_their_peaks() -> None:
    hydrogen = 1.00782503207
    counts = engine_formula_vector("CH4")
    assert counts is not None
    complete = [
        hydrogen,
        2 * hydrogen,
        12.0,
        *(12.0 + h * hydrogen for h in range(1, 5)),
    ]
    assert main_subformula_masses(counts, max_out=2_000_000).tolist() == complete
    # At the cap the masses keep their enumeration order, unsorted.
    assert main_subformula_masses(counts, max_out=3).tolist() == [
        hydrogen,
        2 * hydrogen,
        12.0,
    ]
    proton = main_ion_deltas("[M+H]+")[0]
    peaks = [(np.array([12.0 + 3 * hydrogen + proton, 50.0]), np.ones(2), "[M+H]+")]
    features = main_subformula_features("CH4", peaks=peaks, max_out=2_000_000)
    assert features.tolist() == [0.5, 0.5, 0.5, 0.5]
    for formula, query in (("NaCl", peaks), ("CH4", peaks[:0])):
        assert (
            main_subformula_features(formula, peaks=query, max_out=10).tolist()
            == [
                0.0,
            ]
            * 4
        )


def test_main_engine_empty_candidates_preserve_all94_columns() -> None:
    library, pool, spectra = retrieval_inputs()
    evidence = (
        MainEngine.Config()
        .make()
        .base(
            spectra,
            target=900.0,
            library=library,
            pool=pool,
            # MainEngine.base consumes seven selected bits packed into one byte.
            train_fingerprints=np.zeros((9, 1), np.uint8),
        )
    )
    assert evidence.features.shape == (0, 94)
    assert len(evidence.rows.candidates.smiles) == 0


def test_library_aggregation_skips_empty_queries_but_retains_query_denominator() -> (
    None
):
    library, pool, _ = retrieval_inputs()
    library = replace(
        library,
        offsets=np.array([0, 2], np.int64),
        mz=np.array([20, 30], np.float32),
        intensity=np.array([1, 1], np.float32),
        sid=np.array([0], np.int64),
        library=np.array([2], np.int8),
        mode=np.array([1], np.int8),
        adduct=np.array([0], np.int16),
        precursor=np.array([101.0]),
        peak_count=np.array([2], np.int32),
        mass=np.array([100.0]),
        order=np.array([0], np.int64),
        mass_sorted=np.array([100.0]),
    )
    rows = candidates(pool, target=100.0, ppm=10.0, fallback_ppm=30.0)
    position = np.flatnonzero(np.equal(rows.train_sid, 0)).item()
    search = MainSearch.Config().make()
    store = PeakStore(
        offsets=library.offsets,
        mz=library.mz,
        intensity=library.intensity,
    )

    def request(
        mass: list[float],
        intensity: list[float],
        mode: int | str,
    ) -> SpectrumRequest:
        return {
            "mz": mass,
            "it": intensity,
            "prec": 101.0,
            "mode": mode,
            "adduct": "[M+H]+",
        }

    queries = [
        request([], [], 1),
        request([20, 30], [1, 1], -1),
        request([20, 30], [1, 1], "positive"),
        request([20], [1], 1),
        request([30], [1], 1),
        request([20, 30], [2, 1], 1),
        request([70, 80], [1, 1], 1),
    ]
    scores = np.array(
        [
            search.score(
                (np.array(query["mz"]), np.array(query["it"])),
                store=store,
                candidates=np.array([0], np.int64),
            ).item(0)
            for query in queries[2:6]
        ],
    )

    def evidence(query: list[SpectrumRequest]) -> MainLibraryEvidence:
        return _main_library(
            query,
            rows=rows,
            library=library,
            target=100.0,
            search=search,
            ppm=10.0,
            excluded=None,
        )

    combined = evidence(queries)
    assert combined.mean.item(position) == sum(
        scores.item(i) for i in range(len(scores))
    ) / len(queries)
    assert combined.top3.item(position) == np.mean(np.sort(scores)[-3:])
    positive = evidence(queries[2:3])
    assert positive.mean.item(position) == scores.item(0)
    empty_reference = replace(
        library,
        offsets=np.array([0, 2, 2], np.int64),
        sid=np.array([0, 0], np.int64),
        library=np.array([2, 2], np.int8),
        mode=np.array([1, -1], np.int8),
        adduct=np.array([0, 6], np.int16),
        mass=np.array([100.0, 100.0]),
        order=np.array([0, 1], np.int64),
        mass_sorted=np.array([100.0, 100.0]),
    )
    zero = _main_library(
        [queries[2], queries[1]],
        rows=rows,
        library=empty_reference,
        target=100.0,
        search=search,
        ppm=10.0,
        excluded=None,
    )
    assert zero.top3.item(position) == scores.item(0)
    assert zero.mean.item(position) == scores.item(0) / 2


def test_complete_depth_pair_reader_preserves_duplicates_and_invalid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _fragments_write(tmp_path)
    table = MainFragmentTable(tmp_path)
    rows: tuple[tuple[int, tuple[list[int], list[int]]], ...] = (
        (0, ([1, 3], [3])),
        (1, ([], [])),
        (2, ([4], [1, 4])),
    )
    for row, expected in rows:
        for actual, values in zip(table.row(row), expected, strict=True):
            np.testing.assert_array_equal(actual, values)
            assert actual.dtype == np.float32
            assert not actual.flags.writeable
    for actual, values in zip(table.for_smiles("same"), ([1, 3], [3]), strict=True):
        np.testing.assert_array_equal(actual, values)
    assert all(len(row) == 0 for row in table.for_smiles("invalid"))
    for index in (-1, 3):
        with pytest.raises(IndexError) as caught:
            _ = table.row(index)
        assert caught.value.args == (index,)
    with pytest.raises(KeyError, match="absent"):
        _ = table.for_smiles("absent")


def test_depth_pair_reader_rejects_wrong_types_offsets_and_provenance(
    tmp_path: Path,
) -> None:
    _fragments_write(tmp_path)
    for name, malformed, error in (
        ("frag_mass", np.array([1, 3, 4], dtype=np.float64), TypeError),
        # MainFragmentTable requires a flat mass vector, not a column matrix.
        ("f1_mass", np.zeros((3, 1), dtype=np.float32), ValueError),
        ("frag_off", np.array([0, 2, 2], dtype=np.int64), ValueError),
        ("frag_off", np.array([1, 2, 2, 3], dtype=np.int64), ValueError),
        ("frag_off", np.array([0, 2, 2, 4], dtype=np.int64), ValueError),
        ("f1_off", np.array([0, 2, 1, 3], dtype=np.int64), ValueError),
        ("f1_off", np.array([0, 1, 1, 3], dtype=np.int32), TypeError),
    ):
        path = tmp_path / f"{name}.npy"
        correct = cast("NDArray[np.generic]", np.load(path, allow_pickle=False))
        np.save(path, malformed, allow_pickle=False)
        with pytest.raises(error, match=r"Main fragment|expected"):
            _ = MainFragmentTable(tmp_path)
        np.save(path, correct, allow_pickle=False)
    for version, release in ((2, IDENTITY_RDKIT), (1, SIMULATOR_RDKIT)):
        _ = (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain({"format_version": version, "rdkit": release})),
        )
        with pytest.raises(ValueError, match="Main fragment"):
            _ = MainFragmentTable(tmp_path)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": True, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ReadError):
        _ = MainFragmentTable(tmp_path)


def test_main_base_rows_are_distinct_readonly_float64_and_missing_data_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    _fragments_write(tmp_path)
    values = np.array([1.000001, 3.000001, 4.000001], dtype=np.float64)
    np.save(tmp_path / "base_mass.npy", values, allow_pickle=False)
    table = MainFragmentTable(tmp_path)
    row = table.base_for_smiles("same")
    np.testing.assert_array_equal(row, values[:2])
    assert row.dtype == np.float64
    assert not row.flags.writeable
    assert row[0] != table.for_smiles("same")[0][0]
    assert len(table.base_for_smiles("invalid")) == 0
    with pytest.raises(KeyError, match="absent"):
        _ = table.base_for_smiles("absent")
    for wrong, error in (
        (values.astype(np.float32), TypeError),
        (values[:2], ValueError),
    ):
        np.save(tmp_path / "base_mass.npy", wrong, allow_pickle=False)
        with pytest.raises(error, match=r"Main fragment|expected"):
            _ = MainFragmentTable(tmp_path).base_for_smiles("same")
    (tmp_path / "base_mass.npy").unlink()
    with pytest.raises(FileNotFoundError):
        _ = MainFragmentTable(tmp_path).base_for_smiles("same")


def _fragments_write(directory: Path) -> None:
    parquet.write_table(
        Table.from_pydict({"smiles": ["same", "invalid", "same"]}),
        directory / "molecules.parquet",
    )
    for name, values in (
        ("frag_off", np.array([0, 2, 2, 3], dtype=np.int64)),
        ("frag_mass", np.array([1, 3, 4], dtype=np.float32)),
        ("f1_off", np.array([0, 1, 1, 3], dtype=np.int64)),
        ("f1_mass", np.array([3, 1, 4], dtype=np.float32)),
    ):
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


def test_complete_family_positions_keep_pool_duplicates_generated_aliases_and_sids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    data, rows = _inputs(tmp_path)
    provider: MainFamilyProvider = data
    result = provider.select(
        rows,
        (Analog(sid=2, similarity=0.9), Analog(sid=0, similarity=0.8)),
    )
    assert [row.hashes.item(0) for row in result.candidate_structures] == [
        1,
        2,
        3,
        7,
        8,
    ]
    assert [row.hashes.item(0) for row in result.analog_structures] == [6, 4]
    assert result.train_smiles == ("parent0", "parent1", "parent0")
    for actual, expected in zip(
        result.depth1,
        ([10, 11], [12, 13], [14, 15], [70, 71], list[int]()),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
    for position, actual in enumerate(result.depth2):
        expected = (
            data.pool.fragments(position)
            if position < 3
            else data.generated_fragments.row(position - 3)[0].astype(np.float64)
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
    np.testing.assert_array_equal(
        result.capabilities,
        np.repeat(np.array([1, 2, 3, 7, 0], np.uint8)[:, None], 22, axis=1),
    )
    assert result.capabilities.dtype == np.uint8
    np.testing.assert_array_equal(
        result.candidate_structures[3].formula,
        family_data_formula_vector("C2H6O"),
    )
    assert result.fragnet[-1] is None
    assert [row["mass"] for row in result.fragnet[:-1] if row is not None] == [2] * 4
    assert result.priors is data.priors
    np.testing.assert_array_equal(rows.parent_sid, [-1, -1, -1, 2, 1])
    assert rows.combos == (None, None, None, ("step0",), ("step1",))
    assert rows.raw == (None, None, None, "raw0", "raw1")
    empty = provider.select(
        candidates(data.pool, target=900, ppm=10.0, fallback_ppm=30.0),
        (),
    )
    assert empty.candidate_structures == empty.analog_structures == ()
    assert empty.depth1 == empty.depth2 == empty.fragnet == ()
    assert empty.capabilities.shape == (0, 22)
    assert empty.train_smiles == result.train_smiles


def test_unstaged_or_misaligned_family_evidence_is_an_error(tmp_path: Path) -> None:
    data, rows = _inputs(tmp_path)
    for sid in (-1, 3):
        with pytest.raises(IndexError) as error:
            _ = data.select(rows, (Analog(sid=sid, similarity=0.5),))
        assert error.value.args == (sid,)
    with pytest.raises(ValueError, match="training SID"):
        _ = replace(data, train_smiles=("parent1", "parent0", "parent0"))
    with pytest.raises(ValueError, match="pool positions"):
        _ = replace(data, pool=replace(data.pool, smiles=("wrong", "C", "CCO")))
    with pytest.raises(ValueError, match="capability vocabulary"):
        _ = replace(data, pool_families=replace(data.pool_families, names=("wrong",)))
    with pytest.raises(ValueError, match="capability vocabulary"):
        _ = replace(
            data,
            pool_families=replace(
                data.pool_families,
                names=data.pool_families.names[::-1],
            ),
        )
    with pytest.raises(ValueError, match="pool positions"):
        _ = replace(
            data,
            pool_families=replace(
                data.pool_families,
                offsets=np.array([0, 2, 4], np.int64),
                mass=data.pool_families.mass[:4],
                capabilities=data.pool_families.capabilities[:2],
            ),
        )
    changed = replace(
        rows,
        candidates=replace(
            rows.candidates,
            smiles=("wrong", *rows.candidates.smiles[1:]),
        ),
    )
    with pytest.raises(ValueError, match="pool identity"):
        _ = data.select(changed, ())
    changed = replace(
        rows,
        candidates=replace(
            rows.candidates,
            smiles=(*rows.candidates.smiles[:-1], "missing"),
        ),
    )
    with pytest.raises(KeyError, match="missing"):
        _ = data.select(changed, ())
    with pytest.raises(ValueError, match="generated positions"):
        _ = data.select(
            replace(
                rows,
                candidates=replace(rows.candidates, generated=np.zeros(5, bool)),
            ),
            (),
        )
    with pytest.raises(IndexError) as error:
        _ = data.select(replace(rows, parent_sid=np.array([-1, -1, -1, 3, 1])), ())
    assert error.value.args == (3,)
    with pytest.raises(KeyError, match="unstaged"):
        _ = data.select(replace(rows, raw=(*rows.raw[:-1], "unstaged")), ())
    with pytest.raises(KeyError, match="unstaged"):
        _ = data.select(replace(rows, combos=(*rows.combos[:-1], ("unstaged",))), ())
    with pytest.raises(ValueError, match="candidate positions"):
        _ = data.select(replace(rows, raw=()), ())
    with pytest.raises(IndexError) as error:
        _ = data.select(
            replace(
                rows,
                candidates=replace(
                    rows.candidates,
                    pool_rows=np.array([9, 1, 2, -1, -1]),
                ),
            ),
            (),
        )
    assert error.value.args == (9,)


def test_training_sid_reader_reorders_rows_and_rejects_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "structures.parquet"
    parquet.write_table(
        Table.from_pydict({"sid": [2, 0, 1], "smiles": ["alias", "alias", "other"]}),
        path,
    )
    assert training_smiles(path) == ("alias", "other", "alias")
    for indices in ([0, 0, 2], [0, 1, 3], [-1, 0, 1]):
        parquet.write_table(
            Table.from_pydict({"sid": indices, "smiles": ["a", "b", "c"]}),
            path,
        )
        with pytest.raises(
            ValueError,
            match=r"^Expected one structure per contiguous training SID\.$",
        ) as error:
            _ = training_smiles(path)
        assert error.value.args == (
            "Expected one structure per contiguous training SID.",
        )
    parquet.write_table(Table.from_pydict({"sid": [], "smiles": []}), path)
    assert training_smiles(path) == ()


def test_provided_pool_family_reader_checks_every_shape_dtype_and_offset(
    tmp_path: Path,
) -> None:
    offsets = np.array([0, 2, 2, 5], np.int64)
    mass = np.array([1, 2, 3, 4, 5], np.float32)
    capabilities = np.arange(66, dtype=np.uint8).reshape(3, 22)
    names = tuple(f"cap{index}" for index in range(22))
    for filename, array in (
        ("pool_f1_off.npy", offsets),
        ("pool_f1_mass.npy", mass),
        ("pool_cap.npy", capabilities),
    ):
        np.save(tmp_path / filename, array)
    _ = (tmp_path / "caps.txt").write_text("\n".join(names))
    result = MainPoolFamilies.load(tmp_path, count=3)
    np.testing.assert_array_equal(result.offsets, offsets)
    np.testing.assert_array_equal(result.mass, mass)
    np.testing.assert_array_equal(result.capabilities, capabilities)
    assert result.names == names
    assert not result.mass.flags.writeable
    with pytest.raises(ValueError, match="pool family count"):
        _ = MainPoolFamilies.load(tmp_path, count=2)
    for filename, wrong, error, message in (
        ("pool_f1_off.npy", np.array([0, 2, 1, 5], np.int64), ValueError, "pool"),
        ("pool_f1_off.npy", offsets.astype(np.int32), TypeError, "expected int64"),
        ("pool_f1_off.npy", offsets[:-1], ValueError, "pool"),
        ("pool_f1_off.npy", offsets + 1, ValueError, "pool"),
        ("pool_f1_mass.npy", mass.astype(np.float64), TypeError, "expected float32"),
        ("pool_f1_mass.npy", mass[None], ValueError, "pool"),
        ("pool_cap.npy", capabilities.astype(np.int16), TypeError, "expected uint8"),
        ("pool_cap.npy", capabilities[:-1], ValueError, "pool"),
        ("pool_cap.npy", capabilities[:, :-1], ValueError, "pool"),
    ):
        original = (tmp_path / filename).read_bytes()
        np.save(tmp_path / filename, wrong)
        with pytest.raises(error, match=message):
            _ = MainPoolFamilies.load(tmp_path, count=3)
        _ = (tmp_path / filename).write_bytes(original)


def _inputs(tmp_path: Path) -> tuple[MainFamilyData, CandidateRows]:
    pool = CandidatePool(
        mass=np.full(3, 100.0),
        keys=("a", "b", "a"),
        smiles=("CCO", "C", "CCO"),
        formulas=("C2H6O", "CH4", "C2H6O"),
        heavy_atoms=np.array([3, 1, 3], np.int32),
        source=np.zeros(3, np.int8),
        train_sid=np.array([0, 1, 2], np.int64),
        packed=np.array([[32], [64], [32]], np.uint8),
        bits=np.arange(3, dtype=np.int64),
        fragment_offsets=np.arange(4, dtype=np.int64) * 2,
        fragment_mass=np.arange(6, dtype=np.float32) + np.float32(0.123),
        key_to_pid={"a": 2, "b": 1},
    )
    names = tuple(f"cap{index}" for index in range(22))
    fragment_dir = tmp_path / "fragments"
    fragment_dir.mkdir()
    _header(fragment_dir)
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "bad"]}),
        fragment_dir / "molecules.parquet",
    )
    for name, array in (
        ("frag_off", np.array([0, 2, 2], np.int64)),
        ("f1_off", np.array([0, 2, 2], np.int64)),
        ("frag_mass", np.array([70.123, 71.123], np.float32)),
        ("f1_mass", np.array([70, 71], np.float32)),
    ):
        np.save(fragment_dir / f"{name}.npy", array)
    generated_fragments = MainFragmentTable(fragment_dir)
    intrinsic_dir = tmp_path / "intrinsic"
    intrinsic_dir.mkdir()
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "bad"]}),
        intrinsic_dir / "molecules.parquet",
    )
    np.save(intrinsic_dir / "records.npy", np.zeros((2, 34), np.float32))
    np.save(
        intrinsic_dir / "capabilities.npy",
        np.repeat(np.array([7, 0], np.uint8)[:, None], 22, axis=1),
    )
    _ = (intrinsic_dir / "record_names.txt").write_text("\n".join(record_fields()))
    _ = (intrinsic_dir / "capability_names.txt").write_text("\n".join(names))
    data = MainFamilyData(
        pool=pool,
        train_smiles=("parent0", "parent1", "parent0"),
        pool_structures=_analog(
            tmp_path / "pool",
            pool.smiles,
            pool.formulas,
            (1, 2, 3),
        ),
        train_structures=_analog(
            tmp_path / "train",
            ("parent0", "parent1", "parent0"),
            (None,) * 3,
            (4, 5, 6),
        ),
        generated_structures=_analog(
            tmp_path / "generated",
            ("CCO", "bad"),
            (None, None),
            (7, 8),
        ),
        pool_families=MainPoolFamilies(
            offsets=np.arange(4, dtype=np.int64) * 2,
            mass=np.arange(6, dtype=np.float32) + 10,
            capabilities=np.repeat(np.array([1, 2, 3], np.uint8)[:, None], 22, axis=1),
            names=names,
        ),
        generated_fragments=generated_fragments,
        generated_intrinsic=IntrinsicTable.load(intrinsic_dir),
        fragnet=_fragnet(tmp_path / "fragnet"),
        priors=DerivationPriors(
            {"glob": 0.2, "tf": {}, "env": {1: {}, 2: {}, 3: {}}},
            proposals=_FamilyDataProposals(),
        ),
    )
    generated = tuple(
        GeneratedCandidate(
            smiles=smiles,
            key="a" if index == 0 else "invalid",
            formula="C2H6O" if index == 0 else "",
            mass=100,
            heavy_atoms=3,
            fingerprint=np.zeros(3, np.float32),
            fragments=generated_fragments.row(index)[0].astype(np.float64),
            parent_similarity=0.7,
            steps=1,
            parent_tanimoto=0.5,
            parent_sid=2 - index,
            combo=(f"step{index}",),
            raw=f"raw{index}",
        )
        for index, smiles in enumerate(("CCO", "bad"))
    )
    return data, candidates(
        pool,
        target=100,
        ppm=10.0,
        fallback_ppm=30.0,
        generated=generated,
    )


def _analog(
    directory: Path,
    smiles: tuple[str, ...],
    formulas: tuple[str | None, ...],
    markers: tuple[int, ...],
) -> AnalogTable:
    directory.mkdir()
    _header(directory)
    parquet.write_table(
        Table.from_pylist(
            [
                {
                    "smiles": value,
                    "formula_input": formula,
                    "core_smiles": "",
                    "element_smiles": "",
                    "query_smiles": "",
                }
                for value, formula in zip(smiles, formulas, strict=True)
            ],
        ),
        directory / "molecules.parquet",
    )
    for name, array in (
        ("hashes", np.repeat(np.array(markers, np.int64)[:, None], 8, axis=1)),
        ("numbers", np.zeros((len(smiles), 8), np.int16)),
        (
            "formula",
            np.stack([family_data_formula_vector(value) for value in formulas]),
        ),
        ("core_mask", np.zeros(0, bool)),
    ):
        np.save(directory / f"{name}.npy", array)
    for prefix in ("graph", "core"):
        for name, array in (
            ("elements", np.zeros(0, np.int16)),
            ("bonds", np.zeros((0, 2), np.int32)),
            ("atom_off", np.zeros(len(smiles) + 1, np.int64)),
            ("bond_off", np.zeros(len(smiles) + 1, np.int64)),
        ):
            np.save(directory / f"{prefix}_{name}.npy", array)
    return AnalogTable(directory)


def _fragnet(directory: Path) -> FragnetTable:
    directory.mkdir()
    _header(directory)
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["CCO", "C", "bad"],
                "parsed": [True, True, False],
                "valid": [True, True, False],
            },
        ),
        directory / "molecules.parquet",
    )
    for name, array in (
        ("amass", np.ones(4, np.float64)),
        ("mass", np.array([2, 2, 0], np.float64)),
        ("atom_off", np.array([0, 2, 4, 4], np.int64)),
        ("bond_off", np.array([0, 1, 2, 2], np.int64)),
        ("bonds", np.array([[0, 1], [0, 1]], np.int32)),
        ("bring", np.zeros(2, np.int8)),
        *((name, np.zeros(4, np.int8)) for name in ("kind", "el", "nh")),
        ("el10", np.zeros(4, np.uint8)),
    ):
        np.save(directory / f"{name}.npy", array)
    return FragnetTable(directory)


def _header(directory: Path) -> None:
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


class _FamilyDataProposals:
    def paths(
        self,
        parent_smiles: str,
        combo: tuple[str, ...],
    ) -> tuple[ProposalPath, ...]:
        paths: dict[tuple[str, tuple[str, ...]], tuple[ProposalPath, ...]] = {
            ("parent0", ("step0",)): (),
            ("parent1", ("step1",)): (),
        }
        return paths[parent_smiles, combo]

    def identity(self, smiles: str) -> tuple[str | None, str | None]:
        if smiles not in ("CCO", "C", "bad", "raw0", "raw1"):
            raise KeyError(smiles)
        return (None, None) if smiles == "bad" else ("key", "skeleton")


def screen_analogs() -> tuple[Analog, ...]:
    """Rank four parents; the second falls below the 0.3 similarity floor."""
    return (
        Analog(sid=1, similarity=0.9),
        Analog(sid=0, similarity=0.2),
        Analog(sid=2, similarity=0.5),
        Analog(sid=3, similarity=0.3),
    )


@pytest.mark.parametrize(("parents", "chosen"), [(6, (1, 2, 3)), (2, (1, 2)), (0, ())])
def test_planned_requests_keep_parent_order_and_every_budget(
    parents: int,
    chosen: tuple[int, ...],
) -> None:
    config = ReactionScreen.Config()
    config.parents = parents
    train_mass = np.array([46.0, 60.0, 74.0, 1000.0])
    planned = config.make().plan(
        screen_analogs(),
        target=100.0,
        train_smiles=("p0", "p1", "p2", "p3"),
        train_mass=train_mass,
    )
    assert tuple(analog.sid for analog, _ in planned) == chosen
    for analog, request in planned:
        assert request == {
            "smiles": f"p{analog.sid}",
            "delta": 100.0 - train_mass.item(analog.sid),
            # The 10 ppm window at 100 Da is below the 0.004 Da floor.
            "tolerance": 0.004,
            "max_steps": 2,
            "max_products": 60,
            "allow_isomerization": True,
            "application_budget": 800,
            "stage_parents": 24,
            "first_products": 32,
            "later_products": 16,
        }
    wide = config.make().plan(
        screen_analogs()[:1],
        target=1000.0,
        train_smiles=("p0", "p1"),
        train_mass=np.array([0.0, 900.0]),
    )
    assert [request["tolerance"] for _, request in wide] == [0.01] * min(parents, 1)


class _Generation:
    """Serve every parent the same proposals and plain-table chemistry."""

    def __init__(self, products: Sequence[str]) -> None:
        self.products_ = tuple(products)
        self.requests: list[ReactionRequest] = []

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        self.requests.append(request)
        return tuple({"raw": raw, "combo": ("methyl_O",)} for raw in self.products_)

    def standardize(self, smiles: str) -> str | None:
        return smiles.upper()

    def key(self, smiles: str) -> str | None:
        return "key-" + smiles

    def properties(self, smiles: str) -> tuple[str, float, int] | None:
        return ("C3H8O", 100.0, len(smiles))

    def fingerprint(self, smiles: str) -> NDArray[np.uint8] | None:
        del smiles
        return np.packbits(np.array([1, 0, 1, 0, 0, 0, 0, 0], np.uint8))

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        return np.array([float(len(smiles))])


def test_generation_admits_each_key_once_logs_repeats_and_stops_at_the_total() -> None:
    table = _Generation(["ccoc", "CCOC", "coc"])
    config = ReactionScreen.Config()
    config.total = 3
    result = config.make().run(
        screen_analogs(),
        target=100.0,
        excluded={"key-COC"},
        train_smiles=("p0", "p1", "p2", "p3"),
        train_mass=np.array([46.0, 60.0, 74.0, 86.0]),
        train_fingerprints=np.packbits(np.eye(4, 8, dtype=np.uint8), axis=1),
        bits=np.arange(4, dtype=np.int64),
        proposals=table,
        chemistry=table,
        fragments=table,
    )
    # Parent 1 admits CCOC; its repeat and the pooled COC are logged instead.
    assert [candidate.smiles for candidate in result.candidates] == ["CCOC"]
    assert [(log.parent, log.kind, log.identity) for log in result.logs] == [
        (1, "gen", "key-CCOC"),
        (1, "pool", "key-COC"),
        (2, "gen", "key-CCOC"),
        (2, "gen", "key-CCOC"),
        (2, "pool", "key-COC"),
        (3, "gen", "key-CCOC"),
        (3, "gen", "key-CCOC"),
        (3, "pool", "key-COC"),
    ]
    (candidate,) = result.candidates
    assert (candidate.parent_sid, candidate.parent_similarity) == (1, 0.9)
    assert (candidate.raw, candidate.combo, candidate.heavy_atoms) == (
        "ccoc",
        ("methyl_O",),
        4,
    )
    # Product bits 0 and 2 against parent 1's bit 1 share nothing.
    assert candidate.fingerprint.tolist() == [1, 0, 1, 0]
    assert candidate.parent_tanimoto == 0.0
    assert candidate.fragments.tolist() == [4.0]
    assert len(table.requests) == 3
    config.total = 1
    stopped = config.make().run(
        screen_analogs(),
        target=100.0,
        excluded=set(),
        train_smiles=("p0", "p1", "p2", "p3"),
        train_mass=np.array([46.0, 60.0, 74.0, 86.0]),
        train_fingerprints=np.packbits(np.eye(4, 8, dtype=np.uint8), axis=1),
        bits=np.arange(4, dtype=np.int64),
        proposals=table,
        chemistry=table,
        fragments=table,
    )
    assert [candidate.smiles for candidate in stopped.candidates] == ["CCOC"]
    assert stopped.logs == ()
    assert len(table.requests) == 4


def test_seen_keys_are_logged_before_missing_properties_and_mass_checks() -> None:
    table = _Table({})
    result = (
        ReactionScreen.Config()
        .make()
        .run(
            [Analog(sid=0, similarity=0.75)],
            target=100.0,
            excluded={"seen"},
            train_smiles=("parent",),
            train_mass=np.array([0.0]),
            train_fingerprints=np.array([[0]], np.uint8),
            bits=np.array([0], np.int64),
            proposals=_SeenProducts(),
            chemistry=table,
            fragments=table,
        )
    )
    assert not result.candidates
    assert [
        (row.parent, row.similarity, row.combo, row.kind, row.how, row.identity)
        for row in result.logs
    ] == [
        (0, 0.75, ("first", "second"), "pool", "key", "seen"),
    ]


class _SeenProducts:
    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        assert request["smiles"] == "parent"
        return ({"raw": "seen", "combo": ("first", "second")},)


class _Table:
    """Serve one parent's proposals and every prepared decision from plain tables."""

    def __init__(self, masses: dict[str, float]) -> None:
        self.masses = masses

    def products(self, request: ReactionRequest) -> tuple[ReactionProduct, ...]:
        assert request["smiles"] == "parent"
        return tuple({"raw": raw, "combo": ("rule",)} for raw in self.masses)

    def standardize(self, smiles: str) -> str | None:
        return None if smiles == "unstandardizable" else smiles

    def key(self, smiles: str) -> str | None:
        return {"keyless": None, "seen": "seen"}.get(smiles, "key-" + smiles)

    def properties(self, smiles: str) -> tuple[str, float, int] | None:
        return ("C2H6O", self.masses[smiles], 3)

    def fingerprint(self, smiles: str) -> NDArray[np.uint8] | None:
        if smiles == "unprinted":
            return None
        return np.packbits(np.array([1, 1, 0, 1, 1, 0, 0, 0], np.uint8))

    def base_for_smiles(self, smiles: str) -> NDArray[np.float64]:
        return np.array([self.masses[smiles]])


def test_each_rejected_product_skips_only_itself_inside_the_exact_mass_window() -> None:
    def admit(
        masses: dict[str, float],
        *,
        tolerance: float,
        ppm: float,
        target: float,
    ) -> GenerationResult:
        config = ReactionScreen.Config()
        config.tolerance, config.ppm = tolerance, ppm
        table = _Table(masses)
        return config.make().run(
            [Analog(sid=0, similarity=0.9)],
            target=target,
            excluded={"seen"},
            train_smiles=("parent",),
            train_mass=np.array([0.0]),
            train_fingerprints=np.packbits(
                np.array([[1, 1, 1, 0, 0, 0, 0, 0]], np.uint8),
                axis=1,
            ),
            bits=np.arange(8),
            proposals=table,
            chemistry=table,
            fragments=table,
        )

    result = admit(
        {
            "unstandardizable": 100.0,
            "keyless": 100.0,
            "seen": 100.0,
            "far": 102.0,
            "edge": 100.5,
            "unprinted": 100.0,
            "good": 100.0,
        },
        tolerance=0.5,
        ppm=0.0,
        target=100.0,
    )
    assert [candidate.raw for candidate in result.candidates] == ["edge", "good"]
    assert [(row.kind, row.identity) for row in result.logs] == [("pool", "seen")]
    for candidate in result.candidates:
        assert candidate.parent_tanimoto == float(np.float32(0.4))
        assert (candidate.parent_sid, candidate.steps) == (0, 1)
        assert candidate.combo == ("rule",)
    relative = admit(
        {"inside": 1000.05, "outside": 1000.5},
        tolerance=0.0,
        ppm=100.0,
        target=1000.0,
    )
    assert [candidate.raw for candidate in relative.candidates] == ["inside"]


def pipeline_stage_a() -> StageA:
    """Back off from a shared mode/type table to a carbon-count table."""
    table = np.full((8, 6), 2.0, np.float32)
    return StageA(
        chains=((("mode", "ft"),), (("mode", "ft"), ("c1",))),
        tabs=(
            ((np.arange(8), table),),
            (
                (np.arange(8), table),
                (np.zeros(0, np.int64), np.zeros((0, 6), np.float32)),
            ),
        ),
        alpha=80.0,
    )


def pipeline_stage_b() -> str:
    """Split the Fragnet booster on its sixth column, the explained peak weight."""
    return model_text(stump(5, left=-1.0, right=1.0), features=stage_b_names())


RANKER_NAMES: Final = (
    *main_feature_names(),
    "derivation__dg_n_combo",
    "derivation__dp_hit",
    "frag2__disc_max",
    "analog_struct__st_ap",
    "model_views__fzA_gap",
    "fragnet__fn_score",
)


def ranker_texts(names: Sequence[str] = RANKER_NAMES) -> list[str]:
    """Serialize four three-stump rankers over the released feature names."""
    columns = [names.index(name) for name in ("fz", "lib_max", "derivation__dp_hit")]
    return [
        model_text(
            stump(columns[index % 3], left=-0.5, right=0.5 * (index + 1)),
            stump(columns[(index + 1) % 3], left=0.25, right=-0.25),
            stump(names.index("frag2__disc_max"), left=0.0, right=1.0),
            features=names,
        )
        for index in range(4)
    ]


def _fragnet_record(
    groups: Sequence[tuple[str, int]],
    bonds: Sequence[tuple[int, int]],
) -> FragnetRecord:
    """Weigh heavy atoms with their hydrogens; codes C=0 and O=2."""
    masses = ELEMENT_MASSES
    amass = np.array([masses[symbol] + count * masses["H"] for symbol, count in groups])
    codes = np.array([2 if symbol == "O" else 0 for symbol, _ in groups], np.int8)
    return {
        "amass": amass,
        "kind": np.zeros(len(groups), np.int8),
        "el": codes,
        "nh": np.array([count for _, count in groups], np.int8),
        "bonds": np.array(bonds, np.int32),
        "bring": np.zeros(len(bonds), np.int8),
        "mass": float(amass.sum()),
        "el10": codes.astype(np.uint8),
    }


PIPELINE_MOLECULES: Final = {
    "CCCO": (
        (("C", 3), ("C", 2), ("C", 2), ("O", 1)),
        ((0, 1), (1, 2), (2, 3)),
        "BDERNNFJNOPAEC",
    ),
    "CCOC": (
        (("C", 3), ("C", 2), ("O", 0), ("C", 3)),
        ((0, 1), (1, 2), (2, 3)),
        "XOBKSJJDNFUZPF",
    ),
    "CC(C)O": (
        (("C", 3), ("C", 1), ("C", 3), ("O", 1)),
        ((0, 1), (1, 2), (1, 3)),
        "KFZMGEQAYNKOFK",
    ),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class _Provider:
    """Serve literal prepared rows of propanol, its ether and its isomer."""

    priors: DerivationPriors

    def select(self, rows: CandidateRows, analogs: Sequence[Analog]) -> MainFamilyRows:
        smiles = rows.candidates.smiles
        records = [_fragnet_record(*PIPELINE_MOLECULES[value][:2]) for value in smiles]
        train = (ethanol(), propanol())
        capabilities = np.zeros((len(smiles), 22), np.uint8)
        capabilities[:, 0] = [value != "CCOC" for value in smiles]
        capabilities[:, 7] = [value == "CCOC" for value in smiles]
        return MainFamilyRows(
            candidate_structures=tuple(
                structure(
                    [8 if symbol == "O" else 6 for symbol, _ in groups],
                    bonds,
                    hashes=(0,) * 8,
                    numbers=(0,) * 7 + (4,),
                    formula=(3, 8, 0, 1),
                )
                for groups, bonds, _ in (PIPELINE_MOLECULES[value] for value in smiles)
            ),
            analog_structures=tuple(train[analog.sid] for analog in analogs),
            train_smiles=("CCO", "CCCO"),
            depth1=tuple(
                fragment_masses(row["amass"], bonds=row["bonds"], depth=1)
                for row in records
            ),
            depth2=tuple(
                fragment_masses(row["amass"], bonds=row["bonds"]) for row in records
            ),
            capabilities=capabilities,
            fragnet=tuple(records),
            priors=self.priors,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Pipeline:
    """Hold one complete tiny main-engine release and its three queries."""

    target: float
    spectra: list[SpectrumRequest]
    library: SpectralLibrary
    pool: CandidatePool
    train_fingerprints: NDArray[np.uint8]
    generated: tuple[GeneratedCandidate, ...]
    logs: tuple[DerivationLog, ...]
    provider: _Provider
    bank: MainBank
    views_bank: MainBank


def pipeline(directory: Path) -> Pipeline:
    """Build the propanol query against ethanol and propanol training spectra."""
    train_mass = np.array([46.041864812, 60.057514876])
    mass = np.repeat(train_mass, 2)
    order = np.argsort(mass, kind="mergesort")
    counts = np.array([3, 4, 2, 5], np.int32)
    library = SpectralLibrary(
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        mz=np.concatenate(
            [np.arange(count, dtype=np.float32) * 10 + 20 for count in counts.tolist()],
        ),
        intensity=np.concatenate(
            [np.arange(count, dtype=np.float32) % 3 + 1 for count in counts.tolist()],
        ),
        structure_mass=train_mass,
        keys=("LFQSCWFLJHTTHZ", "BDERNNFJNOPAEC"),
        smiles=("CCO", "CCCO"),
        sid=np.array([0, 0, 1, 1], np.int64),
        library=np.array([2, 3, 0, 2], np.int8),
        mode=np.array([1, -1, 1, -1], np.int8),
        instrument=np.array([0, 1, 2, 0], np.int8),
        adduct=np.array([0, 6, 0, 6], np.int16),
        precursor=mass + 1.00727645217,
        peak_count=counts,
        mass=mass,
        order=order,
        mass_sorted=mass[order],
    )
    propanol_fragments = fragment_masses(
        _fragnet_record(*PIPELINE_MOLECULES["CCCO"][:2])["amass"],
        bonds=np.array(PIPELINE_MOLECULES["CCCO"][1], np.int32),
    )
    bits = np.array([807, 4903, 8192, 10_802, 12_912, 15_164, 16_466], np.int64)
    pool = CandidatePool(
        mass=train_mass[1:],
        keys=("BDERNNFJNOPAEC",),
        smiles=("CCCO",),
        formulas=("C3H8O",),
        heavy_atoms=np.array([4], np.int32),
        source=np.array([0], np.int8),
        train_sid=np.array([1], np.int64),
        packed=np.array([[254]], np.uint8),
        bits=bits,
        fragment_offsets=np.array([0, len(propanol_fragments)], np.int64),
        fragment_mass=propanol_fragments.astype(np.float32),
        key_to_pid={"BDERNNFJNOPAEC": 0},
    )
    generated = tuple(
        GeneratedCandidate(
            smiles=smiles,
            key=PIPELINE_MOLECULES[smiles][2],
            formula="C3H8O",
            mass=60.057514876,
            heavy_atoms=4,
            fingerprint=np.unpackbits(np.array([packed], np.uint8))[:7].astype(
                np.float32,
            ),
            fragments=fragment_masses(
                _fragnet_record(*PIPELINE_MOLECULES[smiles][:2])["amass"],
                bonds=np.array(PIPELINE_MOLECULES[smiles][1], np.int32),
            ),
            parent_similarity=similarity,
            steps=1,
            parent_tanimoto=tanimoto,
            parent_sid=parent,
            combo=combo,
            raw=smiles,
        )
        for smiles, packed, similarity, tanimoto, parent, combo in (
            ("CCOC", 48, 0.98, 0.4, 0, ("methyl_O",)),
            ("CC(C)O", 240, 0.74, 0.57, 1, ("hydroxyl_aliph", "dehydroxyl_aliph")),
        )
    )
    proposals = PreparedPaths(
        {
            ("CCO", ("methyl_O",)): (path("CCOC", "CCOC", (1, 1, 1)),),
            ("CCCO", ("hydroxyl_aliph", "dehydroxyl_aliph")): (
                path("CCCO", "CCCO", (2, 2, 2), (3, 3, 3)),
                path("CC(C)O", "CC(C)O", (4, 4, 4), (5, 5, 5)),
            ),
        },
        {
            **{smiles: (row[2], smiles) for smiles, row in PIPELINE_MOLECULES.items()},
            "COCC": ("XOBKSJJDNFUZPF", "CCOC"),
        },
    )
    logs = (
        DerivationLog(
            parent=1,
            similarity=0.74,
            combo=("hydroxyl_aliph", "dehydroxyl_aliph"),
            kind="pool",
            how="key",
            identity="BDERNNFJNOPAEC",
        ),
        DerivationLog(
            parent=0,
            similarity=0.98,
            combo=("methyl_O",),
            kind="gen",
            how="smi",
            identity="COCC",
        ),
    )
    full = directory / "full0.pt"
    checkpoint = source_checkpoint(index=0)
    state = {
        name: value.clone()
        for name, value in from_plain(checkpoint["model"], dict[str, Tensor]).items()
    }
    # Shift the Full1 heads so the two banks disagree.
    state["head.3.bias"] += 0.125
    state["elem_head.2.bias"] += 0.25
    torch.save({**checkpoint, "model": state}, full)
    config = MainBank.Config()
    config.checkpoints = [checkpoint_config(full, index=0)]
    config.merged_adduct = modal_adduct
    config.single_count = keep_count
    views = directory / "views"
    views.mkdir()
    with torch.random.fork_rng(devices=[]):
        bank, views_bank = config.make(), save_bank(views, [0, 1, 2]).make()
    target = train_mass.item(1)
    spectra: list[SpectrumRequest] = [
        {
            "mz": [20, 30, 40, 105],
            "it": [1, 2, 3, 0.01],
            "prec": target + 1.00727645217,
            "mode": 1,
            "adduct": "[M+H]+",
            "ce": 10.0,
            "ce_n": 0,
        },
        {
            "mz": [20, 40, 50],
            "it": [3, 1, 2],
            "prec": target - 1.00727645217,
            "mode": -1,
            "adduct": "[M-H]-",
            "ce": float("nan"),
            "ce_n": 12,
        },
        {
            "mz": [],
            "it": [],
            "prec": target + 1.00727645217,
            "mode": 1,
            "adduct": "[M+H]+",
            "ce": float("nan"),
            "ce_n": 2,
        },
    ]
    return Pipeline(
        target=target,
        spectra=spectra,
        library=library,
        pool=pool,
        train_fingerprints=np.array([[242], [254]], np.uint8),
        generated=generated,
        logs=logs,
        provider=_Provider(
            priors=DerivationPriors(
                {
                    "glob": 0.2,
                    "tf": {"methyl_O": (3.0, 40.0), "demethyl_O": (4.0, 20.0)},
                    "env": {1: {}, 2: {}, 3: {}},
                },
                proposals=proposals,
            ),
        ),
        bank=bank,
        views_bank=views_bank,
    )


def _run(
    engine: MainEngine,
    inputs: Pipeline,
    rankers: TreeBank,
    *,
    spectra: Sequence[SpectrumRequest] | None = None,
    target: float | None = None,
    merged_counts: Sequence[int] = (),
    generated: Sequence[GeneratedCandidate] | None = None,
) -> EngineResult:
    return engine.run(
        inputs.spectra if spectra is None else spectra,
        target=inputs.target if target is None else target,
        library=inputs.library,
        pool=inputs.pool,
        train_fingerprints=inputs.train_fingerprints,
        bank=inputs.bank,
        views_bank=inputs.views_bank,
        provider=inputs.provider,
        stage_a=pipeline_stage_a(),
        stage_b=LightGBMModel.parse(pipeline_stage_b()),
        rankers=rankers,
        generated=inputs.generated if generated is None else generated,
        logs=inputs.logs,
        merged_counts=merged_counts,
    )


def assert_main_pipeline(tmp_path: Path) -> None:
    """Compose every family into the released ranker schema and four-model mean."""
    inputs = pipeline(tmp_path)
    engine = MainEngine.Config().make()
    base = engine.base(
        inputs.spectra,
        target=inputs.target,
        library=inputs.library,
        pool=inputs.pool,
        train_fingerprints=inputs.train_fingerprints,
        bank=inputs.bank,
        generated=inputs.generated,
    )
    rows = base.rows
    assert rows.candidates.smiles == ("CCCO", "CCOC", "CC(C)O")
    assert rows.candidates.sources == ("pool", "generated", "generated")
    np.testing.assert_array_equal(rows.candidates.pool_rows, [0, -1, -1])
    np.testing.assert_array_equal(rows.train_sid, [1, -1, -1])
    np.testing.assert_array_equal(rows.parent_sid, [-1, 0, 1])
    np.testing.assert_array_equal(
        rows.parent_similarity,
        np.array([0, 0.98, 0.74], np.float32),
    )
    assert rows.raw == (None, "CCOC", "CC(C)O")
    assert base.diagnostics.n_gen == 2
    values, views = family_features(
        base,
        inputs.spectra,
        provider=inputs.provider,
        views_bank=inputs.views_bank,
        train_fingerprints=inputs.train_fingerprints,
        bits=inputs.pool.bits,
        stage_a=pipeline_stage_a(),
        stage_b=LightGBMModel.parse(pipeline_stage_b()),
        logs=inputs.logs,
        search=engine.fragment_search,
        query_floor=engine.config.query_floor,
        query_max_peaks=engine.config.query_max_peaks,
        query_over=engine.config.query_over,
        max_candidates=engine.config.family_max_candidates,
        pairwise_rank_limit=engine.config.pairwise_rank_limit,
    )
    families = ("derivation", "frag2", "analog_struct", "model_views", "fragnet")
    names = tuple(values)
    assert tuple(dict.fromkeys(name.split("__")[0] for name in names)) == families
    for family in families:
        members = [name for name in names if name.startswith(family + "__")]
        assert members == sorted(members)
    assert all(
        value.shape == (3,) and value.dtype == np.float32 for value in values.values()
    )
    assert sum(name.startswith("derivation__") for name in names) == 41
    assert sum(name.startswith("frag2__") for name in names) == 31
    assert sum(name.startswith("analog_struct__") for name in names) == 36
    assert views.single.shape[0] == views.merged.shape[0] == 3
    assert views.modes.tolist() == [1.0, -1.0]
    # Both generated rows hold their own derivation support; the pool row holds a log.
    assert values["derivation__dp_hit"].tolist() == [1, 0, 0]
    assert values["derivation__dg_n_par"].tolist() == [0, 1, 1]
    rankers = TreeBank(
        features=RANKER_NAMES,
        models=tuple(LightGBMModel.parse(text) for text in ranker_texts()),
    )
    combined = {
        name: base.features[:, i] for i, name in enumerate(main_feature_names())
    } | values
    selected = select_features(combined, names=RANKER_NAMES, count=3)
    ranked = _run(engine, inputs, rankers)
    np.testing.assert_array_equal(ranked.features, selected)
    np.testing.assert_array_equal(ranked.scores, rankers.score(selected, RANKER_NAMES))
    assert ranked.feature_names == RANKER_NAMES
    assert ranked.features.dtype == np.float32
    assert ranked.scores.dtype == np.float64
    assert ranked.candidates.smiles == rows.candidates.smiles
    assert ranked.candidates.keys == rows.candidates.keys
    assert ranked.candidates.formulas == rows.candidates.formulas
    assert ranked.candidates.sources == rows.candidates.sources
    np.testing.assert_array_equal(
        ranked.candidates.pool_rows,
        rows.candidates.pool_rows,
    )
    np.testing.assert_array_equal(
        ranked.candidates.generated,
        rows.candidates.generated,
    )
    assert ranked.diagnostics == base.diagnostics
    limited_config = MainEngine.Config()
    limited_config.max_query_spectra = 3
    limited = _run(
        limited_config.make(),
        inputs,
        rankers,
        spectra=[*inputs.spectra, inputs.spectra[0]],
        merged_counts=(0, 12, 2, 99),
    )
    np.testing.assert_array_equal(limited.features, ranked.features)
    np.testing.assert_array_equal(limited.scores, ranked.scores)
    assert [spectrum.get("ce_n") for spectrum in inputs.spectra] == [0, 12, 2]
    empty = _run(engine, inputs, rankers, target=900.0, generated=())
    assert empty.features.shape == (0, len(RANKER_NAMES))
    assert empty.features.dtype == np.float32
    assert empty.scores.shape == (0,)
    assert empty.scores.dtype == np.float64
    assert empty.feature_names == RANKER_NAMES


def test_halogen_subformulas_skip_unavailable_iodine() -> None:
    counts = engine_formula_vector("CF")
    assert counts is not None
    fluorine = 18.99840322
    assert main_subformula_masses(counts, max_out=100).tolist() == [
        12.0,
        fluorine,
        12.0 + fluorine,
    ]


def test_aggregates_fall_back_to_a_product_with_the_same_key_and_skeleton() -> None:
    priors = derivation_priors()
    context = DerivationContext(
        smiles=("C(C)OC", "COC(C)"),
        keys=("K2", "K9"),
        formulas=("C3H8O", "C3H8O"),
        generated=np.array([True, True]),
        parent_sid=np.array([0, 0], np.int64),
        similarities=np.array([0.9, 0.8]),
        combos=(("methyl_O",), ("methyl_O",)),
        analog_sid=np.array([0], np.int64),
        model_score=np.zeros(2, np.float32),
        parent_smiles=("CCO",),
        logs=(),
    )
    values = aggregate(
        context,
        priors=priors,
        evidence=supports(context, proposals=priors.proposals),
    )
    # The first spelling shares CCOC's key and skeleton; the second only its skeleton.
    assert values["site_logfr"].tolist() == pytest.approx(
        [math.log(SITE), math.log(METHYL)],
    )
    assert values["prior_miss"].tolist() == [0.0, 1.0]
    assert values["n_sib"].tolist() == [2.0, 0.0]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
