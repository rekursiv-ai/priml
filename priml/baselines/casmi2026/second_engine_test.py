"""Checks for the second engine module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hashlib
import heapq
import json
import math
import re
import sys

from pyarrow import Table, parquet
from torch import Tensor
from treekle import to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.chemistry import PeakStore
from priml.baselines.casmi2026.custom_types import (
    Analog,
    SpectrumRequest,
)
from priml.baselines.casmi2026.isolated import (
    HYDROGEN_MASS,
    IDENTITY_RDKIT,
    second_adduct_parameters,
)
from priml.baselines.casmi2026.model_test import tiny_second
from priml.baselines.casmi2026.second_engine import (
    EngineIdentityTable,
    FeatureMatrix,
    FragmentMassStore,
    Histogram,
    HistogramBoosting,
    HistogramEnsemble,
    Node,
    QuerySpectrum,
    SecondAssets,
    SecondBank,
    SecondBase,
    SecondChannels,
    SecondEngine,
    SecondFragments,
    SecondLibrary,
    SecondPeaks,
    SecondPool,
    SecondRankerBank,
    SecondRetrieval,
    SimulationRows,
    _adduct_carriers,
    _explanation,
    _polarity_features,
    _proton_carrier,
    _relative,
    _sigmoid,
    analog_features,
    blend_histograms,
    collision_energy,
    instrument_family,
    predict_second,
    prepare_queries,
    second_features,
    second_fragments,
)
from priml.baselines.casmi2026.testing import (
    identity_rows,
    retrieval_inputs,
    write_identity,
)
from priml.model.ae.checkpoint import LocalFile


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        PredictionMatrix,
    )
    from priml.baselines.casmi2026.model import SecondFPNet


def peaks(
    *,
    max_peaks: int = 128,
    window: float = 50.0,
    per_window: int = 8,
) -> SecondPeaks:
    """Build the released peak preparation with optional geometry overrides."""
    config = SecondPeaks.Config()
    config.max_peaks, config.window, config.per_window = max_peaks, window, per_window
    return config.make()


def test_peak_preparation_filters_quotas_and_square_root_normalizes() -> None:
    for mass, intensity in (
        (list[float](), list[float]()),
        ([105.0, 106.0], [2.0, 1.0]),
        ([10.0, 20.0], [0.0, -1.0]),
    ):
        mz, it = peaks().prepare(
            np.asarray(mass, np.float64),
            np.asarray(intensity, np.float64),
            precursor=100.0,
        )
        assert mz.shape == it.shape == (0,)
        assert mz.dtype == it.dtype == np.float32
    # The 1.5 Da margin admits 101.5 Da; the 0.001 floor drops 0.001 of base 3.
    mz, it = peaks().prepare(
        np.array([0.0, -1.0, 100.0, 101.5, 101.6]),
        np.array([1.0, 2.0, 0.001, 3.0, 5.0]),
        precursor=100.0,
    )
    assert mz.tolist() == [-1.0, 0.0, 101.5]
    assert it.tolist() == [
        np.float32((2 / 3) ** 0.5),
        np.float32((1 / 3) ** 0.5),
        1.0,
    ]
    # One peak per 50 Da window by intensity, then trimmed or refilled to the cap.
    mass = np.array([10.0, 20.0, 60.0, 70.0, 110.0])
    intensity = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    for maximum, expected in (
        (3, [10.0, 60.0, 110.0]),
        (2, [10.0, 60.0]),
        (4, [10.0, 20.0, 60.0, 110.0]),
    ):
        mz, _ = peaks(max_peaks=maximum, per_window=1).prepare(
            mass,
            intensity,
            precursor=500.0,
        )
        assert mz.tolist() == expected


def test_merge_normalizes_each_spectrum_then_drops_the_weaker_neighbour() -> None:
    mass, values = peaks().merge(
        [
            (np.array([10.0, 10.003, 10.006, 10.009]), np.array([1.0, 0.8, 0.9, 0.7])),
            (np.array([10.003, 20.0]), np.array([3.0, 1.0])),
        ],
    )
    assert mass.tolist() == [10.0, 10.003, 20.0]
    assert values.tolist() == [1.0, 1.0, 1 / 3]
    mass, values = peaks().merge(
        [
            (np.array([10.0, 20.0]), np.array([0.0, -1.0])),
            (np.zeros(0), np.zeros(0)),
        ],
    )
    assert mass.tolist() == [10.0, 20.0]
    # A nonpositive base peak divides by the 1e-9 floor.
    assert values.tolist() == [0.0, -1.0 / 1e-9]
    mass, values = peaks().merge([])
    assert mass.shape == values.shape == (0,)


def test_instrument_families_follow_the_released_vocabulary() -> None:
    names = (
        None,
        "timsTOF pro",
        "ORBITRAP",
        "qft",
        "ftms",
        "hybrid ft",
        "itft",
        "exactive",
        "QTOF",
        "ion trap",
        "QQ",
        "other",
    )
    assert [instrument_family(name) for name in names] == [
        4,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        3,
        4,
    ]


def second_models() -> list[SecondFPNet]:
    """Build two seeded tiny evaluation models with the bank's adduct vocabulary."""
    models: list[SecondFPNet] = []
    with torch.random.fork_rng(devices=[]):
        for seed in range(2):
            _ = torch.manual_seed(seed)
            config = tiny_second()
            config.adduct.channels_in = 26
            models.append(config.make().eval())
    return models


def second_states() -> list[dict[str, Tensor]]:
    """Return the released states of both seeded tiny models."""
    return [
        {name: value.detach().clone() for name, value in model.released_state().items()}
        for model in second_models()
    ]


def bank_spectra() -> list[QuerySpectrum]:
    """Return a positive timsTOF, an unknown-adduct negative and an empty spectrum."""
    return [
        QuerySpectrum(
            mass=np.array([20.0, 40.0, 60.0, 80.0]),
            intensity=np.array([1.0, 0.5, 0.25, 0.75]),
            precursor=100.0,
            adduct="[M+H]+",
            instrument="timsTOF",
            collision_energy=10.0,
            polarity=1.0,
        ),
        QuerySpectrum(
            mass=np.array([15.0, 45.0]),
            intensity=np.array([2.0, 1.0]),
            precursor=200.0,
            adduct="unknown",
            instrument="QTOF",
            collision_energy=30.0,
            polarity=-1.0,
        ),
        QuerySpectrum(
            mass=np.zeros(0),
            intensity=np.zeros(0),
            precursor=30.0,
            adduct="[M-H]-",
            instrument=None,
            collision_energy=0.0,
            polarity=1.0,
        ),
    ]


def _predict(
    spectra: Sequence[QuerySpectrum],
    *,
    single: Sequence[SecondFPNet],
    merged: Sequence[SecondFPNet],
) -> NDArray[np.float32] | None:
    return predict_second(
        spectra,
        single=single,
        merged=merged,
        device="cpu",
        peaks=peaks(),
        merged_energy=25.0,
    )


def test_bank_arms_average_their_models_and_skip_spectra_without_peaks() -> None:
    first, second = second_models()
    spectra = bank_spectra()
    both = _predict(spectra, single=(first, second), merged=(second,))
    single = _predict(spectra, single=(first, second), merged=())
    merged = _predict(spectra, single=(), merged=(second,))
    first_alone = _predict(spectra, single=(first,), merged=())
    second_alone = _predict(spectra, single=(second,), merged=())
    nonempty = _predict(spectra[:2], single=(first,), merged=())
    assert both is not None
    assert single is not None
    assert merged is not None
    assert first_alone is not None
    assert second_alone is not None
    assert nonempty is not None
    assert both.dtype == np.float32
    assert both.shape == (7,)
    np.testing.assert_array_equal(both, (single + merged) / 2)
    np.testing.assert_array_equal(single, (first_alone + second_alone) / 2)
    np.testing.assert_array_equal(nonempty, first_alone)
    # The merged arm takes the median precursor, first metadata and mean polarity.
    mass, intensity = peaks().merge([(item.mass, item.intensity) for item in spectra])
    pooled = QuerySpectrum(
        mass=mass,
        intensity=intensity,
        precursor=100.0,
        adduct="[M+H]+",
        instrument="timsTOF",
        collision_energy=25.0,
        polarity=1 / 3,
    )
    as_single = _predict([pooled], single=(second,), merged=())
    assert as_single is not None
    np.testing.assert_array_equal(merged, as_single)
    assert _predict(spectra, single=(), merged=()) is None
    silent = [
        replace(item, mass=np.zeros(0), intensity=np.zeros(0)) for item in spectra
    ]
    assert _predict(silent, single=(first,), merged=(second,)) is None


@pytest.mark.parametrize(
    ("mass", "values", "expected_mass", "expected_values"),
    [
        ([0.0, 0.003], [1.0, 2.0], [0.003], [1.0]),
        ([0.0, 0.003], [2.0, 1.0], [0.0], [1.0]),
        ([0.0, 0.003], [1.0, 1.0], [0.003], [1.0]),
        ([0.0, 0.005, 0.5], [1.0, 2.0, 1.0], [0.0, 0.005, 0.5], [0.5, 1.0, 0.5]),
    ],
)
def test_merge_retains_exact_threshold_and_newer_ties(
    mass: list[float],
    values: list[float],
    expected_mass: list[float],
    expected_values: list[float],
) -> None:
    actual_mass, actual_values = peaks().merge(
        [(np.asarray(mass, np.float64), np.asarray(values, np.float64))],
    )
    assert np.array_equal(actual_mass, np.asarray(expected_mass, np.float64))
    assert np.array_equal(actual_values, np.asarray(expected_values, np.float64))


def test_merge_converts_both_float32_inputs_before_normalizing() -> None:
    source_mass = np.array([1.0, 2.0], np.float32)
    source_values = np.array([1.0, 3.0], np.float32)
    mass, values = peaks().merge([(source_mass, source_values)])
    assert mass.dtype == values.dtype == np.float64
    assert np.array_equal(mass, [1.0, 2.0])
    assert np.array_equal(values, [1.0 / 3.0, 1.0])
    assert np.array_equal(source_mass, [1.0, 2.0])
    assert np.array_equal(source_values, [1.0, 3.0])


def test_analog_features_hand_derived_for_two_candidates_and_two_analogs() -> None:
    features = analog_features(
        np.array([[1, 1, 0], [0, 1, 1]], np.float32),
        np.array([0.5, 0.0], np.float32),
        np.array([[1, 1, 0], [0, 0, 1]], np.float32),
        np.array([0.5, -1.0], np.float32),
        logits=np.array([1.0, -1.0, 2.0]),
        fragments=np.array([0.25, 0.75], np.float32),
    )
    assert features.dtype == np.float32
    # Tanimoto to the analogs is [1, 0] and [1/3, 1/2]; weights clip to [0.5, 0].
    third = float(np.float32(1 / 3))
    weak = third * 0.125
    expected = np.array(
        [
            # Library value, rank, maximum, gap and presence.
            [0.5, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.0, -0.5],
            [1.0, 0.0],
            # Cubic propagation, its rank, maximum and gap.
            [0.125, weak],
            [0.0, 1.0],
            [0.125, 0.125],
            [0.0, weak - 0.125],
            # Linear, best and leading Tanimoto, the cubic mean and the top weight.
            [0.5, third * 0.5],
            [1.0, 0.5],
            [1.0, third],
            [1.0, third],
            [0.5, 0.5],
            [math.log(2), math.log(2)],
            # Model scores [0, 1]: z, rank, gap, normalized z and rank, leader flag.
            [-1.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            # Fragment value, rank, gap and z.
            [0.25, 0.75],
            [1.0, 0.0],
            [-0.5, 0.0],
            [-1.0, 1.0],
            # Agreement of library and analogs with the model, then correlation.
            [0.0, 0.0],
            [0.0, weak],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, -1.0],
        ],
    ).T
    np.testing.assert_allclose(features, expected, rtol=0, atol=1e-6)


def test_analog_features_zero_each_absent_evidence_family() -> None:
    fingerprints = np.array([[1, 1, 0], [0, 1, 1]], np.float32)
    library = np.array([0.5, 0.0], np.float32)
    analogs = np.array([[1, 1, 0]], np.float32)
    similarity = np.array([0.5], np.float32)
    logits = np.array([1.0, -1.0, 2.0])
    fragments = np.array([0.25, 0.75], np.float32)
    complete = analog_features(
        fingerprints,
        library,
        analogs,
        similarity,
        logits=logits,
        fragments=fragments,
    )
    assert complete.shape == (2, 31)
    no_model = analog_features(
        fingerprints,
        library,
        analogs,
        similarity,
        fragments=fragments,
    )
    assert not no_model[:, 15:21].any()
    assert not no_model[:, 25:].any()
    np.testing.assert_array_equal(no_model[:, :15], complete[:, :15])
    np.testing.assert_array_equal(no_model[:, 21:25], complete[:, 21:25])
    for absent, weights in ((None, similarity), (analogs[:0], similarity[:0])):
        no_analog = analog_features(
            fingerprints,
            library,
            absent,
            weights,
            logits=logits,
            fragments=fragments,
        )
        # Zero propagation still ranks its tied candidates in encounter order.
        assert no_analog[:, 6].tolist() == [0.0, 1.0]
        assert not no_analog[:, [5, *range(7, 14)]].any()
        np.testing.assert_array_equal(no_analog[:, :5], complete[:, :5])
        # Without propagation the analog agreement falls back to zero.
        assert not no_analog[:, 28].any()
    no_fragment = analog_features(
        fingerprints,
        library,
        analogs,
        similarity,
        logits=logits,
    )
    assert not no_fragment[:, 21:25].any()
    np.testing.assert_array_equal(no_fragment[:, :21], complete[:, :21])
    empty = analog_features(fingerprints[:0], library[:0], None, similarity[:0])
    assert empty.shape == (0, 31)
    constant = analog_features(
        fingerprints * 0,
        library * 0,
        None,
        similarity[:0],
        logits=logits * 0,
        fragments=fragments * 0,
    )
    # Zero variance and an empty library keep z-scores, agreement and correlation at zero.
    for column in (15, 18, 24, 25, 26, 27, 28, 29, 30):
        np.testing.assert_array_equal(constant[:, column], [0.0, 0.0])


def test_model_columns_divide_scores_by_the_root_of_each_bit_count() -> None:
    # Four, one and one bits score 3, 2 and 4; over roots 2, 1 and 1: 1.5, 2 and 4.
    features = analog_features(
        np.array(
            [[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
            np.float32,
        ),
        np.zeros(3, np.float32),
        None,
        np.zeros(0, np.float32),
        logits=np.array([0.75, 0.75, 0.75, 0.75, 2.0, 4.0]),
    )
    z, rank, gap, normalized_z, normalized_rank, leader = features[:, 15:21].T
    # Raw deviations 0, -1 and 1 over sqrt(2/3); float32 deviation is 1 ulp off.
    assert z.tolist() == pytest.approx([0.0, -(1.5**0.5), 1.5**0.5], rel=1e-6)
    assert rank.tolist() == [0.5, 1.0, 0.0]
    assert gap.tolist() == [-1.0, -2.0, 0.0]
    # Normalized deviations -1, -1/2 and 3/2 over sqrt(7/6).
    assert normalized_z.tolist() == [
        float(np.float32(value / (7 / 6) ** 0.5)) for value in (-1.0, -0.5, 1.5)
    ]
    assert normalized_rank.tolist() == [1.0, 0.5, 0.0]
    assert leader.tolist() == [0.0, 0.0, 1.0]


def test_sigmoid_is_the_libm_logistic_and_saturates_without_overflow() -> None:
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(2.0) == 1.0 / (1.0 + math.exp(-2.0))
    assert _sigmoid(-2.0) == 1.0 / (1.0 + math.exp(2.0))
    assert _sigmoid(1000.0) == 1.0
    assert _sigmoid(-1000.0) == 0.0


def test_fragment_proton_carrier_keeps_positive_zero_and_negative_modes() -> None:
    proton = second_adduct_parameters("[M+H]+")[2]
    assert _proton_carrier(1.0) == (proton,)
    assert _proton_carrier(0.0) == (-proton,)
    assert _proton_carrier(-1.0) == (-proton,)


def test_adduct_carriers_are_exact_for_every_named_and_fallback_adduct() -> None:
    proton = second_adduct_parameters("[M+H]+")[2]
    for adduct in ("[M+H]+", "[M-H2O+H]+", "[M-2H2O+H]+"):
        assert _adduct_carriers(adduct) == (proton,)
    for adduct in ("[M-H]-", "[M-H2O-H]-"):
        assert _adduct_carriers(adduct) == (-proton,)
    for adduct in ("[M+NH4]+", "[M+Na]+", "[M+K]+"):
        assert _adduct_carriers(adduct) == (proton, second_adduct_parameters(adduct)[2])
    for adduct in ("[M+CH2O2-H]-", "[M+Cl]-"):
        assert _adduct_carriers(adduct) == (
            -proton,
            second_adduct_parameters(adduct)[2],
        )
    for adduct in ("[2M+H]+", "[M+Li]+", "[M]+"):
        assert _adduct_carriers(adduct) == (proton,)
    # A plus sign only counts within the final two characters.
    for adduct in ("[2M-H]-", "[M+]-", "unknown"):
        assert _adduct_carriers(adduct) == (-proton,)


class _Pool:
    """Serve two fixed four-bit fingerprints by position."""

    def fingerprints(self, indices: NDArray[np.int64], /) -> NDArray[np.uint8]:
        bits = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
        return bits[indices]


def _channels(*analogs: Analog) -> SecondChannels:
    empty = np.zeros(0, dtype=np.float32)
    return SecondChannels(
        pool_rows=np.zeros(0, dtype=np.int64),
        maximum=empty,
        mean=empty,
        shifted=empty,
        count=empty,
        analogs=(),
        polarity_analogs=analogs,
    )


def test_fragment_mass_store_rows_slice_by_offsets() -> None:
    store = FragmentMassStore(
        offsets=np.array([0, 2, 2, 5], dtype=np.int64),
        mass=np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
    )
    assert store.row(0).tolist() == [1.5, 2.5]
    assert store.row(1).tolist() == []
    assert store.row(2).tolist() == [3.5, 4.5, 5.5]


def test_relative_returns_values_best_first_ranks_and_gap_to_maximum() -> None:
    values, ranks, gap = _relative(np.array([0.5, 2.0, 1.0]))
    assert values.tolist() == [0.5, 2.0, 1.0]
    assert ranks.tolist() == [1.0, 0.0, 0.5]
    assert gap.tolist() == [-1.5, 0.0, -1.0]
    assert {values.dtype, ranks.dtype, gap.dtype} == {np.dtype(np.float32)}


def test_polarity_features_are_zero_without_analogs() -> None:
    fingerprints = np.ones((3, 4), dtype=np.float32)
    got = _polarity_features(_Pool(), fingerprints=fingerprints, channels=_channels())
    assert got.shape == (3, 5)
    assert not got.any()


def test_polarity_features_weight_tanimoto_by_analog_similarity() -> None:
    fingerprints = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    got = _polarity_features(
        _Pool(),
        fingerprints=fingerprints,
        channels=_channels(
            Analog(sid=0, similarity=1.0),
            Analog(sid=1, similarity=0.5),
        ),
    )
    np.testing.assert_allclose(
        got,
        [[1.0, 0.0, 0.0, 1.0, 1.0], [1 / 24, 1.0, 1 / 24 - 1, 1 / 192, 1.0]],
        rtol=1e-6,
    )
    assert got.dtype == np.float32


def test_explanation_sums_square_root_intensity_of_matched_peaks() -> None:
    proton = second_adduct_parameters("[M+H]+")[2]
    masses = np.array([100.0])
    peaks = np.array(
        [100.0 + proton, 100.0 + proton + 0.011, 100.0 + 2 * HYDROGEN_MASS + proton],
    )
    intensity = np.array([4.0, 9.0, 1.0])
    got = _explanation(masses, query=(peaks, intensity), carriers=(proton,))
    assert got == (2.0 + 1.0) / (2.0 + 3.0 + 1.0)


def test_explanation_is_zero_without_masses_peaks_or_intensity() -> None:
    peaks = (np.array([100.0]), np.array([4.0]))
    assert _explanation(np.zeros(0), query=peaks, carriers=(1.0,)) == 0.0
    assert (
        _explanation(
            np.array([99.0]),
            query=(np.zeros(0), np.zeros(0)),
            carriers=(1.0,),
        )
        == 0.0
    )
    assert (
        _explanation(np.array([99.0]), query=(peaks[0], np.zeros(1)), carriers=(1.0,))
        == 0.0
    )


def test_fragment_panels_mix_polarities_take_adducts_and_average_own_modes() -> None:
    proton = second_adduct_parameters("[M+H]+")[2]
    fragments = FragmentMassStore(
        offsets=np.array([0, 1, 1], dtype=np.int64),
        mass=np.array([100.0]),
    )
    spectra: list[SpectrumRequest] = [
        {
            "mz": [100.0 + proton, 300.0],
            "it": [4.0, 1.0],
            "mode": 1,
            "prec": 101.0,
            "adduct": "[M+H]+",
        },
        {
            "mz": [100.0 - proton],
            "it": [1.0],
            "mode": -1,
            "prec": 99.0,
            "adduct": "[M-H]-",
        },
    ]
    rows = np.array([0, 1, 0], dtype=np.int64)
    search = SecondEngine.Config().fragment_search.make()
    result = second_fragments(spectra, rows=rows, fragments=fragments, search=search)
    # The positive query explains sqrt(0.8) of sqrt(0.8) + sqrt(0.2), two thirds.
    assert result.maximum.tolist() == [1.0, 0.0, 1.0]
    assert result.adduct.tolist() == [1.0, 0.0, 1.0]
    assert result.polarity.tolist() == [1.0, 0.0, 1.0]
    assert result.mean.tolist() == pytest.approx([5 / 6, 0.0, 5 / 6], abs=1e-6)
    assert result.mean.dtype == np.float32
    negative = second_fragments(
        spectra[1:],
        rows=rows,
        fragments=fragments,
        search=search,
    )
    assert negative.maximum.tolist() == negative.mean.tolist() == [1.0, 0.0, 1.0]
    positive = second_fragments(
        spectra[:1],
        rows=rows,
        fragments=fragments,
        search=search,
    )
    assert positive.maximum.tolist() == pytest.approx([2 / 3, 0.0, 2 / 3], abs=1e-6)
    silent = second_fragments([], rows=rows, fragments=fragments, search=search)
    for values in (silent.maximum, silent.adduct, silent.mean, silent.polarity):
        assert values.tolist() == [0.0, 0.0, 0.0]


def test_second_features_align_channels_and_stack_the_four_blocks() -> None:
    _, pool, _ = retrieval_inputs()
    rows = np.array([2, 0, 4], dtype=np.int64)
    channels = SecondChannels(
        pool_rows=np.array([0, 2], dtype=np.int64),
        maximum=np.array([0.5, 0.9], dtype=np.float32),
        mean=np.array([0.1, 0.3], dtype=np.float32),
        shifted=np.array([0.4, 0.7], dtype=np.float32),
        count=np.array([2, 3], dtype=np.float32),
        analogs=(Analog(sid=1, similarity=0.5),),
        polarity_analogs=(Analog(sid=3, similarity=0.8),),
    )
    fragments = SecondFragments(
        maximum=np.array([0.25, 0.75, 0.0], dtype=np.float32),
        adduct=np.array([0.5, 0.25, 0.0], dtype=np.float32),
        mean=np.array([0.2, 0.4, 0.0], dtype=np.float32),
        polarity=np.array([1.0, 0.5, 0.0], dtype=np.float32),
    )
    logits = ((np.arange(7) - 3) / 7).astype(np.float32)
    features = second_features(
        pool,
        rows=rows,
        channels=channels,
        fragments=fragments,
        logits=logits,
    )
    assert features.shape == (3, 51)
    assert features.dtype == np.float32
    fingerprints = pool.fingerprints(rows).astype(np.float32)
    # Pool row 4 has no library channel, so its aligned evidence is zero.
    np.testing.assert_array_equal(
        features[:, :31],
        analog_features(
            fingerprints,
            np.array([0.9, 0.5, 0.0], np.float32),
            pool.fingerprints(np.array([1], np.int64)).astype(np.float32),
            np.array([0.5], np.float32),
            logits=logits,
            fragments=fragments.maximum,
        ),
    )
    f32 = np.float32
    count = np.log1p(np.array([3.0, 2.0, 0.0])).astype(np.float32)
    library = np.array(
        [
            [f32(0.3), f32(0.1), f32(0.0)],
            [0.0, 0.5, 1.0],
            [0.0, f32(0.1) - f32(0.3), -f32(0.3)],
            [f32(0.7), f32(0.4), f32(0.0)],
            [0.0, 0.5, 1.0],
            [0.0, f32(0.4) - f32(0.7), -f32(0.7)],
            count,
            count - count[0],
        ],
        dtype=np.float32,
    ).T
    np.testing.assert_array_equal(features[:, 31:39], library)
    np.testing.assert_array_equal(
        features[:, 39:44],
        _polarity_features(pool, fingerprints=fingerprints, channels=channels),
    )
    fragment = np.array(
        [
            [0.5, 0.25, 0.0],
            [0.0, 0.5, 1.0],
            [0.0, -0.25, -0.5],
            [f32(0.2), f32(0.4), 0.0],
            [0.5, 0.0, 1.0],
            [f32(0.2) - f32(0.4), 0.0, -f32(0.4)],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    ).T
    np.testing.assert_array_equal(features[:, 44:], fragment)
    unmodelled = second_features(
        pool,
        rows=rows,
        channels=channels,
        fragments=fragments,
        logits=None,
    )
    assert not unmodelled[:, 15:21].any()
    np.testing.assert_array_equal(unmodelled[:, 31:], features[:, 31:])


def second_library() -> SecondLibrary:
    """Label the shared raw library with pool rows, one unpooled row and a NaN mass."""
    library, _, _ = retrieval_inputs()
    return SecondLibrary(
        store=PeakStore(
            offsets=library.offsets,
            mz=library.mz,
            intensity=library.intensity,
        ),
        mass=np.concatenate([library.mass[:-1], [float("nan")]]),
        pool_rows=np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, -1, 4, 2], dtype=np.int64),
        polarity=library.mode,
        timstof=np.isin(np.arange(13), [0, 2, 6, 9]),
        precursor=library.precursor,
    )


def test_retrieval_indexes_timstof_first_representatives_per_polarity() -> None:
    library = second_library()
    config = SecondRetrieval.Config()
    config.n_analog = 2
    retrieval = config.make()
    index = retrieval.index(library)
    # Row 10 has no pool row and row 12 no finite mass; equal masses keep row order.
    assert index.rows.tolist() == [8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11]
    assert index.mass.tolist() == [
        90.0,
        100.0,
        100.0,
        100.0,
        100.0005,
        100.0005,
        100.002,
        100.002,
        103.0,
        110.0,
        300.0,
    ]
    # Each pool row keeps its timsTOF spectrum, then its longest, then its first.
    for representatives, chosen in (
        (index.all, [0, 3, 6, 7, 9]),
        (index.positive, [0, 3, 6, 9]),
        (index.negative, [2, 4, 7]),
    ):
        assert representatives.rows.tolist() == chosen
        assert representatives.pool_rows.tolist() == library.pool_rows[chosen].tolist()
        assert representatives.mass.tolist() == library.mass[chosen].tolist()
        assert len(representatives.store.offsets) == len(chosen) + 1
    _, _, spectra = retrieval_inputs()
    channels = retrieval.channels(spectra, library=library, index=index, target=100.0)
    # The 10 ppm window holds rows 0-4, three of pool row 0 and two of pool row 1.
    assert channels.pool_rows.tolist() == [0, 1]
    assert channels.count.tolist() == [3.0, 2.0]
    assert np.all(channels.mean <= channels.maximum)
    assert channels.maximum.max() > 0
    for analogs in (channels.analogs, channels.polarity_analogs):
        assert 0 < len(analogs) <= 2
        assert len({analog.sid for analog in analogs}) == len(analogs)
        similarity = [analog.similarity for analog in analogs]
        assert similarity == sorted(similarity, reverse=True)
    silent = retrieval.channels([], library=library, index=index, target=100.0)
    assert silent.pool_rows.tolist() == [0, 1]
    for values in (silent.maximum, silent.mean, silent.shifted):
        assert values.tolist() == [0.0, 0.0]
    # Unscored analogs keep their mass order within the 200 Da window.
    assert silent.analogs == (
        Analog(sid=0, similarity=0.0),
        Analog(sid=1, similarity=0.0),
    )
    assert silent.polarity_analogs == ()
    distant = retrieval.channels(spectra, library=library, index=index, target=900.0)
    assert distant.pool_rows.shape == distant.maximum.shape == (0,)
    assert distant.analogs == distant.polarity_analogs == ()


def test_channels_average_query_maxima_and_keep_each_analogs_best_score() -> None:
    # One structure holds a positive and a negative copy of the peaks 10 and 20.
    library = SecondLibrary(
        store=PeakStore(
            offsets=np.array([0, 2, 4], np.int64),
            mz=np.array([10, 20, 10, 20], np.float32),
            intensity=np.ones(4, np.float32),
        ),
        mass=np.full(2, 100.0),
        pool_rows=np.zeros(2, np.int64),
        polarity=np.array([1, -1], np.int8),
        timstof=np.zeros(2, np.bool_),
        precursor=np.full(2, 101.0),
    )
    # The positive query copies both peaks; the negative one shares none.
    spectra: list[SpectrumRequest] = [
        {"mz": [10, 20], "it": [1, 1], "mode": 1, "prec": 101.0, "adduct": "[M+H]+"},
        {"mz": [30, 40], "it": [1, 1], "mode": -1, "prec": 101.0, "adduct": "[M-H]-"},
    ]
    retrieval = SecondRetrieval.Config().make()
    channels = retrieval.channels(
        spectra,
        library=library,
        index=retrieval.index(library),
        target=100.0,
    )
    assert channels.maximum.tolist() == [1.0]
    assert channels.mean.tolist() == [0.5]
    # Equal precursors leave no shift, so the shifted channel keeps the best query.
    assert channels.shifted.tolist() == [1.0]
    assert channels.count.tolist() == [2.0]
    # The polarity searches score the structure 1 and nearly 0; the better one stands.
    best = (Analog(sid=0, similarity=1.0),)
    assert channels.analogs == channels.polarity_analogs == best


def test_minimal_pool_preserves_first_and_last_tied_rows_and_all_packed_bits(
    tmp_path: Path,
) -> None:
    parquet.write_table(
        Table.from_pydict(
            {
                "mass": [100.0, 100.0, 100.004],
                "key": ["a", "a", "b"],
                "smiles": ["C", "CC", "O"],
            },
        ),
        tmp_path / "pool_meta.parquet",
    )
    np.save(tmp_path / "pool_mass.npy", np.array([100.0, 100.0, 100.004]))
    np.save(tmp_path / "pool_fp.npy", np.array([[17], [33], [63]], np.uint8))
    np.save(tmp_path / "fp_bits.npy", np.arange(8, dtype=np.int64))
    pool = SecondPool.load(tmp_path)
    assert pool.keys == ("a", "a", "b")
    assert pool.smiles == ("C", "CC", "O")
    np.testing.assert_array_equal(pool.bits, np.arange(8, dtype=np.int64))
    np.testing.assert_array_equal(pool.window(100.0, ppm=10), [0, 1])
    assert pool.window(900.0, ppm=30).dtype == np.int64
    assert pool.window(900.0, ppm=30).shape == (0,)
    selected = np.array([1, 0, 1], np.int64)
    np.testing.assert_array_equal(
        pool.fingerprints(selected),
        np.unpackbits(np.array([[33], [17], [33]], np.uint8), axis=1),
    )
    assert pool.fingerprints(selected).dtype == np.uint8
    assert pool.mass.flags.writeable is False
    # SecondPool.load rejects two packed rows for three molecules.
    np.save(tmp_path / "pool_fp.npy", np.zeros((2, 1), np.uint8))
    with pytest.raises(ValueError, match=r"^Engine2 pool rows must be aligned\.$"):
        _ = SecondPool.load(tmp_path)
    # SecondPool.load packs the eight declared bits into one byte per row.
    np.save(tmp_path / "pool_fp.npy", np.zeros((3, 1), np.uint8))
    # SecondPool.load rejects a column matrix in place of the flat mass array.
    np.save(tmp_path / "pool_mass.npy", np.zeros((3, 1)))
    with pytest.raises(
        ValueError,
        match=r"^Engine2 array pool_mass.npy has an invalid rank\.$",
    ):
        _ = SecondPool.load(tmp_path)
    np.save(tmp_path / "pool_mass.npy", np.zeros(3, np.float32))
    with pytest.raises(
        TypeError,
        match=r"pool_mass.npy: expected float64, got float32",
    ):
        _ = SecondPool.load(tmp_path)
    lower, upper = 100.0 * (1 - 10 / 1e6), 100.0 * (1 + 10 / 1e6)
    bounded = replace(
        pool,
        mass=np.array(
            [
                np.nextafter(lower, -np.inf),
                lower,
                lower,
                upper,
                upper,
                np.nextafter(upper, np.inf),
            ],
        ),
    )
    np.testing.assert_array_equal(bounded.window(100.0, ppm=10), [1, 2, 3, 4])


def pipeline_fragments() -> FragmentMassStore:
    """Give pool row i the neutral fragments 10(i + 1) and 20(i + 1) Da."""
    return FragmentMassStore(
        offsets=np.arange(6, dtype=np.int64) * 2,
        mass=np.array(
            [10.0 * (row + 1) * scale for row in range(5) for scale in (1, 2)],
        ),
    )


def pipeline_bank(directory: Path) -> SecondBank:
    """Pin two single and two merged eight-bit checkpoints of the seeded states."""
    states = second_states()
    files: list[tuple[Path, str]] = []
    for index, name in enumerate(("single_a", "merged_b", "single_c", "merged_d")):
        state = {key: value.clone() for key, value in states[index % 2].items()}
        state["head.3.weight"] = torch.cat(
            # SecondBank adds one output row to this seven-bit source head.
            [state["head.3.weight"], torch.zeros((1, 9))],
        )
        state["head.3.bias"] = torch.cat([state["head.3.bias"], torch.zeros(1)])
        state["head.3.bias"] += 0.125 * index
        path = directory / f"{name}.pt"
        torch.save({"model": state, "nbits": 8, "d": 8, "layers": 1}, path)
        files.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    config = tiny_bank(single=(files[0], files[2]), merged=(files[1], files[3]))
    config.model.channels_out = 8
    with torch.random.fork_rng(devices=[]):
        return config.make()


def test_second_pipeline_composes_retrieval_fragments_bank_and_rankers(
    tmp_path: Path,
) -> None:
    _, original_pool, _ = retrieval_inputs()
    pool = SecondPool(
        mass=original_pool.mass,
        keys=original_pool.keys,
        smiles=original_pool.smiles,
        packed=original_pool.packed,
        bits=np.arange(8, dtype=np.int64),
    )
    raw = second_library()
    fragments = pipeline_fragments()
    bank = pipeline_bank(tmp_path)
    assert len(bank.single) == len(bank.merged) == 2
    engine = SecondEngine.Config().make()
    index = engine.retrieval.index(raw)
    bases: list[SecondBase] = []
    for target in (100.0, 100.004, 900.0, float("nan")):
        query = pipeline_queries(target)
        base = engine.base(
            query,
            pool=pool,
            library=raw,
            index=index,
            fragments=fragments,
            bank=bank,
        )
        bases.append(base)
        assert base.own_features.dtype == base.pv_features.dtype == np.float32
        assert base.spectra == prepare_queries(query, margin=2.0)
        numeric_query = [s.copy() for s in query]
        for spectrum in numeric_query:
            spectrum["mode"] = 1 if spectrum["mode"] == "positive" else -1
        numeric_base = engine.base(
            numeric_query,
            pool=pool,
            library=raw,
            index=index,
            fragments=fragments,
            bank=bank,
        )
        np.testing.assert_array_equal(numeric_base.logits, base.logits)
        np.testing.assert_array_equal(numeric_base.own_features, base.own_features)
        np.testing.assert_array_equal(numeric_base.pv_features, base.pv_features)
    exact, fallback, distant, unknown = bases
    for base in (exact, fallback, distant):
        assert base.target is not None
        assert base.channels is not None
        assert base.logits is not None
        assert base.logits.shape == (8,)
    assert exact.target == pytest.approx(100.0)
    assert len(exact.rows)
    np.testing.assert_array_equal(exact.rows, pool.window(100.0, ppm=10))
    # No pool mass lies within 10 ppm of 100.004 Da; the 30 ppm fallback reaches two.
    assert fallback.target is not None
    assert not len(pool.window(fallback.target, ppm=10))
    np.testing.assert_array_equal(fallback.rows, [2, 3])
    for base in (exact, fallback):
        assert base.channels is not None
        assert base.fragments is not None
        assert base.own_features.shape == (len(base.rows), 51)
        assert base.pv_features.shape == (len(base.rows), 31)
        evidence = second_fragments(
            base.spectra,
            rows=base.rows,
            fragments=fragments,
            search=engine.fragment_search,
        )
        np.testing.assert_array_equal(base.fragments.maximum, evidence.maximum)
        np.testing.assert_array_equal(base.fragments.mean, evidence.mean)
        np.testing.assert_array_equal(
            base.own_features,
            second_features(
                pool,
                rows=base.rows,
                channels=base.channels,
                fragments=base.fragments,
                logits=None,
            ),
        )
        np.testing.assert_array_equal(
            base.pv_features,
            second_features(
                pool,
                rows=base.rows,
                channels=base.channels,
                fragments=base.fragments,
                logits=base.logits,
            )[:, :31],
        )
    assert len(distant.rows) == 0
    assert distant.fragments is None
    assert distant.own_features.shape == (0, 51)
    assert (unknown.target, unknown.channels, unknown.logits) == (None, None, None)
    assert len(unknown.rows) == 0
    rank_config = SecondRankerBank.Config()
    rank_config.own.priors, rank_config.own.seeds = [0.35, 0.55], [0, 1]
    rank_config.pv.priors, rank_config.pv.seeds = [0.30, 0.60], [0, 1, 2, 3]
    rank_config.weight_fingerprint = 0.88
    for family in (rank_config.own, rank_config.pv):
        family.tree.max_depth = 2
        family.tree.rounds = 3
        family.tree.min_leaf = 1
        family.tree.max_bins = 2
        family.tree.max_leaves = 2
    rankers = rank_config.make()
    positions = np.arange(4)
    training: list[SimulationRows] = []
    for family, width in enumerate((51, 31)):
        data = np.zeros((4, width), np.float32)
        data[:, :2] = np.column_stack([positions / 3, positions % 3])
        training.append(
            SimulationRows(
                data=data.astype(np.float64),
                labels=(positions >= 2).astype(np.float64),
                groups=np.not_equal(positions % 3, 0).astype(np.float64),
                blocks=("base", "libx", "anapol", "fragx") if family == 0 else (),
            ),
        )
    _ = rankers.fit(training[0], pv=training[1])
    for target, base in zip((100.0, 100.004, 900.0, float("nan")), bases, strict=True):
        result = engine.run(
            pipeline_queries(target),
            pool=pool,
            library=raw,
            index=index,
            fragments=fragments,
            bank=bank,
            rankers=rankers,
            formulas=original_pool.formulas,
        )
        assert result.features.shape == (len(base.rows), 82)
        assert len(set(result.feature_names)) == 82
        assert result.feature_names[0] == "own__base_00"
        assert result.feature_names[51] == "pv__base_00"
        np.testing.assert_array_equal(
            result.features,
            np.hstack([base.own_features, base.pv_features]),
        )
        assert result.candidates.smiles == tuple(
            pool.smiles[base.rows.item(i)] for i in range(len(base.rows))
        )
        assert result.candidates.keys == tuple(
            pool.keys[base.rows.item(i)] for i in range(len(base.rows))
        )
        assert result.candidates.formulas == tuple(
            original_pool.formulas[base.rows.item(i)] for i in range(len(base.rows))
        )
        np.testing.assert_array_equal(result.candidates.pool_rows, base.rows)
        assert result.candidates.generated.dtype == np.bool_
        assert not result.candidates.generated.any()
        assert result.candidates.sources == ("pool",) * len(base.rows)
        assert result.scores.dtype == np.float64
        assert result.diagnostics.n_cand == len(base.rows)
        if not len(base.rows):
            assert result.scores.shape == (0,)
            assert result.diagnostics.lib_max == result.diagnostics.fzmax == 0.0
            continue
        scores = rankers.score(base.own_features, pv_features=base.pv_features)
        np.testing.assert_array_equal(result.scores, scores.blend)
        assert result.diagnostics.lib_max == base.own_features[:, 0].max()
    with pytest.raises(
        ValueError,
        match=r"^Engine2 formulas require an explicit aligned pool join\.$",
    ):
        _ = engine.run(
            [],
            pool=pool,
            library=raw,
            index=index,
            fragments=fragments,
            bank=bank,
            rankers=rankers,
            formulas=(),
        )
    for blocks in (("base",), ("base", "base", "libx", "fragx")):
        rankers.blocks = blocks
        with pytest.raises(
            ValueError,
            match=r"^Engine2 own rankers require all four complete feature blocks\.$",
        ):
            _ = engine.run(
                [],
                pool=pool,
                library=raw,
                index=index,
                fragments=fragments,
                bank=bank,
                rankers=rankers,
                formulas=original_pool.formulas,
            )
    overflow_query: list[SpectrumRequest] = [
        {
            "mz": [],
            "it": [],
            "prec": 2**62,
            "mode": 1,
            "adduct": "[M+2H]2+",
        },
    ]
    with torch.random.fork_rng(devices=[]):
        empty_bank = tiny_bank().make()
    overflow = engine.base(
        overflow_query,
        pool=pool,
        library=raw,
        index=index,
        fragments=fragments,
        bank=empty_bank,
    )
    assert overflow.target == float(2**63)
    assert len(overflow.rows) == 0


def test_query_preparation_keeps_zero_peaks_and_original_metadata() -> None:
    raw: list[SpectrumRequest] = [
        {
            "mz": [0.0, -1, 101, 102, 102.001],
            "it": [-1, 0, 0, -2, 5],
            "prec": 100.0,
            "mode": 1,
            "adduct": "[M+H]+",
            "ce_ev": None,
        },
    ]
    prepared = prepare_queries(raw, margin=2.0)
    assert prepared[0]["mz"] == [0.0, -1.0, 101.0, 102.0]
    assert prepared[0]["it"] == [-1.0, 0.0, 0.0, -2.0]
    first = prepared[0]
    assert "ce_ev" in first
    assert first["ce_ev"] is None
    assert raw[0]["mz"] == [0.0, -1, 101, 102, 102.001]
    assert raw[0]["it"] == [-1, 0, 0, -2, 5]
    assert collision_energy([-10.0, 30.0], default=25.0) == 20.0
    assert collision_energy([], default=25.0) == 25.0
    assert collision_energy("invalid", default=7.0) == 7.0
    assert collision_energy(-3.5, default=25.0) == 3.5
    assert collision_energy("12.5", default=25.0) == 12.5
    assert np.isnan(collision_energy(None, default=25.0))


def pipeline_queries(target: float) -> list[SpectrumRequest]:
    """Return the literal pipeline queries around one target neutral mass."""
    queries: list[SpectrumRequest] = [
        {
            "mz": [20, 30, 40, 104],
            "it": [1, 2, 3, 10],
            "prec": 0.0,
            "mode": "positive",
            "adduct": "[M+H]+",
            "instrument": "timsTOF",
            "ce_ev": [-10.0, 30.0],
        },
        {
            "mz": [20, 40, 50, 60, 70],
            "it": [3, 1, 2, 1, 0.001],
            "prec": 0.0,
            "mode": "negative",
            "adduct": "[M-H]-",
            "instrument": "QTOF",
            "ce_ev": [],
        },
        {
            "mz": [],
            "it": [],
            "prec": 0.0,
            "mode": "positive",
            "adduct": "[M+H]+",
            "instrument": None,
            "ce_ev": [35.0],
        },
    ]
    for query in queries:
        query["prec"] = (100.0 if np.isnan(target) else target) + (
            1 if query["mode"] == "positive" else -1
        ) * 1.00727645217
        if np.isnan(target):
            query["adduct"] = "unknown"
    return queries


def _stump(*, max_bins: int = 4) -> HistogramBoosting.Config:
    config = HistogramBoosting.Config()
    config.max_depth, config.rounds, config.min_leaf, config.max_bins = (
        1,
        1,
        1,
        max_bins,
    )
    config.seed = 0
    return config


def test_a_newton_stump_splits_where_the_labels_separate() -> None:
    data = cast("FeatureMatrix", np.array([[0.0], [1.0], [2.0], [3.0]]))
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    model = _stump().make().fit(data, labels, np.ones(4))
    assert model.baseline.item() == 0.0
    assert [values.tolist() for values in model.thresholds] == [[0.5, 1.5, 2.5]]
    assert model.bin_counts.tolist() == [4]
    assert model.binned(data).ravel().tolist() == [0, 1, 2, 3]
    (tree,) = model.trees
    state = tree.state()
    assert state["feature_idx"].tolist() == [0, 0, 0]
    assert state["num_threshold"].tolist() == [1.5, 0.0, 0.0]
    assert state["bin_threshold"].tolist() == [1, 0, 0]
    assert state["left"].tolist() == [1, 0, 0]
    assert state["right"].tolist() == [2, 0, 0]
    assert state["is_leaf"].tolist() == [0, 1, 1]
    assert state["count"].tolist() == [4, 2, 2]
    assert state["depth"].tolist() == [0, 1, 1]
    assert state["missing_go_to_left"].tolist() == [0, 0, 0]
    # Gradients are +-0.5 and Hessians 0.25, so each leaf is -+1 / (0.5 + 1).
    assert state["value"].tolist() == pytest.approx([0.0, -0.02, 0.02], abs=1e-12)
    assert state["gain"].tolist() == pytest.approx([4 / 3, -1.0, -1.0])
    query = np.array([[0.0], [3.0], [np.nan]])
    assert model.stages(query).shape == (1, 3)
    assert model.stages(query)[0].tolist() == pytest.approx(
        [-0.02, 0.02, 0.02],
        abs=1e-12,
    )
    probabilities = model.predict_proba(query)
    assert probabilities[:, 1].tolist() == [
        _sigmoid(value) for value in state["value"][[1, 2, 2]].tolist()
    ]
    np.testing.assert_array_equal(probabilities[:, 0], 1 - probabilities[:, 1])


def test_missing_rows_split_alone_through_an_infinite_threshold() -> None:
    data = cast("FeatureMatrix", np.array([[0.0], [1.0], [np.nan], [np.nan]]))
    model = _stump().make().fit(data, np.array([0.0, 0.0, 1.0, 1.0]), np.ones(4))
    assert [values.tolist() for values in model.thresholds] == [[0.5]]
    assert model.binned(data).ravel().tolist() == [0, 1, 4, 4]
    state = model.trees[0].state()
    assert state["num_threshold"].tolist() == [math.inf, 0.0, 0.0]
    assert state["missing_go_to_left"].tolist() == [0, 0, 0]
    assert state["count"].tolist() == [4, 2, 2]
    stages = model.stages(np.array([[5.0], [np.nan]]))
    assert stages[0].tolist() == pytest.approx([-0.02, 0.02], abs=1e-12)


def test_growth_stops_at_the_leaf_cap_and_skips_hessianless_bins() -> None:
    index = np.arange(8, dtype=np.float64)
    config = _stump(max_bins=8)
    config.max_depth, config.max_leaves = 3, 2
    capped = config.make().fit(
        index[:, None],
        (index % 4 >= 2).astype(np.float64),
        np.ones(8),
    )
    assert capped.trees[0].state()["is_leaf"].tolist() == [0, 1, 1]
    # The 1e-6 weight leaves the first bin below the Hessian floor of 1e-3.
    weights = np.array([1e-6, 1.0, 1.0, 1.0, 1.0, 1.0])
    model = (
        _stump(max_bins=8)
        .make()
        .fit(
            index[:6, None],
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            weights,
        )
    )
    state = model.trees[0].state()
    assert state["num_threshold"].tolist()[0] == 2.5
    assert state["count"].tolist() == [6, 3, 3]
    light = (
        _stump()
        .make()
        .fit(index[:4, None], np.array([0.0, 0.0, 1.0, 1.0]), np.full(4, 1e-6))
    )
    assert light.trees[0].state()["is_leaf"].tolist() == [1]


def test_held_out_rows_score_every_round_until_the_patience_window() -> None:
    index = np.arange(12)
    data = cast(
        "FeatureMatrix",
        np.column_stack([index / 7, index % 3]).astype(np.float64),
    )
    labels = (index % 6 >= 3).astype(np.float64)
    config = HistogramBoosting.Config()
    config.max_depth, config.rounds, config.min_leaf, config.max_bins = 2, 3, 1, 4
    config.seed, config.early_stopping_rows = 0, 7
    model = config.make().fit(data, labels, np.ones(12))
    # Ten of twelve rows train; the initial and three updated scores are kept.
    assert [tree.state()["count"].item(0) for tree in model.trees] == [10, 10, 10]
    assert model.train_score.shape == model.validation_score.shape == (4,)
    assert np.all(model.train_score <= 0)
    # The balanced training rows start at logit zero, a loss of log 2 per row.
    assert model.train_score[0] == pytest.approx(-math.log(2))
    config.tolerance, config.patience = 1e9, 1
    stopped = config.make().fit(data, labels, np.ones(12))
    assert len(stopped.trees) == 1
    assert stopped.validation_score.shape == (2,)
    train, valid = model.validation_indices(labels)
    assert (len(train), len(valid)) == (10, 2)
    assert sorted([*train.tolist(), *valid.tolist()]) == list(range(12))
    assert sorted(labels[valid].tolist()) == [0.0, 1.0]
    plain = HistogramBoosting.Config()
    plain.max_depth, plain.rounds, plain.min_leaf, plain.max_bins = 2, 1, 1, 4
    plain.seed = 0
    assert plain.make().fit(data, labels, np.ones(12)).train_score.shape == (0,)


def test_class_allocation_rounds_expected_counts_and_breaks_ties_randomly() -> None:
    random = np.random.RandomState(0)
    allocation = HistogramBoosting.class_allocation(np.array([3, 5]), 3, random)
    # Expected counts 1.125 and 1.875 round down; the larger remainder takes the rest.
    assert allocation.tolist() == [1, 2]
    tied = HistogramBoosting.class_allocation(np.array([2, 2]), 3, random)
    assert sorted(tied.tolist()) == [1, 2]
    assert HistogramBoosting.class_allocation(np.array([4, 4]), 4, random).tolist() == [
        2,
        2,
    ]


def test_bin_thresholds_take_midpoints_or_weighted_quantiles() -> None:
    model = _stump().make()
    values = np.arange(8, dtype=np.float64)
    assert model.numeric_thresholds(
        np.array([3.0, 1.0, 2.0, np.nan]),
        None,
    ).tolist() == [1.5, 2.5]
    # With eight values and four bins the 25/50/75 % cuts fall between neighbours.
    assert model.numeric_thresholds(values, None).tolist() == [1.5, 3.5, 5.5]
    assert model.numeric_thresholds(values, np.ones(8)).tolist() == [1.5, 3.5, 5.5]
    # A zero weight drops 3; the weighted cuts then land on the values 1, 4 and 6.
    weights = np.ones(8)
    weights[3] = 0
    assert model.numeric_thresholds(values, weights).tolist() == [1.0, 4.0, 6.0]
    assert model.numeric_thresholds(np.array([0.0, math.inf]), None).tolist() == [1e300]


def test_large_inputs_bin_from_a_weighted_resample() -> None:
    index = np.arange(12)
    data = cast(
        "FeatureMatrix",
        np.column_stack([index / 7, index % 3]).astype(np.float64),
    )
    config = _stump()
    config.binning_rows = 3
    model = config.make().fit(data, (index % 6 >= 3).astype(np.float64), np.ones(12))
    # Three resampled rows leave at most two midpoints per feature.
    assert all(len(values) <= 2 for values in model.thresholds)
    assert model.bin_counts.tolist() == [len(values) + 1 for values in model.thresholds]
    assert np.all(model.binned(data) < model.bin_counts)


def test_derivatives_and_loss_take_the_stable_tail_of_each_logit_range() -> None:
    gradient, hessian = HistogramBoosting.derivatives(
        np.array([0.0, -40.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 1.0, 1.0]),
    )
    assert gradient.dtype == hessian.dtype == np.float32
    assert gradient.tolist() == [-0.25, np.float32(math.exp(-40)), 0.5]
    assert hessian.tolist() == [0.125, np.float32(math.exp(-40)), 0.25]
    ones = np.ones(1)
    for logit, label, expected in (
        (-40.0, 0.0, -math.exp(-40)),
        (-3.0, 0.0, -math.log1p(math.exp(-3))),
        (0.0, 1.0, -math.log(2)),
        (40.0, 1.0, 0.0),
    ):
        assert (
            HistogramBoosting.loss_score(np.array([logit]), np.array([label]), ones)
            == expected
        )
    # Above 18 the loss is the logit plus exp(-logit); above 33.3 the logit alone.
    tail = HistogramBoosting.loss_score(np.array([25.0]), ones, ones)
    assert -2e-11 < tail < -1e-11
    weighted = HistogramBoosting.loss_score(
        np.array([0.0, 40.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
    )
    assert weighted == -math.log(2) / 4


def test_stopping_needs_a_full_window_without_improvement() -> None:
    config = _stump()
    config.patience, config.tolerance = 2, 0.5
    model = config.make()
    assert not model.should_stop([0.0, 1.0])
    assert model.should_stop([0.0, 1.0, 1.0, 1.25])
    assert not model.should_stop([0.0, 1.0, 1.0, 1.75])


def test_boosting_requires_the_seed_that_its_owner_assigns() -> None:
    with pytest.raises(ValueError, match="requires a nonnegative seed"):
        _ = HistogramBoosting.Config().make()
    config = HistogramBoosting.Config()
    config.seed = 0
    assert config.make().config.seed == 0


def _small_model(
    *,
    tail_weight: float = 1.0,
) -> tuple[HistogramBoosting, FeatureMatrix]:
    index = np.arange(12, dtype=np.int64)
    data: FeatureMatrix = np.zeros((12, 3), dtype=np.float64)
    data[:, 0], data[:, 1], data[:, 2] = index / 7, (index * 5 % 11) / 3, index % 3
    labels = ((index % 6) >= 3).astype(np.float64)
    weights = np.where(index >= 9, tail_weight, 1.0)
    config = HistogramBoosting.Config()
    config.max_depth, config.rounds, config.min_leaf, config.max_bins = 3, 1, 2, 4
    config.seed = 0
    return config.make().fit(data, labels, weights), data


def _grow(model: HistogramBoosting, data: FeatureMatrix) -> Node:
    gradient, hessian = HistogramBoosting.derivatives(
        np.zeros(12),
        (np.arange(12) % 2).astype(np.float64),
        np.ones(12),
    )
    return model.grow(
        model.binned(data),
        gradient.astype(np.float32),
        hessian.astype(np.float32),
        np.zeros(3, dtype=np.bool_),
    )


def test_a_split_leaving_a_hessianless_complement_ends_the_scan() -> None:
    model, _ = _small_model()
    bins = model.config.max_bins + 1
    hessian = np.tile(np.array([1.0, 1.0, 1e-6, 1e-6, 0.0]), (3, 1))
    assert hessian.shape[1] == bins
    histogram = Histogram(
        gradient=np.tile(np.array([1.0, -1.0, 0.5, -0.5, 0.0]), (3, 1)),
        hessian=hessian,
        count=np.tile(np.array([3, 3, 3, 3, 0], np.uint32), (3, 1)),
    )
    node = Node(
        samples=np.arange(12, dtype=np.uint32),
        depth=0,
        gradient=0.0,
        hessian=2.000002,
        value=0.0,
        histogram=histogram,
    )
    split = model.best_split(node, np.zeros(3, dtype=np.bool_))
    assert split is not None
    assert split.threshold == 0


def test_each_class_needs_two_rows_and_a_place_in_both_splits() -> None:
    model, _ = _small_model()
    with pytest.raises(ValueError, match="two rows"):
        _ = model.validation_indices(np.array([0.0, 0.0, 0.0, 1.0]))


def test_scanning_a_node_without_a_histogram_is_an_error() -> None:
    model, _ = _small_model()
    node = Node(
        samples=np.arange(4, dtype=np.uint32),
        depth=0,
        gradient=0.0,
        hessian=1.0,
        value=0.0,
    )
    with pytest.raises(ValueError, match="histogram"):
        _ = model.best_split(node, np.zeros(3, dtype=np.bool_))


def test_growing_rejects_a_queued_node_without_a_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = _small_model()

    def queue_unsplit(node: Node, pending: list[Node], missing: object) -> None:
        del missing
        heapq.heappush(pending, node)

    monkeypatch.setattr(model, "enqueue", queue_unsplit)
    with pytest.raises(ValueError, match="split"):
        _ = _grow(model, data)


def test_growing_rejects_a_split_parent_that_lost_its_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = _small_model()
    enqueue = model.enqueue

    def drop_histogram(
        node: Node,
        pending: list[Node],
        missing: NDArray[np.bool_],
    ) -> None:
        enqueue(node, pending, missing)
        node.histogram = None

    monkeypatch.setattr(model, "enqueue", drop_histogram)
    with pytest.raises(ValueError, match="parent"):
        _ = _grow(model, data)


def test_nodes_order_by_descending_gain_for_best_first_growth() -> None:
    high, low = (
        Node(
            samples=np.zeros(0, np.uint32),
            depth=0,
            gradient=0.0,
            hessian=0.0,
            value=0.0,
            gain=gain,
        )
        for gain in (2.0, 1.0)
    )
    assert high < low
    assert sorted([low, high]) == [high, low]
    pending = [low, high]
    heapq.heapify(pending)
    assert heapq.heappop(pending) is high


def test_exporting_an_internal_node_without_a_split_is_an_error() -> None:
    model, _ = _small_model()
    leaf = Node(
        samples=np.arange(2, dtype=np.uint32),
        depth=1,
        gradient=0.0,
        hessian=1.0,
        value=0.0,
    )
    root = Node(
        samples=np.arange(4, dtype=np.uint32),
        depth=0,
        gradient=0.0,
        hessian=2.0,
        value=0.0,
        left=leaf,
        right=leaf,
    )
    with pytest.raises(ValueError, match="split"):
        _ = model.export(root)


def test_ensembles_fit_every_prior_then_seed_and_average_probabilities(
    tmp_path: Path,
) -> None:
    indices = np.arange(8, dtype=np.int64)
    data = np.column_stack([indices / 7, indices % 3]).astype(np.float32)
    labels = (indices % 4 >= 2).astype(np.float64)
    groups = np.not_equal(indices % 3, 0).astype(np.int8)
    path = tmp_path / "simulation.npz"
    np.savez(
        path,
        X=data,
        Y=labels,
        S=groups,
        M=1 - groups,
        blocks=np.array(["base", "libx", "anapol", "fragx"]),
    )
    rows = SimulationRows.load(path, "S")
    assert rows.blocks == ("base", "libx", "anapol", "fragx")
    assert rows.groups.tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert SimulationRows.load(path, "M").groups.tolist() == [1, 0, 0, 1, 0, 0, 1, 0]
    config = HistogramEnsemble.Config()
    config.tree.max_depth = 2
    config.tree.rounds = 3
    config.tree.min_leaf = 2
    config.tree.max_bins = 3
    config.priors, config.seeds = [0.25, 0.5], [0, 1]
    ensemble = config.make().fit(rows)
    assert [model.config.seed for model in ensemble.models] == [0, 1, 0, 1]
    # Group-0 rows 0, 3 and 6 carry the prior; rows 2, 3, 6 and 7 are positive.
    for model, positive, total in zip(
        ensemble.models,
        (2.0, 2.0, 2.0, 2.0),
        (4.5, 4.5, 4.0, 4.0),
        strict=True,
    ):
        mean = positive / total
        assert model.baseline.item() == math.log(mean) - math.log1p(-mean)
        assert len(model.trees) == 3
    query = np.vstack(
        [data.astype(np.float64), [[np.nan, np.nan], [-10, -10], [100, 100]]],
    )
    probabilities = np.stack(
        [model.predict_proba(query)[:, 1] for model in ensemble.models],
    )
    mean = ensemble.predict(query)
    assert mean.shape == (11,)
    assert np.all((probabilities.min(0) <= mean) & (mean <= probabilities.max(0)))
    config.priors = [0.25]
    pair = config.make().fit(rows)
    first, second = (model.predict_proba(query)[:, 1] for model in pair.models)
    np.testing.assert_array_equal(pair.predict(query), (first + second) / 2)


def test_blend_mixes_reversed_ordinal_ranks_with_a_fingerprint_tiebreak() -> None:
    fingerprint = np.array([0.2, 0.4, 0.4])
    own = np.array([0.9, 0.1, 0.5])
    blend = blend_histograms(fingerprint, own=own, weight_fingerprint=0.88)
    # Fingerprint ties keep encounter order: ranks [1, 0, 0.5]; own ranks [0, 1, 0.5].
    expected = [0.12 + 0.2e-6, 0.88 + 0.4e-6, 0.5 + 0.4e-6]
    assert blend.tolist() == pytest.approx(expected, abs=1e-12)
    assert np.argsort(-blend).tolist() == [1, 2, 0]
    assert blend_histograms(fingerprint, own=own, weight_fingerprint=0.0).tolist() == (
        pytest.approx([1.0 + 0.2e-6, 0.0 + 0.4e-6, 0.5 + 0.4e-6], abs=1e-12)
    )


def test_simulation_reader_accepts_second_without_block_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "simulation.npz"
    np.savez(
        path,
        X=np.zeros((3, 2), dtype=np.float32),
        Y=np.array([0, 1, 0]),
        M=np.array([0, 0, 1]),
    )
    rows = SimulationRows.load(path, "M")
    assert rows.blocks == ()
    assert rows.data.dtype == rows.labels.dtype == rows.groups.dtype == np.float64


def test_simulation_reader_rejects_misaligned_rows(tmp_path: Path) -> None:
    path = tmp_path / "simulation.npz"
    np.savez(path, X=np.zeros((3, 2), dtype=np.float32), Y=np.zeros(2), S=np.zeros(3))
    with pytest.raises(
        ValueError,
        match=r"^Simulation features, labels and groups need aligned rows\.$",
    ):
        _ = SimulationRows.load(path, "S")


def test_simulation_reader_refuses_object_payloads(tmp_path: Path) -> None:
    path = tmp_path / "simulation.npz"
    np.savez(path, X=np.array([[1, 2]], dtype=object), Y=np.zeros(2), S=np.zeros(2))
    with pytest.raises(ValueError, match="Object arrays"):
        _ = SimulationRows.load(path, "S")


def test_histogram_ensemble_rejects_prediction_before_fitting() -> None:
    config = HistogramEnsemble.Config()
    config.priors, config.seeds = [0.5], [0]
    with pytest.raises(ValueError, match="fitted"):
        _ = config.make().predict(np.zeros((3, 2)))


def test_histogram_ensemble_requires_declared_priors_and_seeds() -> None:
    with pytest.raises(ValueError, match="requires priors and seeds"):
        _ = HistogramEnsemble.Config().make()


def engine_keys() -> dict[str, str | None]:
    """Return staged InChIKey14 decisions: a wildcard, a failure, empty and ions."""
    return {
        "*C": "",
        "invalid": None,
        "": "",
        "C": "VNWKTOKETHGBQD",
        "CC": "OTMSDBZUPAUEDD",
        "CCO": "LFQSCWFLJHTTHZ",
        "CCCC": "IJDNQMDRQITEOD",
        "C/C=C/C": "IAQRGUVFOMOMEM",
        "C[N+](C)(C)C": "QEMXHQIAXOOASZ",
        "CCO.[Na+].[Cl-]": "LOHGGLZYTJNUAL",
        "[13CH3]O": "OKKJLVBELUTLKV",
    }


def test_every_staged_engine_key_and_first_duplicate_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    keys = engine_keys()
    rows = [{"smiles": smiles, "key": value} for smiles, value in keys.items()]
    parquet.write_table(
        Table.from_pylist([*rows, {"smiles": "*C", "key": "second"}]),
        tmp_path / "keys.parquet",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            to_plain(
                {
                    "format_version": 1,
                    "rdkit": IDENTITY_RDKIT,
                    "rows": len(rows) + 1,
                },
            ),
        ),
    )
    table = EngineIdentityTable(tmp_path)
    assert {smiles: table.key(smiles) for smiles in keys} == keys
    assert table.for_smiles("*C") == {"smiles": "*C", "key": ""}
    assert table.key("") == ""
    assert table.key("invalid") is None
    with pytest.raises(KeyError, match=r"^'unstaged'$"):
        table.for_smiles("unstaged")
    for header in (
        {"format_version": 2, "rdkit": IDENTITY_RDKIT, "rows": len(rows) + 1},
        {"format_version": 1, "rdkit": "different", "rows": len(rows) + 1},
        {"format_version": 1, "rdkit": IDENTITY_RDKIT, "rows": len(rows)},
    ):
        (tmp_path / "manifest.json").write_text(json.dumps(to_plain(header)))
        with pytest.raises(
            ValueError,
            match=(
                r"^Engine identities require format 1 and RDKit 2026\.03\.3\.$"
                r"|^Engine identity row count differs from its manifest\.$"
            ),
        ):
            EngineIdentityTable(tmp_path)


def _checkpoint(path: Path, state: dict[str, Tensor]) -> str:
    torch.save({"nbits": 7, "d": 8, "layers": 1, "model": state}, path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tiny_bank(
    single: tuple[tuple[Path, str], ...] = (),
    merged: tuple[tuple[Path, str], ...] = (),
    *,
    device: str = "cpu",
) -> SecondBank.Config:
    """Declare the checkpoints and the geometry of their tiny model."""
    config = SecondBank.Config()
    config.model = tiny_second()
    config.model.adduct.channels_in = 26
    config.single = [LocalFile.Config(path=path, sha256=seal) for path, seal in single]
    config.merged = [LocalFile.Config(path=path, sha256=seal) for path, seal in merged]
    config.device = device
    return config


def test_entire_bank_through_complete_checkpoint_assembly(
    tmp_path: Path,
) -> None:
    states = second_states()
    paths = (
        tmp_path / "first.pt",
        tmp_path / "second.pt",
        tmp_path / "third.pt",
    )
    sources = [*states, states[1]]
    seals = tuple(
        _checkpoint(path, state) for path, state in zip(paths, sources, strict=True)
    )
    spectra = bank_spectra()
    with torch.random.fork_rng(devices=[]):
        bank = tiny_bank(
            single=tuple(zip(paths[:2], seals[:2], strict=True)),
            merged=((paths[2], seals[2]),),
        ).make()
        empty = tiny_bank().make()
    assert len(bank.single) == 2
    assert len(bank.merged) == 1
    for model, state in zip((*bank.single, *bank.merged), sources, strict=True):
        assert not model.training
        released = model.released_state()
        assert released.keys() == state.keys()
        assert all(torch.equal(released[name], value) for name, value in state.items())
    first, second = second_models()
    expected = _predict(spectra, single=(first, second), merged=(second,))
    result = bank.predict(spectra)
    assert result is not None
    assert expected is not None
    np.testing.assert_array_equal(result, expected)
    assert empty.predict(spectra) is None
    silent = QuerySpectrum(
        mass=np.zeros(0),
        intensity=np.zeros(0),
        precursor=100.0,
        adduct="unknown",
        instrument=None,
        collision_energy=0.0,
        polarity=1.0,
    )
    assert bank.predict([silent]) is None
    wrong = "0" * 64
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = tiny_bank(single=((paths[0], wrong),)).make()
    damaged = tmp_path / "damaged.pt"
    bad_state = dict(states[0])
    del bad_state["ins.weight"]
    seal = _checkpoint(damaged, bad_state)
    with pytest.raises(ValueError, match="Checkpoint keys"):
        _ = tiny_bank(single=((damaged, seal),)).make()
    config = tiny_bank(single=((paths[0], seals[0]),))
    config.model.num_layers = 2
    with pytest.raises(ValueError, match="widths differ from the declared model"):
        _ = config.make()


def test_checkpoint_is_read_as_cpu_tensors_before_target_device_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.pt"
    seal = _checkpoint(path, second_states()[0])
    original = torch.load
    calls: list[Path] = []

    def tensor_load(path: Path, *, map_location: str, weights_only: bool) -> object:
        assert map_location == "cpu"
        assert weights_only is True
        calls.append(path)
        return cast(
            object,
            original(path, map_location=map_location, weights_only=weights_only),
        )

    monkeypatch.setattr(torch, "load", tensor_load)
    with torch.random.fork_rng(devices=[]):
        bank = tiny_bank(single=((path, seal),), device="meta").make()
    assert calls == [path]
    assert all(
        value.device.type == "meta"
        for value in bank.single[0].released_state().values()
    )


def test_bank_builds_each_family_from_its_own_independent_slot() -> None:
    config = _tiny_config().copy_tree().finalize()
    assert (config.own.priors, config.own.seeds) == ([0.35, 0.55], [0, 1])
    assert (config.pv.priors, config.pv.seeds) == ([0.30, 0.60], [0, 1, 2, 3])
    assert config.own.tree == config.pv.tree
    bank = config.make()
    assert bank.blocks == ()
    assert bank.own.config == config.own
    assert bank.pv.config == config.pv
    assert bank.own is not bank.pv


def test_bank_fits_both_families_like_standalone_ensembles() -> None:
    bank = _tiny_config().make()
    rows, query = _simulation()
    pv_rows = replace(rows, blocks=())
    assert bank.fit(rows, pv=pv_rows) is bank
    assert bank.blocks == ("base", "libx", "anapol", "fragx")
    assert (len(bank.own.models), len(bank.pv.models)) == (4, 8)
    scores = bank.score(query, pv_features=query[::-1])
    config = _tiny_config()
    np.testing.assert_array_equal(
        scores.own,
        config.own.make().fit(rows).predict(query),
    )
    np.testing.assert_array_equal(
        scores.fingerprint,
        config.pv.make().fit(pv_rows).predict(query[::-1]),
    )
    np.testing.assert_array_equal(
        scores.blend,
        blend_histograms(scores.fingerprint, own=scores.own, weight_fingerprint=0.88),
    )


def test_bank_defers_unfitted_prediction_to_the_fitted_ensemble_contract() -> None:
    _, query = _simulation()
    with pytest.raises(ValueError, match="fitted"):
        _ = _tiny_config().make().score(query, pv_features=query)


def test_ranker_slots_and_blend_weight_are_injectable() -> None:
    config = _tiny_config()
    config.own.priors = [0.35]
    config.own.seeds = [1]
    config.pv.priors = [0.60]
    config.pv.seeds = [2]
    config.weight_fingerprint = 0
    bank = config.make()
    rows, query = _simulation()
    _ = bank.fit(rows, pv=rows)
    scores = bank.score(query, pv_features=query[::-1])
    assert [model.config.seed for model in bank.own.models] == [1]
    assert [model.config.seed for model in bank.pv.models] == [2]
    expected = blend_histograms(
        scores.fingerprint,
        own=scores.own,
        weight_fingerprint=0,
    )
    np.testing.assert_array_equal(scores.blend, expected)
    empty = bank.score(query[:0], pv_features=query[:0])
    assert empty.own.shape == empty.fingerprint.shape == empty.blend.shape == (0,)


def _tiny_config() -> SecondRankerBank.Config:
    config = SecondRankerBank.Config()
    config.own.priors, config.own.seeds = [0.35, 0.55], [0, 1]
    config.pv.priors, config.pv.seeds = [0.30, 0.60], [0, 1, 2, 3]
    config.weight_fingerprint = 0.88
    for family in (config.own, config.pv):
        family.tree.max_depth = 2
        family.tree.rounds = 3
        family.tree.min_leaf = 2
        family.tree.max_bins = 3
    return config


def _simulation() -> tuple[SimulationRows, PredictionMatrix]:
    indices = np.arange(8, dtype=np.int64)
    data = np.column_stack([indices / 7, indices % 3]).astype(np.float32)
    rows = SimulationRows(
        data=cast("FeatureMatrix", data.astype(np.float64)),
        labels=(indices % 4 >= 2).astype(np.float64),
        groups=np.not_equal(indices % 3, 0).astype(np.float64),
        blocks=("base", "libx", "anapol", "fragx"),
    )
    query = np.vstack([data, [[np.nan, np.nan], [-10, -10], [100, 100]]])
    return rows, cast("PredictionMatrix", query.astype(np.float32))


def test_complete_second_assets_preserve_raw_rows_fragments_keys_and_twelve_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = second_config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        inputs = config.make().read()
    chemistry = config.chemistry.copy_tree().make()
    assert inputs.formulas == tuple(
        chemistry.formula(value) for value in inputs.pool.smiles
    )
    assert inputs.identities.key(inputs.pool.smiles[0]) is None
    assert inputs.identities.key(inputs.pool.smiles[1]) == ""
    for name, values in (
        ("nm", inputs.library.mass),
        ("row_p", inputs.library.pool_rows),
        ("pol", inputs.library.polarity),
        ("tims", inputs.library.timstof),
        ("prec", inputs.library.precursor),
        ("off", inputs.library.store.offsets),
        ("mz", inputs.library.store.mz),
        ("it", inputs.library.store.intensity),
    ):
        expected = cast(
            "NDArray[np.generic]",
            np.load(Path(config.library_dir) / f"{name}.npy", allow_pickle=False),
        )
        np.testing.assert_array_equal(values, expected)
        assert not values.flags.writeable
    np.testing.assert_array_equal(inputs.fragments.offsets, [0, 2, 3])
    np.testing.assert_array_equal(
        inputs.fragments.mass,
        [100.0000000001, 11.111111111, 12.222222222],
    )
    assert inputs.fragments.mass.dtype == np.float64
    assert inputs.rankers.blocks == ("base", "libx", "anapol", "fragx")
    assert len(inputs.bank.merged) == len(inputs.bank.single) == 1
    for model, state in zip(
        (*inputs.bank.merged, *inputs.bank.single),
        second_states(),
        strict=True,
    ):
        released = model.released_state()
        assert released.keys() == state.keys()
        for name, value in released.items():
            assert torch.equal(value, state[name])
    rows, query = _simulation()
    scores = inputs.rankers.score(query, pv_features=query)
    direct = (
        _tiny_config()
        .make()
        .fit(rows, pv=replace(rows, blocks=()))
        .score(query, pv_features=query)
    )
    np.testing.assert_array_equal(scores.own, direct.own)
    np.testing.assert_array_equal(scores.fingerprint, direct.fingerprint)
    np.testing.assert_array_equal(scores.blend, direct.blend)
    mass = Path(config.library_dir) / "nm.npy"
    np.save(mass, np.zeros(3, np.float32))
    with pytest.raises(TypeError, match=r"nm.npy: expected float64, got float32\.$"):
        config.make().read()
    # SecondAssets.read requires a flat molecular-mass vector, not a matrix.
    np.save(mass, np.zeros((3, 1), np.float64))
    with pytest.raises(
        ValueError,
        match=r"^Second-arm array nm.npy has an invalid rank\.$",
    ):
        config.make().read()


@pytest.mark.parametrize(
    ("location", "name", "values", "message"),
    [
        ("library", "pol", [1], "Second-arm library metadata rows must be aligned."),
        (
            "library",
            "it",
            [1.0],
            "Second-arm raw peak masses and intensities must be aligned.",
        ),
        (
            "library",
            "off",
            [0, 3, 2, 3],
            "Second-arm offsets must cover every complete ordered row.",
        ),
        (
            "fragments",
            "frag_off",
            [1, 2, 3],
            "Second-arm offsets must cover every complete ordered row.",
        ),
        (
            "fragments",
            "frag_off",
            [0, 2],
            "Second-arm offsets must cover every complete ordered row.",
        ),
        (
            "fragments",
            "frag_off",
            [0, 1, 2],
            "Second-arm offsets must cover every complete ordered row.",
        ),
    ],
)
def test_second_assets_reject_incomplete_ordered_rows(
    tmp_path: Path,
    location: str,
    name: str,
    values: list[int | float],
    message: str,
) -> None:
    config = second_config(tmp_path)
    directory = Path(
        config.library_dir if location == "library" else config.fragments_dir,
    )
    dtype = np.float32 if name == "it" else np.int8 if name == "pol" else np.int64
    np.save(directory / f"{name}.npy", np.array(values, dtype=dtype))
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$") as error:
        config.make().read()
    assert str(error.value) == message


def test_second_defaults_keep_group_archives_and_full_budget() -> None:
    config = SecondAssets.Config().copy_tree().finalize()
    assert Path(config.own_training).name == "sim_rank_rows_nofp.npz"
    assert Path(config.pv_training).name == "rank_train.npz"
    assert config.rankers.own.tree.rounds == config.rankers.pv.tree.rounds == 500


def second_config(root: Path) -> SecondAssets.Config:
    """Publish complete test primitive inputs with seeded tiny bank states."""
    config = SecondAssets.Config()
    pool, library, fragments, chemistry, identities = (
        root / name
        for name in ("pool", "library", "fragments", "chemistry", "identities")
    )
    config.pool_dir, config.library_dir, config.fragments_dir = pool, library, fragments
    config.chemistry.working_dir, config.identities_dir = chemistry, identities
    for directory in (pool, library, fragments, chemistry, identities):
        directory.mkdir()
    write_identity(chemistry)
    smiles = tuple(row["smiles"] for row in identity_rows()[:2])
    parquet.write_table(
        Table.from_pydict({"key": ["duplicate", "duplicate"], "smiles": smiles}),
        pool / "pool_meta.parquet",
    )
    for name, values in (
        ("pool_mass", np.array([100, 100], np.float64)),
        ("pool_fp", np.array([[17], [33]], np.uint8)),
        ("fp_bits", np.arange(7, dtype=np.int64)),
    ):
        np.save(pool / f"{name}.npy", values)
    for name, values in (
        ("nm", np.array([100, np.nan, 100], np.float64)),
        ("row_p", np.array([0, -1, 1], np.int64)),
        ("pol", np.array([1, -1, -1], np.int8)),
        ("tims", np.array([True, False, True])),
        ("prec", np.array([101, 99, 100], np.float64)),
        ("off", np.array([0, 2, 2, 3], np.int64)),
        ("mz", np.array([0, 500, 12], np.float32)),
        ("it", np.array([1, -2, 3], np.float32)),
    ):
        np.save(library / f"{name}.npy", values)
    np.save(fragments / "frag_off.npy", np.array([0, 2, 3], np.int64))
    np.save(
        fragments / "frag_mass.npy",
        np.array([100.0000000001, 11.111111111, 12.222222222], np.float64),
    )
    parquet.write_table(
        Table.from_pydict({"smiles": smiles, "key": [None, ""]}),
        identities / "keys.parquet",
    )
    (identities / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT, "rows": 2})),
    )
    rows, _ = _simulation()
    config.own_training = root / "own.npz"
    config.pv_training = root / "pv.npz"
    np.savez(
        config.own_training,
        X=rows.data,
        Y=rows.labels,
        S=rows.groups,
        M=1 - rows.groups,
        blocks=np.asarray(rows.blocks),
    )
    np.savez(
        config.pv_training,
        X=rows.data,
        Y=rows.labels,
        M=rows.groups,
        S=1 - rows.groups,
    )
    config.rankers.own.priors, config.rankers.own.seeds = [0.35, 0.55], [0, 1]
    config.rankers.pv.priors, config.rankers.pv.seeds = [0.30, 0.60], [0, 1, 2, 3]
    config.rankers.weight_fingerprint = 0.88
    for ensemble in (config.rankers.own, config.rankers.pv):
        ensemble.tree.max_depth = 2
        ensemble.tree.rounds = 3
        ensemble.tree.min_leaf = 2
        ensemble.tree.max_bins = 3
    states = second_states()
    paths: list[Path] = []
    for index, name in enumerate(("fp_merged_m1.pt", "fp_single_s2.pt")):
        path = root / name
        torch.save({"model": states[index], "nbits": 7, "d": 8, "layers": 1}, path)
        paths.append(path)
    seals = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    config.bank = tiny_bank(
        single=((paths[1], seals[1]),),
        merged=((paths[0], seals[0]),),
    )
    return config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
