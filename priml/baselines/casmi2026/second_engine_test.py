"""Checks for the second engine module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import copy
import hashlib
import heapq
import json
import math
import re
import sys

from numpy.typing import NDArray
from pyarrow import Table, parquet
from torch import Tensor
from treekle import json_loads, to_plain

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
from priml.baselines.casmi2026.main_engine import view_features
from priml.baselines.casmi2026.model import SecondFPNet
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
    float32_comparands,
    identity_rows,
    retrieval_inputs,
    write_identity,
)
from priml.model.ae.checkpoint import LocalFile
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    mismatches,
    read_tensors,
)


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from priml.baselines.casmi2026.custom_types import (
        PredictionMatrix,
    )


_BASELINE: Final = _CWD


BANK_GOLDEN: Final = _BASELINE / "testdata" / "bank.pt"


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


def preprocessing_record() -> dict[str, Tensor]:
    """Exercise every filtering, window quota and adjacent-merge branch."""
    record: dict[str, Tensor] = {}
    for name, mass, intensity, precursor, maximum, window, per_window in (
        ("empty", list[float](), list[float](), 100.0, 5, 50.0, 2),
        ("all_over", [105.0, 106.0], [2.0, 1.0], 100.0, 5, 50.0, 2),
        ("zero", [10.0, 20.0], [0.0, -1.0], 100.0, 5, 50.0, 2),
        (
            "floor",
            [0.0, -1.0, 100.0, 101.5, 101.6],
            [1.0, 2.0, 0.001, 3.0, 5.0],
            100.0,
            5,
            50.0,
            2,
        ),
        (
            "quota_trim",
            np.arange(17) * 21.0 + 5,
            np.arange(17) + 1.0,
            500.0,
            5,
            50.0,
            2,
        ),
        ("quota_fill", np.arange(17) + 5.0, np.arange(17) + 1.0, 100.0, 5, 50.0, 2),
        (
            "quota_exact",
            np.arange(17) * 10.0 + 5,
            np.arange(17) + 1.0,
            500.0,
            4,
            50.0,
            1,
        ),
        (
            "production_limit",
            np.arange(145) * 1.7 + 5,
            np.arange(145) + 1.0,
            300.0,
            128,
            50.0,
            8,
        ),
    ):
        mz, it = peaks(
            max_peaks=maximum,
            window=window,
            per_window=per_window,
        ).prepare(
            np.asarray(mass, np.float64),
            np.asarray(intensity, np.float64),
            precursor=precursor,
        )
        record[f"peaks/{name}/mass"] = torch.from_numpy(mz)
        record[f"peaks/{name}/intensity"] = torch.from_numpy(it)
    for name, pairs in (
        ("empty", list[tuple[NDArray[np.float64], NDArray[np.float64]]]()),
        (
            "chain",
            [
                (
                    np.array([10.0, 10.003, 10.006, 10.009]),
                    np.array([1.0, 0.8, 0.9, 0.7]),
                ),
                (np.array([10.003, 20.0]), np.array([3.0, 1.0])),
            ],
        ),
        (
            "nonpositive",
            [
                (np.array([10.0, 20.0]), np.array([0.0, -1.0])),
                (np.zeros(0), np.zeros(0)),
            ],
        ),
    ):
        mz, it = peaks().merge(pairs)
        record[f"merged/{name}/mass"] = torch.from_numpy(mz)
        record[f"merged/{name}/intensity"] = torch.from_numpy(it)
    record["instrument"] = torch.tensor(
        [
            instrument_family(value)
            for value in (
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
        ],
    )
    return record


def test_entire_preprocessing_and_model_bank_values() -> None:
    golden = BANK_GOLDEN
    expected = read_tensors(golden)
    actual = preprocessing_record()
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        models: list[SecondFPNet] = []
        for index in range(2):
            config = tiny_second()
            config.adduct.channels_in = (
                26  # Full published vocabulary, including unknown.
            )
            model = config.make().eval()
            prefix = f"model/{index}/"
            model.load_released_state(
                {
                    name.removeprefix(prefix): value
                    for name, value in expected.items()
                    if name.startswith(prefix)
                },
            )
            for name, value in model.released_state().items():
                actual[f"model/{index}/{name}"] = value.detach().clone()
            models.append(model)
        pairs = [
            (np.array([20.0, 40.0, 60.0, 80.0]), np.array([1.0, 0.5, 0.25, 0.75])),
            (np.array([15.0, 45.0]), np.array([2.0, 1.0])),
            (np.zeros(0), np.zeros(0)),
        ]
        for name, single, merged, inputs in (
            ("both", models, models[1:], pairs),
            ("single", models, list[SecondFPNet](), pairs),
            ("merged", list[SecondFPNet](), models[1:], pairs),
            ("none", list[SecondFPNet](), list[SecondFPNet](), pairs),
            ("empty", models, models[1:], [(np.zeros(0), np.zeros(0))] * 3),
        ):
            metadata = [
                (100.0, "[M+H]+", "timsTOF", 10.0, 1.0),
                (200.0, "unknown", "QTOF", 30.0, -1.0),
                (30.0, "[M-H]-", None, 0.0, 1.0),
            ]
            spectra = [
                QuerySpectrum(
                    mass=mass,
                    intensity=intensity,
                    precursor=precursor,
                    adduct=adduct,
                    instrument=instrument,
                    collision_energy=ce,
                    polarity=mode,
                )
                for (mass, intensity), (precursor, adduct, instrument, ce, mode) in zip(
                    inputs,
                    metadata,
                    strict=True,
                )
            ]
            result = predict_second(
                spectra,
                single=single,
                merged=merged,
                device="cpu",
                peaks=peaks(),
                merged_energy=25.0,
            )
            actual[f"bank/{name}/present"] = torch.tensor(result is not None)
            actual[f"bank/{name}/logits"] = (
                torch.zeros(0) if result is None else torch.from_numpy(result)
            )
    assert_tensor_golden(golden, float32_comparands(actual))


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


FEATURES_GOLDEN: Final = _BASELINE / "testdata" / "features.pt"


def analog_record() -> dict[str, Tensor]:
    """Exercise model, analog, fragment, no-hit, empty and zero-variance branches."""
    fingerprint = (np.arange(21).reshape(3, 7) % 5 <= 1).astype(np.float32)
    analog_fp = np.equal(np.arange(35).reshape(5, 7) % 3, 0).astype(np.float32)
    library = np.array([0.9, 0.0, 0.5], np.float32)
    similarity = np.array([1.0, 0.5, 0.25, 0.0, -1.0], np.float32)
    logits = (np.arange(7, dtype=np.float32) - 3) / 7
    fragment = np.array([0.25, 0.75, 0.0], np.float32)
    record: dict[str, Tensor] = {}
    for name, fp, lib, afp, sim, model, frag in (
        ("complete", fingerprint, library, analog_fp, similarity, logits, fragment),
        (
            "zero_library",
            fingerprint,
            library * 0,
            analog_fp,
            similarity,
            logits,
            fragment,
        ),
        ("no_model", fingerprint, library, analog_fp, similarity, None, fragment),
        ("no_analog", fingerprint, library, None, similarity, logits, fragment),
        (
            "empty_analog",
            fingerprint,
            library,
            analog_fp[:0],
            similarity[:0],
            logits,
            fragment,
        ),
        ("no_fragment", fingerprint, library, analog_fp, similarity, logits, None),
        ("empty", fingerprint[:0], library[:0], None, similarity[:0], None, None),
        (
            "constant",
            fingerprint * 0,
            library * 0,
            None,
            similarity[:0],
            logits * 0,
            fragment * 0,
        ),
    ):
        record[f"analog/{name}"] = torch.from_numpy(
            analog_features(fp, lib, afp, sim, logits=model, fragments=frag),
        )
    return record


def view_record() -> dict[str, Tensor]:
    """Keep every fingerprint block and both polarities while reducing geometry."""
    bits = np.array([0, 4096, 8192, 10_240, 12_288, 14_336, 16_384], dtype=np.int64)
    record: dict[str, Tensor] = {}
    for name, count, ns, nm, nmodels, modes in (
        ("complete", 5, 4, 2, 3, [1, -1]),
        ("two_models", 5, 4, 2, 2, [1, -1]),
        ("missing_negative", 5, 4, 2, 3, [1, 1]),
        ("no_single", 5, 0, 2, 3, [1, -1]),
        ("no_merged", 5, 4, 0, 3, list[int]()),
        ("no_views", 5, 0, 0, 3, list[int]()),
        ("above_pairwise_limit", 9, 4, 2, 3, [1, -1]),
    ):
        fp = (np.arange(count * 7).reshape(count, 7) % 5 <= 1).astype(np.float32)
        single = (
            (np.arange(nmodels * ns * 7).reshape(nmodels, ns, 7) % 11 - 5) / 7
        ).astype(np.float32)
        merged = (
            (np.arange(nmodels * nm * 7).reshape(nmodels, nm, 7) % 13 - 5) / 7
        ).astype(np.float32)
        values = view_features(
            fp,
            single,
            merged,
            np.array(modes, dtype=np.int64),
            bits,
            pairwise_rank_limit=7,
        )
        record.update(
            {
                f"views/{name}/{feature}": torch.from_numpy(value)
                for feature, value in values.items()
            },
        )
    return record


def test_complete_feature_values() -> None:
    assert_tensor_golden(FEATURES_GOLDEN, analog_record() | view_record())


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


def test_whole_second_engine_features_and_fragment_panels() -> None:
    _, pool, _ = retrieval_inputs()
    # pool_fps fixes6930 selected bits and discards six trailing padding bits.
    pool = replace(
        pool,
        bits=np.arange(6930, dtype=np.int64),
        packed=(np.arange(5 * 867).reshape(5, 867) % 251).astype(np.uint8),
    )
    rows = np.array([4, 1, 2, 1], dtype=np.int64)
    channels = SecondChannels(
        pool_rows=np.array([0, 2, 4], dtype=np.int64),
        maximum=np.array([0.25, 0.5, 0.9], dtype=np.float32),
        mean=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        shifted=np.array([0.4, 0.3, 0.7], dtype=np.float32),
        count=np.array([2, 4, 3], dtype=np.float32),
        analogs=(
            Analog(sid=3, similarity=0.9),
            Analog(sid=1, similarity=0.5),
            Analog(sid=2, similarity=-0.1),
        ),
        polarity_analogs=(Analog(sid=0, similarity=0.8), Analog(sid=2, similarity=0.4)),
    )
    fragments = FragmentMassStore(
        offsets=np.arange(6, dtype=np.int64) * 2,
        mass=np.arange(10, dtype=np.float64) * 10 + 10,
    )
    spectra: list[SpectrumRequest] = [
        {
            "mz": [11.00727645217, 21.00727645217, 40.123],
            "it": [1, 0.5, 0.0019],
            "mode": 1,
            "prec": 101,
            "adduct": "[M+H]+",
        },
        {
            "mz": [28.99272354783, 38.99272354783, 8.99272354783],
            "it": [1, 0.5, 0.25],
            "mode": -1,
            "prec": 99,
            "adduct": "[M-H]-",
        },
        {
            "mz": [72.989220701, 82.989220701, 31.00727645217],
            "it": [1, 0.25, 0.4],
            "mode": 1,
            "prec": 123,
            "adduct": "[M+Na]+",
        },
        {"mz": [], "it": [], "mode": 1, "prec": 101, "adduct": "unknown"},
    ]
    actual: dict[str, Tensor] = {}
    for name, analogs, zero in (
        ("active", channels.polarity_analogs, False),
        ("absent", (), False),
        (
            "negative",
            (Analog(sid=0, similarity=-0.8), Analog(sid=2, similarity=-0.4)),
            False,
        ),
        ("zero", channels.polarity_analogs, True),
    ):
        selected_pool = (
            replace(pool, packed=np.zeros_like(pool.packed)) if zero else pool
        )
        actual[f"polarity/{name}"] = torch.from_numpy(
            _polarity_features(
                selected_pool,
                fingerprints=selected_pool.fingerprints(rows).astype(np.float32),
                channels=replace(channels, polarity_analogs=analogs),
            ),
        )
    evidence: list[SecondFragments] = []
    for case, query in enumerate(
        (spectra, spectra[:2], list[SpectrumRequest](), spectra[3:]),
    ):
        result = second_fragments(
            query,
            rows=rows,
            fragments=fragments,
            search=SecondEngine.Config().fragment_search.make(),
        )
        evidence.append(result)
        for name, value in {
            "frag": result.maximum,
            "frag_ad": result.adduct,
            "frag_mean": result.mean,
            "frag_pol": result.polarity,
        }.items():
            actual[f"fragments/{case}/{name}"] = torch.from_numpy(value)
    logits = ((np.arange(6930, dtype=np.float32) % 17 - 8) / 19).astype(np.float32)
    for case, (retrieved, model, fragment) in enumerate(
        (
            (channels, logits, evidence[0]),
            (channels, None, evidence[1]),
            (replace(channels, analogs=(), polarity_analogs=()), logits, evidence[2]),
            (
                replace(
                    channels,
                    maximum=channels.maximum * 0,
                    mean=channels.mean * 0,
                    shifted=channels.shifted * 0,
                    count=channels.count * 0,
                    analogs=(),
                    polarity_analogs=(),
                ),
                logits * 0,
                evidence[3],
            ),
        ),
    ):
        actual[f"features/{case}"] = torch.from_numpy(
            second_features(
                pool,
                rows=rows,
                channels=retrieved,
                fragments=fragment,
                logits=model,
            ),
        )
    assert_tensor_golden(_BASELINE / "testdata" / "fragment_features.pt", actual)


def test_whole_second_engine_retrieval() -> None:
    library, _, spectra = retrieval_inputs()
    actual: dict[str, Tensor] = {}
    second_library = SecondLibrary(
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
    config_second = SecondRetrieval.Config()
    config_second.n_analog = 2
    second = config_second.make()
    index = second.index(second_library)
    actual["second/lib_rows"] = torch.from_numpy(index.rows)
    actual["second/lib_mass"] = torch.from_numpy(index.mass)
    for name, representatives in (
        ("all", index.all),
        ("pos", index.positive),
        ("neg", index.negative),
    ):
        for field, value in {
            "rep": representatives.rows,
            "rep_p": representatives.pool_rows,
            "rep_nm": representatives.mass,
            "roff": representatives.store.offsets,
            "rmz": representatives.store.mz,
            "rit": representatives.store.intensity,
        }.items():
            actual[f"second/{name}/{field}"] = torch.from_numpy(value)
    for case, (query, target) in enumerate(
        (
            (spectra, 100.0),
            (spectra[2:], 100.0),
            (spectra, 900.0),
            (list[SpectrumRequest](), 100.0),
        ),
    ):
        channels = second.channels(
            query,
            library=second_library,
            index=index,
            target=target,
        )
        for name, value in {
            "u": channels.pool_rows,
            "max": channels.maximum,
            "mean": channels.mean,
            "shifted": channels.shifted,
            "count": channels.count,
        }.items():
            actual[f"channels/{case}/{name}"] = torch.from_numpy(value)
        for name, analogs in (
            ("all", channels.analogs),
            ("pol", channels.polarity_analogs),
        ):
            actual[f"channels/{case}/{name}"] = torch.tensor(
                [[analog.sid, analog.similarity] for analog in analogs],
                dtype=torch.float64,
            ).reshape(-1, 2)
    assert_tensor_golden(
        _BASELINE / "testdata" / "second_retrieval.pt",
        float32_comparands(actual),
    )


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


def test_entire_second_pipeline_and_all_twelve_fitted_models(
    tmp_path: Path,
) -> None:
    golden = _BASELINE / "testdata/second_pipeline.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    library, original_pool, _ = retrieval_inputs()
    pool = SecondPool(
        mass=original_pool.mass,
        keys=original_pool.keys,
        smiles=original_pool.smiles,
        packed=original_pool.packed,
        bits=np.arange(8, dtype=np.int64),
    )
    raw = SecondLibrary(
        store=PeakStore(
            offsets=library.offsets,
            mz=library.mz,
            intensity=library.intensity,
        ),
        mass=np.concatenate([library.mass[:-1], [float("nan")]]),
        pool_rows=np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, -1, 4, 2], np.int64),
        polarity=library.mode,
        timstof=np.isin(np.arange(13), [0, 2, 6, 9]),
        precursor=library.precursor,
    )
    rows = [expected[f"input/fragments/{i}"].numpy() for i in range(5)]
    for i, row in enumerate(rows):
        actual[f"input/fragments/{i}"] = torch.from_numpy(row.copy())
    fragments = FragmentMassStore(
        offsets=np.concatenate([[0], np.cumsum([len(row) for row in rows])]).astype(
            np.int64,
        ),
        mass=np.concatenate(rows),
    )
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        old = read_tensors(_BASELINE / "testdata/bank.pt")
        paths: list[Path] = []
        for index, name in enumerate(("single_a", "merged_b", "single_c", "merged_d")):
            prefix = f"model/{index % 2}/"
            state = {
                key.removeprefix(prefix): value.clone()
                for key, value in old.items()
                if key.startswith(prefix)
            }
            state["head.3.weight"] = torch.cat(
                # SecondBank adds one output row to this seven-bit source head.
                [state["head.3.weight"], torch.zeros((1, 9))],
            )
            state["head.3.bias"] = torch.cat([state["head.3.bias"], torch.zeros(1)])
            if index >= 2:
                state["head.3.bias"] += 0.125 * index
            path = tmp_path / f"{name}.pt"
            torch.save({"model": state, "nbits": 8, "d": 8, "layers": 1}, path)
            paths.append(path)
        seals = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        config = tiny_bank(
            single=((paths[0], seals[0]), (paths[2], seals[2])),
            merged=((paths[1], seals[1]), (paths[3], seals[3])),
        )
        config.model.channels_out = 8
        bank = config.make()
        assert len(bank.single) == len(bank.merged) == 2
        engine = SecondEngine.Config().make()
        index = engine.retrieval.index(raw)
        bases: list[SecondBase] = []
        for case, target in enumerate((100.0, 100.004, 900.0, float("nan"))):
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
            actual[f"case/{case}/cand"] = torch.from_numpy(base.rows)
            actual[f"case/{case}/mid"] = torch.tensor(case)
            fields = ["mid", "cand"]
            if base.target is not None:
                fields += ["target", "zlog"]
                if len(base.rows) == 0:
                    fields.append("specs")
                fields += [
                    "mode",
                    "modes",
                    "adducts",
                    "ana",
                    "anapol",
                    "lib",
                    "lib_mean",
                    "libsh",
                    "lib_cnt",
                ]
                assert base.logits is not None
                assert base.channels is not None
                actual[f"case/{case}/target"] = torch.tensor(
                    base.target,
                    dtype=torch.float64,
                )
                actual[f"case/{case}/zlog"] = torch.from_numpy(base.logits)
                modes = np.array(
                    [
                        1.0 if s["mode"] in (1, "positive") else -1.0
                        for s in base.spectra
                    ],
                )
                actual[f"case/{case}/modes"] = torch.from_numpy(modes)
                actual[f"case/{case}/mode"] = torch.tensor(
                    modes.mean(),
                    dtype=torch.float64,
                )
                actual[f"case/{case}/adducts"] = torch.tensor(
                    list("\0".join(s["adduct"] for s in base.spectra).encode()),
                    dtype=torch.uint8,
                )
                if not len(base.rows):
                    for qi, spectrum in enumerate(base.spectra):
                        actual[f"case/{case}/specs/{qi}/mz"] = torch.tensor(
                            spectrum["mz"],
                            dtype=torch.float32,
                        )
                        actual[f"case/{case}/specs/{qi}/it"] = torch.tensor(
                            spectrum["it"],
                            dtype=torch.float32,
                        )
                for name, analogs in (
                    ("ana", base.channels.analogs),
                    ("anapol", base.channels.polarity_analogs),
                ):
                    actual[f"case/{case}/{name}"] = torch.tensor(
                        [[a.sid, a.similarity] for a in analogs],
                        dtype=torch.float64,
                    ).reshape(-1, 2)
                positions = {
                    base.channels.pool_rows.item(i): i
                    for i in range(len(base.channels.pool_rows))
                }
                for name, values in (
                    ("lib", base.channels.maximum),
                    ("lib_mean", base.channels.mean),
                    ("libsh", base.channels.shifted),
                    ("lib_cnt", base.channels.count),
                ):
                    aligned = np.zeros(len(base.rows), np.float32)
                    for i in range(len(base.rows)):
                        if (position := positions.get(base.rows.item(i))) is not None:
                            aligned[i] = values[position]
                    actual[f"case/{case}/{name}"] = torch.from_numpy(aligned)
                if base.fragments is not None:
                    for name, values in (
                        ("frag", base.fragments.maximum),
                        ("frag_ad", base.fragments.adduct),
                        ("frag_mean", base.fragments.mean),
                        ("frag_pol", base.fragments.polarity),
                    ):
                        fields.append(name)
                        actual[f"case/{case}/{name}"] = torch.from_numpy(values)
                    actual[f"case/{case}/own"] = torch.from_numpy(base.own_features)
                    actual[f"case/{case}/pv"] = torch.from_numpy(base.pv_features)
            actual[f"case/{case}/fields"] = torch.tensor(
                list("\0".join(fields).encode()),
                dtype=torch.uint8,
            )
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
        for family, ensemble in enumerate((rankers.own, rankers.pv)):
            query = np.concatenate(
                [
                    base.own_features if family == 0 else base.pv_features
                    for base in bases
                    if len(base.rows)
                ],
            )
            for mi, model in enumerate(ensemble.models):
                for name, value in _model_record(
                    model,
                    rows=training[family],
                    query=query,
                ).items():
                    actual[f"family/{family}/model/{mi}/{name}"] = value
        for case, base in enumerate(bases):
            result = engine.run(
                pipeline_queries((100.0, 100.004, 900.0, float("nan"))[case]),
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
            if not len(base.rows):
                assert result.scores.shape == (0,)
                continue
            scores = rankers.score(base.own_features, pv_features=base.pv_features)
            np.testing.assert_array_equal(result.scores, scores.blend)
            for name, values in (
                ("own", scores.own),
                ("pv", scores.fingerprint),
                ("blend", scores.blend),
            ):
                actual[f"case/{case}/score/{name}"] = torch.from_numpy(values)
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
        rankers.blocks = ("base",)
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
        rankers.blocks = ("base", "base", "libx", "fragx")
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
        overflow = engine.base(
            overflow_query,
            pool=pool,
            library=raw,
            index=index,
            fragments=fragments,
            bank=tiny_bank().make(),
        )
        assert overflow.target == float(2**63)
        assert len(overflow.rows) == 0
    assert_tensor_golden(golden, float32_comparands(actual, inputs=("input/*",)))


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
    """Rebuild the literal query inputs shared by the pipeline golden."""
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


HGB_GOLDEN: Final = _BASELINE / "testdata" / "hgb.pt"


def test_every_histogram_tree_and_three_boosting_rounds() -> None:
    actual: dict[str, Tensor] = {}
    for case in range(12):
        rows = 16 if case == 5 else 12
        index: NDArray[np.int64] = np.arange(rows, dtype=np.int64)
        data: FeatureMatrix = np.zeros((rows, 3), dtype=np.float64)
        data[:, 0], data[:, 1], data[:, 2] = index / 7, (index * 5 % 11) / 3, index % 3
        labels = ((index % 6) >= 3).astype(np.float64)
        prior = (0.55, 0.30, 0.60)[case - 3] if 3 <= case <= 5 else 0.35
        weights = np.where(np.equal(index % 4, 0), prior, 1.0 - prior)
        if case % 3 == 1:
            data[labels == 1, 0] = np.nan
        elif case % 3 == 2:
            data[labels == 0, 1] = np.nan
            weights[index % 3 == 0] = 0
        if case == 8:
            weights.fill(1e-6)
        elif case in (9, 10):
            data.fill(0)
        elif case == 11:
            weights[index < 3] = 1e-6
        query = np.vstack(
            [data, [[-10, -10, -10], [100, 100, 100], [np.nan, np.nan, np.nan]]],
        )
        config = HistogramBoosting.Config()
        config.max_depth = 3
        config.rounds = 3
        config.min_leaf = 2
        config.max_bins = 4
        config.seed = case % 4
        if 3 <= case <= 5 or case == 10:
            config.early_stopping_rows = 7
            config.binning_rows = 11
        if case == 6:
            config.min_leaf = 8
        elif case == 7:
            config.max_leaves = 2
        elif case == 10:
            config.patience = 3
        model = config.make().fit(data, labels, weights)
        prefix = f"case{case}"
        actual[f"{prefix}/baseline"] = torch.from_numpy(model.baseline.copy())
        actual[f"{prefix}/random_seed"] = torch.tensor(model.random_seed)
        generator = copy.deepcopy(model.generator)
        actual[f"{prefix}/rng"] = torch.from_numpy(
            generator.integers(0x7FFF_FFFF, size=8, dtype=np.int64),
        )
        actual[f"{prefix}/bin_counts"] = torch.from_numpy(model.bin_counts.copy())
        actual[f"{prefix}/bin_categorical"] = torch.zeros(
            data.shape[1],
            dtype=torch.uint8,
        )
        actual[f"{prefix}/bin_missing"] = torch.tensor(config.max_bins)
        actual[f"{prefix}/binned"] = torch.from_numpy(model.binned(data))
        for feature, thresholds in enumerate(model.thresholds):
            actual[f"{prefix}/thresholds/{feature}"] = torch.from_numpy(
                thresholds.copy(),
            )
        for step, tree in enumerate(model.trees):
            for name, values in tree.state().items():
                actual[f"{prefix}/trees/{step}/{name}"] = torch.from_numpy(
                    values.copy(),
                )
        actual[f"{prefix}/train_score"] = torch.from_numpy(model.train_score.copy())
        actual[f"{prefix}/validation_score"] = torch.from_numpy(
            model.validation_score.copy(),
        )
        actual[f"{prefix}/stages"] = torch.from_numpy(model.stages(query))
        actual[f"{prefix}/probabilities"] = torch.from_numpy(model.predict_proba(query))
    logits = np.array(
        [-1000.0, -40.0, -37.0, -2.0, 0.0, 2.0, 18.0, 25.0, 33.3, 40.0, 1000.0],
    )
    targets = np.arange(len(logits), dtype=np.float64) % 2
    weights = np.linspace(0.1, 0.9, len(logits))
    gradient, hessian = HistogramBoosting.derivatives(logits, targets, weights)
    actual["derivative/gradient"] = torch.from_numpy(gradient)
    actual["derivative/hessian"] = torch.from_numpy(hessian)
    actual["loss/score"] = torch.tensor(
        HistogramBoosting.loss_score(logits, targets, weights),
        dtype=torch.float64,
    )
    assert_tensor_golden(HGB_GOLDEN, float32_comparands(actual))


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


ENSEMBLE_GOLDEN: Final = _BASELINE / "testdata" / "rankers.pt"


def test_all_twelve_fitted_rankers_and_blended_scores(
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
        M=groups,
        blocks=np.array(["base", "libx", "anapol", "fragx"]),
    )
    query = np.vstack(
        [data.astype(np.float64), [[np.nan, np.nan], [-10, -10], [100, 100]]],
    )
    actual: dict[str, Tensor] = {}
    means: list[NDArray[np.float64]] = []
    for family, column in enumerate(("S", "M")):
        rows = SimulationRows.load(path, column)
        assert rows.blocks == ("base", "libx", "anapol", "fragx")
        config = HistogramEnsemble.Config()
        config.tree.max_depth = 2
        config.tree.rounds = 3
        config.tree.min_leaf = 2
        config.tree.max_bins = 3
        if family == 0:
            config.priors = [0.35, 0.55]
            config.seeds = [0, 1]
        else:
            config.priors = [0.30, 0.60]
            config.seeds = [0, 1, 2, 3]
        ensemble = config.make().fit(rows)
        assert len(ensemble.models) == (4, 8)[family]
        for index, model in enumerate(ensemble.models):
            prefix = f"family{family}/model{index}"
            actual[f"{prefix}/baseline"] = torch.from_numpy(model.baseline.copy())
            actual[f"{prefix}/random_seed"] = torch.tensor(model.random_seed)
            generator = copy.deepcopy(model.generator)
            actual[f"{prefix}/rng"] = torch.from_numpy(
                generator.integers(0x7FFF_FFFF, size=8, dtype=np.int64),
            )
            actual[f"{prefix}/bin_counts"] = torch.from_numpy(model.bin_counts.copy())
            actual[f"{prefix}/bin_categorical"] = torch.zeros(
                rows.data.shape[1],
                dtype=torch.uint8,
            )
            actual[f"{prefix}/bin_missing"] = torch.tensor(config.tree.max_bins)
            actual[f"{prefix}/binned"] = torch.from_numpy(model.binned(rows.data))
            for feature, thresholds in enumerate(model.thresholds):
                actual[f"{prefix}/thresholds/{feature}"] = torch.from_numpy(
                    thresholds.copy(),
                )
            for step, tree in enumerate(model.trees):
                for name, values in tree.state().items():
                    actual[f"{prefix}/trees/{step}/{name}"] = torch.from_numpy(
                        values.copy(),
                    )
            actual[f"{prefix}/train_score"] = torch.from_numpy(model.train_score.copy())
            actual[f"{prefix}/validation_score"] = torch.from_numpy(
                model.validation_score.copy(),
            )
            actual[f"{prefix}/stages"] = torch.from_numpy(model.stages(query))
            actual[f"{prefix}/probabilities"] = torch.from_numpy(
                model.predict_proba(query),
            )
        probability = ensemble.predict(query)
        means.append(probability)
        actual[f"family{family}/mean"] = torch.from_numpy(probability.copy())
    actual["blend"] = torch.from_numpy(
        blend_histograms(means[1], means[0], weight_fingerprint=0.88),
    )
    for case, size in enumerate((19, 31, 37)):
        positions = np.arange(size, dtype=np.float64)
        first = (positions * (case + 2)) % 5
        second = positions % 3
        actual[f"ties{case}"] = torch.from_numpy(
            blend_histograms(first, second, weight_fingerprint=0.88),
        )
    assert_tensor_golden(ENSEMBLE_GOLDEN, float32_comparands(actual))


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
    """Read the staged engine keys without chemical reinterpretation."""
    expected = read_tensors(_BASELINE / "testdata/recipe_policy.pt")
    return json_loads(
        expected["engine/identities"].numpy().tobytes().decode(),
        dict[str, str | None],
    )


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
    expected = read_tensors(_BASELINE / "testdata" / "bank.pt")
    states = [
        {
            name.removeprefix(f"model/{index}/"): value
            for name, value in expected.items()
            if name.startswith(f"model/{index}/")
        }
        for index in range(2)
    ]
    paths = (
        tmp_path / "first.pt",
        tmp_path / "second.pt",
        tmp_path / "third.pt",
    )
    seals = tuple(
        _checkpoint(path, state)
        for path, state in zip(paths, [*states, states[1]], strict=True)
    )
    spectra = [
        QuerySpectrum(
            mass=mass,
            intensity=intensity,
            precursor=precursor,
            adduct=adduct,
            instrument=instrument,
            collision_energy=ce,
            polarity=mode,
        )
        for mass, intensity, precursor, adduct, instrument, ce, mode in (
            (
                np.array([20.0, 40.0, 60.0, 80.0]),
                np.array([1.0, 0.5, 0.25, 0.75]),
                100.0,
                "[M+H]+",
                "timsTOF",
                10.0,
                1.0,
            ),
            (
                np.array([15.0, 45.0]),
                np.array([2.0, 1.0]),
                200.0,
                "unknown",
                "QTOF",
                30.0,
                -1.0,
            ),
            (np.zeros(0), np.zeros(0), 30.0, "[M-H]-", None, 0.0, 1.0),
        )
    ]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        bank = tiny_bank(
            single=tuple(zip(paths[:2], seals[:2], strict=True)),
            merged=((paths[2], seals[2]),),
        ).make()
        assert len(bank.single) == 2
        assert len(bank.merged) == 1
        actual: dict[str, Tensor] = {}
        for index, model in enumerate(bank.single):
            actual.update(
                {
                    f"model/{index}/{name}": value.detach()
                    for name, value in model.released_state().items()
                },
            )
        result = bank.predict(spectra)
        assert result is not None
        actual["bank/both/present"] = torch.tensor(True)
        actual["bank/both/logits"] = torch.from_numpy(result)
        selected = {
            name: value
            for name, value in expected.items()
            if name.startswith(("model/", "bank/both/"))
        }
        assert not mismatches(selected, actual), "\n".join(mismatches(selected, actual))
        empty = tiny_bank().make()
        assert empty.predict(spectra) is None
        for name, chosen in (("single", (0, 1)), ("merged", (2,))):
            files = tuple((paths[i], seals[i]) for i in chosen)
            arm = tiny_bank(
                single=files if name == "single" else (),
                merged=files if name == "merged" else (),
            ).make()
            result = arm.predict(spectra)
            assert result is not None
            assert torch.equal(
                torch.from_numpy(result),
                expected[f"bank/{name}/logits"],
            )
        assert (
            bank.predict(
                [
                    QuerySpectrum(
                        mass=np.zeros(0),
                        intensity=np.zeros(0),
                        precursor=100.0,
                        adduct="unknown",
                        instrument=None,
                        collision_energy=0.0,
                        polarity=1.0,
                    ),
                ],
            )
            is None
        )
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
    expected = read_tensors(_BASELINE / "testdata" / "bank.pt")
    path = tmp_path / "model.pt"
    seal = _checkpoint(
        path,
        {
            name.removeprefix("model/0/"): value
            for name, value in expected.items()
            if name.startswith("model/0/")
        },
    )
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


def test_bank_replays_every_fitted_state_and_score() -> None:
    bank = _tiny_config().make()
    rows, query = _simulation()
    pv_rows = SimulationRows(
        data=rows.data,
        labels=rows.labels,
        groups=rows.groups,
        blocks=(),
    )
    assert bank.fit(rows, pv=pv_rows) is bank
    assert bank.blocks == ("base", "libx", "anapol", "fragx")
    scores = bank.score(query, pv_features=query)
    actual: dict[str, Tensor] = {}
    for family, ensemble in enumerate((bank.own, bank.pv)):
        for index, model in enumerate(ensemble.models):
            actual.update(
                {
                    f"family{family}/model{index}/{name}": value
                    for name, value in _model_record(
                        model,
                        rows=rows,
                        query=query,
                    ).items()
                },
            )
    actual["family0/mean"] = torch.from_numpy(scores.own.copy())
    actual["family1/mean"] = torch.from_numpy(scores.fingerprint.copy())
    actual["blend"] = torch.from_numpy(scores.blend.copy())
    for case, size in enumerate((19, 31, 37)):
        positions = np.arange(size, dtype=np.float64)
        first, second = (positions * (case + 2)) % 5, positions % 3
        actual[f"ties{case}"] = torch.from_numpy(
            blend_histograms(first, own=second, weight_fingerprint=0.88),
        )
    report = mismatches(read_tensors(ENSEMBLE_GOLDEN), float32_comparands(actual))
    assert not report, "\n".join(report)


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


def _model_record(
    model: HistogramBoosting,
    *,
    rows: SimulationRows,
    query: PredictionMatrix,
) -> dict[str, Tensor]:
    generator = copy.deepcopy(model.generator)
    record = {
        "baseline": torch.from_numpy(model.baseline.copy()),
        "random_seed": torch.tensor(model.random_seed),
        "rng": torch.from_numpy(
            generator.integers(0x7FFF_FFFF, size=8, dtype=np.int64),
        ),
        "bin_counts": torch.from_numpy(model.bin_counts.copy()),
        "bin_categorical": torch.zeros(rows.data.shape[1], dtype=torch.uint8),
        "bin_missing": torch.tensor(model.config.max_bins),
        "binned": torch.from_numpy(model.binned(rows.data)),
        "train_score": torch.from_numpy(model.train_score.copy()),
        "validation_score": torch.from_numpy(model.validation_score.copy()),
        "stages": torch.from_numpy(model.stages(query)),
        "probabilities": torch.from_numpy(
            model.predict_proba(query),
        ),
    }
    for feature, thresholds in enumerate(model.thresholds):
        record[f"thresholds/{feature}"] = torch.from_numpy(thresholds.copy())
    for step, tree in enumerate(model.trees):
        record.update(
            {
                f"trees/{step}/{name}": torch.from_numpy(values.copy())
                for name, values in tree.state().items()
            },
        )
    return record


def test_complete_second_assets_preserve_raw_rows_fragments_keys_and_twelve_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = second_config(tmp_path)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
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
    source = read_tensors(_BASELINE / "testdata/bank.pt")
    assert len(inputs.bank.merged) == len(inputs.bank.single) == 1
    for index, model in enumerate((*inputs.bank.merged, *inputs.bank.single)):
        state = model.released_state()
        keys = {
            name.removeprefix(f"model/{index}/")
            for name in source
            if name.startswith(f"model/{index}/")
        }
        assert state.keys() == keys
        for name, value in state.items():
            assert torch.equal(value, source[f"model/{index}/{name}"])
    rows, query = _simulation()
    scores = inputs.rankers.score(query, pv_features=query)
    actual: dict[str, Tensor] = {}
    for family, ensemble in enumerate((inputs.rankers.own, inputs.rankers.pv)):
        for index, model in enumerate(ensemble.models):
            actual.update(
                (f"family{family}/model{index}/{name}", value)
                for name, value in _model_record(model, rows=rows, query=query).items()
            )
    actual["family0/mean"] = torch.from_numpy(scores.own)
    actual["family1/mean"] = torch.from_numpy(scores.fingerprint)
    actual["blend"] = torch.from_numpy(scores.blend)
    for case, size in enumerate((19, 31, 37)):
        positions = np.arange(size, dtype=np.float64)
        actual[f"ties{case}"] = torch.from_numpy(
            blend_histograms(
                (positions * (case + 2)) % 5,
                own=positions % 3,
                weight_fingerprint=0.88,
            ),
        )
    report = mismatches(read_tensors(ENSEMBLE_GOLDEN), float32_comparands(actual))
    assert not report, "\n".join(report)
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
    """Publish complete test primitive inputs with the stored bank states."""
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
    source = read_tensors(_BASELINE / "testdata/bank.pt")
    paths: list[Path] = []
    for index, name in enumerate(("fp_merged_m1.pt", "fp_single_s2.pt")):
        prefix = f"model/{index}/"
        state = {
            name.removeprefix(prefix): value
            for name, value in source.items()
            if name.startswith(prefix)
        }
        path = root / name
        torch.save({"model": state, "nbits": 7, "d": 8, "layers": 1}, path)
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
