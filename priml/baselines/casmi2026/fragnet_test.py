"""Checks for the fragnet module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from zipfile import ZipFile

import hashlib
import io
import json
import pickle
import sys
import types

from pyarrow import Table, parquet
from treekle import ReadError, to_plain

import numpy as np
import pytest

from priml.baselines.casmi2026 import fragnet
from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.boosting_test import model_text, stump
from priml.baselines.casmi2026.fragnet import (
    Fragments,
    FragnetTable,
    StageA,
    _matches,
    aggregates,
    categorical_fields,
    features,
    fragments,
    prepare_candidates,
    prepare_peaks,
    stage_b_features,
    stage_b_names,
)
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        FragnetRecord,
        SpectrumRequest,
    )


def test_complete_reader_preserves_duplicates_invalid_rows_and_local_bonds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    original = parquet.read_table

    def projected(path: Path, *, columns: list[str]) -> Table:
        assert columns == ["smiles", "parsed", "valid"]
        return original(path, columns=columns)

    monkeypatch.setattr(
        fragnet,
        "parquet",
        types.SimpleNamespace(read_table=projected),
    )
    _write(tmp_path)
    table = FragnetTable(tmp_path)
    assert table.row(1) is None
    assert table.for_smiles("invalid") is None
    first, third = table.row(0), table.row(2)
    assert first is not None
    assert third is not None
    selected = table.for_smiles("same")
    assert selected is not None
    assert selected["mass"] == first["mass"]
    assert first["mass"] == 30.0
    assert third["mass"] == 17.0
    for actual, expected in (
        (first["amass"], [13.0, 17.0]),
        (first["kind"], [0, 9]),
        (first["el"], [0, 2]),
        (first["nh"], [1, 1]),
        (first["el10"], [0, 2]),
        (first["bonds"], [[0, 1]]),
        (first["bring"], [0]),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    np.testing.assert_array_equal(third["bonds"], np.empty((0, 2), dtype=np.int32))
    with pytest.raises(KeyError, match="absent"):
        _ = table.for_smiles("absent")
    for index in (-1, 3):
        with pytest.raises(IndexError):
            _ = table.row(index)


def test_reader_refuses_malformed_ragged_arrays_and_scientific_versions(
    tmp_path: Path,
) -> None:
    _write(tmp_path)
    for name, malformed, message in (
        ("kind", np.zeros(3, dtype=np.int32), "dtype or rank"),
        # FragnetTable requires a flat element vector, not a column matrix.
        ("el", np.zeros((3, 1), dtype=np.int8), "dtype or rank"),
        (
            "nh",
            np.zeros(2, dtype=np.int8),
            r"^Fragnet atom arrays differ from the complete atom layout\.$",
        ),
        (
            "el10",
            np.zeros(4, dtype=np.uint8),
            r"^Fragnet atom arrays differ from the complete atom layout\.$",
        ),
        (
            "mass",
            np.zeros(2),
            r"^Fragnet molecular masses differ from the metadata rows\.$",
        ),
        (
            "bring",
            np.zeros(2, dtype=np.int8),
            r"^Fragnet bond arrays differ from the complete edge layout\.$",
        ),
        (
            "bonds",
            # FragnetTable requires two atom indices per bond, not three.
            np.zeros((1, 3), dtype=np.int32),
            r"^Fragnet bond arrays differ from the complete edge layout\.$",
        ),
        ("atom_off", np.array([0, 2, 3]), "ragged offsets"),
        ("atom_off", np.array([1, 2, 2, 3]), "ragged offsets"),
        ("atom_off", np.array([0, 2, 2, 4]), "ragged offsets"),
        ("atom_off", np.array([0, 2, 1, 3]), "ragged offsets"),
        ("bond_off", np.array([0, 1, 2, 1]), "ragged offsets"),
    ):
        path = tmp_path / f"{name}.npy"
        correct = cast("NDArray[np.generic]", np.load(path))
        np.save(path, malformed)
        with pytest.raises(ValueError, match=message):
            _ = FragnetTable(tmp_path)
        np.save(path, correct)
    for version, release in ((2, IDENTITY_RDKIT), (1, SIMULATOR_RDKIT)):
        _ = (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain({"format_version": version, "rdkit": release})),
        )
        with pytest.raises(
            ValueError,
            match=r"^Fragnet data requires format 1 and RDKit 2026\.03\.3\.$",
        ):
            _ = FragnetTable(tmp_path)
    _ = (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": True, "rdkit": IDENTITY_RDKIT})),
    )
    with pytest.raises(ReadError):
        _ = FragnetTable(tmp_path)


def _write(directory: Path) -> None:
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["same", "invalid", "same"],
                "parsed": [True, False, True],
                "valid": [True, False, True],
            },
        ),
        directory / "molecules.parquet",
    )
    for name, values in (
        ("amass", np.array([13.0, 17.0, 17.0])),
        ("kind", np.array([0, 9, 9], dtype=np.int8)),
        ("el", np.array([0, 2, 2], dtype=np.int8)),
        ("nh", np.array([1, 1, 1], dtype=np.int8)),
        ("el10", np.array([0, 2, 2], dtype=np.uint8)),
        ("bonds", np.array([[0, 1]], dtype=np.int32)),
        ("bring", np.array([0], dtype=np.int8)),
        ("mass", np.array([30.0, np.nan, 17.0])),
        ("atom_off", np.array([0, 2, 2, 3])),
        ("bond_off", np.array([0, 1, 1, 1])),
    ):
        np.save(directory / f"{name}.npy", values)
    _ = (directory / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )


def chain_record() -> FragnetRecord:
    """Return a three-atom chain whose cleavages weigh 60, 70, 80, 130 and 150."""
    return {
        "amass": np.array([60.0, 70.0, 80.0]),
        "kind": np.array([0, 6, 9], np.int8),
        "el": np.array([0, 1, 2], np.int8),
        "nh": np.zeros(3, np.int8),
        "bonds": np.array([[0, 1], [1, 2]], np.int32),
        "bring": np.zeros(2, np.int8),
        "mass": 210.0,
        "el10": np.zeros(3, np.uint8),
    }


def test_features_keep_tied_leaders_and_score_only_explained_candidates() -> None:
    chain = chain_record()
    assert fragments(chain)[0].tolist() == [60.0, 150.0, 70.0, 130.0, 80.0]
    pair: FragnetRecord = {
        **chain,
        "amass": np.array([100.0, 110.0]),
        "kind": np.zeros(2, np.int8),
        "el": np.zeros(2, np.int8),
        "nh": np.zeros(2, np.int8),
        "bonds": np.array([[0, 1]], np.int32),
        "bring": np.zeros(1, np.int8),
        "el10": np.zeros(2, np.uint8),
    }
    # The chain explains both protonated peaks; the pair explains neither.
    spectrum: SpectrumRequest = {
        "mz": [61.00727645217, 81.00727645217],
        "it": [1.0, 1.0],
        "adduct": "[M+H]+",
        "mode": 1,
        "prec": 220.0,
        "ce": 20.0,
    }
    # Explained fragments score 2 and unexplained 0: raw peak scores 2 and 0.
    table = np.tile(np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0], np.float32), (8, 1))
    stage_a = StageA(
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
    # StageB splits on its sixth column, the explained peak weight.
    stage_b = LightGBMModel.parse(
        model_text(stump(5, left=-1.0, right=1.0), features=stage_b_names()),
    )
    values = features(
        [chain, pair, None],
        [spectrum, spectrum],
        formulas=["X", "X", "Y"],
        model_score=np.array([0.3, 0.3, 0.1], np.float32),
        stage_a=stage_a,
        stage_b=stage_b,
        maximum=2,
    )
    assert tuple(values)[:4] == ("fn_score", "fn_score_max", "fn_rank", "fn_gap")
    assert len(values) == 14
    assert all(array.dtype == np.float32 for array in values.values())
    # The cap keeps both tied leaders; the third row scores one below the least.
    assert values["fn_score"].tolist() == [1.0, -1.0, -2.0]
    assert values["fn_score_max"].tolist() == [1.0, -1.0, -2.0]
    assert values["fn_gap"].tolist() == [0.0, -2.0, -3.0]
    # Both leaders tie on the base score, so the lead mean is zero.
    assert values["fn_vs_lead"].tolist() == [1.0, -1.0, -2.0]
    assert values["fn_grp_gap"].tolist() == [0.0, -2.0, 0.0]
    # Raw scores 2, 0 and 0 - 1 sit against the lead mean (2 + 0) / 2.
    assert values["fn_raw_vs_lead"].tolist() == [1.0, -1.0, -2.0]
    assert values["fn_raw_gap"].tolist() == [0.0, -2.0, -3.0]
    assert values["fn_raw_grp_gap"].tolist() == [0.0, -2.0, 0.0]
    for name in ("fn_rank", "fn_raw_rank", "fn_pk_rank", "fn_fc_rank"):
        assert values[name].tolist() == [0.0, 0.5, 1.0], name
    assert values["fn_grp_rank"].tolist() == [0.0, 1.0, 0.0]
    assert values["fn_sp_rank"].tolist() == [0.0, 1.0, 1.0]
    empty = features(
        [chain],
        [],
        formulas=["X"],
        model_score=np.ones(1, np.float32),
        stage_a=stage_a,
        stage_b=stage_b,
    )
    assert all(array.tolist() == [0.0] for array in empty.values())


@pytest.mark.parametrize(
    "case",
    ["pin", "chains", "field", "width", "dtype", "order", "base"],
)
def test_released_stage_a_requires_pinned_complete_categorical_tables(
    tmp_path: Path,
    case: str,
) -> None:
    chains = np.array(
        [[["mode", "ft"], ["c1"], ["c2"]], [["mode", "ft"], ["size"], ["frac"]]],
        dtype=object,
    )
    if case == "chains":
        chains = np.array([], dtype=object)
    elif case == "field":
        chains[1, 2] = np.array(["unknown"])
    elif case == "base":
        chains[1, 0] = np.array(["ce", "ft"])
    arrays: dict[str, NDArray[np.generic]] = {"chains": chains, "alpha": np.array(80)}
    for ci in range(len(chains)):
        for li in range(3):
            arrays[f"k_{ci}_{li}"] = np.array([0, 1], np.int64)
            arrays[f"l_{ci}_{li}"] = np.zeros((2, 6), np.float32)
    if case == "width":
        arrays["l_0_0"] = np.zeros((2, 5), np.float32)
    elif case == "dtype":
        arrays["k_0_0"] = np.array([0, 1], np.int32)
    elif case == "order":
        arrays["k_0_0"] = np.array([1, 0], np.int64)
    path = tmp_path / "model.npz"
    np.savez(path, allow_pickle=True, **arrays)
    with pytest.raises(ValueError, match=r"digest|chains|schema|dtype|first"):
        _ = StageA.load(
            path,
            "invalid"
            if case == "pin"
            else hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_empty_contexts_and_missing_energy_remain_explicit() -> None:
    assert all(not len(array) for array in fragments(None))
    assert prepare_peaks([]) is None
    assert stage_b_features(np.zeros((0, 19))).shape == (0, 31)
    spectrum: SpectrumRequest = {
        "mz": [100.0, 120.0],
        "it": [1.0, 0.5],
        "adduct": "[M+H]+",
        "mode": 1,
        "prec": 150.0,
        "ce": None,
    }
    with pytest.raises(TypeError, match="energy"):
        _ = prepare_peaks([spectrum])


def test_all_six_match_classes_unobservable_windows_and_complete_matches() -> None:
    # The window is the lightest peak - 0.5 to precursor - 2; 59.4 and 197.5 miss it.
    masses = np.array(
        [20.0, 59.4, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 196.9, 197.5, 220.0],
    )
    ion = 1.00727645217
    hydrogen = 1.00782503207
    spectrum: SpectrumRequest = {
        "mz": [
            60.0 + ion,
            80.0 + ion - hydrogen,
            100.0 + ion + hydrogen,
            120.0 + ion - 2 * hydrogen,
            140.0 + ion + 2 * hydrogen,
        ],
        "it": [1.0, 0.5, 0.8, 0.6, 0.7],
        "adduct": "[M+H]+",
        "mode": 1,
        "prec": 200.0,
    }
    peaks = prepare_peaks([spectrum])
    assert peaks is not None
    categories, best, offsets, positions, classes = _matches(masses, peaks)
    assert categories[:, 0].tolist() == [-1, -1, 1, 2, 3, 4, 5, 0, 0, -1, -1]
    assert best[:, 0].tolist() == [-1, -1, 0, 1, 2, 3, 4, -1, -1, -1, -1]
    assert offsets.tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5]
    assert positions.tolist() == [0, 1, 2, 3, 4]
    assert classes.tolist() == [1, 2, 3, 4, 5]


def test_categorical_boundaries_and_nitrogen_oxygen_saturation() -> None:
    rows: Fragments = (
        np.arange(10.0, 101.0, 10.0),
        np.arange(10, dtype=np.int8) % 4,
        np.arange(10, dtype=np.int16),
        np.arange(-1, 9, dtype=np.int16),
        np.array([2, 3, 4, 5, 7, 10, 14, 20, 30, 31], np.int16),
        np.arange(10, dtype=np.int16),
        np.arange(10, dtype=np.int16),
    )
    for energy, category in (
        (float("nan"), 0),
        (-1.0, 1),
        (15.0, 1),
        (15.000000000000002, 2),
        (30.0, 2),
        (30.000000000000004, 3),
        (50.0, 3),
        (51.0, 4),
        (float("inf"), 0),
    ):
        values = categorical_fields(
            rows,
            np.full(10, 100.0),
            mode=-1,
            energy=energy,
        )
        assert values["ce"].tolist() == [category] * 10
        assert values["mode"].tolist() == [0] * 10
        assert values["ft"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        assert values["c1"].tolist() == list(range(10))
        assert values["c2"].tolist() == list(range(10))
        assert values["size"].tolist() == [0, 1, 1, 2, 3, 4, 5, 6, 7, 7]
        assert values["frac"].tolist() == list(range(10))
        assert values["nN"].tolist() == [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
        assert values["nO"].tolist() == [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]
        assert all(value.dtype == np.int64 for value in values.values())
    small: Fragments = (
        np.array([0.1, 5e-10, 1.1]),
        rows[1][:3],
        rows[2][:3],
        rows[3][:3],
        rows[4][:3],
        rows[5][:3],
        rows[6][:3],
    )
    values = categorical_fields(
        small,
        np.array([0.25, 0.0, 0.25]),
        mode=0,
        energy=30.0,
    )
    assert values["mode"].tolist() == [0, 0, 0]
    assert values["frac"].tolist() == [3, 4, 9]


def test_positive_explanations_mask_and_ring_chain_capacity() -> None:
    record: FragnetRecord = {
        "amass": np.array([20.0, 30.0, 40.0, 50.0]),
        "kind": np.array([0, 6, 9, 0], np.int8),
        "el": np.array([0, 1, 2, 0], np.int8),
        "nh": np.zeros(4, np.int8),
        "bonds": np.array([[0, 1], [1, 2], [2, 0], [2, 3]], np.int32),
        "bring": np.array([1, 1, 1, 0], np.int8),
        "mass": 140.0,
        "el10": np.zeros(4, np.uint8),
    }
    complete = fragments(record)
    capped = fragments(record, maximum=1)
    assert len(capped[0]) == 1
    assert capped[0].item(0) == complete[0].item(0)
    assert 2 in complete[1]
    assert 1 in complete[1]
    assert complete[5].max() == 1
    assert complete[6].max() == 1
    peaks = prepare_peaks(
        [
            {
                "mz": [51.00727645217, 91.00727645217],
                "it": [1.0, 0.5],
                "adduct": "[M+H]+",
                "mode": 1,
                "prec": 180.0,
            },
        ],
    )
    assert peaks is not None
    table = np.tile(np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0], np.float32), (8, 1))
    stage_a = StageA(
        chains=((("mode", "ft"),),),
        tabs=(((np.arange(8), table),),),
        alpha=80.0,
    )
    prepared = prepare_candidates([record], peaks)
    values = aggregates(stage_a, prepared, peaks, mask=np.array([False]))
    assert values[0, 0, 5] > 0
    assert values[0, 0, 5] == values[0, 0, 6] == values[0, 0, 16] == values[0, 0, 17]
    with pytest.raises(ValueError, match="StageB"):
        _ = features(
            [record],
            [
                {
                    "mz": [100.0],
                    "it": [1.0],
                    "adduct": "[M+H]+",
                    "mode": 1,
                    "prec": 150.0,
                },
            ],
            formulas=["C"],
            model_score=np.array([1.0], np.float32),
            stage_a=stage_a,
            stage_b=LightGBMModel(
                features=(),
                trees=(),
                objective="lambdarank",
                sigmoid=1.0,
                average=False,
            ),
        )


def test_aggregates_sum_class_kind_peak_and_frequency_channels_by_hand() -> None:
    # Protonated 60 and 80 Da peaks at intensities 1 and 1/4 weigh 1 and 1/2.
    peaks = prepare_peaks(
        [
            {
                "mz": [61.00727645217, 81.00727645217],
                "it": [1.0, 0.25],
                "adduct": "[M+H]+",
                "mode": 1,
                "prec": 220.0,
                "ce": 20.0,
            },
        ],
    )
    assert peaks is not None
    assert peaks.pk_w.tolist() == [1.0, 0.5]
    # Positive mode keys row 4 + kind: single cuts score -1 unexplained and 3
    # explained, the double cut -2 unexplained.
    table = np.zeros((8, 6), np.float32)
    table[5, :2] = [-1.0, 3.0]
    table[7, 0] = -2.0
    stage_a = StageA(
        chains=((("mode", "ft"),),),
        tabs=(((np.arange(8), table),),),
        alpha=80.0,
    )
    # The chain cuts 60, 150, 70 (double), 130 and 80 Da; 60 and 80 explain both
    # peaks, and the record-less candidate explains neither.
    prepared = prepare_candidates([chain_record(), None], peaks)
    fragment = [2.0, 6.0, -4.0, 5.0, 2.0]
    peak = [4.5, 4.5, 1.5, 6.0, 2.0]
    kind = [4.0, 0.0, -2.0, 4.0, 2.0, 0.0]
    for mask, frequency in (
        (None, 0.5),
        (np.array([False, True]), 0.0),
        (np.array([True, False]), 1.0),
    ):
        values = aggregates(stage_a, prepared, peaks, mask=mask)
        assert values.shape == (2, 1, 19)
        # The last three channels discount by the masked explaining share.
        discount = [value * (1.0 - frequency) for value in (4.5, 4.5, 1.5)]
        assert values[0, 0].tolist() == [*fragment, *peak, *kind, *discount]
        # Unexplained top peaks only add their weight to the record-less row.
        assert values[1, 0].tolist() == [0.0] * 15 + [1.5] + [0.0] * 3


def test_llr_backs_off_each_chain_and_counts_the_shared_level_once() -> None:
    # Positive single and ring cuts key the shared level 5 and 6; key 7 is unseen.
    fields = {
        "mode": np.ones(4, np.int64),
        "ft": np.array([1, 1, 2, 3], np.int64),
        "c1": np.array([0, 4, 0, 0], np.int64),
        "size": np.array([0, 2, 0, 0], np.int64),
    }
    classes = np.arange(1, 7, dtype=np.float32)
    shared = np.array([5, 6], np.int64)
    model = StageA(
        chains=((("mode", "ft"), ("size",)), (("mode", "ft"), ("c1",))),
        # Size 2 refines key 5 to 5 * 8 + 2, and c1 0 refines it to 5 * 507.
        tabs=(
            (
                (shared, np.stack([classes, 2 * classes])),
                (np.array([42], np.int64), 10 * classes[None]),
            ),
            (
                (shared, np.stack([100 * classes, 200 * classes])),
                (np.array([2535], np.int64), 1000 * classes[None]),
            ),
        ),
        alpha=80.0,
    )
    # Chain matches less the shared level: 1+1000-1, 10+100-1, 2+200-2 and none.
    np.testing.assert_array_equal(
        model.llr(fields),
        np.array([1000.0, 109.0, 200.0, 0.0])[:, None] * classes,
    )


def test_peaks_flag_the_five_strongest_of_each_spectrum() -> None:
    peaks = prepare_peaks(
        [
            {
                "mz": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                "it": [1.0, 6.0, 3.0, 5.0, 2.0, 4.0],
                "adduct": "[M+H]+",
                "mode": 1,
                "prec": 200.0,
            },
            {
                "mz": [60.0, 70.0, 80.0],
                "it": [1.0, 1.0, 1.0],
                "adduct": "[M+H]+",
                "mode": 1,
                "prec": 200.0,
            },
        ],
    )
    assert peaks is not None
    assert peaks.pk_off.tolist() == [0, 6, 9]
    # Only the weakest peak of the first spectrum falls outside its top five.
    assert peaks.top.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("case", ["version2", "version3", "numeric", "value"])
def test_stage_a_chain_envelope_uses_bounded_numpy_data(
    tmp_path: Path,
    case: str,
) -> None:
    chain = np.array([[["mode", "ft"]]], dtype=object)
    payload = io.BytesIO()
    if case == "value":
        np.lib.format.write_array_header_1_0(
            payload,
            {"descr": "|O", "fortran_order": False, "shape": (2,)},
        )
        pickle.dump(["invalid", "object"], payload)
    else:
        np.lib.format.write_array(
            payload,
            np.arange(2) if case == "numeric" else chain,
            version=(2, 0)
            if case == "version2"
            else (1, 0)
            if case == "numeric"
            else (3, 0),
            allow_pickle=True,
        )
    path = tmp_path / "chains.npz"
    with ZipFile(path, "w") as archive:
        _ = archive.writestr("chains.npy", payload.getvalue())
        for name, value in (
            ("alpha", np.array(80)),
            ("k_0_0", np.arange(8)),
            ("l_0_0", np.zeros((8, 6), np.float32)),
        ):
            array = io.BytesIO()
            np.save(array, value, allow_pickle=False)
            _ = archive.writestr(name + ".npy", array.getvalue())
    seal = hashlib.sha256(path.read_bytes()).hexdigest()
    if case == "version2":
        assert StageA.load(path, seal).chains == ((("mode", "ft"),),)
    else:
        with pytest.raises(
            (ValueError, TypeError),
            match=r"format|object array|NumPy array",
        ):
            _ = StageA.load(path, seal)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
