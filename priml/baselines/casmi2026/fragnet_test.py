"""Checks for the fragnet module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from zipfile import ZipFile

import hashlib
import io
import json
import pickle
import sys
import types

from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026 import fragnet
from priml.baselines.casmi2026.boosting import LightGBMModel
from priml.baselines.casmi2026.custom_types import (
    SpectrumRequest,
)
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
from priml.baselines.casmi2026.testing import float32_comparands
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import (
        FragnetRecord,
    )


_BASELINE: Final = _CWD


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


GOLDEN: Final = _BASELINE / "testdata" / "fragnet.pt"


def test_complete_fragmentation_tables_three_updates_and_all_family_outputs(
    tmp_path: Path,
) -> None:
    golden = GOLDEN
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    records: list[FragnetRecord | None] = []
    for index in range(5):
        prefix = f"input/{index}/"
        for name, tensor in expected.items():
            if name.startswith(prefix):
                actual[name] = tensor.clone()
        if not expected[prefix + "valid"].item():
            records.append(None)
            continue
        row: FragnetRecord = {
            "amass": expected[prefix + "amass"].numpy(),
            "kind": expected[prefix + "kind"].numpy(),
            "el": expected[prefix + "el"].numpy(),
            "nh": expected[prefix + "nh"].numpy(),
            "bonds": expected[prefix + "bonds"].numpy(),
            "bring": expected[prefix + "bring"].numpy(),
            "mass": expected[prefix + "mass"].item(),
            "el10": expected[prefix + "el10"].numpy(),
        }
        records.append(row)
        for maximum in (2, 30_000):
            for column, array in enumerate(fragments(row, maximum=maximum)):
                actual[f"enumeration/{index}/{maximum}/{column}"] = torch.from_numpy(
                    array,
                )
    arrays: dict[str, NDArray[np.generic]] = {"alpha": expected["stageA/alpha"].numpy()}
    chains: list[NDArray[np.object_]] = []
    for ci in range(2):
        chain: list[NDArray[np.str_]] = []
        for li in range(3):
            names = (
                expected[f"stageA/chain/{ci}/{li}"]
                .numpy()
                .tobytes()
                .decode()
                .split("\0")
            )
            chain.append(np.array(names))
            arrays[f"k_{ci}_{li}"] = expected[f"stageA/keys/{ci}/{li}"].numpy()
            arrays[f"l_{ci}_{li}"] = expected[f"stageA/values/{ci}/{li}"].numpy()
        chains.append(np.array(chain, dtype=object))
    arrays["chains"] = np.array(chains, dtype=object)
    asset = tmp_path / "stageA.npz"
    np.savez(asset, allow_pickle=True, **arrays)
    model_a = StageA.load(asset, hashlib.sha256(asset.read_bytes()).hexdigest())
    actual["stageA/alpha"] = torch.tensor(model_a.alpha, dtype=torch.float64)
    for ci, loaded_chain in enumerate(model_a.chains):
        for li, names in enumerate(loaded_chain):
            actual[f"stageA/chain/{ci}/{li}"] = torch.tensor(
                list("\0".join(names).encode()),
                dtype=torch.uint8,
            )
            keys, values = model_a.tabs[ci][li]
            actual[f"stageA/keys/{ci}/{li}"] = torch.from_numpy(keys)
            actual[f"stageA/values/{ci}/{li}"] = torch.from_numpy(values)
    text = expected["stageB/header"].numpy().tobytes().decode()
    for step in range(3):
        text += f"Tree={step}\n"
        prefix = f"stageB/tree/{step}/"
        for name, tensor in expected.items():
            if name.startswith(prefix):
                text += (
                    name.removeprefix(prefix)
                    + "="
                    + " ".join(str(value) for value in tensor.tolist())
                    + "\n"
                )
    model_b = LightGBMModel.parse(text + "end of trees\n")
    assert model_b.features == stage_b_names()
    actual["stageB/header"] = expected["stageB/header"].clone()
    actual["stageB/train"] = expected["stageB/train"].clone()
    for step, tree in enumerate(model_b.trees):
        for name, value in tree.columns.items():
            actual[f"stageB/tree/{step}/{name}"] = torch.from_numpy(value)
        actual[f"stageB/round/{step + 1}"] = torch.from_numpy(
            model_b.predict(expected["stageB/train"].numpy(), rounds=step + 1),
        )
    spectra: list[SpectrumRequest] = []
    for index in range(3):
        prefix = f"spectra/{index}/"
        spectra.append(
            {
                "mz": [
                    expected[prefix + "mz"][index].item()
                    for index in range(expected[prefix + "mz"].numel())
                ],
                "it": [
                    expected[prefix + "it"][index].item()
                    for index in range(expected[prefix + "it"].numel())
                ],
                "prec": 250.0,
                "adduct": ("[M+Na]+", "[M-H]-", "unknown")[index],
                "mode": (1, -1, 1)[index],
                "ce": (10.0, 35.0, float("nan"))[index],
            },
        )
        for name in ("mz", "it"):
            actual[prefix + name] = expected[prefix + name].clone()
    peaks = prepare_peaks(spectra)
    assert peaks is not None
    peak_arrays = {
        "pk_mz": peaks.pk_mz,
        "pk_w": peaks.pk_w,
        "pk_off": peaks.pk_off,
        "pk_spec": peaks.pk_spec,
        "sp_del": peaks.sp_del,
        "sp_nd": peaks.sp_nd,
        "lo": peaks.lo,
        "hi": peaks.hi,
        "top": peaks.top,
        "wtot": peaks.wtot,
        "mode": peaks.mode,
        "ce": peaks.ce,
    }
    actual["peaks/S"] = torch.tensor(peaks.S)
    for name, value in peak_arrays.items():
        actual[f"peaks/{name}"] = torch.from_numpy(value)
    prepared = prepare_candidates(records, peaks)
    for index, column in enumerate(prepared.cat):
        actual[f"prepared/cat/{index}"] = torch.from_numpy(column)
    prepared_arrays = {
        "coff": prepared.coff,
        "M": prepared.M,
        "cls": prepared.cls,
        "pkix": prepared.pkix,
        "inv": prepared.inv,
        "moff": prepared.moff,
        "mk": prepared.mk,
        "mc": prepared.mc,
    }
    actual["prepared/n"] = torch.tensor(prepared.n)
    for name, value in prepared_arrays.items():
        actual[f"prepared/{name}"] = torch.from_numpy(value)
    for mode in (1, -1):
        fd = categorical_fields(
            prepared.cat,
            prepared.M,
            mode=mode,
            energy=float("nan") if mode < 0 else 25.0,
        )
        for name, array in fd.items():
            actual[f"fields/{mode}/{name}"] = torch.from_numpy(array)
        actual[f"stageA/llr/{mode}"] = torch.from_numpy(model_a.llr(fd))
    for name, mask in (
        ("all", None),
        ("mask", np.array([True, False, True, False, False])),
    ):
        agg = aggregates(model_a, prepared, peaks, mask=mask)
        actual[f"aggregate/{name}"] = torch.from_numpy(agg)
        for spectrum in range(2):
            actual[f"stageB/features/{name}/{spectrum}"] = torch.from_numpy(
                stage_b_features(agg[:, spectrum]),
            )
    scores = np.array([0.3, 0.3, 0.2, 0.1, 0], np.float32)
    formulas = ["C2H5Br", "C2H4S", "C2H5I", "C2H5Br", ""]
    for case, (query, cap, size) in enumerate(
        (
            (spectra, 400, 5),
            (spectra, 2, 5),
            (spectra, 3, 5),
            (list[SpectrumRequest](), 400, 5),
            (spectra, 400, 0),
        ),
    ):
        output = features(
            records[:size],
            query,
            formulas=formulas[:size],
            model_score=scores[:size],
            stage_a=model_a,
            stage_b=model_b,
            maximum=cap,
        )
        actual[f"features/{case}"] = torch.from_numpy(
            np.column_stack(list(output.values())),
        )
    assert_tensor_golden(
        golden,
        float32_comparands(
            actual,
            inputs=(
                "input/*",
                "spectra/*",
                "stageA/alpha",
                "stageB/train",
                "stageB/tree/*",
            ),
        ),
    )


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
    masses = np.array([20.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 220.0])
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
    assert categories[:, 0].tolist() == [-1, 1, 2, 3, 4, 5, 0, -1]
    assert best[:, 0].tolist() == [-1, 0, 1, 2, 3, 4, -1, -1]
    assert offsets.tolist() == [0, 0, 1, 2, 3, 4, 5, 5, 5]
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
