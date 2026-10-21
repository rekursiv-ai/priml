"""Checks for the pubchem module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json
import math
import re
import sys

from pyarrow import Table, parquet
from torch import Tensor
from treekle import ReadError, from_plain, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026 import pubchem
from priml.baselines.casmi2026.chemistry import CandidatePool
from priml.baselines.casmi2026.dreams import MainBank, keep_count
from priml.baselines.casmi2026.dreams_test import save_bank
from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    precursor_mass,
)
from priml.baselines.casmi2026.pubchem import (
    PubChemAssets,
    PubChemGate,
    PubChemScreen,
    PubChemStore,
    PubChemTable,
    _order,
    _partial_rows,
    best_pool_fz,
    first_modal_adduct,
    logger,
    molecule_logits,
)
from priml.baselines.casmi2026.testing import (
    float32_comparands,
    retrieval_inputs,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import SpectrumRequest
    from priml.baselines.casmi2026.pubchem import PubChemDiagnostics, PubChemRecord


_BASELINE: Final = _CWD


def test_entire_pubchem_runner_bank_screen_and_all_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    golden = _BASELINE / "testdata/pubchem.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    strings = write_pubchem_table(prepared, expected=expected, prefix="chemistry")
    chemistry = PubChemTable(prepared)
    records = [chemistry.row(index) for index in range(len(strings))]
    observe(
        actual,
        "chemistry",
        {
            "smiles": [row.smiles for row in records],
            "parsed": np.array([row.parsed for row in records], np.bool_),
            "raw_valid": np.array([row.raw_valid for row in records], np.bool_),
            "ecfp4": np.stack([row.ecfp4 for row in records]),
            "raw_fp": np.stack([row.raw_fp for row in records]),
            "keys": [row.key for row in records],
        },
    )
    for name, filename in (
        ("mass", "pc_mass"),
        ("off", "pc_off"),
        ("buf", "pc_smiles"),
        ("ls", "pc_lsid"),
        ("lp", "pc_lpmid"),
    ):
        np.save(tmp_path / (filename + ".npy"), expected["store/" + name].numpy())
    store = PubChemStore.load(tmp_path, popularity=tmp_path)
    assert store.ls is not None
    assert store.lp is not None
    for mapped in (store.mass, store.offsets, store.buffer, store.ls, store.lp):
        assert not mapped.flags.writeable
    bits = expected["store/bits"].numpy()
    for name, value in (
        ("mass", store.mass),
        ("off", store.offsets),
        ("buf", store.buffer),
        ("ls", store.ls),
        ("lp", store.lp),
        ("bits", bits),
    ):
        observe(actual, "store/" + name, value)
    pool_keys = {decode(expected["pool_keys/0"])}
    observe(actual, "pool_keys", sorted(pool_keys))
    for name, target, maximum, weight, union, limit, present in (
        ("normal", 100.0, 3, 0.25, 2, 2, True),
        ("all", 100.0, 99, 0.0, 2, 25, True),
        ("missing_pop", 100.0, 3, 0.25, 2, 25, False),
        ("no_union", 100.0, 3, 0.25, 0, 25, True),
        ("negative_weight", 100.0, 3, -0.5, 2, 25, True),
        ("empty", 900.0, 3, 0.25, 2, 25, True),
        ("invalid", 80.0, 3, 0.25, 2, 25, True),
        ("no_full", 100.0, 3, 0.25, 2, 25, True),
        ("all_pool", 100.0, 99, 0.25, 2, 25, True),
    ):
        config = PubChemScreen.Config()
        config.pass_one, config.popularity_weight = maximum, weight
        config.popularity_union, config.limit = union, limit
        screen = config.make()
        selected_store = store if present else PubChemStore.load(tmp_path)
        selected_chemistry = chemistry
        if name == "no_full":
            failed = tmp_path / "failed"
            failed.mkdir()
            write_pubchem_table(
                failed,
                expected=expected,
                prefix="chemistry",
                raw_valid=False,
            )
            selected_chemistry = PubChemTable(failed)
        clock = iter([10.0, 10.125])
        result = screen.probe(
            name,
            target,
            np.array([-0.7, 0.2, 1.3, -0.1, 0.6, -0.3, 0.8], np.float32),
            store=selected_store,
            chemistry=selected_chemistry,
            bits=bits,
            pool_keys={row.key for row in records if row.key is not None}
            if name == "all_pool"
            else pool_keys,
            clock=clock.__next__,
        )
        observe(actual, "probe/" + name, result)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        bank = tiny_pubchem_bank(tmp_path)
        spectra: list[SpectrumRequest] = [
            {
                "mz": [0, 25, 45, 102, 103],
                "it": [1, 4, 2, 0.004, 9],
                "prec": 100.0,
                "adduct": "[M+Na]+",
                "mode": 1,
                "ce": 10.0,
                "ce_n": 0,
            },
            {
                "mz": [25.003, 45, 70],
                "it": [2, 5, 1],
                "prec": 120.0,
                "adduct": "[M+H]+",
                "mode": 1,
                "ce": None,
                "ce_n": 12,
            },
            {
                "mz": [20, 35],
                "it": [3, 1],
                "prec": 80.0,
                "adduct": "unknown",
                "mode": -1,
                "ce": float("nan"),
                "ce_n": 2,
            },
            {
                "mz": [],
                "it": [],
                "prec": 90.0,
                "adduct": "[M-H]-",
                "mode": -1,
                "ce": 50.0,
                "ce_n": 1,
            },
        ]
        source_spectra = [
            dict(row, mz=np.array(row["mz"]), it=np.array(row["it"])) for row in spectra
        ]
        observe(actual, "spectra", source_spectra)
        single, merged = bank.items(spectra)
        for call, items in enumerate((single, merged)):
            observe(
                actual,
                f"bank/{call}/items",
                [
                    {
                        "mz": item.mass,
                        "it": item.intensity,
                        "prec": item.precursor,
                        "adduct_ix": item.adduct,
                        "ce": item.energy,
                        "ce_known": item.energy_known,
                        "n_merged": item.merged_count,
                        "mode": item.mode,
                    }
                    for item in items
                ],
            )
            observe(actual, f"bank/{call}/heads", bank.predict(items))
        observe(actual, "logits", molecule_logits(bank, spectra))
        observe(actual, "empty_logits", molecule_logits(bank, []))
        pool = tiny_pubchem_pool(expected)
        observe(actual, "pool/mass", pool.mass)
        observe(actual, "pool/packed", pool.packed)
        input_rows: list[dict[str, object]] = []
        queries: dict[str, list[SpectrumRequest]] = {}
        for mid, count, target, adduct in (
            ("full", 17, 100.0, "[M+H]+"),
            ("fallback", 2, 100.004, "[M+H]+"),
            ("no_pool", 3, 150.0, "[M-H]-"),
            ("unknown", 2, 100.0, "unknown"),
        ):
            queries[mid] = []
            for index in range(count):
                precursor = (
                    precursor_mass(target + index * 0.0001, adduct=adduct)
                    if adduct != "unknown"
                    else target
                )
                input_rows.append(
                    {
                        "molecule_id": mid,
                        "precursor_mz": precursor,
                        "adduct": adduct,
                        "ionization_mode": "positive"
                        if mid != "no_pool"
                        else "negative",
                        "ms2_mzs": [25.0, 45.0, 80.0],
                        "ms2_normalized_intensities": [3.0, 1.0, 0.2],
                        "collision_energy_ev": [10.0, 20.0] if index % 2 else None,
                    },
                )
                queries[mid].append(
                    {
                        "mz": [25.0, 45.0, 80.0],
                        "it": [3.0, 1.0, 0.2],
                        "prec": precursor,
                        "adduct": adduct,
                        "mode": 1 if mid != "no_pool" else -1,
                        "ce": 15.0 if index % 2 else float("nan"),
                        "ce_n": 2 if index % 2 else 1,
                    },
                )
        observe(actual, "runner_inputs", input_rows)
        config = PubChemScreen.Config()
        config.pass_one, config.popularity_union, config.limit = 3, 2, 2
        screen = config.make()
        results = {
            mid: value
            for mid, query in queries.items()
            if (
                value := screen.run(
                    mid,
                    query,
                    store=store,
                    chemistry=chemistry,
                    pool=pool,
                    bank=bank,
                )
            )
            is not None
        }
        observe(actual, "runner", results)
    assert_tensor_golden(
        golden,
        float32_comparands(actual, inputs=("pool/mass", "store/mass")),
    )


def test_pubchem_majority_ties_retain_first_encounter_and_single_zero_count(
    tmp_path: Path,
) -> None:
    single, merged = tiny_pubchem_bank(tmp_path).items(
        [
            {
                "mz": [],
                "it": [],
                "prec": 100.0,
                "adduct": "unknown",
                "mode": 1,
                "ce": None,
                "ce_n": 0,
            },
            {
                "mz": [],
                "it": [],
                "prec": 120.0,
                "adduct": "[M+H]+",
                "mode": 1,
                "ce": 20.0,
                "ce_n": 12,
            },
        ],
    )
    assert single[0].merged_count == 0
    assert single[1].merged_count == 8
    assert len(merged) == 1
    assert merged[0].adduct == single[0].adduct
    assert merged[0].precursor == 110.0
    assert merged[0].merged_count == 8
    assert merged[0].energy == 20.0
    assert merged[0].energy_known == 1.0


def test_pubchem_majority_default_counts_unknown_energy_and_string_polarities(
    tmp_path: Path,
) -> None:
    spectra: list[SpectrumRequest] = [
        {"mz": [], "it": [], "prec": 100.0, "adduct": adduct, "mode": mode, "ce": None}
        for adduct, mode in (
            ("[M+H]+", "positive"),
            ("[M+Na]+", "positive"),
            ("[M+H]+", "positive"),
            ("[M-H]-", "negative"),
        )
    ]
    single, merged = tiny_pubchem_bank(tmp_path).items(spectra)
    assert [row.mode for row in single] == [1, 1, 1, -1]
    assert [row.merged_count for row in single] == [1, 1, 1, 1]
    assert [row.mode for row in merged] == [1, -1]
    assert merged[0].adduct == single[0].adduct
    assert [row.merged_count for row in merged] == [3, 1]
    assert [row.energy for row in merged] == [0, 0]
    assert [row.energy_known for row in merged] == [0, 0]


class _FailingBank:
    """Stand in for a two-model bank whose prediction raises."""

    models = (object(), object())

    def items(self, spectra: object) -> object:
        del spectra
        raise RuntimeError("prediction failed")


def _screen_inputs(
    tmp_path: Path,
) -> tuple[PubChemStore, PubChemTable, CandidatePool]:
    expected = read_tensors(_BASELINE / "testdata/pubchem.pt")
    write_pubchem_table(tmp_path, expected=expected, prefix="chemistry")
    for name, filename in (
        ("mass", "pc_mass"),
        ("off", "pc_off"),
        ("buf", "pc_smiles"),
    ):
        np.save(tmp_path / (filename + ".npy"), expected["store/" + name].numpy())
    return (
        PubChemStore.load(tmp_path),
        PubChemTable(tmp_path),
        tiny_pubchem_pool(expected),
    )


def test_run_requires_exactly_two_models(tmp_path: Path) -> None:
    store, chemistry, pool = _screen_inputs(tmp_path)

    class _OneModel:
        models = (object(),)

    with pytest.raises(
        ValueError,
        match=r"^PubChem requires exactly two V3 fingerprint models\.$",
    ):
        _ = (
            PubChemScreen.Config()
            .make()
            .run(
                "m",
                [],
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=cast("MainBank", _OneModel()),
            )
        )


def test_run_logs_the_failed_molecule_and_skips_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, chemistry, pool = _screen_inputs(tmp_path)
    spectra: list[SpectrumRequest] = [
        {"mz": [25.0], "it": [1.0], "prec": 101.0, "adduct": "[M+H]+", "mode": 1},
    ]
    with caplog.at_level("ERROR", logger=logger.name):
        result = (
            PubChemScreen.Config()
            .make()
            .run(
                "molecule-7",
                spectra,
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=cast("MainBank", _FailingBank()),
            )
        )
    assert result is None
    assert [record.getMessage() for record in caplog.records] == [
        "PubChem fingerprint prediction failed for molecule-7.",
    ]


def test_run_converts_integer_precursors_in_double_precision_and_names_the_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, chemistry, pool = _screen_inputs(tmp_path)
    bank = tiny_pubchem_bank(tmp_path)
    screen = PubChemScreen.Config().make()
    identities: list[str] = []
    original = PubChemScreen.probe

    def probe(
        self: PubChemScreen,
        molecule_id: str,
        target: float,
        logits: NDArray[np.float32],
        *,
        store: PubChemStore,
        chemistry: PubChemTable,
        bits: NDArray[np.int64],
        pool_keys: set[str],
    ) -> tuple[str, list[str], list[float], list[str], PubChemDiagnostics]:
        identities.append(molecule_id)
        return original(
            self,
            molecule_id,
            target,
            logits,
            store=store,
            chemistry=chemistry,
            bits=bits,
            pool_keys=pool_keys,
        )

    monkeypatch.setattr(PubChemScreen, "probe", probe)
    spectra: list[SpectrumRequest] = [
        {
            "mz": [25.0],
            "it": [1.0],
            "prec": 2**62,
            "adduct": "[M+2H]2+",
            "mode": 1,
        },
    ]
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        result = screen.run(
            "molecule-9",
            spectra,
            store=store,
            chemistry=chemistry,
            pool=pool,
            bank=bank,
        )
    assert result is not None
    assert result["target"] == float(2**63)
    assert identities == ["molecule-9"]


def test_prepared_rows_require_every_field_and_preserve_first_duplicate(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem_preparation.pt")
    values = write_pubchem_table(tmp_path, expected=expected, prefix="rows")
    table = PubChemTable(tmp_path)
    assert table.for_smiles("*").parsed
    assert table.for_smiles("*").raw_valid
    assert table.for_smiles("*").key is None
    assert table.for_smiles("").raw_valid
    assert table.for_smiles("").key is None
    assert not table.for_smiles("invalid").parsed
    for index, value in enumerate(values):
        row = table.row(index)
        assert row.smiles == value
        assert not row.ecfp4.flags.writeable
        assert not row.raw_fp.flags.writeable
    with pytest.raises(KeyError, match="absent"):
        table.for_smiles("absent")
    for index in (-1, len(values)):
        with pytest.raises(IndexError):
            table.row(index)
    metadata = from_plain(
        parquet.read_table(tmp_path / "molecules.parquet").to_pylist(),
        list[dict[str, object]],
    )
    metadata[1].update(smiles="CCO", key="second")
    parquet.write_table(Table.from_pylist(metadata), tmp_path / "molecules.parquet")
    table = PubChemTable(tmp_path)
    assert table.row(1).key == "second"
    assert table.for_smiles("CCO").key == table.row(0).key
    ecfp4 = cast("NDArray[np.generic]", np.load(tmp_path / "ecfp4.npy"))
    np.save(tmp_path / "ecfp4.npy", np.zeros((8, 512), np.int8))
    with pytest.raises(TypeError, match=r"expected uint8, got int8\.$"):
        PubChemTable(tmp_path)
    np.save(tmp_path / "ecfp4.npy", ecfp4)
    for name, malformed, message in (
        (
            "ecfp4",
            np.zeros(512, np.uint8),
            "PubChem array ecfp4.npy has an invalid rank.",
        ),
        (
            "raw_fp",
            np.zeros((7, 2070), np.uint8),
            "PubChem complete fingerprint widths or rows differ from metadata.",
        ),
        (
            "raw_fp",
            np.zeros((8, 2068), np.uint8),
            "PubChem complete fingerprint widths or rows differ from metadata.",
        ),
    ):
        path = tmp_path / (name + ".npy")
        original = cast("NDArray[np.generic]", np.load(path, allow_pickle=False))
        np.save(path, malformed)
        with pytest.raises(ValueError, match="^" + re.escape(message) + "$") as error:
            PubChemTable(tmp_path)
        assert error.value.args == (message,)
        np.save(path, original)
    header: dict[str, object] = {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "raw_bits": 16_551,
        "ecfp4_bits": 4096,
    }
    for name, value in (("raw_bits", 16_552), ("ecfp4_bits", 4095)):
        (tmp_path / "manifest.json").write_text(
            json.dumps(to_plain(header | {name: value})),
        )
        with pytest.raises(
            ValueError,
            match=r"^PubChem data requires complete",
        ) as error:
            PubChemTable(tmp_path)
        assert error.value.args == (
            "PubChem data requires complete 16551-bit raw and 4096-bit ECFP4 layouts.",
        )
    for version, release in ((2, IDENTITY_RDKIT), (1, SIMULATOR_RDKIT)):
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                to_plain(header | {"format_version": version, "rdkit": release}),
            ),
        )
        with pytest.raises(ValueError, match=r"^PubChem data requires format") as error:
            PubChemTable(tmp_path)
        assert error.value.args == (
            f"PubChem data requires format 1 and RDKit {IDENTITY_RDKIT}.",
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(to_plain(header | {"format_version": True})),
    )
    with pytest.raises(ReadError):
        PubChemTable(tmp_path)


def test_pubchem_native_boundaries_keep_missing_staging_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem.pt")
    write_pubchem_table(tmp_path, expected=expected, prefix="chemistry")
    chemistry = PubChemTable(tmp_path)
    pool = tiny_pubchem_pool(expected)
    assert math.isnan(
        best_pool_fz(
            pool,
            target=999.0,
            logits=np.ones(7, np.float32),
            ppm=10.0,
            fallback_ppm=30.0,
        ),
    )
    invalid = [PubChemScreen.Config() for _ in range(3)]
    invalid[0].pass_one = 0
    invalid[1].limit = 0
    invalid[2].max_spectra = 0
    for config in invalid:
        with pytest.raises(ValueError, match="positive"):
            config.make()
    config = PubChemScreen.Config()
    config.popularity_union = -1
    with pytest.raises(ValueError, match="nonnegative"):
        config.make()
    screen = PubChemScreen.Config().make()
    store = PubChemStore(
        mass=np.array([100.0]),
        offsets=np.array([0, 6]),
        buffer=np.frombuffer(b"absent", np.uint8),
    )
    with pytest.raises(KeyError, match="absent"):
        screen.probe(
            "missing",
            100.0,
            np.ones(7, np.float32),
            store=store,
            chemistry=chemistry,
            bits=pool.bits,
            pool_keys=set(),
        )
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        bank = tiny_pubchem_bank(tmp_path)
        assert (
            screen.run(
                "empty",
                [],
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=bank,
            )
            is None
        )
        query: list[SpectrumRequest] = [
            {
                "mz": [],
                "it": [],
                "prec": precursor_mass(100.0, adduct="[M+H]+"),
                "adduct": "[M+H]+",
                "mode": "positive",
                "ce": None,
            },
        ]
        with pytest.raises(KeyError, match="absent"):
            screen.run(
                "missing",
                query,
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=bank,
            )
        with monkeypatch.context() as absent_logits:
            absent_logits.setattr(pubchem, "molecule_logits", missing_prediction)
            assert (
                screen.run(
                    "none",
                    query,
                    store=store,
                    chemistry=chemistry,
                    pool=pool,
                    bank=bank,
                )
                is None
            )
        monkeypatch.setattr(bank, "predict", fail_prediction)
        assert (
            screen.run(
                "failed",
                query,
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=bank,
            )
            is None
        )
        bank.models = bank.models[:1]
        with pytest.raises(ValueError, match="two"):
            screen.run(
                "one",
                query,
                store=store,
                chemistry=chemistry,
                pool=pool,
                bank=bank,
            )


def test_pubchem_store_rejects_malformed_arrays_and_pickle_payloads(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem_preparation.pt")
    write_pubchem_table(tmp_path, expected=expected, prefix="rows")
    (tmp_path / "ecfp4.npy").write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(ValueError, match=r"pickled.*data"):
        PubChemTable(tmp_path)
    values = {
        "pc_mass": np.array([100.0, 100.0001]),
        "pc_off": np.array([0, 3, 6], np.int64),
        "pc_smiles": np.frombuffer(b"CCOCOC", np.uint8),
        "pc_lsid": np.array([1, 2], np.float16),
        "pc_lpmid": np.array([3, 4], np.float32),
    }
    for name, value in values.items():
        np.save(tmp_path / (name + ".npy"), value)
    store = PubChemStore.load(tmp_path, popularity=tmp_path)
    assert store.window(100.0, ppm=10.0) == (0, ("CCO", "COC"))
    pc_mass = cast("NDArray[np.generic]", np.load(tmp_path / "pc_mass.npy"))
    np.save(tmp_path / "pc_mass.npy", np.zeros(2, np.float32))
    with pytest.raises(TypeError, match=r"expected float64, got float32\.$"):
        PubChemStore.load(tmp_path, popularity=tmp_path)
    np.save(tmp_path / "pc_mass.npy", pc_mass)
    for name, malformed, message in (
        (
            "pc_mass",
            # PubChemStore.load requires a flat mass vector, not a column matrix.
            np.zeros((2, 1)),
            "PubChem array pc_mass.npy has an invalid rank.",
        ),
        (
            "pc_off",
            np.array([0, 6], np.int64),
            "PubChem offsets differ from the complete mass and ASCII layout.",
        ),
        (
            "pc_off",
            np.array([1, 3, 6], np.int64),
            "PubChem offsets differ from the complete mass and ASCII layout.",
        ),
        (
            "pc_off",
            np.array([0, 3, 7], np.int64),
            "PubChem offsets differ from the complete mass and ASCII layout.",
        ),
        (
            "pc_lpmid",
            np.array([1.0]),
            "PubChem popularity shape or floating dtype differs from its mass rows.",
        ),
    ):
        np.save(tmp_path / (name + ".npy"), malformed)
        with pytest.raises(ValueError, match="^" + re.escape(message) + "$") as error:
            PubChemStore.load(tmp_path, popularity=tmp_path)
        assert error.value.args == (message,)
        np.save(tmp_path / (name + ".npy"), values[name])
    (tmp_path / "pc_lsid.npy").write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(ValueError, match=r"pickled.*data"):
        PubChemStore.load(tmp_path, popularity=tmp_path)
    for offsets in ([0, 7, 6], [-1, 3, 6], [0, 3, 7]):
        broken = replace(store, offsets=np.array(offsets, np.int64))
        with pytest.raises(
            ValueError,
            match=r"^PubChem window offsets exceed",
        ) as error:
            broken.window(100.0, ppm=10.0)
        assert error.value.args == (
            "PubChem window offsets exceed the declared ASCII layout.",
        )


def test_pubchem_pool_and_popularity_promotions_keep_full_precision(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem.pt")
    pool = tiny_pubchem_pool(expected)
    pool = replace(pool, packed=np.array([[0b11100000]] * 3, np.uint8))
    logits = np.array([1e8, 1, -1e8, 0, 0, 0, 0], np.float32)
    assert (
        best_pool_fz(
            pool,
            target=100.0,
            logits=logits,
            ppm=10.0,
            fallback_ppm=30.0,
        )
        == 1.0
    )
    write_pubchem_table(tmp_path, expected=expected, prefix="chemistry")
    chemistry = PubChemTable(tmp_path)
    first, second = (
        row
        for row in chemistry.metadata
        if row["key"] is not None and row["smiles"] in ("CCN", "COC")
    )
    strings = (first["smiles"], second["smiles"])
    raw = "".join(strings).encode("ascii")
    store = PubChemStore(
        mass=np.array([100.0, 100.0]),
        offsets=np.array([0, len(strings[0]), len(raw)], np.int64),
        buffer=np.frombuffer(raw, np.uint8),
        ls=np.array([1.0, 1.0], np.float16),
        lp=np.array([2**-11, 0], np.float16),
    )
    result = (
        PubChemScreen.Config()
        .make()
        .probe(
            "promotion",
            100.0,
            np.zeros(7, np.float32),
            store=store,
            chemistry=chemistry,
            bits=pool.bits,
            pool_keys=set(),
        )
    )
    assert result[1][0] == strings[0]
    assert result[4].get("top_pop") == 1.0 + 2**-11


def test_screening_edges_keep_the_mask_union_and_limit(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem.pt")
    strings = write_pubchem_table(tmp_path, expected=expected, prefix="chemistry")
    raw = np.zeros((len(strings), 16_551), np.uint8)
    raw[strings.index("CCN"), :3] = 1
    raw[strings.index("COC"), 1] = 1
    np.save(tmp_path / "ecfp4.npy", np.packbits(raw[:, :4096], axis=1))
    np.save(tmp_path / "raw_fp.npy", np.packbits(raw, axis=1))
    chemistry = PubChemTable(tmp_path)
    store = PubChemStore(
        mass=np.full(3, 100.0),
        offsets=np.array([0, 3, 6, 13], np.int64),
        buffer=np.frombuffer(b"CCNCOCinvalid", np.uint8),
    )
    bits = np.array([0, 1, 2, 4096], np.int64)
    logits = np.array([1e8, 1, -1e8, 0], np.float32)
    config = PubChemScreen.Config(
        pass_one=1,
        popularity_weight=0.0,
        limit=1,
        max_spectra=1,
    )
    result = config.make().probe(
        "partial",
        100.0,
        logits,
        store=store,
        chemistry=chemistry,
        bits=bits,
        pool_keys=set(),
    )
    # Both rows score 1e8 + 1 - 1e8 = 1 exactly; the stable tie keeps CCN.
    assert result[1] == ["CCN"]
    assert result[4]["fz_top"] == 1.0
    config.pass_one = 3
    result = config.make().probe(
        "full",
        100.0,
        logits,
        store=store,
        chemistry=chemistry,
        bits=bits,
        pool_keys=set(),
    )
    assert result[1] == ["CCN"]
    assert result[4].get("S") == 0.0
    config.popularity_weight, config.popularity_union, config.limit = 0.25, 1, 25
    config.pass_one = 1
    popular = replace(
        store,
        ls=np.array([0, 10, 100], np.float16),
        lp=np.zeros(3, np.float16),
    )
    result = config.make().probe(
        "union",
        100.0,
        logits,
        store=popular,
        chemistry=chemistry,
        bits=bits,
        pool_keys=set(),
    )
    assert result[1] == ["COC", "CCN"]
    assert result[4]["n_pass2"] == 2
    with pytest.raises(ValueError, match="Expected"):
        config.make().probe(
            "incomplete",
            100.0,
            logits,
            store=replace(popular, lp=None),
            chemistry=chemistry,
            bits=bits,
            pool_keys=set(),
        )
    for dtype in (np.float16, np.float32, np.float64):
        delta = 2**-30 if dtype is np.float64 else 0.0
        population = np.array([1.0 + delta, 1.0, 0.0], dtype)
        config.pass_one, config.popularity_union = 3, 0
        result = config.make().probe(
            "promotion",
            100.0,
            np.zeros(4, np.float32),
            store=replace(store, ls=population, lp=population),
            chemistry=chemistry,
            bits=bits,
            pool_keys=set(),
        )
        assert result[4].get("top_pop") == 2.0
    raw.fill(0)
    raw[strings.index("CCN"), [0, 4096]] = 1
    raw[strings.index("COC"), [1, 4097]] = 1
    np.save(tmp_path / "ecfp4.npy", np.packbits(raw[:, :4096], axis=1))
    np.save(tmp_path / "raw_fp.npy", np.packbits(raw, axis=1))
    chemistry = PubChemTable(tmp_path)
    config.pass_one, config.popularity_union = 2, 0
    tied = PubChemStore(
        mass=np.full(3, 100.0),
        offsets=np.array([0, 3, 6, 9], np.int64),
        buffer=np.frombuffer(b"CCNCOCCCO", np.uint8),
        ls=np.zeros(3, np.float16),
        lp=np.zeros(3, np.float16),
    )
    result = config.make().probe(
        "no-union-ties",
        100.0,
        np.array([1, 2, -1, -2], np.float32),
        store=tied,
        chemistry=chemistry,
        bits=np.array([0, 1, 4096, 4097], np.int64),
        pool_keys=set(),
    )
    assert result[1] == ["COC", "CCN"]


def test_first_stage_order_stable_full_order_and_inclusive_mass_bounds(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_BASELINE / "testdata/pubchem.pt")
    write_pubchem_table(tmp_path, expected=expected, prefix="chemistry")
    table = PubChemTable(tmp_path)
    template = table.for_smiles("CCN")
    records: list[PubChemRecord] = []
    for positions in ((1,), (0, 1, 2), ()):
        fingerprint = np.zeros(4096, np.uint8)
        fingerprint[list(positions)] = 1
        records.append(replace(template, ecfp4=np.packbits(fingerprint)))
    bits = np.array([0, 1, 2, 4096], np.int64)
    logits = np.array([1e8, 1, -1e8, 0], np.float32)
    np.testing.assert_array_equal(
        _partial_rows(records, bits=bits, logits=logits, maximum=1),
        [0],
    )
    np.testing.assert_array_equal(
        _partial_rows(records[1::-1], bits=bits, logits=logits, maximum=2),
        [0, 1],
    )
    invalid = [replace(row, parsed=False) for row in records]
    empty = _partial_rows(invalid, bits=bits, logits=logits, maximum=1)
    np.testing.assert_array_equal(np.arange(3)[empty], [])
    np.testing.assert_array_equal(
        _partial_rows([], bits=bits, logits=logits, maximum=1),
        [],
    )
    ranked: list[PubChemRecord] = []
    for index in range(6):
        fingerprint = np.zeros(4096, np.uint8)
        fingerprint[index] = 1
        ranked.append(replace(template, ecfp4=np.packbits(fingerprint)))
    selected = _partial_rows(
        ranked,
        bits=np.arange(6, dtype=np.int64),
        logits=np.array([1, 6, 2, 5, 3, 4], np.float32),
        maximum=3,
    )
    assert set(selected) == {1, 3, 5}
    values = np.array([float(index % 3) for index in range(50)])
    assert _order(values) == [
        index for group in (2, 1, 0) for index in range(50) if index % 3 == group
    ]
    target = 1000.0
    tolerance = target * 10.0 * 1e-6
    mass = np.array(
        [
            target - tolerance * 1.05,
            target - tolerance,
            target,
            target + tolerance,
            target + tolerance * 1.05,
        ],
    )
    store = PubChemStore(
        mass=mass,
        offsets=np.arange(6, dtype=np.int64),
        buffer=np.frombuffer(b"CNOSP", np.uint8),
    )
    assert store.window(target, ppm=10.0) == (1, ("N", "O", "S"))
    pool = replace(
        tiny_pubchem_pool(expected),
        mass=mass[1:4],
        packed=np.array([[128], [128], [64]], np.uint8),
    )
    assert (
        best_pool_fz(
            pool,
            target=target,
            logits=np.array([1, 5, 0, 0, 0, 0, 0], np.float32),
            ppm=10.0,
            fallback_ppm=30.0,
        )
        == 5.0
    )


def write_pubchem_table(
    directory: Path,
    *,
    expected: dict[str, Tensor],
    prefix: str,
    raw_valid: bool = True,
) -> list[str]:
    """Write the stored golden inputs in the native file schema."""
    count = int(expected[prefix + "/smiles/length"])
    smiles = [decode(expected[f"{prefix}/smiles/{index}"]) for index in range(count)]
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": smiles,
                "parsed": expected[prefix + "/parsed"].tolist(),
                "raw_valid": expected[prefix + "/raw_valid"].tolist()
                if raw_valid
                else [False] * count,
                "key": [
                    None
                    if f"{prefix}/keys/{index}/none" in expected
                    else decode(expected[f"{prefix}/keys/{index}"])
                    for index in range(count)
                ],
            },
        ),
        directory / "molecules.parquet",
    )
    for name in ("ecfp4", "raw_fp"):
        np.save(directory / (name + ".npy"), expected[prefix + "/" + name].numpy())
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
    return smiles


def tiny_pubchem_bank(directory: Path) -> MainBank:
    """Load both stored checkpoint inputs, with every state checked."""
    expected = read_tensors(_BASELINE / "testdata/neural_bank.pt")
    config = save_bank(directory, [0, 1])
    config.merged_adduct, config.single_count = first_modal_adduct, keep_count
    bank = config.make()
    for index, model in enumerate(bank.models):
        for name, value in model.released_state().items():
            if not torch.equal(value, expected[f"model/{index}/{name}"]):
                raise ValueError(f"Source checkpoint field changed: {index}/{name}")
    return bank


def tiny_pubchem_pool(expected: dict[str, Tensor]) -> CandidatePool:
    """Supply the complete positional pool inputs used by the source runner."""
    keys = (decode(expected["pool_keys/0"]), "other", "last")
    return CandidatePool(
        mass=expected["pool/mass"].numpy(),
        keys=keys,
        smiles=("CCO", "C", "N"),
        formulas=("C2H6O", "CH4", "H3N"),
        heavy_atoms=np.array([3, 1, 1], np.int32),
        source=np.zeros(3, np.int8),
        train_sid=np.full(3, -1, np.int64),
        packed=expected["pool/packed"].numpy(),
        bits=expected["store/bits"].numpy(),
        fragment_offsets=np.zeros(4, np.int64),
        fragment_mass=np.zeros(0, np.float32),
        key_to_pid={key: index for index, key in enumerate(keys)},
    )


def fail_prediction(*args: object, **kwargs: object) -> object:
    """Reach the original runner's explicit failed-logits skip boundary."""
    del args, kwargs
    raise RuntimeError("independent model failure")


def missing_prediction(*args: object, **kwargs: object) -> None:
    """Reach the original runner's explicit absent-logits skip boundary."""
    del args, kwargs


def decode(value: Tensor) -> str:
    """Decode a stored string without changing its contents."""
    return bytes(from_plain(value.tolist(), list[int])).decode()


def observe(actual: dict[str, Tensor], prefix: str, value: object) -> None:
    """Compare every field through a dtype-preserving replay tree."""
    if isinstance(value, np.ndarray):
        actual[prefix] = torch.from_numpy(value.copy())
    elif isinstance(value, Tensor):
        actual[prefix] = value.detach().clone()
    elif isinstance(value, dict):
        for name, child in cast(dict[str, object], value).items():
            observe(actual, prefix + "/" + name, child)
    elif isinstance(value, (list, tuple)):
        values = cast(Sequence[object], value)
        actual[prefix + "/length"] = torch.tensor(len(values))
        for index, child in enumerate(values):
            observe(actual, f"{prefix}/{index}", child)
    elif isinstance(value, str):
        actual[prefix] = torch.tensor(list(value.encode()), dtype=torch.uint8)
    elif value is None:
        actual[prefix + "/none"] = torch.tensor(True)
    elif isinstance(value, (bool, int)):
        actual[prefix] = torch.tensor(value)
    elif isinstance(value, float):
        actual[prefix] = torch.tensor(value, dtype=torch.float64)
    else:
        raise TypeError(type(value))


def test_pubchem_constructor_keeps_every_store_row_model_state_and_strict_chemistry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = PubChemAssets.Config()
    pool_dir, store_dir, chemistry_dir = (
        tmp_path / name for name in ("pool", "store", "chemistry")
    )
    config.pool_dir, config.store_dir = pool_dir, store_dir
    config.popularity_dir, config.chemistry_dir = store_dir, chemistry_dir
    pool_dir.mkdir()
    _, original, _ = retrieval_inputs()
    write_pool(pool_dir, pool=original)
    store_dir.mkdir()
    for name, values in (
        ("pc_mass", np.array([100, 100], np.float64)),
        ("pc_off", np.array([0, 3, 4], np.int64)),
        ("pc_smiles", np.frombuffer(b"CCOC", np.uint8)),
        ("pc_lsid", np.array([0.5, 1.5], np.float16)),
        ("pc_lpmid", np.array([2.5, 3.5], np.float16)),
    ):
        np.save(store_dir / f"{name}.npy", values)
    chemistry_dir.mkdir()
    parquet.write_table(
        Table.from_pydict(
            {
                "smiles": ["CCO", "C"],
                "parsed": [True, True],
                "raw_valid": [True, False],
                "key": ["ethanol", None],
            },
        ),
        chemistry_dir / "molecules.parquet",
    )
    np.save(
        chemistry_dir / "ecfp4.npy",
        np.arange(1024, dtype=np.uint8).reshape(2, 512),
    )
    np.save(
        chemistry_dir / "raw_fp.npy",
        np.arange(4138, dtype=np.uint8).reshape(2, 2069),
    )
    (chemistry_dir / "manifest.json").write_text(
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
    source = read_tensors(_BASELINE / "testdata/neural_bank.pt")
    config.bank = save_bank(tmp_path, [0, 1])
    with torch.random.fork_rng(devices=[]):
        inputs = config.make().read()
    assert inputs.pool.smiles == original.smiles
    assert inputs.pool.keys == original.keys
    np.testing.assert_array_equal(inputs.pool.packed, original.packed)
    assert inputs.store.window(100, ppm=10.0) == (0, ("CCO", "C"))
    assert inputs.store.ls is not None
    assert inputs.store.lp is not None
    assert inputs.store.ls.dtype == inputs.store.lp.dtype == np.float16
    np.testing.assert_array_equal(inputs.store.ls, [0.5, 1.5])
    np.testing.assert_array_equal(inputs.store.lp, [2.5, 3.5])
    assert inputs.chemistry.for_smiles("CCO").key == "ethanol"
    assert not inputs.chemistry.for_smiles("C").raw_valid
    np.testing.assert_array_equal(
        inputs.chemistry.ecfp4,
        np.arange(1024, dtype=np.uint8).reshape(2, 512),
    )
    np.testing.assert_array_equal(
        inputs.chemistry.raw_fp,
        np.arange(4138, dtype=np.uint8).reshape(2, 2069),
    )
    for index, model in enumerate(inputs.bank.models):
        state = model.released_state()
        keys = {
            name.removeprefix(f"model/{index}/")
            for name in source
            if name.startswith(f"model/{index}/")
        }
        assert state.keys() == keys
        for name, value in state.items():
            assert torch.equal(value, source[f"model/{index}/{name}"])
    assert len(inputs.bank.models) == 2
    with pytest.raises(KeyError, match=r"^'unstaged'$"):
        inputs.chemistry.for_smiles("unstaged")
    config.bank.checkpoints[0].sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="hashes to"):
        config.make().read()


def write_pool(directory: Path, *, pool: CandidatePool) -> None:
    """Publish complete test primitive inputs in the provided pool's native schema."""
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
        directory / "pool_meta.parquet",
    )
    for name, values in (
        ("pool_fp", pool.packed),
        ("fp_bits", pool.bits),
        ("pool_frag_off", pool.fragment_offsets),
        ("pool_frag_mass", pool.fragment_mass),
    ):
        np.save(directory / f"{name}.npy", values)


def test_complete_pubchem_gate_all_thresholds_rows_and_counters() -> None:
    cases = (
        ("absent", 0.0, 0.0, None, 0.0, 0.0),
        ("empty", 0.0, 0.0, None, 0.0, 0.0),
        ("library_equal", 0.9, 1000.0, 7.0, 5.0, 0.0),
        ("library_below", math.nextafter(0.9, 0.0), 1000.0, 7.0, 5.0, 0.0),
        ("margin250_equal", 0.0, 250.0, None, 0.0, 0.0),
        ("margin250_above", 0.0, math.nextafter(250.0, math.inf), None, 0.0, 0.0),
        ("margin600_equal", 0.0, 600.0, None, 0.0, 0.0),
        ("margin600_above", 0.0, math.nextafter(600.0, math.inf), None, 0.0, 0.0),
        ("S_equal", 0.0, 0.0, 6.0, 5.0, 0.0),
        ("S_above", 0.0, 0.0, math.nextafter(6.0, math.inf), 5.0, 0.0),
        ("pop_below", 0.0, 0.0, 7.0, math.nextafter(5.0, 0.0), 0.0),
        ("pop_equal", 0.0, 0.0, 7.0, 5.0, 0.0),
        ("nan_S", 0.0, 0.0, math.nan, 99.0, 0.0),
        ("pool_nan", 0.0, -1000.0, None, 0.0, math.nan),
        ("pool_inf", 0.0, -1000.0, None, 0.0, math.inf),
        ("top_zero", 0.0, 0.0, None, 0.0, 0.0),
        ("fallback_fz", 0.0, None, None, 0.0, 0.0),
        ("base_empty", 0.0, 0.0, None, 0.0, 0.0),
        ("library_empty", 0.9, 1000.0, 7.0, 5.0, 0.0),
        ("only_duplicate", 0.0, 0.0, None, 0.0, 0.0),
    )
    gate = PubChemGate.Config().make()
    rows: list[tuple[str, str]] = []
    stats = {"untouched": 0, "gentle": 0, "aggressive": 0, "no_pc": 0, "promoted": 0}
    for name, library, fit, score, popularity, pool in cases:
        base = [f"base{index}" for index in range(27)]
        base_keys = [f"key{index}" for index in range(27)]
        if name in {"base_empty", "library_empty"}:
            base, base_keys = base[:0], base_keys[:0]
        proposals = ["duplicate", *[f"proposal{index}" for index in range(1, 7)]]
        keys = ["key0", *[f"new{index}" for index in range(1, 7)]]
        if name in {"absent", "empty"}:
            proposals, keys = proposals[:0], keys[:0]
        if name == "only_duplicate":
            proposals, keys = proposals[:1], keys[:1]
        result = gate.screen(
            base,
            base_keys,
            library_max=library,
            proposals=proposals,
            proposal_keys=keys,
            proposal_scores=[1000.0] * len(proposals),
            top_fit=fit,
            pool_fit=pool,
            outlier=score,
            popularity=popularity,
        )
        rows.append((name, ";".join(result.candidates)))
        stats[result.mode] += 1
        stats["promoted"] += int(result.promoted)
    golden = _BASELINE / "testdata/pubchem_gate.pt"
    actual = {
        "rows": torch.tensor(list(str(rows).encode()), dtype=torch.uint8),
        "stats": torch.tensor(list(str(stats).encode()), dtype=torch.uint8),
    }
    assert_tensor_golden(golden, actual)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
