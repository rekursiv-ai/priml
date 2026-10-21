"""Checks for the data module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast
from unittest.mock import patch

import argparse
import hashlib
import json
import math
import pickle
import sys
import tempfile
import types

from pyarrow import Table, parquet
from torch import Tensor
from treekle import from_plain, json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026 import data as data_module
from priml.baselines.casmi2026.chemistry import ChemicalTable, IdentityTable
from priml.baselines.casmi2026.custom_types import GapGraph
from priml.baselines.casmi2026.data import (
    SpectrumData,
    TrainingPreparation,
    build_spectral_cache,
    molecule_bucket,
    prepare_data,
)
from priml.baselines.casmi2026.fragnet import FragnetTable
from priml.baselines.casmi2026.fusion import GapGraphTable
from priml.baselines.casmi2026.isolated import (
    ADMISSION_DIR,
    DATASET_ROOT,
    ELEMENT_MASSES,
    FRAGMENT_GAP_DIR,
    GLACIER_ROOT_DIR,
    ICE_ROOT_DIR,
    IDENTITY_DIR,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    AnalogTable,
    IntrinsicTable,
    molecular_record,
    record_fields,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.main_engine import (
    DerivationTable,
    MainFragmentTable,
)
from priml.baselines.casmi2026.pubchem import PubChemTable
from priml.baselines.casmi2026.pubchem_test import decode, observe
from priml.baselines.casmi2026.scripts import (
    admission,
    analog_data,
    atomic_data,
    derivation_data,
    engine_identity,
    fragment_gap_data,
    fragnet_data,
    glacier_data,
    ice_data,
    intrinsic,
    main_fragments,
    prepare_data as scripts_prepare_data,
    pubchem_data,
    second_fragments,
    second_pool,
)
from priml.baselines.casmi2026.scripts.acquire_assets import (
    Asset,
    AssetAcquisition,
    acquire_assets,
    read_assets,
)
from priml.baselines.casmi2026.scripts.atomic_data import prepare_graphs
from priml.baselines.casmi2026.scripts.ice_data import prepare_ice_graphs
from priml.baselines.casmi2026.scripts.intrinsic import capability_rules
from priml.baselines.casmi2026.scripts.main_fragments import (
    base_fragment_masses,
    fragment_masses,
    fragment_rows,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    sha256_file,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.pubchem_data import PubChemChemistry
from priml.baselines.casmi2026.scripts.ragged import save_stream
from priml.baselines.casmi2026.scripts.rdkit_backend import StructureRecord
from priml.baselines.casmi2026.scripts.second_fragments import (
    fragment_masses as second_fragments_fragment_masses,
)
from priml.baselines.casmi2026.scripts.second_library import (
    formula_mass,
    prepare_second_library,
)
from priml.baselines.casmi2026.scripts.second_pool import prepare_second_pool
from priml.baselines.casmi2026.second_engine import EngineIdentityTable
from priml.baselines.casmi2026.second_engine_test import engine_keys
from priml.baselines.casmi2026.simulators import (
    SimulatorAdmissionTable,
)
from priml.baselines.casmi2026.testing import (
    float32_comparands,
    identity_rows,
    write_identity,
)
from priml.testing.golden import assert_tensor_golden, read_tensors


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray
    from pyarrow import RecordBatch
    from rdkit import Chem
    from torch import Tensor

    from priml.baselines.casmi2026.custom_types import (
        AtomicRow,
        EngineIdentityRow,
        GlacierRow,
        IceRow,
        SimulatorAdmissionRow,
    )
    from priml.baselines.casmi2026.scripts.derivation_chemistry import (
        DerivationChemistry,
    )


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray


_BASELINE: Final = _CWD


def cache(root: Path) -> SpectrumData.Config:
    """Write a deterministic, tiny public-cache schema with repeated molecules."""
    np.save(root / "spec_off.npy", np.arange(9, dtype=np.int64) * 5)
    np.save(root / "spec_mz.npy", np.arange(40, dtype=np.float32) + 20)
    np.save(root / "spec_it.npy", np.arange(40, dtype=np.float32))
    np.save(
        root / "train_fp_sel.npy",
        # Source pool_fp/train_fp_sel stores eight fingerprint bits per packed byte.
        np.array([[42], [55], [27], [39]], dtype=np.uint8),
    )
    np.save(root / "fp_bits.npy", np.arange(7, dtype=np.int64))
    parquet.write_table(
        Table.from_pydict(
            {"sid": [0, 1, 2, 3], "metric_identity": ["a", "b", "c", "d"]},
        ),
        where=root / "train_metric_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": np.arange(8) // 2,
                "precursor_mz": [100.0] * 8,
                "ce_mean": [10.0, np.nan] * 4,
                "ce_n": [1, 2] * 4,
                "adduct_ix": [0, 2] * 4,
                "mode": [1, -1] * 4,
                "nm": [99.0] * 8,
            },
        ),
        where=root / "spec_meta.parquet",
    )
    config = SpectrumData.Config()
    config.working_dir = root
    config.batch_size = 3
    config.validation_modulus = 2
    return config


def test_cache_split_and_exact_resume(tmp_path: Path) -> None:
    config = cache(tmp_path)
    data = config.make()
    train_sid = set(data.sids[data.train_indices.numpy()])
    eval_sid = set(data.sids[data.eval_indices.numpy()])
    assert train_sid
    assert eval_sid
    assert train_sid.isdisjoint(eval_sid)
    loader = iter(data.train_dataloader())
    first = next(loader)
    assert first["mz"].shape == (3, 5)
    assert first["label"].shape == (3, 7)
    snapshot = data.state_dict()
    restored = config.make()
    restored.load_state_dict(snapshot)
    expected, actual = list(loader), list(restored.train_dataloader())
    assert len(actual) == len(expected)
    assert all(
        torch.equal(a["mz"], b["mz"]) for a, b in zip(actual, expected, strict=True)
    )
    assert data.cursor == 0
    assert data.order is None
    held_out = list(data.eval_dataloader())
    expected_indices = data.eval_indices
    assert torch.equal(
        torch.cat([batch["sid"] for batch in held_out]),
        torch.from_numpy(data.sids[expected_indices.numpy()]),
    )
    assert torch.equal(
        torch.cat([batch["mz"] for batch in held_out]),
        data.collate(expected_indices)["mz"],
    )
    assert len(list(data.train_dataloader())) > 0


def test_empty_peak_mask_and_missing_energy(tmp_path: Path) -> None:
    config = cache(tmp_path)
    data = config.make()
    data.intensities[:] = 0
    output = data.collate(torch.tensor([0, 1, 2]))
    # SpectrumData.collate mirrors source collate's forced one-slot empty spectrum.
    assert output["pad"].shape == (3, 1)
    assert not bool(output["pad"].any())
    assert output["ce"].tolist() == [10, 0, 10]
    assert output["ce_known"].tolist() == [1, 0, 1]


def test_invalid_batch_geometry_and_identity_buckets(tmp_path: Path) -> None:
    config = SpectrumData.Config()
    config.working_dir = tmp_path
    config.batch_size = 0
    with pytest.raises(ValueError, match="batch_size"):
        _ = config.make()
    assert molecule_bucket("same-molecule", modulus=10) == molecule_bucket(
        "same-molecule",
        modulus=10,
    )


@pytest.mark.parametrize("bucket", [-1, 10])
def test_invalid_validation_bucket_fails_before_loading(
    tmp_path: Path,
    bucket: int,
) -> None:
    config = SpectrumData.Config()
    config.working_dir = tmp_path
    config.validation_bucket = bucket
    with pytest.raises(ValueError, match="validation_bucket"):
        _ = config.make()


@pytest.mark.parametrize("bucket", [0, 1])
def test_empty_required_population_is_rejected(tmp_path: Path, bucket: int) -> None:
    config = cache(tmp_path)
    parquet.write_table(
        Table.from_pydict({"sid": [0, 1, 2, 3], "metric_identity": ["a"] * 4}),
        where=tmp_path / "train_metric_structs.parquet",
    )
    config.validation_bucket = bucket
    with pytest.raises(ValueError, match="nonempty train and evaluation"):
        _ = config.make()


def test_complete_cleaned_cache_record(
    tmp_path: Path,
) -> None:
    parquet.write_table(
        Table.from_pydict({"inchikey14": ["first", "last", "first"], "sid": [0, 1, 2]}),
        where=tmp_path / "structures.parquet",
    )
    rows = 8
    masses: list[list[float]] = [
        [-1.0, 0.0, 10.0, 20.0, 32.0, 32.001],
        [],
        [10.0, 20.0],
        [10.0, 20.0, 30.0, 40.0],
        [float(index) / 8 + 1 for index in range(530)],
        [20.0, 10.0, 10.0, 30.0],
        [0.0, -1.0, 200.0],
        [15.0, 25.0, 35.0],
    ]
    intensity: list[list[float]] = [
        [100.0, 200.0, 1.0, 0.2, 0.1999, 1000.0],
        [],
        [0.0, -1.0],
        [4.0, 8.0, 0.008, 0.00799],
        [float(index % 7 + 1) for index in range(530)],
        [3.0, 1.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.002, 0.0009],
    ]
    raw = Table.from_pydict(
        {
            "inchikey14": ["first", "last", "missing", "first"] * 2,
            "ingest_lib": [
                "enveda-180",
                "pluskal_ms2",
                "riken",
                "gnps",
                "massbank",
                "mona",
                "spectraverse",
                "masaryk",
            ],
            "instrument_type": [
                "timsTOF",
                "QFT",
                "QTOF",
                "QIT",
                "QQQ",
                "unknown",
                None,
                "hybrid FT",
            ],
            "adduct": [
                "[M+H]+",
                "[M-H]-",
                "unknown",
                "[2M+Ca]2+",
                "[M+Na]+",
                "[M+Cl]-",
                "[M]+",
                "[M-H2O+H]+",
            ],
            "ionization_mode": ["positive", "negative", None, "positive"] * 2,
            "precursor_mz": [30.0, 50.0, 60.0, 70.0, 100.0, 50.0, 40.0, 90.0],
            "precursor_error_ppm": list(np.arange(rows, dtype=np.float64) / 3),
            "collision_energy_ev": [
                [10.0, 20.0],
                [],
                None,
                [25.0],
                [1.0, np.nan],
                [3.0, 4.0, 8.0],
                [0.0],
                [],
            ],
            "collision_energy_orig_units": ["eV"] * rows,
            "base_peak_intensity": [100.0] * rows,
            "num_peaks": [len(peaks) for peaks in masses],
            "ms2_mzs": masses,
            "ms2_normalized_intensities": intensity,
        },
    )
    parquet.write_table(raw, where=tmp_path / "train.parquet", row_group_size=3)
    output = tmp_path / "prepared"
    build_spectral_cache(
        tmp_path / "train.parquet",
        structures=tmp_path / "structures.parquet",
        output=output,
        intensity_floor=0.001,
        max_peaks=512,
        precursor_margin=2.0,
    )
    offsets, mz, it = (
        cast("NDArray[np.generic]", np.load(output / f"spec_{field}.npy"))
        for field in ("off", "mz", "it")
    )
    assert (offsets.dtype, mz.dtype, it.dtype) == (np.int64, np.float32, np.float32)
    np.testing.assert_array_equal(offsets, [0, 2, 2, 2, 5, 517, 521, 521, 523])
    # The 512-peak cap drops the 18 earliest of the 76 weakest peaks.
    kept = [index for index in range(530) if index % 7 or index >= 7 * 18]
    np.testing.assert_array_equal(
        mz,
        np.array(
            [10, 20, 10, 20, 30, *(masses[4][i] for i in kept), 10, 10, 20, 30, 15, 25],
            np.float32,
        ),
    )
    np.testing.assert_array_equal(
        it,
        np.array(
            [
                *(1 / 200, 0.2 / 200, 4 / 8, 8 / 8, 0.008 / 8),
                *((index % 7 + 1) / 7 for index in kept),
                *(1 / 3, 2 / 3, 3 / 3, 1 / 3, 1.0, 0.002),
            ],
            np.float32,
        ),
    )
    metadata = parquet.read_table(output / "spec_meta.parquet")
    assert {
        name: metadata.column(name).to_pylist() for name in metadata.column_names
    } == {
        "adduct": raw.column("adduct").to_pylist(),
        "precursor_mz": [30.0, 50.0, 60.0, 70.0, 100.0, 50.0, 40.0, 90.0],
        "precursor_error_ppm": [index / 3 for index in range(rows)],
        "collision_energy_orig_units": ["eV"] * rows,
        "base_peak_intensity": [100.0] * rows,
        "num_peaks": [6, 0, 2, 4, 530, 4, 3, 3],
        "ce_mean": [15.0, None, None, 25.0, None, 5.0, 0.0, None],
        "ce_min": [10.0, None, None, 25.0, None, 3.0, 0.0, None],
        "ce_max": [20.0, None, None, 25.0, None, 8.0, 0.0, None],
        "ce_n": [2, 0, 0, 1, 2, 3, 1, 0],
        "sid": [2, 1, -1, 2, 2, 1, -1, 2],
        "lib": [0, 1, 2, 3, 4, 5, 6, 10],
        "instr": [0, 1, 2, 3, 4, 5, 5, 1],
        "adduct_ix": [0, 6, 59, 50, 4, 9, 12, 2],
        "mode": [1, -1, -1, 1, 1, -1, -1, 1],
        "nm": [
            30.0 - 1.00727645217,
            50.0 + 1.00727645217,
            None,
            (140.0 - 39.961493820200005) / 2,
            100.0 - 22.989220701,
            50.0 - 34.969401259899996,
            40.0 + 0.0005485799,
            90.0 + 17.00328823157,
        ],
        "n_clean": [2, 0, 0, 3, 512, 4, 0, 2],
    }


def test_native_cold_cache_and_every_verified_output_without_rdkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    config = _config(tmp_path)
    preparation = config.make()
    assert not preparation.cache_ready()
    with patch.object(data_module, "to_plain", wraps=to_plain) as plain:
        preparation.run()
    assert preparation.cache_ready()
    assert any(call.kwargs.get("dialect") == "python" for call in plain.call_args_list)
    receipt_text = (config.working_dir / "cache_receipt.json").read_text()
    assert (
        receipt_text
        == json.dumps(
            json_loads(receipt_text, dict[str, object]),
            indent=2,
        )
        + "\n"
    )
    work = config.working_dir / "work"
    for filename in (
        "pool_meta.parquet",
        "pool_fp.npy",
        "pool_frag_off.npy",
        "pool_frag_mass.npy",
        "fp_bits.npy",
        "train_structs.parquet",
        "train_fp_sel.npy",
    ):
        assert (work / filename).is_symlink()
        assert (work / filename).read_bytes() == (config.pool / filename).read_bytes()
    for filename, expected in (
        ("spec_off.npy", [0.0, 2.0, 4.0, 6.0]),
        ("spec_mz.npy", [10.0, 20.0] * 3),
        ("spec_it.npy", [1.0, 0.5] * 3),
    ):
        actual = cast("NDArray[np.generic]", np.load(work / filename))
        np.testing.assert_array_equal(actual, expected)
    identities = config.identities.make()
    before = parquet.read_table(config.pool / "train_structs.parquet")
    after = parquet.read_table(work / "train_metric_structs.parquet")
    for name in before.column_names:
        assert before.column(name).to_pylist() == after.column(name).to_pylist()
    assert after.column("metric_identity").to_pylist() == [
        json.dumps(
            to_plain(identities.identity(smiles), dialect="python"),
            separators=(",", ":"),
        )
        for smiles in from_plain(before.column("smiles").to_pylist(), list[str])
    ]
    receipt = json_loads(
        (config.working_dir / "cache_receipt.json").read_text(),
        dict[str, object],
    )
    outputs = from_plain(receipt["outputs"], dict[str, str])
    assert len(outputs) == 12
    for filename, seal in outputs.items():
        path = config.working_dir / filename
        original = path.read_bytes()
        assert seal == hashlib.sha256(original).hexdigest()
        if path.is_symlink():
            path.unlink()
        path.write_bytes(original + b"corruption")
        assert not preparation.cache_ready()
        path.write_bytes(original)
        assert preparation.cache_ready()
    with pytest.raises(
        ValueError,
        match=r"^TrainingPreparation accepts configuration fields only\.$",
    ):
        preparation.run("--truncate")


def test_input_changes_incomplete_receipts_and_owned_refresh_keep_previous_bytes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    preparation = config.make()
    preparation.run()
    marker = config.working_dir / "work/spec_it.npy"
    marker.write_bytes(b"partial")
    assert not preparation.cache_ready()
    preparation.run()
    assert preparation.cache_ready()
    previous = list(tmp_path.glob("prepared.incomplete-*/previous/work/spec_it.npy"))
    assert len(previous) == 1
    assert previous[0].read_bytes() == b"partial"
    source = config.pool / "train_fp_sel.npy"
    original = source.read_bytes()
    source.write_bytes(original + b"changed input")
    assert not preparation.cache_ready()
    source.write_bytes(original)
    assert preparation.cache_ready()
    receipt = config.working_dir / "cache_receipt.json"
    original_receipt = receipt.read_text()
    for malformed in (
        "[]",
        '{"owner":4}',
        original_receipt.replace('"format_version": 1', '"format_version": 2'),
    ):
        receipt.write_text(malformed)
        assert not preparation.cache_ready()
    receipt.write_text(original_receipt)
    (config.working_dir / "work/spec_mz.npy").unlink()
    assert not preparation.cache_ready()


@pytest.mark.parametrize(
    ("owner", "message"),
    [
        (None, r"^Unowned preparation directory: "),
        (
            '{"training":"different","pool":"different","identities":"different"}',
            r"^Unowned preparation directory: ",
        ),
        (
            (
                '{"training":"a","pool":"b","identities":"c","intensity_floor":0.001,'
                '"max_peaks":512,"precursor_margin":2.0}'
            ),
            r"^Preparation belongs to different inputs: ",
        ),
    ],
)
def test_unowned_destination_is_never_rotated(
    tmp_path: Path,
    owner: str | None,
    message: str,
) -> None:
    config = _config(tmp_path)
    config.working_dir.mkdir()
    marker = config.working_dir / "keep"
    marker.write_text("foreign data")
    if owner is not None:
        (config.working_dir / ".preparation.json").write_text(owner)
    with pytest.raises(FileExistsError, match=message):
        config.make().run()
    assert marker.read_text() == "foreign data"
    assert not list(tmp_path.glob("prepared.incomplete-*"))


def test_owner_record_and_each_receipt_section_gate_the_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    preparation = config.make()
    preparation.run()
    owner = json_loads(
        (config.working_dir / ".preparation.json").read_text(),
        dict[str, object],
    )
    assert owner == {
        "training": str(config.training.resolve()),
        "pool": str(config.pool.resolve()),
        "identities": str(Path(config.identities.working_dir).resolve()),
        "intensity_floor": config.intensity_floor,
        "max_peaks": config.max_peaks,
        "precursor_margin": config.precursor_margin,
    }
    receipt_path = config.working_dir / "cache_receipt.json"
    original = receipt_path.read_text()
    for section, name in (
        ("owner", "pool"),
        ("sources", "data.py"),
        ("inputs", str(config.training.resolve())),
    ):
        receipt_path.write_text(_tampered(original, section, name, "tampered"))
        assert not preparation.cache_ready()
    receipt_path.write_text(_tampered(original, "outputs", "work/surplus", "0" * 64))
    assert not preparation.cache_ready()
    receipt_path.write_text(original)
    assert preparation.cache_ready()


def test_preparation_rejects_contained_inputs_and_invalid_official_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.working_dir = tmp_path
    with pytest.raises(
        ValueError,
        match=r"^Preparation output must not contain protected input artifacts\.$",
    ):
        config.make().run()
    config.working_dir = config.training
    with pytest.raises(ValueError, match=r"^output path aliases protected input"):
        config.make().run()
    config.working_dir = tmp_path / "prepared"
    rows = identity_rows()
    rows[0]["metric"] = "null"
    parquet.write_table(
        Table.from_pylist([dict(row) for row in rows]),
        Path(config.identities.working_dir) / "molecules.parquet",
    )
    with pytest.raises(ValueError, match=r"^Invalid prepared training identity: .+$"):
        config.make().run()
    assert not (config.working_dir / "cache_receipt.json").exists()


def _tampered(receipt: str, section: str, entry: str, value: str) -> str:
    sections = json_loads(receipt, dict[str, object])
    entries = from_plain(sections[section], dict[str, object])
    entries[entry] = value
    sections[section] = entries
    return json.dumps(sections)


def _config(root: Path) -> TrainingPreparation.Config:
    config = TrainingPreparation.Config()
    config.base_dir = root
    config.training = Path("train.parquet")
    config.pool = Path("pool")
    config.identities.working_dir = "identities"
    config.working_dir = Path("prepared")
    config = config.finalized()
    config.pool.mkdir()
    identities = Path(config.identities.working_dir)
    identities.mkdir()
    write_identity(identities)
    smiles = [row["smiles"] for row in identity_rows()[:2]]
    parquet.write_table(
        Table.from_pydict(
            {
                "sid": [0, 1],
                "inchikey14": ["k0", "k1"],
                "smiles": smiles,
                "mass": [100.0, 101.0],
            },
        ),
        config.pool / "train_structs.parquet",
    )
    parquet.write_table(
        Table.from_pydict({"smiles": smiles}),
        config.pool / "pool_meta.parquet",
    )
    # TrainingPreparation.run consumes three selected bits in one packed byte.
    for name, value in (
        ("pool_fp", np.zeros((2, 1), np.uint8)),
        ("train_fp_sel", np.zeros((2, 1), np.uint8)),
        ("pool_frag_off", np.array([0, 1, 2], np.int64)),
        ("pool_frag_mass", np.array([100.0, 101.0], np.float32)),
        ("fp_bits", np.arange(3, dtype=np.int64)),
    ):
        np.save(config.pool / f"{name}.npy", value)
    parquet.write_table(
        Table.from_pydict(
            {
                "inchikey14": ["k0", "k1", "k0"],
                "ingest_lib": ["gnps"] * 3,
                "instrument_type": ["timsTOF"] * 3,
                "adduct": ["[M+H]+"] * 3,
                "ionization_mode": ["positive"] * 3,
                "precursor_mz": [101.0, 102.0, 101.0],
                "precursor_error_ppm": [0.0] * 3,
                "collision_energy_ev": [[25.0]] * 3,
                "collision_energy_orig_units": ["eV"] * 3,
                "base_peak_intensity": [1.0] * 3,
                "num_peaks": [2] * 3,
                "ms2_mzs": [[20.0, 10.0]] * 3,
                "ms2_normalized_intensities": [[0.5, 1.0]] * 3,
            },
        ),
        config.training,
    )
    return config


def test_staging_factory_declares_resolved_inputs() -> None:
    config = prepare_data().finalized()
    root = Path("/opt/scratch/datasets/casmi2026")
    assert config.working_dir == root / "training"
    assert (
        config.training
        == root / "inputs/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
    )
    assert config.pool == root / "inputs/datasets/ahmedberatozer/casmi26-v2-pool"
    assert config.identities.working_dir == root / "native-chemistry/identities"


_SCRIPTS_DIR: Final = _CWD / "scripts"


# acquire_assets_test.py.
def _pins(tmp_path: Path, *contents: bytes) -> Path:
    path = tmp_path / "assets.tsv"
    path.write_text(
        "".join(
            f"owner{index}/slug{index}\t{index + 1}\t{hashlib.sha256(data).hexdigest()}\n"
            for index, data in enumerate(contents)
        ),
    )
    return path


def _acquire_assets_config(tmp_path: Path, pins: Path) -> AssetAcquisition.Config:
    config = AssetAcquisition.Config()
    config.base_dir = tmp_path / "root"
    config.assets = pins
    return config


def test_the_shipped_pin_table_names_seventeen_archives_with_full_digests() -> None:
    assets = read_assets(AssetAcquisition.Config().assets)
    assert len(assets) == 17
    assert all(len(asset.sha256) == 64 and asset.version >= 1 for asset in assets)
    assert assets[0] == Asset(
        owner="ahmedberatozer",
        slug="casmi26-fpnet-full1",
        version=1,
        sha256="c310e8d445b23c706a2b8db3bd92f71f429258a9d0ca4d080bc14363e9ee5c0f",
    )
    assert assets[0].archive == "casmi26-fpnet-full1.zip"


def test_placed_archives_are_verified_and_stamped(tmp_path: Path) -> None:
    pins = _pins(tmp_path, b"first", b"second")
    config = _acquire_assets_config(tmp_path, pins)
    resolved = config.copy_tree().finalize()
    assert resolved.working_dir == tmp_path / "root/datasets/casmi2026/archives"
    for index, data in enumerate((b"first", b"second")):
        folder = Path(resolved.working_dir) / f"owner{index}" / f"slug{index}"
        folder.mkdir(parents=True)
        (folder / f"slug{index}.zip").write_bytes(data)
    config.make().run()
    for index in range(2):
        folder = Path(resolved.working_dir) / f"owner{index}" / f"slug{index}"
        assert (folder / ".ensure_complete").is_file()


def test_an_absent_archive_is_named_for_download(tmp_path: Path) -> None:
    config = _acquire_assets_config(tmp_path, _pins(tmp_path, b"first"))
    with pytest.raises(
        FileNotFoundError,
        match=r"Download version 1 of owner0/slug0 to .*slug0\.zip",
    ):
        config.make().run()


def test_a_corrupt_archive_is_moved_aside_and_requested_again(tmp_path: Path) -> None:
    config = _acquire_assets_config(tmp_path, _pins(tmp_path, b"first"))
    folder = tmp_path / "root/datasets/casmi2026/archives/owner0/slug0"
    folder.mkdir(parents=True)
    (folder / "slug0.zip").write_bytes(b"damaged")
    with pytest.raises(FileNotFoundError, match="owner0/slug0"):
        config.make().run()
    moved = list(folder.parent.glob("slug0.corrupt.*"))
    assert len(moved) == 1
    assert (moved[0] / "slug0.zip").read_bytes() == b"damaged"


def test_arguments_are_refused_and_the_launcher_declares_the_job(
    tmp_path: Path,
) -> None:
    config = _acquire_assets_config(tmp_path, _pins(tmp_path, b"first"))
    with pytest.raises(
        ValueError,
        match=r"^AssetAcquisition accepts configuration fields only\.$",
    ):
        config.make().run("--force")
    assert isinstance(acquire_assets(), AssetAcquisition.Config)
    assert acquire_assets().assets.name == "assets.tsv"


# admission_test.py.


def test_admission_writer_keeps_all_duplicate_invalid_and_empty_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admission, "AdmissionChemistry", _Decisions)
    request, output = tmp_path / "request.parquet", tmp_path / "output"
    values = ["valid", "invalid", "valid", "other"]
    parquet.write_table(Table.from_pydict({"smiles": values}), request)
    admission.prepare_admission(
        request,
        output=output,
        version=SIMULATOR_RDKIT,
        batch_rows=3,
    )
    table = SimulatorAdmissionTable(output)
    assert [row["smiles"] for row in table.rows] == values
    assert [row["ice_accepted"] for row in table.rows] == [True, False, True, True]
    assert table.for_smiles("valid")["glacier_heavy_atoms"] == 2
    assert parquet.ParquetFile(output / "admission.parquet").num_row_groups == 2
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["rows"] == 4
    assert manifest["format_version"] == 1
    with pytest.raises(FileExistsError):
        admission.prepare_admission(request, output=output, version=SIMULATOR_RDKIT)
    for version, batch in ((IDENTITY_RDKIT, 3), (SIMULATOR_RDKIT, 0)):
        with pytest.raises(ValueError, match=r"requires|positive"):
            admission.prepare_admission(
                request,
                output=tmp_path / "invalid",
                version=version,
                batch_rows=batch,
            )
    parquet.write_table(Table.from_pydict({"smiles": []}), request)
    admission.prepare_admission(
        request,
        output=tmp_path / "empty",
        version=SIMULATOR_RDKIT,
    )
    assert SimulatorAdmissionTable(tmp_path / "empty").rows == ()


def test_actual_admission_paths_and_canonical_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def molecule(name: str) -> types.SimpleNamespace:
        symbol = "Xe" if name == "unsupported" else "N"
        size = 101 if name == "large" else (0 if name == "" else 2)
        return types.SimpleNamespace(
            name=name,
            GetAtoms=lambda: [types.SimpleNamespace(GetSymbol=lambda: symbol)],
            GetNumAtoms=lambda: size,
            GetNumHeavyAtoms=lambda: size,
        )

    def parse(value: str) -> types.SimpleNamespace | None:
        if value == "exception":
            raise RuntimeError("scientific failure")
        return None if value in ("invalid", "reparse") else molecule(value)

    def canonical(value: types.SimpleNamespace) -> str:
        name = from_plain(cast(object, value.name), str)
        return "reparse" if name == "roundtrip" else name

    def inchi(value: str) -> types.SimpleNamespace | None:
        return None if value == "inchi" else molecule(value)

    def encode_inchi(value: types.SimpleNamespace) -> str:
        return from_plain(cast(object, value.name), str)

    def unchanged(value: types.SimpleNamespace) -> types.SimpleNamespace:
        return value

    def tautomer(value: types.SimpleNamespace) -> types.SimpleNamespace:
        name = from_plain(cast(object, value.name), str)
        return molecule("reparse") if name == "tautomer" else value

    def discard(value: object) -> None:
        del value

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        MolToInchi=encode_inchi,
        MolFromInchi=inchi,
        RemoveStereochemistry=discard,
        RemoveHs=unchanged,
    )

    def context(version: str, *, directory: Path) -> types.SimpleNamespace:
        del version, directory
        return types.SimpleNamespace(
            chem=chemistry,
            tautomers=types.SimpleNamespace(Canonicalize=tautomer),
            elements=("N",),
        )

    monkeypatch.setattr(admission, "IceChemistry", context)
    actual = admission.AdmissionChemistry(SIMULATOR_RDKIT, directory=tmp_path)
    for value, ice, ice_reason, glacier, glacier_reason, heavy in (
        ("accepted", True, "", "accepted", "", 2),
        ("", False, "rdkit parse", None, "empty", 0),
        ("invalid", False, "rdkit parse", None, "rdkit parse", 0),
        (
            "unsupported",
            False,
            "elements ['Xe']",
            None,
            "element outside VALID_ATOM_NUM",
            0,
        ),
        ("large", True, "", None, "> 100 heavy atoms", 0),
        ("roundtrip", False, "inchi round trip", None, "canonical re-parse", 0),
        ("inchi", False, "inchi round trip", "inchi", "", 2),
        ("tautomer", False, "inchi round trip", "tautomer", "", 2),
        (
            "exception",
            False,
            "exception RuntimeError('scientific failure')",
            None,
            "exception RuntimeError('scientific failure')",
            0,
        ),
    ):
        assert actual.row(value) == {
            "smiles": value,
            "ice_accepted": ice,
            "ice_reason": ice_reason,
            "glacier_accepted": glacier is not None,
            "glacier_reason": glacier_reason,
            "glacier_canonical": glacier,
            "glacier_heavy_atoms": heavy,
        }

    def unavailable(value: str) -> None:
        del value
        raise RuntimeError("backend became unavailable")

    chemistry.MolFromSmiles = unavailable
    cached = actual.row("accepted")
    assert cached["ice_accepted"] is False
    assert cached["glacier_canonical"] == "accepted"
    assert cached["glacier_heavy_atoms"] == 2
    assert cached["glacier_reason"] == ""


class _Decisions:
    def __init__(self, version: str, *, directory: Path) -> None:
        assert version == SIMULATOR_RDKIT
        _ = (directory / "tautomer-transforms.txt").write_text("declared test boundary")

    def row(self, smiles: str) -> SimulatorAdmissionRow:
        accepted = smiles != "invalid"
        return {
            "smiles": smiles,
            "ice_accepted": accepted,
            "ice_reason": "" if accepted else "rdkit parse",
            "glacier_accepted": accepted,
            "glacier_reason": "" if accepted else "rdkit parse",
            "glacier_canonical": "canonical" if accepted else None,
            "glacier_heavy_atoms": 2 if accepted else 0,
        }


# analog_data_test.py.
def test_actual_element_core_ring_and_formula_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    chemistry = analog_data.AnalogChemistry(IDENTITY_RDKIT)
    row = chemistry.record("input", formula="C2H5NO")
    hashes = [
        analog_data._identity(prefix + name)
        for prefix, name in (("E:", "core"), ("G:", "generic"), ("R:", "core"))
    ]
    np.testing.assert_array_equal(row.hashes, [*hashes, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row.numbers, [1, 1, 3, 1, 1, 0, 3, 4])
    np.testing.assert_array_equal(row.formula, [2, 5, 1, 1, 0, 0, 0, 0, 0, 0])
    assert (row.core_smiles, row.element_smiles, row.query_smiles) == (
        "core",
        "full",
        "core",
    )
    assert row.graph is not None
    assert row.core is not None
    np.testing.assert_array_equal(row.graph.elements, [6, 7, 8, 6])
    np.testing.assert_array_equal(row.graph.bonds, [[0, 1], [1, 2], [0, 2], [2, 3]])
    np.testing.assert_array_equal(row.core.elements, [6, 7, 8])
    np.testing.assert_array_equal(row.core.bonds, [[0, 1], [1, 2], [0, 2]])
    np.testing.assert_array_equal(row.core_mask, [True, True, True, False])
    chain = chemistry.record("chain", formula=None)
    np.testing.assert_array_equal(chain.hashes, np.zeros(8, dtype=np.int64))
    np.testing.assert_array_equal(chain.numbers, [0, 0, 0, 0, 0, 0, 0, 2])
    np.testing.assert_array_equal(chain.core_mask, [False, False])
    assert chain.core is None
    assert chain.query_smiles == chain.core_smiles == ""
    for value in ("", "invalid", "empty"):
        rejected = chemistry.record(value, formula=None)
        np.testing.assert_array_equal(rejected.hashes, np.zeros(8))
        np.testing.assert_array_equal(rejected.numbers, np.zeros(8))
        np.testing.assert_array_equal(rejected.formula, np.full(10, -1))
        assert rejected.graph is None
        assert rejected.core is None
        assert rejected.core_mask.size == 0
    assert chemistry._numeric("invalid") is None
    assert analog_data._core([[1], [0]]) == []
    assert analog_data._components([[1], [0], []], active={0, 1, 2}) == [[0, 1], [2]]
    with pytest.raises(ValueError, match=r"requires RDKit 2026\.03\.3"):
        analog_data.AnalogChemistry(SIMULATOR_RDKIT)


def test_analog_writer_retains_all_rows_numeric_graphs_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _analog_boundary(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "analog"
    values = ["input", "invalid", "input", "chain"]
    formulas = ["C2H5NO", None, "C2H5NO", "CH3N"]
    parquet.write_table(
        Table.from_pydict({"smiles": values, "formula": formulas}),
        request,
    )
    analog_data.prepare_analog(request, output=output, version=IDENTITY_RDKIT)
    table = AnalogTable(output)
    assert [row["smiles"] for row in table.metadata] == values
    assert [row["formula_input"] for row in table.metadata] == formulas
    assert [row["element_smiles"] for row in table.metadata] == [
        "full",
        "",
        "full",
        "chain",
    ]
    assert [row["core_smiles"] for row in table.metadata] == ["core", "", "core", ""]
    for name, expected in (
        ("graph_elements", [6, 7, 8, 6, 6, 7, 8, 6, 6, 7]),
        ("graph_bonds", [[0, 1], [1, 2], [0, 2], [2, 3]] * 2 + [[0, 1]]),
        ("core_elements", [6, 7, 8] * 2),
        ("core_bonds", [[0, 1], [1, 2], [0, 2]] * 2),
        ("core_mask", [True, True, True, False] * 2 + [False, False]),
        ("graph_atom_off", [0, 4, 4, 8, 10]),
        ("graph_bond_off", [0, 4, 4, 8, 9]),
        ("core_atom_off", [0, 3, 3, 6, 6]),
        ("core_bond_off", [0, 3, 3, 6, 6]),
    ):
        actual = cast("NDArray[np.generic]", np.load(output / f"{name}.npy"))
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        table.numbers,
        [
            [1, 1, 3, 1, 1, 0, 3, 4],
            [0] * 8,
            [1, 1, 3, 1, 1, 0, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 2],
        ],
    )
    np.testing.assert_array_equal(
        table.formula,
        [
            [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
            [-1] * 10,
            [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert table.row(1).graph is None
    assert table.row(1).core is None
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {name: manifest[name] for name in ("format_version", "rdkit", "rows")} == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": 4,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "analog_data.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
            _SCRIPTS_DIR.parent / "chemistry.py",
        )
    }
    with pytest.raises(FileExistsError):
        analog_data.prepare_analog(request, output=output, version=IDENTITY_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [], "formula": []}), request)
    analog_data.prepare_analog(
        request,
        output=tmp_path / "empty",
        version=IDENTITY_RDKIT,
    )
    empty = AnalogTable(tmp_path / "empty")
    assert empty.metadata == []
    assert empty.hashes.shape == (0, 8)
    assert empty.graph.elements.shape == (0,)
    assert empty.graph.bonds.shape == (0, 2)


def _analog_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def discard(value: object) -> None:
        del value

    def atom(index: int, number: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            GetIdx=lambda: index,
            GetAtomicNum=lambda: number,
            SetNoImplicit=discard,
        )

    def molecule(
        numbers: list[int],
        edges: list[tuple[int, int]],
    ) -> types.SimpleNamespace:
        atoms = [atom(index, number) for index, number in enumerate(numbers)]
        bonds = [
            types.SimpleNamespace(
                GetBeginAtomIdx=lambda left=left: left,
                GetEndAtomIdx=lambda right=right: right,
                IsInRing=lambda left=left, right=right: (
                    len(numbers) >= 3 and left < 3 and right < 3
                ),
            )
            for left, right in edges
        ]

        def at(index: int) -> types.SimpleNamespace:
            return atoms[index]

        def update(*, strict: bool) -> None:
            del strict

        return types.SimpleNamespace(
            numbers=numbers,
            GetAtoms=lambda: atoms,
            GetBonds=lambda: bonds,
            GetAtomWithIdx=at,
            GetNumAtoms=lambda: len(atoms),
            GetNumBonds=lambda: len(bonds),
            UpdatePropertyCache=update,
        )

    structures = {
        "input": molecule([6, 7, 8, 6, 1], [(0, 1), (1, 2), (2, 0), (2, 3), (0, 4)]),
        "chain": molecule([6, 7], [(0, 1)]),
        "empty": molecule([], []),
    }

    def editable() -> types.SimpleNamespace:
        numbers: list[int] = []
        edges: list[tuple[int, int]] = []

        def add_atom(value: Chem.Atom) -> int:
            numbers.append(value.GetAtomicNum())
            return len(numbers) - 1

        def add_bond(left: int, right: int, kind: object) -> None:
            del kind
            edges.append((left, right))

        return types.SimpleNamespace(
            AddAtom=add_atom,
            AddBond=add_bond,
            GetMol=lambda: molecule(numbers, edges),
        )

    def canonical(value: types.SimpleNamespace, **kwargs: object) -> str:
        del kwargs
        numbers = from_plain(cast(object, value.numbers), list[int])
        name = (
            "full"
            if len(numbers) == 4
            else (
                "chain"
                if len(numbers) == 2
                else ("generic" if numbers == [6, 6, 6] else "core")
            )
        )
        structures[name] = value
        return name

    def parse(value: str, *, sanitize: bool) -> types.SimpleNamespace | None:
        del sanitize
        return structures.get(value)

    def construct_atom(number: int) -> types.SimpleNamespace:
        return atom(0, number)

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        Atom=construct_atom,
        RWMol=editable,
        FastFindRings=discard,
        BondType=types.SimpleNamespace(SINGLE="single"),
    )

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(chem=chemistry)

    monkeypatch.setattr(analog_data, "MolecularContext", context)


def test_graph_hashes_are_seeded_blake2b_and_zero_is_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for text, expected in (
        ("E:C1CCCCC1", 4_101_090_148_000_295_725),
        ("G:C1CCCCC1", 2_963_313_213_660_459_189),
        ("R:C1CCCCC1", 2_587_690_919_976_953_160),
        ("E:C1CCNCC1", 1_399_155_810_780_611_077),
        ("E:C1CCC(CCC2CCCCC2)CC1", 1_073_036_906_936_525_605),
        ("E:C1CC1.C1CC1", 807_823_770_527_924_393),
        (
            "E:C1CC1.C1CCC1.C1CCCC1.C1CCCCC1.C1CCCCCC1.C1CN1.C1CO1",
            3_953_499_957_839_148_272,
        ),
    ):
        assert analog_data._identity(text) == expected

    def reserved(data: bytes, *, digest_size: int) -> types.SimpleNamespace:
        assert data == b"reserved"
        assert digest_size == 8
        return types.SimpleNamespace(digest=lambda: bytes(8))

    monkeypatch.setattr(
        analog_data,
        "hashlib",
        types.SimpleNamespace(blake2b=reserved),
    )
    assert analog_data._identity("reserved") == 1


# atomic_data_test.py.
@pytest.mark.parametrize(
    (
        "number",
        "degree",
        "hydrogens",
        "charge",
        "hybrid",
        "chiral",
        "partial_charge",
        "active",
    ),
    [
        (
            6,
            4,
            4,
            0,
            "SP3",
            "CHI_TETRAHEDRAL_CW",
            "0.25",
            [0, 23, 28, 34, 38, 43, 46, 48, 50, 52, 53, 56, 58, 60, 62, 69],
        ),
        (
            999,
            6,
            5,
            3,
            "OTHER",
            "CHI_OTHER",
            "nan",
            [18, 25, 31, 37, 44, 45, 48, 50, 52, 54, 56, 58, 59, 64],
        ),
        (
            7,
            -1,
            -1,
            -3,
            "SP2",
            "CHI_UNSPECIFIED",
            "inf",
            [1, 31, 33, 45, 48, 50, 52, 54, 56, 58, 59, 61, 68],
        ),
        (
            7,
            -1,
            -1,
            -3,
            "SP2",
            "CHI_UNSPECIFIED",
            None,
            [1, 31, 33, 45, 48, 50, 52, 54, 56, 58, 59, 61, 68],
        ),
    ],
)
def test_complete_atom_layout_unknown_overflows_and_nonfinite_charges(
    *,
    number: int,
    degree: int,
    hydrogens: int,
    charge: int,
    hybrid: str,
    chiral: str,
    partial_charge: str | None,
    active: list[int],
) -> None:
    atom = _atom(
        number=number,
        degree=degree,
        hydrogens=hydrogens,
        charge=charge,
        hybrid=hybrid,
        chiral=chiral,
        partial_charge=partial_charge,
        ring=number == 6,
    )
    expected = torch.zeros(74)
    expected[active] = 1
    expected[73] = 0.25 if number == 6 else 0.0
    assert torch.equal(
        torch.tensor(
            atomic_data._node_features(atom, ring_count=2 if number == 6 else 0),
        ),
        expected,
    )


@pytest.mark.parametrize("kind", ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"])
@pytest.mark.parametrize(
    "stereo",
    ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "OTHER"],
)
def test_complete_bond_layout_all_types_stereochemistry_and_fused_rings(
    kind: str,
    stereo: str,
) -> None:
    conjugated, ring = kind == "AROMATIC", stereo == "STEREOZ"

    def ring_size(size: int) -> bool:
        return ring and size == 5

    bond = cast(
        "Chem.Bond",
        types.SimpleNamespace(
            GetBondType=lambda: kind,
            GetStereo=lambda: stereo,
            GetIsConjugated=lambda: conjugated,
            IsInRing=lambda: ring,
            IsInRingSize=ring_size,
        ),
    )
    expected = [0.0] * 28
    expected[("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER").index(kind)] = 1.0
    expected[5 + int(conjugated)] = 1.0
    expected[7 + int(ring)] = 1.0
    for position, size in enumerate(range(3, 9)):
        expected[9 + 2 * position + int(not (ring and size == 5))] = 1.0
    expected[21 + int(ring)] = 1.0
    expected[
        23 + ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "OTHER").index(stereo)
    ] = 1.0
    assert atomic_data._bond_features(bond, ring_count=2 if ring else 0) == expected


@pytest.mark.parametrize("hybrid", ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNKNOWN"])
@pytest.mark.parametrize(
    "chiral",
    [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "UNKNOWN",
    ],
)
def test_atom_hybridization_and_chirality_use_complete_ordered_vocabularies(
    hybrid: str,
    chiral: str,
) -> None:
    actual = atomic_data._node_features(
        _atom(hybrid=hybrid, chiral=chiral),
        ring_count=1,
    )
    assert actual[32:38] == [
        float(hybrid == value)
        for value in ("SP", "SP2", "SP3", "SP3D", "SP3D2", "UNKNOWN")
    ]
    assert actual[61:65] == [
        float(chiral == value)
        for value in (
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER",
        )
    ]
    assert actual[59:61] == [1.0, 0.0]


def test_atom_integer_channel_boundaries_and_every_element_group() -> None:
    for degree in range(-1, 7):
        actual = atomic_data._node_features(_atom(degree=degree), ring_count=0)
        assert actual[19:26] == [
            *[float(degree == value) for value in range(6)],
            float(degree > 5),
        ]
    for charge in range(-3, 4):
        actual = atomic_data._node_features(_atom(charge=charge), ring_count=0)
        assert actual[26:32] == [
            *[float(charge == value) for value in range(-2, 3)],
            float(abs(charge) > 2),
        ]
    for hydrogens in range(-1, 6):
        actual = atomic_data._node_features(_atom(hydrogens=hydrogens), ring_count=0)
        assert actual[39:45] == [
            *[float(hydrogens == value) for value in range(5)],
            float(hydrogens > 4),
        ]
    elements = (
        (6, 4),
        (7, 3),
        (15, 3),
        (8, 5),
        (16, 5),
        (14, 4),
        (53, 6),
        (1, 0),
        (17, 6),
        (9, 6),
        (35, 6),
        (5, 2),
        (34, 5),
        (26, 7),
        (27, 7),
        (33, 3),
        (11, 1),
        (19, 1),
    )
    assert tuple(atomic_data._elements().items()) == elements
    for position, (number, group) in enumerate(elements):
        actual = atomic_data._node_features(_atom(number=number), ring_count=0)
        assert actual[:19] == [float(index == position) for index in range(19)]
        assert actual[65:73] == [float(index == group) for index in range(8)]


def test_graph_writer_full_rows_manifest_and_glacier_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _atom()
    molecule = types.SimpleNamespace(
        GetAtoms=lambda: [atom],
        GetBonds=list,
        GetNumAtoms=lambda: 1,
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return None if value == "invalid" else molecule

    def canonical(_: object) -> str:
        return "canonical"

    def remove_hydrogens(value: object) -> object:
        return value

    def rings(_: object) -> list[list[int]]:
        return []

    def partial_charges(_: object) -> None:
        return None

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=remove_hydrogens,
        GetSymmSSSR=rings,
    )
    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    monkeypatch.setattr(atomic_data, "ComputeGasteigerCharges", partial_charges)

    def context(version: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(version=version, chem=chemistry)

    monkeypatch.setattr(
        atomic_data,
        "MolecularContext",
        context,
    )
    request = tmp_path / "request.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["N", "invalid", None, "N"]}),
        request,
    )
    output = tmp_path / "output"
    prepare_graphs(request, output=output, version=SIMULATOR_RDKIT)
    rows = parquet.read_table(output / "graphs.parquet").to_pylist()
    assert [row["input_smiles"] for row in rows] == ["N", "invalid"]
    assert [row["smiles"] for row in rows] == ["canonical", ""]
    assert [row["valid"] for row in rows] == [True, False]
    assert rows[0]["degrees"] == [0]
    assert rows[0]["bonds"] == []
    assert rows[0]["weights"] == []
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["format_version"] == 1
    assert manifest["raw"] is False
    assert manifest["rows"] == 2
    assert manifest["valid_rows"] == 1
    assert manifest["node_features"] == 74
    assert manifest["bond_features"] == 28
    with pytest.raises(FileExistsError):
        prepare_graphs(request, output=output, version=SIMULATOR_RDKIT)

    parquet.write_table(
        Table.from_pydict({"smiles": ["invalid", None, "N", "N", "O"]}),
        request,
    )
    batched = tmp_path / "batched"
    prepare_graphs(
        request,
        output=batched,
        version=SIMULATOR_RDKIT,
        batch_rows=1,
    )
    source = parquet.ParquetFile(batched / "graphs.parquet")
    assert source.num_row_groups == 3
    assert source.read().to_pylist() == [
        rows[1],
        rows[0],
        {**rows[0], "input_smiles": "O"},
    ]
    assert source.read_row_group(0).to_pylist() == [rows[1]]
    assert source.read_row_group(1).to_pylist() == [rows[0]]
    for batch_rows in (0, -1):
        with pytest.raises(ValueError, match="batch_rows must be positive"):
            prepare_graphs(
                request,
                output=tmp_path / "bad-batch",
                version=SIMULATOR_RDKIT,
                batch_rows=batch_rows,
            )
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    empty = tmp_path / "empty"
    prepare_graphs(request, output=empty, version=SIMULATOR_RDKIT, batch_rows=2)
    assert parquet.read_table(empty / "graphs.parquet").to_pylist() == []
    manifest = json_loads((empty / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 0
    assert manifest["valid_rows"] == 0


def test_actual_bonded_ring_graph_has_complete_features_and_ring_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ring_size(size: int) -> bool:
        return size == 3

    atom = _atom(degree=2, hydrogens=1, ring=True, ring_size_filled=3)
    bonds = [
        types.SimpleNamespace(
            GetBeginAtomIdx=lambda left=left: left,
            GetEndAtomIdx=lambda right=right: right,
            GetBondTypeAsDouble=lambda: 1.0,
            GetBondType=lambda: "SINGLE",
            GetIsConjugated=lambda: False,
            IsInRing=lambda: True,
            IsInRingSize=ring_size,
            GetStereo=lambda: "STEREONONE",
        )
        for left, right in ((0, 1), (1, 2), (2, 0))
    ]
    molecule = types.SimpleNamespace(
        GetAtoms=lambda: [atom] * 3,
        GetNumAtoms=lambda: 3,
        GetBonds=lambda: bonds,
    )

    def parse(value: str) -> types.SimpleNamespace:
        del value
        return molecule

    def canonical(value: object) -> str:
        del value
        return "N1NN1"

    def unchanged(value: types.SimpleNamespace) -> types.SimpleNamespace:
        return value

    def rings(value: object) -> list[list[int]]:
        del value
        return [[0, 1, 2]]

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=unchanged,
        GetSymmSSSR=rings,
    )

    def unavailable(value: object) -> None:
        del value
        raise RuntimeError("missing Gasteiger parameters")

    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    monkeypatch.setattr(atomic_data, "ComputeGasteigerCharges", unavailable)
    row = atomic_data._graph_row("requested", raw=False)
    node = [0.0] * 74
    for index in (1, 21, 28, 34, 38, 40, 46, 47, 50, 52, 54, 56, 58, 59, 61, 68):
        node[index] = 1.0
    bond = [0.0] * 28
    for index in (0, 5, 8, 9, 12, 14, 16, 18, 20, 21, 23):
        bond[index] = 1.0
    assert row == {
        "input_smiles": "requested",
        "smiles": "N1NN1",
        "valid": True,
        "features": [node] * 3,
        "bonds": [[0, 1], [1, 2], [2, 0]],
        "bond_features": [bond] * 3,
        "weights": [1.0] * 3,
        "degrees": [2] * 3,
    }


@pytest.mark.parametrize("case", ["unsupported", "empty", "large", "reparse"])
def test_actual_graph_admission_and_failed_canonical_reparse(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _atom(number=999 if case == "unsupported" else 7)
    molecule = types.SimpleNamespace(
        GetAtoms=lambda: [atom],
        GetNumAtoms=lambda: 0 if case == "empty" else (101 if case == "large" else 2),
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return molecule if value == "input" else None

    def canonical(value: object) -> str:
        del value
        return "canonical"

    def unchanged(value: types.SimpleNamespace) -> types.SimpleNamespace:
        return value

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical,
        RemoveHs=unchanged,
    )
    monkeypatch.setattr(atomic_data, "Chem", chemistry)
    if case == "reparse":
        with pytest.raises(ValueError, match="Expected parsed is not None"):
            atomic_data._graph_row("input", raw=False)
    else:
        assert atomic_data._graph_row("input", raw=False) == {
            "input_smiles": "input",
            "smiles": "",
            "valid": False,
            "features": [],
            "bonds": [],
            "bond_features": [],
            "weights": [],
            "degrees": [],
        }


def _atom(
    *,
    number: int = 7,
    degree: int = 0,
    hydrogens: int = 3,
    charge: int = 0,
    hybrid: str = "SP3",
    chiral: str = "CHI_UNSPECIFIED",
    partial_charge: str | None = "0.0",
    ring: bool = False,
    ring_size_filled: int = 6,
) -> Chem.Atom:
    def ring_size(size: int) -> bool:
        return ring and size == ring_size_filled

    def property_value(name: str) -> str:
        return _property(partial_charge, name)

    return cast(
        "Chem.Atom",
        types.SimpleNamespace(
            GetAtomicNum=lambda: number,
            GetTotalDegree=lambda: degree + hydrogens,
            GetTotalNumHs=lambda: hydrogens,
            GetFormalCharge=lambda: charge,
            GetHybridization=lambda: hybrid,
            GetChiralTag=lambda: chiral,
            GetIsAromatic=lambda: ring,
            IsInRing=lambda: ring,
            IsInRingSize=ring_size,
            GetProp=property_value,
        ),
    )


def _property(value: str | None, name: str) -> str:
    if value is None or name != "_GasteigerCharge":
        raise KeyError(name)
    return value


def test_graph_preparation_rejects_the_main_engine_chemistry_version(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"requires RDKit 2025\.03\.6"):
        prepare_graphs(
            tmp_path / "input.parquet",
            output=tmp_path / "output",
            version=IDENTITY_RDKIT,
        )


@pytest.mark.parametrize("extra_identities", [False, True])
def test_prepared_paths_keep_order_failures_limits_and_complete_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_identities: bool,
) -> None:
    def canonical(value: str) -> str:
        if value == "unserializable":
            raise RuntimeError("invalid terminal")
        return value

    def products(value: str, *, rule: str, limit: int) -> list[tuple[str, int]]:
        assert limit == {("root", "first"): 32, ("branch", "second"): 16}[(value, rule)]
        return {
            ("root", "first"): [("branch", 3), ("unused", 9)],
            ("branch", "second"): [
                ("leaf", 4),
                ("rejected", 8),
                ("unserializable", 13),
            ],
        }[(value, rule)]

    def environments(value: str) -> dict[int, tuple[int, int, int]]:
        return {3: (1, 2, 3)} if value == "root" else {4: (4, 5, 6)}

    def parse(value: str) -> str | None:
        return None if value == "invalid" else value

    def standardize(value: str) -> str | None:
        return None if value == "rejected" else "normalized-" + value

    def identity(value: str) -> tuple[str, str]:
        return "key-" + value, "skeleton-" + value

    def context(version: str) -> DerivationChemistry:
        assert version == IDENTITY_RDKIT
        return cast(
            "DerivationChemistry",
            types.SimpleNamespace(
                chem=types.SimpleNamespace(
                    MolFromSmiles=parse,
                    MolToSmiles=canonical,
                ),
                rules={"first": "first", "second": "second"},
                products=products,
                environments=environments,
                standard=types.SimpleNamespace(standardize=standardize),
                identity=identity,
            ),
        )

    monkeypatch.setattr(derivation_data, "DerivationChemistry", context)
    request = tmp_path / "requests.parquet"
    parquet.write_table(
        Table.from_pylist(
            [
                {"parent_smiles": "root", "combo": "first+second", "max_mols": 1},
                {"parent_smiles": "invalid", "combo": "first", "max_mols": None},
            ],
        ),
        request,
    )
    identities = tmp_path / "identities.parquet"
    parquet.write_table(Table.from_pydict({"smiles": ["extra", "leaf"]}), identities)
    output = tmp_path / "prepared"
    derivation_data.prepare_derivations(
        request,
        identities=identities if extra_identities else None,
        output=output,
        version=IDENTITY_RDKIT,
    )
    table = DerivationTable(output)
    assert table.requests == (
        {
            "parent_smiles": "root",
            "combo": ("first", "second"),
            "max_first": 32,
            "max_second": 16,
            "max_mols": 1,
            "terminal_count": 3,
            "applications": "root\tfirst\t32\t2\nbranch\tsecond\t16\t3",
        },
        {
            "parent_smiles": "invalid",
            "combo": ("first",),
            "max_first": 32,
            "max_second": 16,
            "max_mols": 24,
            "terminal_count": 0,
            "applications": "",
        },
    )
    assert table.paths("root", ("first", "second")) == (
        {
            "request": 0,
            "position": 0,
            "raw": "leaf",
            "standardized": "normalized-leaf",
            "key": "key-normalized-leaf",
            "skeleton": "skeleton-normalized-leaf",
            "environments": ((1, 2, 3), (4, 5, 6)),
        },
        {
            "request": 0,
            "position": 1,
            "raw": "rejected",
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": ((1, 2, 3), (-1, -1, -1)),
        },
        {
            "request": 0,
            "position": 2,
            "raw": None,
            "standardized": None,
            "key": None,
            "skeleton": None,
            "environments": ((1, 2, 3), (-1, -1, -1)),
        },
    )
    assert table.paths("invalid", ("first",)) == ()
    wanted = ["root", "leaf", "normalized-leaf", "rejected", "invalid"]
    if extra_identities:
        wanted.append("extra")
    assert parquet.read_table(output / "identities.parquet").to_pylist() == [
        {"smiles": value, "key": "key-" + value, "skeleton": "skeleton-" + value}
        for value in wanted
    ]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "requests", "paths", "identities")
    } == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "requests": 2,
        "paths": 3,
        "identities": len(wanted),
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ([request, identities] if extra_identities else [request])
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        Path(name).name: hashlib.sha256((_SCRIPTS_DIR / name).read_bytes()).hexdigest()
        for name in (
            "derivation_data.py",
            "derivation_chemistry.py",
            "rdkit_backend.py",
            "../chemistry.py",
            "../v6_reactions.tsv",
        )
    }


# engine_identity_test.py.
class _Molecule:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles
        self.canonical = False


class Primitives:
    """Supply stored SDK outcomes and explicit exceptional operations."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.keys = engine_keys()
        self.chem = types.SimpleNamespace(
            MolFromSmiles=self.parse,
            MolToInchiKey=self.inchi,
        )
        self.tautomers = types.SimpleNamespace(Canonicalize=self.canonicalize)

    def parse(self, smiles: str) -> _Molecule | None:
        if smiles == "parse-error":
            raise ValueError("known parse failure")
        return None if smiles == "invalid" else _Molecule(smiles)

    def canonicalize(self, molecule: _Molecule) -> _Molecule:
        if molecule.smiles == "canonical-error":
            raise RuntimeError("known tautomer failure")
        molecule.canonical = True
        return molecule

    def inchi(self, molecule: _Molecule) -> str:
        assert molecule.canonical
        if molecule.smiles == "key-error":
            raise RuntimeError("known InChI failure")
        key = self.keys[molecule.smiles]
        assert key is not None
        return key + "-UHFFFAOYSA-N" if key else ""


def test_complete_engine_identity_sdk_and_known_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_identity, "Chemistry", Primitives)
    chemistry = engine_identity.EngineKeyChemistry(IDENTITY_RDKIT)
    assert {smiles: chemistry.key(smiles) for smiles in engine_keys()} == engine_keys()
    for value in ("parse-error", "canonical-error", "key-error"):
        assert chemistry.key(value) is None
    with pytest.raises(
        ValueError,
        match=r"^Engine identity preparation requires RDKit 2026\.03\.3\.$",
    ):
        engine_identity.EngineKeyChemistry(SIMULATOR_RDKIT)


class RecordedKeys:
    """Retain every stored SDK result at the writer's primitive boundary."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def key(self, smiles: str) -> str | None:
        """Return the exact whole recorded SDK decision for a declared input."""
        return engine_keys()[smiles]


def test_writer_preserves_all_rows_duplicates_empty_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_identity, "EngineKeyChemistry", RecordedKeys)
    keys = engine_keys()
    strings = [*keys, "*C", "invalid"]
    source, output = tmp_path / "request.parquet", tmp_path / "prepared"
    parquet.write_table(Table.from_pydict({"smiles": strings}), source)
    engine_identity.prepare_engine_identity(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    rows: list[EngineIdentityRow] = [
        {"smiles": value, "key": keys[value]} for value in strings
    ]
    assert parquet.read_table(output / "keys.parquet").to_pylist() == rows
    table = EngineIdentityTable(output)
    assert {smiles: table.key(smiles) for smiles in keys} == keys
    header = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert header["rows"] == len(strings)
    assert header["rdkit"] == IDENTITY_RDKIT
    assert header["format_version"] == 1
    with pytest.raises(FileExistsError):
        engine_identity.prepare_engine_identity(
            source,
            output=output,
            version=IDENTITY_RDKIT,
        )
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        engine_identity.prepare_engine_identity(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    engine_identity.prepare_engine_identity(
        source,
        output=tmp_path / "empty",
        version=IDENTITY_RDKIT,
    )
    assert parquet.read_table(tmp_path / "empty/keys.parquet").num_rows == 0
    _ = EngineIdentityTable(tmp_path / "empty")


# fragment_gap_data_test.py.
def test_gap_writer_preserves_bounded_groups_duplicates_and_empty_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fragment_gap_data, "GapChemistry", _FragmentGapDataGraphs)
    source, output = tmp_path / "input.parquet", tmp_path / "output"
    parquet.write_table(Table.from_pydict({"smiles": ["CC", "invalid", "CC"]}), source)
    fragment_gap_data.prepare_fragment_gap(
        source,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=2,
    )
    raw = parquet.ParquetFile(output / "graphs.parquet")
    assert raw.num_row_groups == 2
    assert raw.read().column("smiles").to_pylist() == ["CC", "invalid", "CC"]
    assert raw.read().column("valid").to_pylist() == [True, False, True]
    table = GapGraphTable.Config(working_dir=output).make()
    graph = table.for_smiles("CC")
    assert graph is not None
    assert graph.pieces == 1
    np.testing.assert_array_equal(graph.weights, [1.0, 2.0])
    np.testing.assert_array_equal(graph.bonds, [[0, 1]])
    assert table.for_smiles("invalid") is None
    header = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert header["format_version"] == 1
    assert header["rdkit"] == IDENTITY_RDKIT
    assert header["rows"] == 3
    assert header["numpy"] == np.__version__
    assert set(header) == {
        "format_version",
        "rdkit",
        "numpy",
        "rows",
        "inputs",
        "generator",
    }
    assert from_plain(header["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert set(from_plain(header["generator"], dict[str, str])) == {
        "fragment_gap_data.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    with pytest.raises(FileExistsError):
        fragment_gap_data.prepare_fragment_gap(
            source,
            output=output,
            version=IDENTITY_RDKIT,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), source)
    empty = tmp_path / "empty"
    fragment_gap_data.prepare_fragment_gap(source, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "graphs.parquet").num_rows == 0
    with pytest.raises(ValueError, match=r"^batch_rows must be positive\.$"):
        fragment_gap_data.prepare_fragment_gap(
            source,
            output=tmp_path / "bad",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )


def test_gap_sdk_retains_atom_indices_isotope_ignorance_and_periodic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = (
        types.SimpleNamespace(
            GetIdx=lambda: 1,
            GetSymbol=lambda: "C",
            GetTotalNumHs=lambda: 3,
        ),
        types.SimpleNamespace(
            GetIdx=lambda: 0,
            GetSymbol=lambda: "Cu",
            GetTotalNumHs=lambda: 0,
        ),
    )
    molecule = types.SimpleNamespace(
        GetNumAtoms=lambda: 2,
        GetAtoms=lambda: atoms,
        GetBonds=lambda: (
            types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        ),
    )
    fallback: list[str] = []

    def isotope(element: str) -> float:
        fallback.append(element)
        return 43.0

    def parse(value: str) -> types.SimpleNamespace | None:
        if value == "error":
            raise ValueError("Invalid scientific input.")
        return molecule if value != "invalid" else None

    def components(value: types.SimpleNamespace) -> tuple[tuple[int, ...], ...]:
        assert value is molecule
        return ((0, 1),)

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(
            chem=types.SimpleNamespace(
                MolFromSmiles=parse,
                GetPeriodicTable=lambda: types.SimpleNamespace(
                    GetMostCommonIsotopeMass=isotope,
                ),
                GetMolFrags=components,
            ),
        )

    monkeypatch.setattr(fragment_gap_data, "MolecularContext", context)
    chemistry = fragment_gap_data.GapChemistry(IDENTITY_RDKIT)
    graph = chemistry.graph("isotope")
    assert graph is not None
    np.testing.assert_array_equal(graph.weights, [43.0, 12.0 + 3 * 1.00782503207])
    np.testing.assert_array_equal(graph.bonds, [[1, 0]])
    assert graph.pieces == 1
    assert fallback == ["Cu"]
    assert chemistry.graph("invalid") is None
    assert chemistry.graph("error") is None
    assert chemistry.graph("") is None
    molecule.GetNumAtoms = lambda: 0
    assert chemistry.graph("empty") is None
    with pytest.raises(ValueError, match=r"^Gap graphs require RDKit 2026\.03\.3\.$"):
        _ = fragment_gap_data.GapChemistry(SIMULATOR_RDKIT)


class _FragmentGapDataGraphs:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def graph(self, smiles: str) -> GapGraph | None:
        return (
            None
            if smiles == "invalid"
            else GapGraph(
                weights=np.array([1.0, 2.0]),
                bonds=np.array([[0, 1]], dtype=np.int64),
                pieces=1,
            )
        )


# fragnet_data_test.py.
def test_standard_atom_projection_retains_every_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _observations(monkeypatch)
    chemistry = fragnet_data.FragnetChemistry(IDENTITY_RDKIT)
    actual: dict[str, Tensor] = {}
    for index in range(27):
        prefix = str(index)
        value = _fragnet_data_decode(expected[f"{prefix}/input"])
        atoms = chemistry.atoms(value)
        actual[f"{prefix}/input"] = _fragnet_data_text(value)
        actual[f"{prefix}/parsed"] = torch.tensor(atoms is not None)
        if atoms is not None:
            actual[f"{prefix}/raw/symbols"] = _fragnet_data_text(
                "\0".join(atoms.symbols),
            )
            for name, values in (
                ("hydrogens", atoms.hydrogens),
                ("degree", atoms.degree),
                ("charge", atoms.charge),
                ("aromatic", atoms.aromatic),
                ("bonds", atoms.bonds),
                ("bond_aromatic", atoms.bond_aromatic),
                ("bond_ring", atoms.bond_ring),
            ):
                actual[f"{prefix}/raw/{name}"] = torch.from_numpy(values)
        record = molecular_record(atoms, masses=ELEMENT_MASSES)
        actual[f"{prefix}/valid"] = torch.tensor(record is not None)
        if record is not None:
            for name in ("amass", "kind", "el", "nh", "bonds", "bring", "el10"):
                actual[f"{prefix}/{name}"] = torch.from_numpy(record[name])
            actual[f"{prefix}/mass"] = torch.tensor(record["mass"], dtype=torch.float64)
    assert_tensor_golden(
        _SCRIPTS_DIR.parent / "testdata/fragnet_records.pt",
        float32_comparands(actual),
    )
    with pytest.raises(
        ValueError,
        match=r"^Fragnet preparation requires RDKit 2026\.03\.3\.$",
    ):
        _ = fragnet_data.FragnetChemistry(SIMULATOR_RDKIT)


def test_writer_retains_all_duplicate_rows_and_empty_stream_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _observations(monkeypatch)
    value = _fragnet_data_decode(expected["0/input"])
    reference = molecular_record(
        fragnet_data.FragnetChemistry(IDENTITY_RDKIT).atoms(value),
        masses=ELEMENT_MASSES,
    )
    assert reference is not None
    invalid = next(
        _fragnet_data_decode(expected[f"{index}/input"])
        for index in range(27)
        if not bool(expected[f"{index}/parsed"])
    )
    request, output = tmp_path / "input.parquet", tmp_path / "nested" / "output"
    smiles = [value] * 3 + [invalid]
    parquet.write_table(
        Table.from_pydict({"smiles": smiles, "unrelated": ["unused"] * len(smiles)}),
        request,
    )
    fragnet_data.prepare_fragnet(
        request,
        output=output,
        version=IDENTITY_RDKIT,
        batch_rows=3,
    )
    metadata = parquet.ParquetFile(output / "molecules.parquet")
    assert metadata.num_row_groups == 2
    assert metadata.read().column("smiles").to_pylist() == smiles
    table = FragnetTable(output)
    for index in range(3):
        record = table.row(index)
        assert record is not None
        for name in ("amass", "kind", "el", "nh", "bonds", "bring", "el10"):
            np.testing.assert_array_equal(record[name], reference[name], strict=True)
        assert record["mass"] == reference["mass"]
    assert table.row(3) is None
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        fragnet_data.prepare_fragnet(
            request,
            output=tmp_path / "none",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    with pytest.raises(FileExistsError):
        fragnet_data.prepare_fragnet(request, output=output, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        fragnet_data.prepare_fragnet(request, output=request, version=IDENTITY_RDKIT)
    for index, values in enumerate((list[str](), [invalid])):
        source, destination = (
            tmp_path / f"empty-{index}.parquet",
            tmp_path / f"empty-{index}",
        )
        parquet.write_table(Table.from_pydict({"smiles": values}), source)
        fragnet_data.prepare_fragnet(source, output=destination, version=IDENTITY_RDKIT)
        loaded = FragnetTable(destination)
        assert len(loaded.metadata) == len(values)
        assert loaded.bonds.shape == (0, 2)
        assert loaded.amass.shape == (0,)
        assert loaded.atom_offsets.tolist() == [0] * (len(values) + 1)
        assert loaded.bond_offsets.tolist() == [0] * (len(values) + 1)
        if values:
            assert loaded.row(0) is None


def _observations(monkeypatch: pytest.MonkeyPatch) -> dict[str, Tensor]:
    expected = read_tensors(_SCRIPTS_DIR.parent / "testdata/fragnet_records.pt")
    molecules: dict[str, _ObservedMolecule | None] = {}
    for index in range(27):
        prefix = str(index)
        value = _fragnet_data_decode(expected[f"{prefix}/input"])
        if not bool(expected[f"{prefix}/parsed"]):
            molecules[value] = None
            continue
        symbols = _fragnet_data_decode(expected[f"{prefix}/raw/symbols"]).split("\0")
        if symbols == [""]:
            symbols = list[str]()
        atoms = [
            _ObservedAtom(expected, prefix=prefix, index=position, symbol=symbol)
            for position, symbol in enumerate(symbols)
        ]
        bonds = [
            _ObservedBond(expected, prefix=prefix, index=position)
            for position in range(len(expected[f"{prefix}/raw/bonds"]))
        ]
        molecules[value] = _ObservedMolecule(atoms, bonds=bonds)

    def parse(value: str) -> _ObservedMolecule | None:
        return molecules[value]

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(chem=types.SimpleNamespace(MolFromSmiles=parse))

    monkeypatch.setattr(fragnet_data, "MolecularContext", context)
    return expected


class _ObservedMolecule:
    def __init__(
        self,
        atoms: list[_ObservedAtom],
        *,
        bonds: list[_ObservedBond],
    ) -> None:
        self.atoms, self.bonds = atoms, bonds

    def GetAtoms(self) -> list[_ObservedAtom]:  # noqa: N802 -- Standard scientific-library boundary.
        return self.atoms

    def GetBonds(self) -> list[_ObservedBond]:  # noqa: N802 -- Standard scientific-library boundary.
        return self.bonds


class _ObservedAtom:
    def __init__(
        self,
        expected: dict[str, Tensor],
        *,
        prefix: str,
        index: int,
        symbol: str,
    ) -> None:
        self.expected, self.prefix, self.index, self.symbol = (
            expected,
            prefix,
            index,
            symbol,
        )

    def GetSymbol(self) -> str:  # noqa: N802 -- Standard scientific-library boundary.
        return self.symbol

    def GetTotalNumHs(self) -> int:  # noqa: N802 -- Standard scientific-library boundary.
        return int(self.expected[f"{self.prefix}/raw/hydrogens"][self.index])

    def GetDegree(self) -> int:  # noqa: N802 -- Standard scientific-library boundary.
        return int(self.expected[f"{self.prefix}/raw/degree"][self.index])

    def GetFormalCharge(self) -> int:  # noqa: N802 -- Standard scientific-library boundary.
        return int(self.expected[f"{self.prefix}/raw/charge"][self.index])

    def GetIsAromatic(self) -> bool:  # noqa: N802 -- Standard scientific-library boundary.
        return bool(self.expected[f"{self.prefix}/raw/aromatic"][self.index])


class _ObservedBond:
    def __init__(self, expected: dict[str, Tensor], *, prefix: str, index: int) -> None:
        self.expected, self.prefix, self.index = expected, prefix, index

    def GetBeginAtomIdx(self) -> int:  # noqa: N802 -- Standard scientific-library boundary.
        return int(self.expected[f"{self.prefix}/raw/bonds"][self.index, 0])

    def GetEndAtomIdx(self) -> int:  # noqa: N802 -- Standard scientific-library boundary.
        return int(self.expected[f"{self.prefix}/raw/bonds"][self.index, 1])

    def GetIsAromatic(self) -> bool:  # noqa: N802 -- Standard scientific-library boundary.
        return bool(self.expected[f"{self.prefix}/raw/bond_aromatic"][self.index])

    def IsInRing(self) -> bool:  # noqa: N802 -- Standard scientific-library boundary.
        return bool(self.expected[f"{self.prefix}/raw/bond_ring"][self.index])


def _fragnet_data_decode(value: Tensor) -> str:
    return bytes(from_plain(cast(object, value.tolist()), list[int])).decode()


def _fragnet_data_text(value: str) -> Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


# glacier_data_test.py.
def test_actual_glacier_physical_rows_and_writer_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atoms = [
        types.SimpleNamespace(
            GetSymbol=lambda: "C",
            GetIsotope=lambda: 13,
            GetNumImplicitHs=lambda: 0,
            GetNumExplicitHs=lambda: 2,
        ),
        types.SimpleNamespace(
            GetSymbol=lambda: "N",
            GetIsotope=lambda: 0,
            GetNumImplicitHs=lambda: 3,
            GetNumExplicitHs=lambda: 0,
        ),
    ]
    molecule = types.SimpleNamespace(GetAtoms=lambda: atoms)
    masses = {"C": 12.0, "N": 14.0, "H": 1.0}

    def most_common_mass(element: str) -> float:
        return masses.get(element, 20.0)

    def isotope_mass(element: str, isotope: int) -> float:
        del element, isotope
        return 13.0

    periodic = types.SimpleNamespace(
        GetMostCommonIsotopeMass=most_common_mass,
        GetMassForIsotope=isotope_mass,
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return molecule if value == "canonical" else None

    def kekulize(value: object, clear: bool) -> None:
        del value, clear

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        GetPeriodicTable=lambda: periodic,
        Kekulize=kekulize,
    )

    def context(version: str) -> types.SimpleNamespace:
        assert version == SIMULATOR_RDKIT
        return types.SimpleNamespace(chem=chemistry)

    def formula_string(value: object) -> str:
        del value
        return "CH5N"

    descriptors = types.ModuleType("rdkit.Chem.rdMolDescriptors")
    vars(descriptors)["CalcMolFormula"] = formula_string
    monkeypatch.setitem(sys.modules, descriptors.__name__, descriptors)
    monkeypatch.setattr(glacier_data, "MolecularContext", context)
    valid: AtomicRow = {
        "input_smiles": "requested",
        "smiles": "canonical",
        "valid": True,
        "features": [],
        "bonds": [],
        "bond_features": [],
        "weights": [],
        "degrees": [],
    }
    invalid: AtomicRow = {
        **valid,
        "input_smiles": "invalid",
        "smiles": "",
        "valid": False,
    }
    formula = [0.0] * 18
    formula[0] = formula[1] = 1.0
    formula[7] = 5.0
    atom_formulas = [[0.0] * 18 for _ in range(2)]
    atom_formulas[0][0] = atom_formulas[1][1] = 1.0
    atom_formulas[0][7], atom_formulas[1][7] = 2.0, 3.0
    expected: GlacierRow = {
        "input_smiles": "requested",
        "smiles": "canonical",
        "valid": True,
        "formula": formula,
        "atom_formulas": atom_formulas,
        "hydrogens": [2, 3],
        "atom_masses": [13.0, 14.0],
        "hydrogenated_masses": [15.0, 17.0],
        "total_hydrogens": 5,
    }
    rejected: GlacierRow = {
        "input_smiles": "invalid",
        "smiles": "",
        "valid": False,
        "formula": [],
        "atom_formulas": [],
        "hydrogens": [],
        "atom_masses": [],
        "hydrogenated_masses": [],
        "total_hydrogens": 0,
    }
    physical = glacier_data.GlacierChemistry(SIMULATOR_RDKIT)
    assert physical.row(valid) == expected
    assert physical.row(invalid) == rejected
    with pytest.raises(ValueError, match="Expected molecule is not None"):
        physical.row({**valid, "smiles": "unparsable"})

    def graphs(
        input_path: Path,
        *,
        output: Path,
        version: str,
        batch_rows: int,
    ) -> None:
        del input_path, version, batch_rows
        output.mkdir()
        source = Table.from_pylist([valid, invalid])
        with parquet.ParquetWriter(output / "graphs.parquet", source.schema) as writer:
            writer.write_table(source.slice(0, 1))
            writer.write_table(source.slice(1, 1))
        _ = (output / "manifest.json").write_text('{"format_version": 1}\n')

    monkeypatch.setattr(glacier_data, "prepare_graphs", graphs)
    request, output = tmp_path / "input.parquet", tmp_path / "glacier"
    parquet.write_table(
        Table.from_pydict({"smiles": ["requested", "invalid"]}),
        request,
    )
    glacier_data.prepare_glacier_graphs(
        request,
        output=output,
        version=SIMULATOR_RDKIT,
        batch_rows=1,
    )
    source = parquet.ParquetFile(output / "physical.parquet")
    assert source.num_row_groups == 2
    assert source.read().to_pylist() == [expected, rejected]
    np.testing.assert_array_equal(
        cast("NDArray[np.float64]", np.load(output / "element_masses.npy")),
        np.array([12.0, 14.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0] + [20.0] * 10)[
            :,
            None,
        ],
    )
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "rows", "valid_rows")
    } == {
        "format_version": 1,
        "rdkit": SIMULATOR_RDKIT,
        "rows": 2,
        "valid_rows": 1,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert (
        manifest["graph_manifest_sha256"]
        == hashlib.sha256((output / "graphs/manifest.json").read_bytes()).hexdigest()
    )
    assert from_plain(manifest["generator"], dict[str, str]) == {
        name: hashlib.sha256((_SCRIPTS_DIR / name).read_bytes()).hexdigest()
        for name in ("glacier_data.py", "graph_properties.py", "rdkit_backend.py")
    }
    for version, batch_rows in ((IDENTITY_RDKIT, 1), (SIMULATOR_RDKIT, 0)):
        with pytest.raises(ValueError, match=r"requires|positive"):
            glacier_data.prepare_glacier_graphs(
                request,
                output=tmp_path / "bad",
                version=version,
                batch_rows=batch_rows,
            )


# ice_data_test.py.
@dataclass(slots=True, kw_only=True)
class _Parameters:
    tautomerTransformsFile: str = ""  # noqa: N815 -- RDKit public cleanup parameter.


def test_ice_preparation_requires_the_simulator_version_and_positive_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"requires RDKit 2025\.03\.6"):
        prepare_ice_graphs(
            tmp_path / "request.parquet",
            output=tmp_path / "roots",
            version=IDENTITY_RDKIT,
        )
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        prepare_ice_graphs(
            tmp_path / "request.parquet",
            output=tmp_path / "roots",
            version=SIMULATOR_RDKIT,
            batch_rows=0,
        )


def test_canonical_root_data_retains_isotopes_hydrogens_and_complete_bond_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carbon = types.SimpleNamespace(
        GetSymbol=lambda: "C",
        GetIsotope=lambda: 13,
        GetNumImplicitHs=lambda: 2,
        GetNumExplicitHs=lambda: 1,
    )
    nitrogen = types.SimpleNamespace(
        GetSymbol=lambda: "N",
        GetIsotope=lambda: 0,
        GetNumImplicitHs=lambda: 1,
        GetNumExplicitHs=lambda: 2,
    )
    bonds = [_bond(kind) for kind in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")]
    molecule = types.SimpleNamespace(
        GetAtoms=lambda: [carbon, nitrogen],
        GetBonds=lambda: bonds,
    )
    events: list[str] = []

    def parse(value: str) -> object:
        events.append("parse:" + value)
        return None if value == "invalid" else molecule

    def remove_stereo(value: object) -> None:
        assert value is molecule
        events.append("remove_stereo")

    def inchi(value: object) -> str:
        assert value is molecule
        events.append("inchi")
        return "inchi"

    def canonicalize(value: object) -> object:
        events.append("tautomer")
        return value

    def kekulize(value: object, clear: bool) -> None:
        assert value is molecule
        assert clear
        events.append("kekulize")

    def standard_mass(element: str) -> float:
        return float(len(element))

    def isotope_mass(element: str, isotope: int) -> float:
        return 13.5 if (element, isotope) == ("C", 13) else 0.0

    def canonical_smiles(value: object) -> str:
        del value
        return "canonical"

    def from_inchi(value: str) -> types.SimpleNamespace:
        del value
        return molecule

    periodic = types.SimpleNamespace(
        GetMostCommonIsotopeMass=standard_mass,
        GetMassForIsotope=isotope_mass,
    )
    chem = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolToSmiles=canonical_smiles,
        MolToInchi=inchi,
        MolFromInchi=from_inchi,
        RemoveStereochemistry=remove_stereo,
        Kekulize=kekulize,
        GetPeriodicTable=lambda: periodic,
    )
    context = types.SimpleNamespace(chem=chem)

    def construct(version: str) -> types.SimpleNamespace:
        assert version == SIMULATOR_RDKIT
        return context

    monkeypatch.setattr(ice_data, "MolecularContext", construct)
    parameters = _Parameters()
    monkeypatch.setattr(ice_data, "CleanupParameters", lambda: parameters)

    def enumerator(value: object) -> types.SimpleNamespace:
        del value
        return types.SimpleNamespace(Canonicalize=canonicalize)

    monkeypatch.setattr(ice_data, "TautomerEnumerator", enumerator)

    def formula(value: object) -> str:
        del value
        return "CH6N+"

    monkeypatch.setattr(ice_data, "CalcMolFormula", formula)
    chemistry = ice_data.IceChemistry(SIMULATOR_RDKIT, directory=tmp_path)
    assert parameters.tautomerTransformsFile == str(
        tmp_path / "tautomer-transforms.txt",
    )
    np.testing.assert_array_equal(
        chemistry.element_masses,
        [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1],
    )
    row = chemistry.row("input")
    assert events == [
        "parse:input",
        "remove_stereo",
        "parse:canonical",
        "inchi",
        "tautomer",
        "parse:canonical",
        "kekulize",
    ]
    assert row["input_smiles"] == "input"
    assert row["smiles"] == "canonical"
    assert row["valid"]
    expected = np.zeros((2, 32))
    expected[0, [0, 22, 29]] = 1
    expected[1, [1, 21, 29]] = 1
    np.testing.assert_array_equal(row["features"], expected)
    assert row["formula"] == [1, 1, 0, 0, 0, 0, 0, 6, *([0] * 10)]
    assert row["bonds"] == [[0, 1]] * 4
    assert row["bond_orders"] == [1, 2, 3, 2]
    assert row["hydrogens"] == [3, 3]
    assert row["atom_masses"] == [13.5, 1.0]
    assert row["hydrogenated_masses"] == [16.5, 4.0]
    assert row["mass"] == 20.5
    assert not chemistry.row("invalid")["valid"]
    for stage in ("stereo", "inchi", "canonical"):
        calls = (
            iter((molecule, None))
            if stage == "stereo"
            else iter((molecule, molecule, None))
        )
        chem.MolFromSmiles = partial(_next_molecule, calls)
        chem.MolFromInchi = _missing_molecule if stage == "inchi" else from_inchi
        with pytest.raises(ValueError, match="molecule is not None"):
            _ = chemistry.row("input")


def test_batched_root_writer_retains_invalid_rows_empty_inputs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["invalid", "C", None, "C", "O"]}),
        request,
    )
    monkeypatch.setattr(ice_data, "IceChemistry", _Encoder)
    output = tmp_path / "roots"
    prepare_ice_graphs(request, output=output, version=SIMULATOR_RDKIT, batch_rows=2)
    source = parquet.ParquetFile(output / "ice.parquet")
    assert source.num_row_groups == 2
    assert source.read_row_group(0).column("input_smiles").to_pylist() == [
        "invalid",
        "C",
    ]
    assert source.read_row_group(1).column("input_smiles").to_pylist() == ["O"]
    assert source.read().column("valid").to_pylist() == [False, True, True]
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 3
    assert manifest["valid_rows"] == 2
    assert manifest["node_features"] == 32
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == SIMULATOR_RDKIT
    assert manifest["numpy"] == np.__version__
    assert len(from_plain(manifest["inputs"], dict[str, str])) == 1
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "ice_data.py",
        "rdkit_backend.py",
    }
    assert len(from_plain(manifest["tautomer_transforms_sha256"], str)) == 64
    np.testing.assert_array_equal(
        cast(
            "NDArray[np.generic]",
            np.load(output / "element_masses.npy", allow_pickle=False),
        ),
        np.arange(18)[:, None],
    )
    with pytest.raises(FileExistsError):
        prepare_ice_graphs(request, output=output, version=SIMULATOR_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    empty = tmp_path / "empty"
    prepare_ice_graphs(request, output=empty, version=SIMULATOR_RDKIT)
    assert parquet.read_table(empty / "ice.parquet").to_pylist() == []
    manifest = json_loads((empty / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == manifest["valid_rows"] == 0


def _bond(kind: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        GetBeginAtomIdx=lambda: 0,
        GetEndAtomIdx=lambda: 1,
        GetBondType=lambda: kind,
    )


def _next_molecule(values: Iterator[object], smiles: str) -> object:
    del smiles
    return next(values)


def _missing_molecule(inchi: str) -> None:
    del inchi


class _Encoder:
    def __init__(self, version: str, *, directory: Path) -> None:
        assert version == SIMULATOR_RDKIT
        _ = (directory / "tautomer-transforms.txt").write_text("test input data")
        self.element_masses = np.arange(18, dtype=np.float64)

    def row(self, smiles: str) -> IceRow:
        return {
            "input_smiles": smiles,
            "smiles": smiles,
            "valid": smiles != "invalid",
            "features": [],
            "bonds": [],
            "bond_orders": [],
            "formula": [],
            "hydrogens": [],
            "atom_masses": [],
            "hydrogenated_masses": [],
            "mass": 0.0,
        }


# intrinsic_test.py.
def test_capability_vocabulary_is_the_released_one() -> None:
    rules = capability_rules()
    assert [name for name, _ in rules] == [
        "oh_aliph",
        "cooh",
        "phenol",
        "nh2",
        "carbonyl",
        "ester",
        "acetyl",
        "ome",
        "ome_ar",
        "nme",
        "hex",
        "dhex",
        "pent",
        "glca",
        "cglyc",
        "malonyl",
        "coum",
        "caff",
        "ferul",
        "galloyl",
        "sulf",
        "phos",
    ]
    text = "\n".join(f"{name}\t{smarts}" for name, smarts in rules)
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == "9116486b2bde483380295eaa13049fe841361fd737ae4cdb7dbf6f21e36ac5d4"
    )


def test_actual_intrinsic_records_capability_caps_and_failure_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chemistry, record, capabilities = _intrinsic_boundary(monkeypatch)
    np.testing.assert_array_equal(chemistry.descriptors("valid"), record)
    np.testing.assert_array_equal(chemistry.capabilities("valid"), capabilities)
    for value in ("invalid", "exception"):
        np.testing.assert_array_equal(chemistry.descriptors(value), np.zeros(34))
        np.testing.assert_array_equal(chemistry.capabilities(value), np.zeros(22))
    for value in ("empty-scaffold", "missing-scaffold", "failed-scaffold"):
        empty = record.copy()
        for prefix, offset in (("S:", 0), ("G:", 2)):
            hashed = int.from_bytes(
                hashlib.blake2b(prefix.encode(), digest_size=6).digest(),
                "little",
            )
            empty[offset : offset + 2] = [hashed >> 24, hashed & 0xFF_FFFF]
        empty[6:8] = 0
        np.testing.assert_array_equal(chemistry.descriptors(value), empty)
    with pytest.raises(ValueError, match=r"requires RDKit 2026\.03\.3"):
        intrinsic.IntrinsicChemistry(SIMULATOR_RDKIT)


def test_intrinsic_writer_keeps_complete_arrays_vocabularies_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, record, capabilities = _intrinsic_boundary(monkeypatch)
    request, output = tmp_path / "input.parquet", tmp_path / "intrinsic"
    parquet.write_table(
        Table.from_pydict({"smiles": ["valid", "invalid", "valid", None]}),
        request,
    )
    intrinsic.prepare_intrinsic(request, output=output, version=IDENTITY_RDKIT)
    table = IntrinsicTable.load(output)
    assert table.smiles == ("valid", "invalid")
    np.testing.assert_array_equal(table.records, np.stack([record, np.zeros(34)]))
    np.testing.assert_array_equal(
        table.capabilities,
        np.stack([capabilities, np.zeros(22)]),
    )
    assert table.record_names == record_fields()
    assert table.capability_names == tuple(name for name, _ in capability_rules())
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {name: manifest[name] for name in ("format_version", "rdkit", "rows")} == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": 2,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(request.resolve()): hashlib.sha256(request.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "intrinsic.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
            _SCRIPTS_DIR.parent / "fragnet.py",
        )
    }
    with pytest.raises(FileExistsError):
        intrinsic.prepare_intrinsic(request, output=output, version=IDENTITY_RDKIT)
    parquet.write_table(Table.from_pydict({"smiles": [None]}), request)
    intrinsic.prepare_intrinsic(
        request,
        output=tmp_path / "empty",
        version=IDENTITY_RDKIT,
    )
    empty = IntrinsicTable.load(tmp_path / "empty")
    assert empty.smiles == ()
    assert empty.records.shape == (0, 34)
    assert empty.capabilities.shape == (0, 22)


def _intrinsic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[intrinsic.IntrinsicChemistry, NDArray[np.float32], NDArray[np.uint8]]:
    descriptor_counts = {
        query: index + 1
        for index, (_, query) in enumerate(intrinsic.descriptor_rules())
    }

    def matches(pattern: str, **kwargs: int) -> tuple[tuple[int, ...], ...]:
        return ((0,),) * (
            200 if kwargs.get("maxMatches") == 255 else descriptor_counts[pattern]
        )

    def parse(value: str) -> types.SimpleNamespace | None:
        if value == "exception":
            raise RuntimeError("scientific failure")
        if value == "invalid":
            return None
        return types.SimpleNamespace(
            name=value,
            GetNumHeavyAtoms=lambda: 4,
            GetAtoms=lambda: [
                types.SimpleNamespace(GetSymbol=lambda: "N", GetFormalCharge=lambda: 1),
                types.SimpleNamespace(
                    GetSymbol=lambda: "O",
                    GetFormalCharge=lambda: -1,
                ),
                types.SimpleNamespace(GetSymbol=lambda: "C", GetFormalCharge=lambda: 0),
                types.SimpleNamespace(GetSymbol=lambda: "C", GetFormalCharge=lambda: 0),
            ],
            GetRingInfo=lambda: types.SimpleNamespace(NumRings=lambda: 1),
            GetSubstructMatches=matches,
        )

    def scaffold(molecule: types.SimpleNamespace) -> types.SimpleNamespace | None:
        name = from_plain(cast(object, molecule.name), str)
        if name == "failed-scaffold":
            raise RuntimeError("scaffold failure")
        if name == "missing-scaffold":
            return None
        size = 0 if name == "empty-scaffold" else 3
        return types.SimpleNamespace(
            name="scaffold" if size else "",
            GetNumAtoms=lambda: size,
            GetNumHeavyAtoms=lambda: size,
        )

    def unchanged(value: str) -> str:
        return value

    def canonical(molecule: types.SimpleNamespace) -> str:
        return from_plain(cast(object, molecule.name), str)

    def generic(value: object) -> types.SimpleNamespace:
        del value
        return types.SimpleNamespace(name="generic")

    def descriptor(value: object, *, result: float) -> float:
        del value
        return result

    chemistry = types.SimpleNamespace(
        MolFromSmiles=parse,
        MolFromSmarts=unchanged,
        MolToSmiles=canonical,
    )

    def context(version: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(version=version, chem=chemistry)

    monkeypatch.setattr(intrinsic, "MolecularContext", context)
    modules: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "rdkit.Chem.rdMolDescriptors",
            {
                "CalcNumAromaticRings": partial(descriptor, result=1),
                "CalcTPSA": partial(descriptor, result=float("inf")),
                "CalcNumHBD": partial(descriptor, result=2),
                "CalcNumHBA": partial(descriptor, result=3),
                "CalcNumRotatableBonds": partial(descriptor, result=4),
                "CalcFractionCSP3": partial(descriptor, result=0.5),
            },
        ),
        ("rdkit.Chem.Crippen", {"MolLogP": partial(descriptor, result=float("nan"))}),
        (
            "rdkit.Chem.Scaffolds.MurckoScaffold",
            {
                "GetScaffoldForMol": scaffold,
                "MakeScaffoldGeneric": generic,
            },
        ),
    )
    for name, methods in modules:
        module = types.ModuleType(name)
        vars(module).update(methods)
        monkeypatch.setitem(sys.modules, name, module)
    expected = np.zeros(34, dtype=np.float32)
    for prefix, offset in (("S:scaffold", 0), ("G:generic", 2)):
        hashed = int.from_bytes(
            hashlib.blake2b(prefix.encode(), digest_size=6).digest(),
            "little",
        )
        expected[offset : offset + 2] = [hashed >> 24, hashed & 0xFF_FFFF]
    expected[4:10] = [1, 1, 3, 3 / 4, 1, 1]
    expected[10:19] = np.arange(1, 10)
    expected[19:30] = [1, 1, 0, 0, 0, 0, 0, 2, 3, 4, 0.5]
    expected[30:34] = [10, 11, 12, 4]
    capabilities = np.full(22, 200, dtype=np.uint8)
    capabilities[[10, 12, 14]] = 255
    return intrinsic.IntrinsicChemistry(IDENTITY_RDKIT), expected, capabilities


def test_chemistry_preserves_atom_order_and_rejects_unknown_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carbon = types.SimpleNamespace(
        GetIdx=lambda: 1,
        GetSymbol=lambda: "C",
        GetTotalNumHs=lambda: 3,
    )
    oxygen = types.SimpleNamespace(
        GetIdx=lambda: 0,
        GetSymbol=lambda: "O",
        GetTotalNumHs=lambda: 1,
    )
    molecule = types.SimpleNamespace(
        GetNumAtoms=lambda: 2,
        GetAtoms=lambda: (carbon, oxygen),
        GetBonds=lambda: (
            types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        ),
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return None if value == "invalid" else molecule

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(chem=types.SimpleNamespace(MolFromSmiles=parse))

    monkeypatch.setattr(main_fragments, "MolecularContext", context)
    chemistry = main_fragments.FragmentChemistry(IDENTITY_RDKIT)
    graph = chemistry.graph("input")
    assert graph is not None
    np.testing.assert_array_equal(
        graph[0],
        [15.9949146196 + 1.00782503207, 12.0 + 3 * 1.00782503207],
    )
    np.testing.assert_array_equal(graph[1], [[1, 0]])
    assert graph[1].dtype == np.int32
    assert chemistry.graph("invalid") is None
    hydrogen = types.SimpleNamespace(
        GetIdx=lambda: 2,
        GetSymbol=lambda: "H",
        GetTotalNumHs=lambda: 0,
    )
    molecule.GetNumAtoms = lambda: 3
    molecule.GetAtoms = lambda: (carbon, oxygen, hydrogen)
    molecule.GetBonds = lambda: (
        types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        types.SimpleNamespace(GetBeginAtomIdx=lambda: 0, GetEndAtomIdx=lambda: 2),
    )
    graph = chemistry.graph("two bonds")
    assert graph is not None
    np.testing.assert_array_equal(
        graph[0],
        [15.9949146196 + 1.00782503207, 12.0 + 3 * 1.00782503207, 1.00782503207],
    )
    np.testing.assert_array_equal(graph[1], [[1, 0], [0, 2]])
    assert graph[1].dtype == np.int32
    molecule.GetBonds = lambda: ()
    graph = chemistry.graph("no bonds")
    assert graph is not None
    assert graph[1].shape == (0, 2)
    assert graph[1].dtype == np.int32
    oxygen.GetSymbol = lambda: "unknown"
    assert chemistry.graph("unknown") is None
    with pytest.raises(ValueError, match="requires RDKit"):
        _ = main_fragments.FragmentChemistry(SIMULATOR_RDKIT)


def test_writer_retains_all_rows_bounded_batches_and_empty_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_fragments, "FragmentChemistry", _MainFragmentsGraphs)
    request, output = tmp_path / "input.parquet", tmp_path / "output"
    values = ["CC"] * 257 + ["invalid"]
    parquet.write_table(Table.from_pydict({"smiles": values}), request)
    main_fragments.prepare_main_fragments(
        request,
        output=output,
        version=IDENTITY_RDKIT,
    )
    monkeypatch.setitem(sys.modules, "rdkit", None)
    source = parquet.ParquetFile(output / "molecules.parquet")
    assert source.num_row_groups == 2
    assert source.read().column("smiles").to_pylist() == values
    assert source.read().column("valid").to_pylist() == [True] * 257 + [False]
    table = MainFragmentTable(output)
    for index in range(258):
        for row in table.row(index):
            np.testing.assert_array_equal(row, [1, 2, 3] if index < 257 else [])
    np.testing.assert_array_equal(table.base_for_smiles("CC"), [1, 2, 3])
    assert table.base_for_smiles("CC").dtype == np.float64
    assert len(table.base_for_smiles("invalid")) == 0
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 258
    assert manifest["depth_two_masses"] == manifest["depth_one_masses"] == 3 * 257
    assert manifest["base_masses"] == 3 * 257
    assert manifest["capacity"] == 200_000
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["numpy"] == np.__version__
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "main_fragments.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    assert len(from_plain(manifest["inputs"], dict[str, str])) == 1
    with pytest.raises(FileExistsError):
        main_fragments.prepare_main_fragments(
            request,
            output=output,
            version=IDENTITY_RDKIT,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), request)
    empty = tmp_path / "empty"
    main_fragments.prepare_main_fragments(request, output=empty, version=IDENTITY_RDKIT)
    assert parquet.read_table(empty / "molecules.parquet").to_pylist() == []
    for prefix in ("frag", "f1"):
        offsets = cast("NDArray[np.int64]", np.load(empty / f"{prefix}_off.npy"))
        masses = cast("NDArray[np.float32]", np.load(empty / f"{prefix}_mass.npy"))
        np.testing.assert_array_equal(offsets, [0])
        assert masses.shape == (0,)
    base = cast(
        "NDArray[np.float64]",
        np.load(empty / "base_mass.npy", allow_pickle=False),
    )
    assert base.shape == (0,)
    assert base.dtype == np.float64


def test_complete_rounded_fragment_rows_and_cap_encounter_order() -> None:
    golden = _SCRIPTS_DIR.parent / "testdata" / "main_fragments.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    for index in range(13):
        prefix = str(index)
        for name in ("input", "valid"):
            actual[f"{prefix}/{name}"] = expected[f"{prefix}/{name}"].clone()
        graph = None
        if bool(expected[f"{prefix}/valid"]):
            for name in ("weights", "bonds"):
                actual[f"{prefix}/{name}"] = expected[f"{prefix}/{name}"].clone()
            weights = cast("NDArray[np.float64]", actual[f"{prefix}/weights"].numpy())
            bonds = cast("NDArray[np.int32]", actual[f"{prefix}/bonds"].numpy())
            graph = weights, bonds
            for depth, capacity in ((0, 5), (1, 5), (2, 5), (2, 1)):
                actual[f"{prefix}/kernel/{depth}/{capacity}"] = torch.from_numpy(
                    fragment_masses(
                        weights,
                        bonds=bonds,
                        depth=depth,
                        capacity=capacity,
                    ),
                )
        for depth, row in zip((2, 1), fragment_rows(graph)[1:], strict=True):
            actual[f"{prefix}/depth{depth}"] = torch.from_numpy(row.astype(np.float64))
        actual[f"{prefix}/fragments"] = torch.from_numpy(base_fragment_masses(graph))
    assert_tensor_golden(golden, float32_comparands(actual, inputs=("*/weights",)))


def test_fragment_graph_preserves_weights_through_nonroot_bond_endpoints() -> None:
    masses = fragment_masses(
        np.array([1.0, 2.0, 4.0, 8.0]),
        bonds=np.array([[3, 2], [2, 1], [0, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(masses, [1, 2, 4, 6, 8, 9, 12, 13, 14, 15])


class _MainFragmentsGraphs:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def graph(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]] | None:
        return (
            None
            if smiles == "invalid"
            else (np.array([1.0, 2.0]), np.array([[0, 1]], dtype=np.int32))
        )


# manifest_test.py.
def test_file_digest_matches_the_whole_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc" * 100_000)
    assert sha256_file(path) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_inputs_are_keyed_by_resolved_path_and_sources_by_name(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    first, second = tmp_path / "sub" / "b.txt", tmp_path / "a.txt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    relative = tmp_path / "sub" / ".." / "a.txt"
    assert list(input_digests([first, relative])) == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert (
        input_digests([first])[str(first.resolve())]
        == hashlib.sha256(b"one").hexdigest()
    )
    assert source_digests([first, second]) == {
        "b.txt": hashlib.sha256(b"one").hexdigest(),
        "a.txt": hashlib.sha256(b"two").hexdigest(),
    }


def test_manifest_is_indented_json_with_a_trailing_newline(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"format_version": 1, "inputs": {"x": "y"}})
    assert (tmp_path / "manifest.json").read_text() == (
        '{\n  "format_version": 1,\n  "inputs": {\n    "x": "y"\n  }\n}\n'
    )


# prepare_data_test.py.
class Molecules:
    version = IDENTITY_RDKIT

    def __init__(self) -> None:
        self.standardized: list[tuple[str, bool]] = []

    def record(self, smiles: str) -> StructureRecord | None:
        if smiles == "invalid":
            return None
        values = np.zeros(scripts_prepare_data.RAW_BITS, dtype=np.uint8)
        values[[0, 4096, 8192, scripts_prepare_data.RAW_BITS - 1]] = 1
        return StructureRecord(
            smiles=smiles,
            formula="C2H6O",
            mass=46.041864812,
            heavy_atoms=3,
            key="engine",
            raw_key="raw",
            metric_key="metric",
            composition=(("C", 2), ("O", 1)),
            packed=np.packbits(values),
        )

    def standardize(self, smiles: str, *, keep_charged: bool = False) -> str | None:
        self.standardized.append((smiles, keep_charged))
        return None if smiles == "invalid" else smiles + "-standard"


def test_generated_assets_retain_every_input_and_full_fingerprint(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.parquet", tmp_path / "second.pkl"
    parquet.write_table(
        Table.from_pydict({"normalized_smiles": ["CCO", "invalid", None, "CCO"]}),
        first,
    )
    _ = second.write_bytes(pickle.dumps({"keys": ["A", "B"], "smiles": ["OCC", "CCO"]}))
    output = tmp_path / "generated"
    scripts_prepare_data.prepare(
        [first, second],
        output=output,
        chemistry=Molecules(),
    )
    table = ChemicalTable.load(output)
    assert table.input_smiles == ("CCO", "invalid", "OCC")
    assert table.smiles == ("CCO", "", "OCC")
    np.testing.assert_array_equal(table.valid, [True, False, True])
    np.testing.assert_array_equal(table.heavy_atoms, [3, 0, 3])
    assert table.formula == ("C2H6O", "", "C2H6O")
    assert table.keys == ("engine", "", "engine")
    assert table.raw_keys == ("raw", "", "raw")
    assert table.metric_keys == ("metric", "", "metric")
    assert table.metric_identity("OCC") == ("metric", (("C", 2), ("O", 1)))
    assert table.metric_identity("invalid") is None
    np.testing.assert_array_equal(
        table.mass,
        [46.041864812, float("nan"), 46.041864812],
    )
    np.testing.assert_array_equal(
        table.fingerprints(
            np.array([2, 1, 0]),
            np.array([0, 1, scripts_prepare_data.RAW_BITS - 1]),
        ),
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
    )
    assert table.packed.shape == (3, (scripts_prepare_data.RAW_BITS + 7) // 8)
    with pytest.raises(FileExistsError):
        scripts_prepare_data.prepare([first], output=output, chemistry=Molecules())
    chemistry = Molecules()
    output = tmp_path / "standardized"
    scripts_prepare_data.prepare(
        [first],
        output=output,
        chemistry=chemistry,
        standardize=True,
        keep_charged=True,
    )
    assert chemistry.standardized == [
        ("CCO", False),
        ("CCO", True),
        ("invalid", False),
        ("invalid", True),
    ]
    assert parquet.read_table(output / "structures.parquet").column(
        "standardized_smiles",
    ).to_pylist() == ["CCO-standard", None]
    assert parquet.read_table(output / "structures.parquet").column(
        "charged_smiles",
    ).to_pylist() == ["CCO-standard", None]
    assert ChemicalTable.load(output).smiles == ("CCO-standard", "")
    explicit = tmp_path / "explicit.parquet"
    parquet.write_table(Table.from_pydict({"structure": ["C"]}), explicit)
    scripts_prepare_data.prepare(
        [explicit],
        output=tmp_path / "explicit",
        chemistry=Molecules(),
        column="structure",
    )
    assert ChemicalTable.load(tmp_path / "explicit").input_smiles == ("C",)
    with pytest.raises(ValueError, match="A SMILES column is required"):
        scripts_prepare_data.prepare(
            [explicit],
            output=tmp_path / "missing",
            chemistry=Molecules(),
        )


def _parse(*argv: str) -> scripts_prepare_data.Flags:
    parser = argparse.ArgumentParser()
    scripts_prepare_data._add_arguments(parser)
    return cast(scripts_prepare_data.Flags, parser.parse_args(list(argv)))


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["prepare", *argv])
    return scripts_prepare_data.main()


def _usage_error(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    with pytest.raises(SystemExit) as error:
        _ = _run(monkeypatch, *argv)
    assert error.value.code == 2


def test_structures_command_keeps_every_input_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags = _parse("structures", "a.parquet", "b.pkl", "--output", "out")
    assert flags.inputs == [Path("a.parquet"), Path("b.pkl")]
    assert flags.output == Path("out")
    assert (flags.column, flags.version) == (None, IDENTITY_RDKIT)
    assert (flags.standardize, flags.keep_charged) == (False, False)
    source = tmp_path / "source.parquet"
    parquet.write_table(Table.from_pydict({"smiles": ["CCO"]}), source)

    def construct(version: str) -> Molecules:
        assert version == IDENTITY_RDKIT
        return Molecules()

    monkeypatch.setattr(scripts_prepare_data, "Chemistry", construct)
    assert (
        _run(monkeypatch, "structures", str(source), "--output", str(tmp_path / "o"))
        == 0
    )
    assert ChemicalTable.load(tmp_path / "o").input_smiles == ("CCO",)
    _usage_error(monkeypatch)
    _usage_error(monkeypatch, "structures", "--output", "out")
    _usage_error(monkeypatch, "structures", str(source))
    _usage_error(monkeypatch, "unknown", str(source), "--output", "out")


def test_simulator_commands_default_to_the_directories_their_readers_use() -> None:
    for command, directory in (
        ("admission", ADMISSION_DIR),
        ("ice", ICE_ROOT_DIR),
        ("glacier", GLACIER_ROOT_DIR),
    ):
        flags = _parse(command, "in.parquet")
        assert flags.output == Path(DATASET_ROOT) / directory
        assert flags.version == SIMULATOR_RDKIT
        assert flags.batch_rows == scripts_prepare_data.BATCH_ROWS
    for command, directory, config in (
        ("fragment-gap", FRAGMENT_GAP_DIR, GapGraphTable.Config()),
        ("identity", IDENTITY_DIR, IdentityTable.Config()),
    ):
        assert directory == config.working_dir
        flags = _parse(command, "in.parquet")
        assert flags.output == Path(DATASET_ROOT) / directory
        assert flags.version == IDENTITY_RDKIT
    for command in ("engine-identity", "pubchem"):
        flags = _parse(command, "in.parquet", "--output", "out")
        assert flags.version == IDENTITY_RDKIT
        with pytest.raises(SystemExit):
            _ = _parse(command, "in.parquet")
    graphs = _parse("graphs", "in.parquet", "--output", "out")
    assert (graphs.version, graphs.raw, graphs.batch_rows) == (
        SIMULATOR_RDKIT,
        False,
        256,
    )


def test_reactions_commands_dispatch_with_the_isolated_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def prepare(input_path: Path, **kwargs: object) -> None:
        calls.append((input_path, kwargs))

    def chemistry(version: str) -> str:
        assert version == IDENTITY_RDKIT
        return "isolated"

    monkeypatch.setattr(scripts_prepare_data, "prepare_reactions", prepare)
    monkeypatch.setattr(scripts_prepare_data, "MolecularContext", chemistry)
    assert _run(monkeypatch, "reactions", "requests.parquet", "--output", "out") == 0
    assert _run(monkeypatch, "reactions-v6", "v6.parquet", "--output", "out") == 0
    assert calls == [
        (Path("requests.parquet"), {"output": Path("out"), "chemistry": "isolated"}),
        (
            Path("v6.parquet"),
            {
                "output": Path("out"),
                "chemistry": "isolated",
                "rules": v6_reaction_rules(),
                "default_products": 300,
            },
        ),
    ]
    _usage_error(monkeypatch, "reactions", "requests.parquet")


def test_second_engine_commands_keep_explicit_prepared_asset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def prepare(input_path: Path, **kwargs: object) -> None:
        calls.append((input_path, kwargs))

    for name in (
        "prepare_second_library",
        "prepare_second_fingerprints",
        "prepare_second_pool",
    ):
        monkeypatch.setattr(scripts_prepare_data, name, prepare)
    for argv in (
        ("second-library", "train.parquet", "--pool", "pool.parquet"),
        ("second-fingerprints", "train.parquet", "--bits", "bits.npy"),
        (
            "second-pool",
            "train.parquet",
            "--fingerprints",
            "fingerprints",
            "--coconut",
            "coco",
            "--bio",
            "bio",
        ),
    ):
        assert _run(monkeypatch, *argv, "--output", "output") == 0
        assert calls[-1][0] == Path("train.parquet")
    assert [kwargs for _, kwargs in calls] == [
        {"pool": Path("pool.parquet"), "output": Path("output")},
        {"bits": Path("bits.npy"), "output": Path("output"), "version": IDENTITY_RDKIT},
        {
            "coconut": Path("coco"),
            "bio": Path("bio"),
            "fingerprints": Path("fingerprints"),
            "output": Path("output"),
        },
    ]
    _usage_error(monkeypatch, "second-library", "train.parquet", "--output", "o")
    _usage_error(monkeypatch, "second-pool", "train.parquet", "--output", "o")
    pool = scripts_prepare_data._provided("prvsiyan/coconut-casmi26-candidates")
    flags = _parse("second-pool", "t.parquet", "--fingerprints", "f", "--output", "o")
    assert flags.coconut == pool
    assert flags.bio == scripts_prepare_data._provided(
        "prvsiyan/chebi-lipidmaps-casmi26",
    )
    assert (
        _parse("second-fingerprints", "t.parquet", "--output", "o").bits
        == pool / "fp_bits.npy"
    )


def test_derivations_command_retains_explicit_pairs_and_identity_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def prepare(input_path: Path, **kwargs: object) -> None:
        calls.append((input_path, kwargs))

    monkeypatch.setattr(scripts_prepare_data, "prepare_derivations", prepare)
    argv = ("derivations", "requests.parquet", "--identities", "identities.parquet")
    assert _run(monkeypatch, *argv, "--output", "products") == 0
    assert calls == [
        (
            Path("requests.parquet"),
            {
                "identities": Path("identities.parquet"),
                "output": Path("products"),
                "version": IDENTITY_RDKIT,
            },
        ),
    ]
    _usage_error(monkeypatch, "derivations", "requests.parquet", "--output", "o")


def test_graphs_command_preserves_the_isolated_version_and_raw_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, str, bool, int]] = []

    def prepare(
        input_path: Path,
        *,
        output: Path,
        version: str,
        raw: bool,
        batch_rows: int,
    ) -> None:
        calls.append((input_path, output, version, raw, batch_rows))

    monkeypatch.setattr(scripts_prepare_data, "prepare_graphs", prepare)
    assert (
        _run(
            monkeypatch,
            "graphs",
            "graphs.parquet",
            "--output",
            "graphs",
            "--raw",
            "--batch-rows",
            "17",
        )
        == 0
    )
    assert calls == [
        (Path("graphs.parquet"), Path("graphs"), SIMULATOR_RDKIT, True, 17),
    ]


@pytest.mark.parametrize(
    ("command", "function", "version", "batched"),
    [
        ("descriptors", "prepare_intrinsic", IDENTITY_RDKIT, False),
        ("ice", "prepare_ice_graphs", SIMULATOR_RDKIT, True),
        ("glacier", "prepare_glacier_graphs", SIMULATOR_RDKIT, True),
        ("main-fragments", "prepare_main_fragments", IDENTITY_RDKIT, False),
        ("second-fragments", "prepare_second_fragments", IDENTITY_RDKIT, False),
        ("analog", "prepare_analog", IDENTITY_RDKIT, False),
        ("fragnet", "prepare_fragnet", IDENTITY_RDKIT, False),
        ("admission", "prepare_admission", SIMULATOR_RDKIT, True),
        ("pubchem", "prepare_pubchem", IDENTITY_RDKIT, False),
        ("fragment-gap", "prepare_fragment_gap", IDENTITY_RDKIT, True),
        ("identity", "prepare_identity", IDENTITY_RDKIT, True),
        ("engine-identity", "prepare_engine_identity", IDENTITY_RDKIT, True),
    ],
)
def test_every_scientific_command_dispatches_its_preparer_with_path_types(
    command: str,
    function: str,
    version: str,
    batched: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def prepare(input_path: Path, **kwargs: object) -> None:
        calls.append((input_path, kwargs))

    monkeypatch.setattr(scripts_prepare_data, function, prepare)
    assert _run(monkeypatch, command, "input.parquet", "--output", "out") == 0
    expected: dict[str, object] = {"output": Path("out"), "version": version}
    if batched:
        expected["batch_rows"] = 256
    assert calls == [(Path("input.parquet"), expected)]
    _usage_error(monkeypatch, command, "--output", "out")


# pubchem_data_test.py.
def test_pubchem_preparation_requires_the_exact_scientific_release() -> None:
    with pytest.raises(
        ValueError,
        match=r"^PubChem preparation requires RDKit 2026\.03\.3\.$",
    ):
        _ = PubChemChemistry(SIMULATOR_RDKIT)


def test_complete_primitive_records_and_independent_sdk_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubchem_data, "Chemistry", ObservedChemistry)
    monkeypatch.setattr(
        pubchem_data,
        "GenMACCSKeys",
        ObservedChemistry(IDENTITY_RDKIT).maccs_keys,
    )
    golden = _SCRIPTS_DIR.parent / "testdata/pubchem_preparation.pt"
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    chemistry = PubChemChemistry(IDENTITY_RDKIT)
    strings = [decode(expected[f"rows/smiles/{index}"]) for index in range(8)]
    parsed: list[bool] = []
    partial: list[NDArray[np.uint8]] = []
    valid: list[bool] = []
    full: list[NDArray[np.uint8]] = []
    keys: list[str | None] = []
    for value in strings:
        is_parsed, ecfp4, is_valid, raw_fp, key = chemistry.record(value)
        parsed.append(is_parsed)
        partial.append(ecfp4)
        valid.append(is_valid)
        full.append(raw_fp)
        keys.append(key)
    observe(
        actual,
        "rows",
        {
            "smiles": strings,
            "parsed": np.array(parsed, np.bool_),
            "raw_valid": np.array(valid, np.bool_),
            "ecfp4": np.stack(partial),
            "raw_fp": np.stack(full),
            "keys": keys,
        },
    )
    assert_tensor_golden(golden, actual)
    parsed_value, partial_value, valid_value, full_value, key = chemistry.record(
        "raw_error",
    )
    assert parsed_value
    assert not valid_value
    assert key == decode(expected["rows/keys/0"])
    np.testing.assert_array_equal(partial_value, expected["rows/ecfp4"][0].numpy())
    np.testing.assert_array_equal(full_value, np.zeros(2069, np.uint8))
    parsed_value, _, valid_value, full_value, key = chemistry.record("key_error")
    assert parsed_value
    assert valid_value
    assert key is None
    np.testing.assert_array_equal(full_value, expected["rows/raw_fp"][0].numpy())


def test_writer_retains_duplicate_rows_full_layouts_and_empty_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubchem_data, "Chemistry", ObservedChemistry)
    monkeypatch.setattr(
        pubchem_data,
        "GenMACCSKeys",
        ObservedChemistry(IDENTITY_RDKIT).maccs_keys,
    )
    expected = read_tensors(_SCRIPTS_DIR.parent / "testdata/pubchem_preparation.pt")
    source, destination = tmp_path / "input.parquet", tmp_path / "nested" / "prepared"
    smiles = ["CCO"] * 3 + ["invalid", "", "raw_error", "key_error"]
    parquet.write_table(
        Table.from_pydict(
            {"metadata": [42] * len(smiles), "normalized_smiles": smiles},
        ),
        source,
    )
    pubchem_data.prepare_pubchem(
        source,
        output=destination,
        version=IDENTITY_RDKIT,
        batch_rows=4,
    )
    metadata = parquet.ParquetFile(destination / "molecules.parquet")
    assert metadata.num_row_groups == 2
    assert metadata.read().column("smiles").to_pylist() == smiles
    table = PubChemTable(destination)
    for index in range(3):
        record = table.row(index)
        assert record.parsed
        assert record.raw_valid
        assert record.key == decode(expected["rows/keys/0"])
        np.testing.assert_array_equal(record.ecfp4, expected["rows/ecfp4"][0].numpy())
        np.testing.assert_array_equal(record.raw_fp, expected["rows/raw_fp"][0].numpy())
    assert not table.row(3).parsed
    assert table.row(4).parsed
    assert table.row(4).raw_valid
    assert table.row(4).key is None
    assert table.row(5).parsed
    assert not table.row(5).raw_valid
    assert table.row(6).raw_valid
    assert table.row(6).key is None
    manifest = json_loads(
        (destination / "manifest.json").read_text(),
        dict[str, object],
    )
    assert {
        name: manifest[name]
        for name in ("format_version", "rdkit", "rows", "raw_bits", "ecfp4_bits")
    } == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "rows": len(smiles),
        "raw_bits": 16_551,
        "ecfp4_bits": 4096,
    }
    assert manifest["numpy"] == np.__version__
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            _SCRIPTS_DIR / "pubchem_data.py",
            _SCRIPTS_DIR / "rdkit_backend.py",
        )
    }
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        pubchem_data.prepare_pubchem(
            source,
            output=tmp_path / "none",
            version=IDENTITY_RDKIT,
            batch_rows=0,
        )
    with pytest.raises(FileExistsError):
        pubchem_data.prepare_pubchem(source, output=destination, version=IDENTITY_RDKIT)
    with pytest.raises(ValueError, match="aliases protected input"):
        pubchem_data.prepare_pubchem(source, output=source, version=IDENTITY_RDKIT)
    for index, values in enumerate((list[str](), ["invalid"])):
        parquet.write_table(Table.from_pydict({"smiles": values}), source)
        output = tmp_path / f"empty-{index}"
        pubchem_data.prepare_pubchem(source, output=output, version=IDENTITY_RDKIT)
        loaded = PubChemTable(output)
        assert len(loaded.metadata) == len(values)
        assert loaded.ecfp4.shape == (len(values), 512)
        assert loaded.raw_fp.shape == (len(values), 2069)


@pytest.mark.parametrize("field", ["partial", "key"])
def test_partial_radius_and_full_upstream_key_are_independent_contracts(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = ObservedChemistry(IDENTITY_RDKIT)
    observed.raw[0, 4096] ^= 1

    def chemistry(version: str) -> ObservedChemistry:
        assert version == IDENTITY_RDKIT
        return observed

    def upstream_key(value: str) -> str:
        assert value == "CCO"
        return decode(observed.expected["rows/keys/0"]) + "-UHFFFAOYSA-N"

    observed.chem.MolToInchiKey = upstream_key
    monkeypatch.setattr(pubchem_data, "Chemistry", chemistry)
    monkeypatch.setattr(pubchem_data, "GenMACCSKeys", observed.maccs_keys)
    prepared = PubChemChemistry(IDENTITY_RDKIT)
    _, partial, _, _, key = prepared.record("CCO")
    if field == "partial":
        np.testing.assert_array_equal(
            partial,
            observed.expected["rows/ecfp4"][0].numpy(),
        )
    else:
        assert key == decode(observed.expected["rows/keys/0"])


class ObservedChemistry:
    """Replay every full source SDK block through the native standard-library adapter."""

    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT
        self.expected = read_tensors(
            _SCRIPTS_DIR.parent / "testdata/pubchem_preparation.pt",
        )
        self.raw = np.unpackbits(self.expected["rows/raw_fp"].numpy(), axis=1)
        self.positions = {
            decode(self.expected[f"rows/smiles/{index}"]): index for index in range(8)
        }
        self.positions.update(raw_error=0, key_error=0)
        self.chem = types.SimpleNamespace(
            MolFromSmiles=self.parse,
            MolToInchiKey=self.key,
        )
        self.tautomers = types.SimpleNamespace(Canonicalize=self.canonicalize)
        self.generators = tuple(
            types.SimpleNamespace(
                GetFingerprintAsNumPy=ObservedFingerprint(
                    self,
                    index=index,
                ).fingerprint,
            )
            for index in range(6)
        )

    def parse(self, value: str) -> str | None:
        index = self.positions[value]
        return value if bool(self.expected["rows/parsed"][index]) else None

    def canonicalize(self, value: str) -> str:
        return value

    def key(self, value: str) -> str:
        if value == "key_error":
            raise OSError("independent key error")
        index = self.positions[value]
        return (
            decode(self.expected[f"rows/keys/{index}"])
            if f"rows/keys/{index}/none" not in self.expected
            else ""
        )

    def maccs_keys(self, value: str) -> list[int]:
        if value == "raw_error":
            raise OSError("independent raw-fingerprint error")
        return from_plain(
            cast(
                object,
                self.raw[self.positions[value], 16_384:16_551].tolist(),
            ),
            list[int],
        )


class ObservedFingerprint:
    def __init__(self, owner: ObservedChemistry, *, index: int) -> None:
        self.owner, self.index = owner, index

    def fingerprint(self, value: str) -> NDArray[np.uint8]:
        bounds = (0, 4096, 8192, 10_240, 12_288, 14_336, 16_384)
        row = self.owner.positions[value]
        return self.owner.raw[row, bounds[self.index] : bounds[self.index + 1]]


# ragged_test.py.
def _load(path: Path) -> NDArray[np.generic]:
    return cast("NDArray[np.generic]", np.load(path, allow_pickle=False))


def test_vectors_and_matrices_round_trip_with_their_dtype(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.int32)
    with tempfile.TemporaryFile() as stream:
        values.tofile(stream)
        save_stream(stream, tmp_path / "vector.npy", dtype=np.int32)
        save_stream(stream, tmp_path / "matrix.npy", dtype=np.int32, width=2)
        save_stream(stream, tmp_path / "wide.npy", dtype=np.int32, width=3)
    vector = _load(tmp_path / "vector.npy")
    assert vector.dtype == np.int32
    np.testing.assert_array_equal(vector, values)
    np.testing.assert_array_equal(
        _load(tmp_path / "matrix.npy"),
        values.reshape(3, 2),
    )
    assert _load(tmp_path / "wide.npy").shape == (2, 3)


def test_an_empty_stream_saves_an_empty_array_of_the_declared_shape(
    tmp_path: Path,
) -> None:
    with tempfile.TemporaryFile() as stream:
        save_stream(stream, tmp_path / "vector.npy", dtype=np.float64)
        save_stream(stream, tmp_path / "matrix.npy", dtype=np.int8, width=2)
    vector = _load(tmp_path / "vector.npy")
    matrix = _load(tmp_path / "matrix.npy")
    assert (vector.shape, vector.dtype) == ((0,), np.float64)
    assert (matrix.shape, matrix.dtype) == ((0, 2), np.int8)


# second_fragments_test.py.


def test_fixed_masses_keep_atom_indices_empty_roots_and_unknown_atoms_with_hydrogen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carbon = types.SimpleNamespace(
        GetIdx=lambda: 1,
        GetSymbol=lambda: "C",
        GetTotalNumHs=lambda: 3,
    )
    exotic = types.SimpleNamespace(
        GetIdx=lambda: 0,
        GetSymbol=lambda: "Xe",
        GetTotalNumHs=lambda: 1,
    )
    molecule = types.SimpleNamespace(
        GetNumAtoms=lambda: 2,
        GetAtoms=lambda: (carbon, exotic),
        GetBonds=lambda: (
            types.SimpleNamespace(GetBeginAtomIdx=lambda: 1, GetEndAtomIdx=lambda: 0),
        ),
    )

    def parse(value: str) -> types.SimpleNamespace | None:
        return molecule if value != "invalid" else None

    chem = types.SimpleNamespace(MolFromSmiles=parse)

    def context(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(chem=chem)

    monkeypatch.setattr(second_fragments, "MolecularContext", context)
    chemistry = second_fragments.FragmentChemistry(IDENTITY_RDKIT)
    graph = chemistry.graph("input")
    assert graph is not None
    weights, bonds = graph
    np.testing.assert_array_equal(weights, [1.00782503207, 12.0 + 3 * 1.00782503207])
    np.testing.assert_array_equal(bonds, [[1, 0]])
    assert chemistry.graph("invalid") is None
    exotic.GetTotalNumHs = lambda: 0
    assert chemistry.graph("zero") is None
    with pytest.raises(ValueError, match="require RDKit"):
        _ = second_fragments.FragmentChemistry(SIMULATOR_RDKIT)


@pytest.mark.parametrize(
    ("edges", "breaks", "limit", "expected"),
    [
        ([], 2, 34, [7.0]),
        ([[0, 1], [1, 2]], 2, 1, [7.0]),
        ([[0, 1], [1, 2]], 1, 34, [1.0, 3.0, 4.0, 6.0, 7.0]),
        ([[0, 1], [1, 2]], 2, 34, [1.0, 2.0, 3.0, 4.0, 6.0, 7.0]),
        ([[0, 1], [1, 2], [2, 0]], 1, 34, [7.0]),
        ([[0, 1], [1, 2], [2, 0]], 2, 34, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ([[0, 1]], 2, 34, [1.0, 2.0, 4.0, 7.0]),
    ],
)
def test_complete_cut_mass_unions_and_parent_only_fallback(
    edges: list[list[int]],
    breaks: int,
    limit: int,
    expected: list[float],
) -> None:
    actual = second_fragments_fragment_masses(
        np.array([1.0, 2.0, 4.0]),
        bonds=np.array(edges, dtype=np.int64).reshape(-1, 2),
        max_breaks=breaks,
        max_bonds=limit,
    )
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.float64))
    assert actual.dtype == np.float64


def test_fragment_writer_preserves_duplicates_bounded_batches_and_empty_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(second_fragments, "FragmentChemistry", _SecondFragmentsGraphs)
    request, output = tmp_path / "input.parquet", tmp_path / "output"
    values = ["CC"] * 257 + ["invalid"]
    parquet.write_table(Table.from_pydict({"smiles": values}), request)
    second_fragments.prepare_second_fragments(
        request,
        output=output,
        version=IDENTITY_RDKIT,
    )
    source = parquet.ParquetFile(output / "graphs.parquet")
    assert source.num_row_groups == 2
    assert source.read().column("smiles").to_pylist() == values
    assert source.read().column("valid").to_pylist() == [True] * 257 + [False]
    offsets = cast(
        "NDArray[np.int64]",
        np.load(output / "frag_off.npy", allow_pickle=False),
    )
    masses = cast(
        "NDArray[np.float64]",
        np.load(output / "frag_mass.npy", allow_pickle=False),
    )
    np.testing.assert_array_equal(offsets, [*range(0, 3 * 257 + 1, 3), 3 * 257])
    np.testing.assert_array_equal(masses, np.tile([1.0, 2.0, 3.0], 257))
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 258
    assert manifest["masses"] == 3 * 257
    assert manifest["max_bonds"] == 34
    assert manifest["max_breaks"] == 2
    assert manifest["format_version"] == 1
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["numpy"] == np.__version__
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "second_fragments.py",
        "rdkit_backend.py",
    }
    assert len(from_plain(manifest["inputs"], dict[str, str])) == 1
    with pytest.raises(FileExistsError):
        second_fragments.prepare_second_fragments(
            request,
            output=output,
            version=IDENTITY_RDKIT,
        )
    parquet.write_table(Table.from_pydict({"smiles": []}), request)
    output = tmp_path / "empty"
    second_fragments.prepare_second_fragments(
        request,
        output=output,
        version=IDENTITY_RDKIT,
    )
    assert parquet.read_table(output / "graphs.parquet").to_pylist() == []
    np.testing.assert_array_equal(
        cast("NDArray[np.int64]", np.load(output / "frag_off.npy", allow_pickle=False)),
        [0],
    )
    assert (
        len(
            cast(
                "NDArray[np.float64]",
                np.load(output / "frag_mass.npy", allow_pickle=False),
            ),
        )
        == 0
    )


class _SecondFragmentsGraphs:
    def __init__(self, version: str) -> None:
        assert version == IDENTITY_RDKIT

    def graph(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
        return (
            None
            if smiles == "invalid"
            else (np.array([1.0, 2.0]), np.array([[0, 1]], dtype=np.int64))
        )


# second_library_test.py.
def test_complete_raw_records_and_first_duplicate_pool_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    formulas = ["C2H6O", None, "Xe", "", "D2O", "junk"]
    columns: dict[str, object] = {
        "inchikey14": ["a", None, "b", "a", "c", "missing"],
        "normalized_smiles": ["CCO", "N", "[Xe]", "CCO", None, "C"],
        "adduct": ["[M+H]+", "[M-H]-", "unknown", "[2M+Na]+", None, "[M]+"],
        "precursor_mz": [47.0, 18.0, 131.0, 94.0, 21.0, 17.0],
        "ionization_mode": [
            "positive",
            "negative",
            None,
            "POSITIVE",
            "pos",
            "positive",
        ],
        "ms2_mzs": [[-2, 0, 30, 30, 101], [], [20, 10], [200], [4, 5, 6], [1000]],
        "ms2_normalized_intensities": [
            [-1, 0, 4, 9, 0.0001],
            [],
            [0.5, 0.25],
            [1],
            [3, 2, 1],
            [float("nan")],
        ],
        "molecular_formula": formulas,
        "instrument_type": [
            "timsTOF",
            "TIMS- TOF",
            None,
            "Orbitrap",
            " tims tof ",
            "Q-TOF",
        ],
        "extra_metadata": [[1, 2], None, [], [5], [6, 7], [8]],
    }
    source, pool, output = (
        tmp_path / "train.parquet",
        tmp_path / "pool.parquet",
        tmp_path / "nested" / "prepared",
    )
    original = Table.from_pydict(columns)
    parquet.write_table(original, source, row_group_size=2)
    parquet.write_table(Table.from_pydict({"key": ["b", "a", "a", "unused"]}), pool)
    original_batches = parquet.ParquetFile.iter_batches
    original_read = parquet.read_table

    def bounded_batches(
        self: parquet.ParquetFile,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> object:
        assert 0 < batch_size <= 256
        return original_batches(self, batch_size=batch_size, columns=columns)

    def selected_columns(path: Path, *, columns: Sequence[str] | None = None) -> Table:
        if path == pool:
            assert columns == ["key"]
        return original_read(path, columns=columns)

    monkeypatch.setattr(parquet.ParquetFile, "iter_batches", bounded_batches)
    monkeypatch.setattr(parquet, "read_table", selected_columns)
    prepare_second_library(source, pool=pool, output=output)
    mass = ELEMENT_MASSES
    for name, expected in (
        ("off", np.array([0, 5, 5, 7, 8, 11, 12], np.int64)),
        (
            "mz",
            np.array([-2, 0, 30, 30, 101, 20, 10, 200, 4, 5, 6, 1000], np.float32),
        ),
        (
            "it",
            np.array(
                [-1, 0, 4, 9, 0.0001, 0.5, 0.25, 1, 3, 2, 1, math.nan],
                np.float32,
            ),
        ),
        ("prec", np.array([47.0, 18.0, 131.0, 94.0, 21.0, 17.0])),
        (
            "nm",
            np.array(
                [
                    mass["C"] * 2 + mass["H"] * 6 + mass["O"],
                    18.0 + 1.00727645217,
                    math.nan,
                    (94.0 - 22.989220701) / 2,
                    mass["D"] * 2 + mass["O"],
                    0.0,
                ],
            ),
        ),
        ("pol", np.array([1, -1, -1, -1, -1, 1], np.int8)),
        ("tims", np.array([True, True, False, False, True, False])),
    ):
        actual = cast(
            "NDArray[np.generic]",
            np.load(output / f"{name}.npy", allow_pickle=False),
        )
        assert actual.dtype == expected.dtype, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    metadata = parquet.read_table(output / "metadata.parquet")
    assert metadata.num_rows == original.num_rows
    assert metadata.schema.names == [
        name
        for name in original.schema.names
        if name not in {"ms2_mzs", "ms2_normalized_intensities"}
    ]
    assert (
        metadata.to_pylist()
        == original.drop("ms2_mzs").drop("ms2_normalized_intensities").to_pylist()
    )
    np.testing.assert_array_equal(
        [
            formula_mass(value)
            for value in (*formulas, "C+H-2", "C(OH)2", "C0H0", "H3C2H3", "+")
        ],
        [
            mass["C"] * 2 + mass["H"] * 6 + mass["O"],
            math.nan,
            math.nan,
            math.nan,
            mass["D"] * 2 + mass["O"],
            0.0,
            mass["C"] + mass["H"],
            mass["C"] + mass["O"] + mass["H"],
            0.0,
            mass["H"] * 3 + mass["C"] * 2 + mass["H"] * 3,
            0.0,
        ],
    )
    assert formula_mass("Cl2BrFI") == (
        mass["Cl"] * 2 + mass["Br"] + mass["F"] + mass["I"]
    )
    assert formula_mass("AsFeSeSiNaKB") == (
        mass["As"]
        + mass["Fe"]
        + mass["Se"]
        + mass["Si"]
        + mass["Na"]
        + mass["K"]
        + mass["B"]
    )
    row_p = cast("NDArray[np.int64]", np.load(output / "row_p.npy", allow_pickle=False))
    assert row_p.dtype == np.int64
    np.testing.assert_array_equal(row_p, [1, -1, 0, 1, -1, -1])
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["format_version"] == 1
    assert manifest["numpy"] == np.__version__
    assert manifest["rows"] == 6
    assert manifest["peaks"] == 12
    assert manifest["raw_peaks"] is True
    inputs = from_plain(manifest["inputs"], dict[str, str])
    assert inputs == {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, pool)
    }
    assert from_plain(manifest["generator"], dict[str, str]) == {
        "second_library.py": hashlib.sha256(
            (_SCRIPTS_DIR / "second_library.py").read_bytes(),
        ).hexdigest(),
    }
    with pytest.raises(FileExistsError):
        prepare_second_library(source, pool=pool, output=output)
    for protected in (source, pool):
        with pytest.raises(ValueError, match="aliases protected input"):
            prepare_second_library(source, pool=pool, output=protected)


def test_empty_library_and_misaligned_peak_lists(tmp_path: Path) -> None:
    source, pool = tmp_path / "train.parquet", tmp_path / "pool.parquet"
    parquet.write_table(Table.from_pydict({"key": []}), pool)
    values: dict[str, object] = {
        name: []
        for name in (
            "inchikey14",
            "normalized_smiles",
            "adduct",
            "precursor_mz",
            "ionization_mode",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "molecular_formula",
            "instrument_type",
        )
    }
    parquet.write_table(Table.from_pydict(values), source)
    output = tmp_path / "empty"
    prepare_second_library(source, pool=pool, output=output)
    for name in ("mz", "it", "prec", "nm", "pol", "tims", "row_p"):
        assert cast(
            "NDArray[np.generic]",
            np.load(output / f"{name}.npy", allow_pickle=False),
        ).shape == (0,)
    np.testing.assert_array_equal(
        cast("NDArray[np.int64]", np.load(output / "off.npy", allow_pickle=False)),
        [0],
    )
    assert parquet.read_table(output / "metadata.parquet").num_rows == 0
    malformed = Table.from_pydict(
        dict(
            values,
            ms2_mzs=[[1, 2]],
            ms2_normalized_intensities=[[1]],
            **{
                name: [None]
                for name in values
                if name not in {"ms2_mzs", "ms2_normalized_intensities"}
            },
        ),
    )
    parquet.write_table(malformed, source)
    with pytest.raises(ValueError, match=r"^Peak mass and intensity lengths differ\.$"):
        prepare_second_library(source, pool=pool, output=tmp_path / "bad")


# second_pool_test.py.


def test_four_block_boundaries_invalid_rows_and_fingerprint_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parse(value: str) -> str | None:
        return None if value == "invalid" else value

    def first(value: object) -> NDArray[np.uint8]:
        if value == "fp-error":
            raise RuntimeError("scientific fingerprint failure")
        result = np.zeros(4096, dtype=np.uint8)
        result[0] = 1
        return result

    def second(value: object) -> NDArray[np.uint8]:
        del value
        result = np.zeros(4096, dtype=np.uint8)
        result[-1] = 1
        return result

    def third(value: object) -> NDArray[np.uint8]:
        del value
        result = np.zeros(2048, dtype=np.uint8)
        result[[0, -1]] = 1
        return result

    def maccs(value: object) -> list[int]:
        del value
        return [0] * 166 + [1]

    def mass(value: object) -> float:
        if value == "broken":
            raise ValueError("scientific mass failure")
        return 42.125

    def chemistry(version: str) -> types.SimpleNamespace:
        assert version == IDENTITY_RDKIT
        return types.SimpleNamespace(
            chem=types.SimpleNamespace(MolFromSmiles=parse),
            generators=(
                types.SimpleNamespace(GetFingerprintAsNumPy=first),
                types.SimpleNamespace(GetFingerprintAsNumPy=second),
                None,
                types.SimpleNamespace(GetFingerprintAsNumPy=third),
            ),
        )

    monkeypatch.setattr(second_pool, "Chemistry", chemistry)
    monkeypatch.setattr(second_pool, "GenMACCSKeys", maccs)
    monkeypatch.setattr(second_pool, "ExactMolWt", mass)
    request, bits, output = (
        tmp_path / "input.parquet",
        tmp_path / "bits.npy",
        tmp_path / "nested" / "cache",
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "normalized_smiles": [
                    "good",
                    None,
                    "invalid",
                    "good",
                    "broken",
                    "fp-error",
                    "",
                ],
            },
        ),
        request,
    )
    np.save(
        bits,
        np.array(
            [10_406, 0, 4095, 4096, 8191, 8192, 10_239, 10_240, 12],
            dtype=np.int64,
        ),
    )
    second_pool.prepare_second_fingerprints(
        request,
        bits=bits,
        output=output,
        version=IDENTITY_RDKIT,
    )
    table = parquet.read_table(output / "fingerprints.parquet")
    assert table.column("smiles").to_pylist() == [
        "good",
        "invalid",
        "broken",
        "fp-error",
        "",
    ]
    assert table.column("valid").to_pylist() == [True, False, False, False, True]
    np.testing.assert_array_equal(
        np.array(
            from_plain(cast(object, table.column("mass").to_pylist()), list[float]),
        ),
        [42.125, float("nan"), float("nan"), float("nan"), 42.125],
    )
    values = cast(
        "NDArray[np.uint8]",
        np.load(output / "fingerprints.npy", allow_pickle=False),
    )
    assert values.dtype == np.uint8
    np.testing.assert_array_equal(values, [[206, 0], [0, 0], [0, 0], [0, 0], [206, 0]])
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert {
        key: manifest[key]
        for key in (
            "format_version",
            "rdkit",
            "numpy",
            "raw_bits",
            "rows",
            "selected_bits",
        )
    } == {
        "format_version": 1,
        "rdkit": IDENTITY_RDKIT,
        "numpy": np.__version__,
        "raw_bits": 10_407,
        "rows": 5,
        "selected_bits": 9,
    }
    assert from_plain(manifest["inputs"], dict[str, str]) == {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (request, bits)
    }
    assert set(from_plain(manifest["generator"], dict[str, str])) == {
        "second_pool.py",
        "rdkit_backend.py",
        "chemistry.py",
    }
    with pytest.raises(FileExistsError):
        second_pool.prepare_second_fingerprints(
            request,
            bits=bits,
            output=output,
            version=IDENTITY_RDKIT,
        )
    with pytest.raises(
        ValueError,
        match=r"^Second-engine fingerprints require RDKit 2026\.03\.3\.$",
    ):
        second_pool.prepare_second_fingerprints(
            request,
            bits=bits,
            output=tmp_path / "bad-version",
            version=SIMULATOR_RDKIT,
        )
    for selected in (
        np.array([], dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
        np.array([0, 1], dtype=np.int32),
        np.array([-1, 0]),
        np.array([0, 10_407]),
    ):
        np.save(bits, selected)
        with pytest.raises(
            ValueError,
            match=r"^Selected bits require an int64 vector within the four-block layout\.$",
        ):
            second_pool.prepare_second_fingerprints(
                request,
                bits=bits,
                output=tmp_path / "bad-bits",
                version=IDENTITY_RDKIT,
            )


def test_pool_refuses_missing_chemistry_and_misaligned_provided_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    training, fingerprint_dir = tmp_path / "train.parquet", tmp_path / "fingerprints"
    fingerprint_dir.mkdir()
    parquet.write_table(
        Table.from_pydict(
            {"inchikey14": ["new", None], "normalized_smiles": ["unprepared", "C"]},
        ),
        training,
    )
    for prefix in ("coco", "bio"):
        directory = tmp_path / prefix
        directory.mkdir()
        _ = (directory / f"{prefix}_meta.pkl").write_bytes(
            pickle.dumps({"keys": ["a", "b"], "smiles": ["C", "O"]}),
        )
        np.save(directory / f"{prefix}_mass.npy", np.array([16.0, 18.0]))
        # prepare_second_pool packs the three selected bits into one byte.
        np.save(directory / f"{prefix}_fp.npy", np.zeros((2, 1), dtype=np.uint8))
    parquet.write_table(
        Table.from_pydict(
            {"smiles": ["C", "O"], "valid": [True, True], "mass": [16.0, 18.0]},
        ),
        fingerprint_dir / "fingerprints.parquet",
    )
    np.save(fingerprint_dir / "fp_bits.npy", np.arange(3, dtype=np.int64))
    # prepare_second_pool requires the same one-byte packed cache width.
    np.save(fingerprint_dir / "fingerprints.npy", np.zeros((2, 1), dtype=np.uint8))
    metadata = {"format_version": 1, "rdkit": IDENTITY_RDKIT}
    _ = (fingerprint_dir / "manifest.json").write_text(json.dumps(to_plain(metadata)))
    with pytest.raises(KeyError, match="unprepared"):
        prepare_second_pool(
            training,
            coconut=tmp_path / "coco",
            bio=tmp_path / "bio",
            fingerprints=fingerprint_dir,
            output=tmp_path / "missing",
        )
    for label, path, malformed, message in (
        (
            "cached",
            fingerprint_dir / "fingerprints.npy",
            # prepare_second_pool rejects two bytes for three selected bits.
            np.zeros((2, 2), dtype=np.uint8),
            r"^Fingerprint cache rows or packed width differ from metadata\.$",
        ),
        (
            "provided",
            tmp_path / "coco/coco_fp.npy",
            # prepare_second_pool rejects the same incorrect provided width.
            np.zeros((2, 2), dtype=np.uint8),
            r"^Provided fingerprint rows or packed width differ from metadata\.$",
        ),
        (
            "mass",
            tmp_path / "coco/coco_mass.npy",
            np.ones(3),
            r"^Provided mass rows differ from metadata\.$",
        ),
    ):
        correct = cast("NDArray[np.generic]", np.load(path, allow_pickle=False))
        np.save(path, malformed)
        with pytest.raises(ValueError, match=message):
            prepare_second_pool(
                training,
                coconut=tmp_path / "coco",
                bio=tmp_path / "bio",
                fingerprints=fingerprint_dir,
                output=tmp_path / label,
            )
        np.save(path, correct)
    metadata["format_version"] = 2
    _ = (fingerprint_dir / "manifest.json").write_text(json.dumps(to_plain(metadata)))
    with pytest.raises(
        ValueError,
        match=r"^Fingerprint cache requires format 1 and RDKit 2026\.03\.3\.$",
    ):
        prepare_second_pool(
            training,
            coconut=tmp_path / "coco",
            bio=tmp_path / "bio",
            fingerprints=fingerprint_dir,
            output=tmp_path / "version",
        )
    parquet.write_table(
        Table.from_pydict({"inchikey14": ["a"], "normalized_smiles": ["C"]}),
        training,
    )
    metadata["format_version"] = True
    _ = (fingerprint_dir / "manifest.json").write_text(json.dumps(to_plain(metadata)))
    with pytest.raises(ValueError, match="cannot read True as int"):
        prepare_second_pool(
            training,
            coconut=tmp_path / "coco",
            bio=tmp_path / "bio",
            fingerprints=fingerprint_dir,
            output=tmp_path / "boolean-version",
        )


def test_pool_stable_ties_readonly_fingerprints_and_bounded_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    inputs = {name: tmp_path / name for name in ("coconut", "bio", "fingerprints")}
    inputs["input_path"] = tmp_path / "train.parquet"
    for name in ("coconut", "bio", "fingerprints"):
        inputs[name].mkdir()
    count = 80
    masses = [float(index % 3) for index in range(count)]
    for name, prefix, keys, mass, values in (
        (
            "coconut",
            "coco",
            [f"c{index}" for index in range(count)],
            masses,
            np.arange(count, dtype=np.uint8).reshape(-1, 1),
        ),
        (
            "bio",
            "bio",
            ["b", "b"],
            [1.0, 1.0],
            np.array([[201], [202]], dtype=np.uint8),
        ),
    ):
        directory = inputs[name]
        _ = (directory / f"{prefix}_meta.pkl").write_bytes(
            pickle.dumps({"keys": keys, "smiles": keys}),
        )
        np.save(directory / f"{prefix}_mass.npy", mass)
        np.save(directory / f"{prefix}_fp.npy", values)
    fingerprints = inputs["fingerprints"]
    np.save(fingerprints / "fp_bits.npy", np.arange(8, dtype=np.int64))
    np.save(fingerprints / "fingerprints.npy", np.array([[203]], dtype=np.uint8))
    parquet.write_table(
        Table.from_pydict({"smiles": ["train"], "valid": [True], "mass": [1.0]}),
        fingerprints / "fingerprints.parquet",
    )
    _ = (fingerprints / "manifest.json").write_text(
        json.dumps(to_plain({"format_version": 1, "rdkit": IDENTITY_RDKIT})),
    )
    parquet.write_table(
        Table.from_pydict(
            {
                "unused": [False] * 258,
                "inchikey14": ["c0"] * 257 + ["new"],
                "normalized_smiles": ["c0"] * 257 + ["train"],
            },
        ),
        inputs["input_path"],
    )
    file_type, load = parquet.ParquetFile, np.load
    reads: list[Path] = []

    def open_training(path: Path) -> _BoundedTraining:
        return _BoundedTraining(file_type(path))

    def load_readonly(
        path: Path,
        *,
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
        allow_pickle: bool = False,
    ) -> NDArray[np.generic]:
        result = cast(
            "NDArray[np.generic]",
            load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle),
        )
        if path.name in {"fingerprints.npy", "coco_fp.npy", "bio_fp.npy"}:
            assert isinstance(result, np.memmap)
            assert not result.flags.writeable
            reads.append(path)
        return result

    monkeypatch.setattr(
        second_pool,
        "parquet",
        types.SimpleNamespace(
            ParquetFile=open_training,
            read_table=parquet.read_table,
            write_table=parquet.write_table,
        ),
    )
    monkeypatch.setattr(np, "load", load_readonly)
    output = tmp_path / "nested" / "pool"
    prepare_second_pool(**inputs, output=output)
    assert len(reads) == 3
    order = sorted(
        range(count + 3),
        key=lambda row: masses[row] if row < count else 1.0,
    )
    values = cast("NDArray[np.uint8]", load(output / "pool_fp.npy"))
    np.testing.assert_array_equal(
        values[:, 0],
        [row if row < count else 201 + row - count for row in order],
    )
    metadata = parquet.read_table(output / "pool_meta.parquet")
    assert metadata.column("key").to_pylist() == [
        ([f"c{i}" for i in range(count)] + ["b", "b", "new"])[row] for row in order
    ]
    for protected in (
        inputs["input_path"],
        inputs["coconut"] / "coco_fp.npy",
        fingerprints / "manifest.json",
    ):
        with pytest.raises(
            ValueError,
            match="output path aliases protected input artifact",
        ):
            prepare_second_pool(**inputs, output=protected)
    monkeypatch.setattr(np, "load", load)
    for index, path in enumerate(
        (
            fingerprints / "fp_bits.npy",
            inputs["coconut"] / "coco_mass.npy",
            inputs["bio"] / "bio_mass.npy",
        ),
    ):
        correct = cast("NDArray[np.generic]", load(path))
        np.save(path, correct.astype(object))
        with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
            prepare_second_pool(**inputs, output=tmp_path / f"object-{index}")
        np.save(path, correct)


class _BoundedTraining:
    def __init__(self, source: parquet.ParquetFile) -> None:
        self.source = source

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: list[str] | None = None,
    ) -> Iterator[RecordBatch]:
        assert 0 < batch_size <= 256
        assert columns == ["inchikey14", "normalized_smiles"]
        yield from self.source.iter_batches(batch_size=batch_size, columns=columns)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
