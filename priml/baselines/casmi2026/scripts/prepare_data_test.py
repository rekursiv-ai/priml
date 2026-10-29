"""Checks for the prepare_data module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import argparse
import pickle
import sys

from pyarrow import Table, parquet
from treekle import json_dumps, json_loads

import numpy as np
import pytest

from priml.baselines.casmi2026.isolated import (
    ADMISSION_DIR,
    DATASET_ROOT,
    FRAGMENT_GAP_DIR,
    GLACIER_ROOT_DIR,
    ICE_ROOT_DIR,
    IDENTITY_DIR,
    IDENTITY_RDKIT,
    SIMULATOR_RDKIT,
    v6_reaction_rules,
)
from priml.baselines.casmi2026.scripts import prepare_data
from priml.baselines.casmi2026.scripts.rdkit_backend import StructureRecord


if TYPE_CHECKING:
    from numpy.typing import NDArray


class Molecules:
    version = IDENTITY_RDKIT

    def __init__(self) -> None:
        self.standardized: list[tuple[str, bool]] = []

    def record(self, smiles: str) -> StructureRecord | None:
        if smiles == "invalid":
            return None
        values = np.zeros(prepare_data.RAW_BITS, dtype=np.uint8)
        values[[0, 4096, 8192, prepare_data.RAW_BITS - 1]] = 1
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


def _column(output: Path, name: str) -> list[object]:
    table = parquet.read_table(output / "structures.parquet")
    return table.column(name).to_pylist()


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
    prepare_data.prepare([first, second], output=output, chemistry=Molecules())
    assert _column(output, "input_smiles") == ["CCO", "invalid", "OCC"]
    assert _column(output, "smiles") == ["CCO", "", "OCC"]
    assert _column(output, "valid") == [True, False, True]
    assert _column(output, "heavy_atoms") == [3, 0, 3]
    assert _column(output, "formula") == ["C2H6O", "", "C2H6O"]
    assert _column(output, "key") == ["engine", "", "engine"]
    assert _column(output, "raw_key") == ["raw", "", "raw"]
    assert _column(output, "metric_key") == ["metric", "", "metric"]
    composition = json_dumps((("C", 2), ("O", 1)))
    assert _column(output, "composition") == [composition, json_dumps(()), composition]
    assert json_loads(composition, tuple[tuple[str, int], ...]) == (("C", 2), ("O", 1))
    np.testing.assert_array_equal(
        _column(output, "mass"),
        [46.041864812, float("nan"), 46.041864812],
    )
    packed: NDArray[np.uint8] = np.load(output / "raw_fp.npy")
    assert packed.shape == (3, (prepare_data.RAW_BITS + 7) // 8)
    bits = np.unpackbits(packed, axis=1)
    np.testing.assert_array_equal(
        bits[:, [0, 1, 4096, 8192, prepare_data.RAW_BITS - 1]],
        [[1, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 1, 1]],
    )
    assert (output / "fingerprint_bits.txt").read_text() == f"{prepare_data.RAW_BITS}\n"
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, object])
    assert manifest["rows"] == 3
    assert manifest["valid_rows"] == 2
    assert manifest["rdkit"] == IDENTITY_RDKIT
    assert manifest["raw_bits"] == prepare_data.RAW_BITS
    assert manifest["numpy"] == np.__version__
    assert (manifest["standardize"], manifest["keep_charged"]) == (False, False)
    assert manifest["generator"] != {}
    assert manifest["inputs"] != {}
    with pytest.raises(FileExistsError):
        prepare_data.prepare([first], output=output, chemistry=Molecules())


def test_standardization_chooses_the_charged_or_uncharged_form(tmp_path: Path) -> None:
    source = tmp_path / "first.parquet"
    parquet.write_table(
        Table.from_pydict({"smiles": ["CCO", "invalid", None, "CCO"]}),
        source,
    )
    chemistry = Molecules()
    output = tmp_path / "standardized"
    prepare_data.prepare(
        [source],
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
    assert _column(output, "standardized_smiles") == ["CCO-standard", None]
    assert _column(output, "charged_smiles") == ["CCO-standard", None]
    assert _column(output, "smiles") == ["CCO-standard", ""]
    plain = tmp_path / "plain"
    prepare_data.prepare(
        [source],
        output=plain,
        chemistry=Molecules(),
        standardize=True,
    )
    assert _column(plain, "smiles") == ["CCO-standard", ""]
    manifest = json_loads((plain / "manifest.json").read_text(), dict[str, object])
    assert (manifest["standardize"], manifest["keep_charged"]) == (True, False)


def test_an_explicit_column_overrides_inference_and_absence_is_rejected(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "explicit.parquet"
    parquet.write_table(Table.from_pydict({"structure": ["C"]}), explicit)
    prepare_data.prepare(
        [explicit],
        output=tmp_path / "explicit",
        chemistry=Molecules(),
        column="structure",
    )
    assert _column(tmp_path / "explicit", "input_smiles") == ["C"]
    with pytest.raises(ValueError, match="A SMILES column is required"):
        prepare_data.prepare(
            [explicit],
            output=tmp_path / "missing",
            chemistry=Molecules(),
        )


def _parse(*argv: str) -> prepare_data.Flags:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    return cast(prepare_data.Flags, parser.parse_args(list(argv)))


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["prepare", *argv])
    return prepare_data.main()


def _usage_error(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    with pytest.raises(SystemExit) as error:
        _ = _run(monkeypatch, *argv)
    assert error.value.code == 2


def test_structures_command_keeps_every_input_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags = _parse("structures", "a.parquet", "b.pkl", "--output", "out")
    assert flags.inputs == [Path("a.parquet"), Path("b.pkl")]
    assert flags.output == Path("out")
    assert (flags.column, flags.version) == (None, IDENTITY_RDKIT)
    assert (flags.standardize, flags.keep_charged) == (False, False)
    calls: list[tuple[list[Path], dict[str, object]]] = []

    def prepare(inputs: list[Path], **kwargs: object) -> None:
        calls.append((inputs, kwargs))

    def construct(version: str) -> str:
        assert version == IDENTITY_RDKIT
        return "chemistry"

    monkeypatch.setattr(prepare_data, "prepare", prepare)
    monkeypatch.setattr(prepare_data, "Chemistry", construct)
    argv = ("structures", "a.parquet", "b.pkl", "--output", "out")
    assert _run(monkeypatch, *argv) == 0
    flagged = ("--column", "structure", "--standardize", "--keep-charged")
    assert _run(monkeypatch, *argv, *flagged) == 0
    inputs = [Path("a.parquet"), Path("b.pkl")]
    options: dict[str, object] = {"output": Path("out"), "chemistry": "chemistry"}
    assert calls == [
        (
            inputs,
            {**options, "column": None, "standardize": False, "keep_charged": False},
        ),
        (
            inputs,
            {
                **options,
                "column": "structure",
                "standardize": True,
                "keep_charged": True,
            },
        ),
    ]
    _usage_error(monkeypatch)
    _usage_error(monkeypatch, "structures", "--output", "out")
    _usage_error(monkeypatch, "structures", "a.parquet")
    _usage_error(monkeypatch, "unknown", "a.parquet", "--output", "out")


def test_simulator_commands_default_to_the_directories_their_readers_use() -> None:
    for command, directory in (
        ("admission", ADMISSION_DIR),
        ("ice", ICE_ROOT_DIR),
        ("glacier", GLACIER_ROOT_DIR),
    ):
        flags = _parse(command, "in.parquet")
        assert flags.output == Path(DATASET_ROOT) / directory
        assert flags.version == SIMULATOR_RDKIT
        assert flags.batch_rows == prepare_data.BATCH_ROWS
    for command, directory in (
        ("fragment-gap", FRAGMENT_GAP_DIR),
        ("identity", IDENTITY_DIR),
    ):
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

    monkeypatch.setattr(prepare_data, "prepare_reactions", prepare)
    monkeypatch.setattr(prepare_data, "MolecularContext", chemistry)
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
        monkeypatch.setattr(prepare_data, name, prepare)
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
    pool = prepare_data._provided("prvsiyan/coconut-casmi26-candidates")
    flags = _parse("second-pool", "t.parquet", "--fingerprints", "f", "--output", "o")
    assert flags.coconut == pool
    assert flags.bio == prepare_data._provided("prvsiyan/chebi-lipidmaps-casmi26")
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

    monkeypatch.setattr(prepare_data, "prepare_derivations", prepare)
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

    monkeypatch.setattr(prepare_data, "prepare_graphs", prepare)
    argv = ("graphs", "graphs.parquet", "--output", "graphs")
    assert _run(monkeypatch, *argv, "--raw", "--batch-rows", "17") == 0
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

    monkeypatch.setattr(prepare_data, function, prepare)
    assert _run(monkeypatch, command, "input.parquet", "--output", "out") == 0
    expected: dict[str, object] = {"output": Path("out"), "version": version}
    if batched:
        expected["batch_rows"] = 256
    assert calls == [(Path("input.parquet"), expected)]
    _usage_error(monkeypatch, command, "--output", "out")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
