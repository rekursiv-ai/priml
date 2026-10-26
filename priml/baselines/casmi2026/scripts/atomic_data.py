"""Generate standard atomic properties in an isolated RDKit interpreter."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final

import math

from treekle import from_plain

from priml.baselines.casmi2026.isolated import (
    SIMULATOR_GROUPS,
    SIMULATOR_NUMBERS,
    SIMULATOR_RDKIT,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import MolecularContext
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyarrow import Table, parquet
    from rdkit import Chem
    from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import AtomicRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")
    Chem = lazy_import("rdkit.Chem")
    ComputeGasteigerCharges = lazy_import(
        "rdkit.Chem.rdPartialCharges",
        "ComputeGasteigerCharges",
    )


_THIS: Final = Path(__file__).resolve()


def prepare_graphs(
    input_path: Path,
    *,
    output: Path,
    version: str,
    raw: bool = False,
    batch_rows: int = 256,
) -> None:
    """Write every requested molecule's numerical graph and preparation provenance.

    Args:
      input_path: Parquet table with a SMILES column.
      output: New external dataset directory.
      version: Required isolated RDKit release.
      raw: Preserve raw input atom order for direct scientific controls.
      batch_rows: Maximum generated molecular rows retained before writing.

    """
    if version != SIMULATOR_RDKIT:
        raise ValueError(
            f"Simulator graph preparation requires RDKit {SIMULATOR_RDKIT}.",
        )
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    context = MolecularContext(version)
    source = parquet.read_table(input_path)
    values = from_plain(
        source.column("smiles").to_pylist(),
        list[str | None],
    )
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    schema = pa.schema(
        [
            ("input_smiles", pa.string()),
            ("smiles", pa.string()),
            ("valid", pa.bool_()),
            ("features", pa.list_(pa.list_(pa.float64()))),
            ("bonds", pa.list_(pa.list_(pa.int64()))),
            ("bond_features", pa.list_(pa.list_(pa.float64()))),
            ("weights", pa.list_(pa.float64())),
            ("degrees", pa.list_(pa.int64())),
        ],
    )
    writer = parquet.ParquetWriter(destination / "graphs.parquet", schema)
    rows: list[AtomicRow] = []
    total, valid = 0, 0
    try:
        for value in dict.fromkeys(values):
            if value is None:
                continue
            row = _graph_row(value, raw=raw)
            rows.append(row)
            total += 1
            valid += row["valid"]
            if len(rows) == batch_rows:
                writer.write_table(Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": context.version,
        "raw": raw,
        "rows": total,
        "valid_rows": valid,
        "node_features": 74,
        "bond_features": 28,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("rdkit_backend.py"))),
    }
    write_manifest(destination, manifest)


def _graph_row(
    smiles: str,
    *,
    raw: bool,
) -> AtomicRow:
    row: AtomicRow = {
        "input_smiles": smiles,
        "smiles": "",
        "valid": False,
        "features": [],
        "bonds": [],
        "bond_features": [],
        "weights": [],
        "degrees": [],
    }
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return row
    if not raw and any(
        atom.GetAtomicNum() not in _elements() for atom in molecule.GetAtoms()
    ):
        return row
    molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumAtoms() == 0 or (not raw and molecule.GetNumAtoms() > 100):
        return row
    canonical = Chem.MolToSmiles(molecule)
    if not raw:
        parsed = Chem.MolFromSmiles(canonical)
        if parsed is None:
            raise ValueError("Expected parsed is not None.")
        molecule = Chem.RemoveHs(parsed)
    # Unsupported elements can reject Gasteiger parameterization; missing charges
    # then become zero in the atom feature vector.
    with suppress(Exception):
        ComputeGasteigerCharges(molecule)
    atom_rings = [0] * molecule.GetNumAtoms()
    bond_rings: dict[tuple[int, int], int] = {}
    for ring in Chem.GetSymmSSSR(molecule):
        atoms = list(ring)
        for atom in atoms:
            atom_rings[atom] += 1
        for first, second in zip(atoms, [*atoms[1:], atoms[0]], strict=True):
            key = (min(first, second), max(first, second))
            bond_rings[key] = bond_rings.get(key, 0) + 1
    row["valid"] = True
    row["smiles"] = canonical
    for index, atom in enumerate(molecule.GetAtoms()):
        row["features"].append(_node_features(atom, ring_count=atom_rings[index]))
        row["degrees"].append(atom.GetTotalDegree() - atom.GetTotalNumHs())
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row["bonds"].append([first, second])
        row["weights"].append(bond.GetBondTypeAsDouble())
        row["bond_features"].append(
            _bond_features(
                bond,
                ring_count=bond_rings.get((min(first, second), max(first, second)), 0),
            ),
        )
    return row


def _node_features(atom: Chem.Atom, *, ring_count: int) -> list[float]:
    number = atom.GetAtomicNum()
    degree = atom.GetTotalDegree() - atom.GetTotalNumHs()
    charge = atom.GetFormalCharge()
    hydrogens = atom.GetTotalNumHs()
    in_ring = atom.IsInRing()
    try:
        partial_charge = float(atom.GetProp("_GasteigerCharge"))
    except (KeyError, ValueError, RuntimeError):
        partial_charge = 0.0
    if math.isnan(partial_charge) or math.isinf(partial_charge):
        partial_charge = 0.0
    groups = _elements()
    return [
        *_categorical(number, tuple(groups)),
        *_one_hot(degree, range(6)),
        float(degree > 5),
        *_one_hot(charge, range(-2, 3)),
        float(abs(charge) > 2),
        *_categorical(
            str(atom.GetHybridization()),
            ("SP", "SP2", "SP3", "SP3D", "SP3D2"),
        ),
        float(atom.GetIsAromatic()),
        *_one_hot(hydrogens, range(5)),
        float(hydrogens > 4),
        float(not in_ring),
        float(in_ring),
        *(
            float(value)
            for size in range(3, 9)
            for value in (atom.IsInRingSize(size), not atom.IsInRingSize(size))
        ),
        float(ring_count < 2),
        float(ring_count >= 2),
        *_one_hot(
            str(atom.GetChiralTag()),
            (
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ),
        ),
        *(float(number in groups and groups[number] == group) for group in range(8)),
        partial_charge,
    ]


def _bond_features(bond: Chem.Bond, *, ring_count: int) -> list[float]:
    conjugated, in_ring = bond.GetIsConjugated(), bond.IsInRing()
    return [
        *_categorical(
            str(bond.GetBondType()),
            ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
        ),
        float(not conjugated),
        float(conjugated),
        float(not in_ring),
        float(in_ring),
        *(
            float(value)
            for size in range(3, 9)
            for value in (bond.IsInRingSize(size), not bond.IsInRingSize(size))
        ),
        float(ring_count < 2),
        float(ring_count >= 2),
        *_categorical(
            str(bond.GetStereo()),
            ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"),
        ),
    ]


def _one_hot(
    value: int | str,
    vocabulary: Sequence[int] | Sequence[str],
) -> list[float]:
    return [float(value == member) for member in vocabulary]


def _categorical(
    value: int | str,
    vocabulary: Sequence[int] | Sequence[str],
) -> list[float]:
    return [*_one_hot(value, vocabulary), float(value not in vocabulary)]


def _elements() -> dict[int, int]:
    """Keep the released element vocabulary order and its eight chemical groups."""
    return dict(zip(SIMULATOR_NUMBERS, SIMULATOR_GROUPS, strict=True))
