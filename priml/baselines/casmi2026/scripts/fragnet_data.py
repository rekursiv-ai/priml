"""Prepare complete Fragnet molecular fields in an isolated chemistry interpreter."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import tempfile

from treekle import from_plain

from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    IDENTITY_RDKIT,
    FragnetAtoms,
    molecular_record,
)
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.ragged import save_stream
from priml.baselines.casmi2026.scripts.rdkit_backend import MolecularContext
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from pyarrow import Table, parquet

    import numpy as np
    import pyarrow as pa
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    pa = lazy_import("pyarrow")
    np = lazy_import("numpy")


_THIS: Final = Path(__file__).resolve()


def prepare_fragnet(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Write every ordered molecular record and explicit rejected row.

    Args:
      input_path: Parquet containing every requested SMILES in its original order.
      output: New external asset directory, preserving duplicate and invalid rows.
      version: Exact isolated scientific-library release.
      batch_rows: Maximum complete rows per Arrow group.

    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    chemistry = FragnetChemistry(version)
    masses = ELEMENT_MASSES
    source = parquet.ParquetFile(input_path)
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True)
    count = source.metadata.num_rows
    mass = np.lib.format.open_memmap(
        destination / "mass.npy",
        mode="w+",
        dtype=np.float64,
        shape=(count,),
    )
    atom_offsets, bond_offsets = (np.zeros(count + 1, dtype=np.int64) for _ in range(2))
    layouts = (
        ("amass", np.float64, 1),
        ("kind", np.int8, 1),
        ("el", np.int8, 1),
        ("nh", np.int8, 1),
        ("el10", np.uint8, 1),
        ("bonds", np.int32, 2),
        ("bring", np.int8, 1),
    )
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("parsed", pa.bool_()),
            ("valid", pa.bool_()),
            ("symbols", pa.list_(pa.string())),
            *(
                (name, pa.list_(pa.int64()))
                for name in ("hydrogens", "degree", "charge", "aromatic")
            ),
            ("bonds", pa.list_(pa.list_(pa.int64()))),
            *((name, pa.list_(pa.bool_())) for name in ("bond_aromatic", "bond_ring")),
        ],
    )
    row_index = 0
    with ExitStack() as stack:
        raw = {
            name: stack.enter_context(tempfile.TemporaryFile(dir=destination))
            for name, _, _ in layouts
        }
        writer = stack.enter_context(
            parquet.ParquetWriter(destination / "molecules.parquet", schema),
        )
        for batch in source.iter_batches(batch_size=batch_rows, columns=["smiles"]):
            values = from_plain(cast(object, batch.column(0).to_pylist()), list[str])
            metadata: list[dict[str, object]] = []
            for value in values:
                atoms = chemistry.atoms(value)
                record = molecular_record(atoms, masses=masses)
                mass[row_index] = record["mass"] if record is not None else np.nan
                atom_offsets[row_index + 1] = atom_offsets[row_index] + (
                    len(record["amass"]) if record is not None else 0
                )
                bond_offsets[row_index + 1] = bond_offsets[row_index] + (
                    len(record["bonds"]) if record is not None else 0
                )
                if record is not None:
                    for name in ("amass", "kind", "el", "nh", "el10", "bonds", "bring"):
                        record[name].tofile(raw[name])
                metadata.append(_metadata(value, atoms=atoms, valid=record is not None))
                row_index += 1
            writer.write_table(Table.from_pylist(metadata, schema=schema))
        for name, dtype, width in layouts:
            save_stream(
                raw[name],
                destination / f"{name}.npy",
                dtype=dtype,
                width=width,
            )
    np.save(destination / "atom_off.npy", atom_offsets)
    np.save(destination / "bond_off.npy", bond_offsets)
    mass.flush()
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": row_index,
        "inputs": input_digests([input_path]),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "chemistry.py",
                _THIS.parent.parent / "fragnet.py",
            ),
        ),
    }
    write_manifest(destination, manifest)


class FragnetChemistry:
    """Observe only standard parsed atoms and bonds before native normalization."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Fragnet preparation requires RDKit {IDENTITY_RDKIT}.")
        self.chem = MolecularContext(version).chem

    def atoms(self, smiles: str) -> FragnetAtoms | None:
        """Return every unmodified molecular observation in library order.

        Args:
          smiles: Structure string passed unchanged to the scientific parser.

        Returns:
          atoms: Complete ordered observations, or an explicit parse failure.

        """
        molecule = self.chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        atoms, bonds = list(molecule.GetAtoms()), list(molecule.GetBonds())
        return FragnetAtoms(
            symbols=tuple(atom.GetSymbol() for atom in atoms),
            hydrogens=np.array(
                [atom.GetTotalNumHs() for atom in atoms],
                dtype=np.int64,
            ),
            degree=np.array([atom.GetDegree() for atom in atoms], dtype=np.int64),
            charge=np.array([atom.GetFormalCharge() for atom in atoms], dtype=np.int64),
            aromatic=np.array([atom.GetIsAromatic() for atom in atoms], dtype=np.int64),
            bonds=np.array(
                [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds],
                dtype=np.int64,
            ).reshape(len(bonds), 2),
            bond_aromatic=np.array(
                [bond.GetIsAromatic() for bond in bonds],
                dtype=np.bool_,
            ),
            bond_ring=np.array([bond.IsInRing() for bond in bonds], dtype=np.bool_),
        )


def _metadata(
    smiles: str,
    *,
    atoms: FragnetAtoms | None,
    valid: bool,
) -> dict[str, object]:
    return {
        "smiles": smiles,
        "parsed": atoms is not None,
        "valid": valid,
        "symbols": list(atoms.symbols) if atoms is not None else [],
        "hydrogens": atoms.hydrogens.tolist() if atoms is not None else [],
        "degree": atoms.degree.tolist() if atoms is not None else [],
        "charge": atoms.charge.tolist() if atoms is not None else [],
        "aromatic": atoms.aromatic.tolist() if atoms is not None else [],
        "bonds": atoms.bonds.tolist() if atoms is not None else [],
        "bond_aromatic": atoms.bond_aromatic.tolist() if atoms is not None else [],
        "bond_ring": atoms.bond_ring.tolist() if atoms is not None else [],
    }
