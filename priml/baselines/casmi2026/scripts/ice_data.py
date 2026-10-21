"""Prepare canonical ICE molecular graphs in the isolated scientific interpreter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.isolated import (
    SIMULATOR_GROUPS,
    SIMULATOR_RDKIT,
    SIMULATOR_SYMBOLS,
)
from priml.baselines.casmi2026.scripts.graph_properties import atom_properties
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import MolecularContext
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyarrow import Table, parquet
    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        CleanupParameters,
        TautomerEnumerator,
    )
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import IceRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")
    CleanupParameters = lazy_import(
        "rdkit.Chem.MolStandardize.rdMolStandardize",
        "CleanupParameters",
    )
    TautomerEnumerator = lazy_import(
        "rdkit.Chem.MolStandardize.rdMolStandardize",
        "TautomerEnumerator",
    )
    CalcMolFormula = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcMolFormula")


_THIS: Final = Path(__file__).resolve()


def prepare_ice_graphs(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Write complete canonical root graphs and isotope-aware chemical data.

    Args:
      input_path: Parquet table with a SMILES column.
      output: New external dataset directory.
      version: Required isolated RDKit release.
      batch_rows: Maximum number of graph rows retained before writing.

    """
    if version != SIMULATOR_RDKIT:
        raise ValueError(f"ICE graph preparation requires RDKit {SIMULATOR_RDKIT}.")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    chemistry = IceChemistry(version, directory=destination)
    source = parquet.read_table(input_path)
    values = from_plain(
        cast(object, source.column("smiles").to_pylist()),
        list[str | None],
    )
    schema = pa.schema(
        [
            ("input_smiles", pa.string()),
            ("smiles", pa.string()),
            ("valid", pa.bool_()),
            ("features", pa.list_(pa.list_(pa.float64()))),
            ("bonds", pa.list_(pa.list_(pa.int64()))),
            ("bond_orders", pa.list_(pa.int64())),
            ("formula", pa.list_(pa.float64())),
            ("hydrogens", pa.list_(pa.int64())),
            ("atom_masses", pa.list_(pa.float64())),
            ("hydrogenated_masses", pa.list_(pa.float64())),
            ("mass", pa.float64()),
        ],
    )
    rows: list[IceRow] = []
    total, valid = 0, 0
    writer = parquet.ParquetWriter(destination / "ice.parquet", schema)
    try:
        for value in dict.fromkeys(values):
            if value is None:
                continue
            row = chemistry.row(value)
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
    np.save(destination / "element_masses.npy", chemistry.element_masses[:, None])
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": total,
        "valid_rows": valid,
        "node_features": 32,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("rdkit_backend.py"))),
        "tautomer_transforms_sha256": hashlib.sha256(
            (destination / "tautomer-transforms.txt").read_bytes(),
        ).hexdigest(),
    }
    write_manifest(destination, manifest)


class IceChemistry:
    """Own isolated canonicalization primitives and the ICE chemical vocabulary."""

    def __init__(self, version: str, *, directory: Path) -> None:
        context = MolecularContext(version)
        self.chem = context.chem
        self.table = self.chem.GetPeriodicTable()
        self.elements = SIMULATOR_SYMBOLS
        self.groups = SIMULATOR_GROUPS
        self.element_masses: NDArray[np.float64] = np.array(
            [self.table.GetMostCommonIsotopeMass(element) for element in self.elements],
            dtype=np.float64,
        )
        transforms = directory / "tautomer-transforms.txt"
        _ = transforms.write_text(
            "// Name SMARTS Bonds Charges\n"
            "1,3 hetero atom H shift\t[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]\n"
            "1,3 (thio)keto/enol r\t[O,S,Se,Te;X2!H0]-[C]=[C]\n",
        )
        parameters = CleanupParameters()
        parameters.tautomerTransformsFile = str(transforms)
        self.tautomers = TautomerEnumerator(parameters)

    def row(self, smiles: str) -> IceRow:
        """Return complete canonical graph data, or mark an unparsable input.

        Args:
          smiles: Original input string before the stereo-free InChI roundtrip.

        Returns:
          row: Every reached constructor property and neural graph channel.

        """
        row: IceRow = {
            "input_smiles": smiles,
            "smiles": "",
            "valid": False,
            "features": [],
            "bonds": [],
            "bond_orders": [],
            "formula": [],
            "hydrogens": [],
            "atom_masses": [],
            "hydrogenated_masses": [],
            "mass": float("nan"),
        }
        molecule = self.chem.MolFromSmiles(smiles)
        if molecule is None:
            return row
        self.chem.RemoveStereochemistry(molecule)
        stereo_free = self.chem.MolToSmiles(molecule)
        molecule = self.chem.MolFromSmiles(stereo_free)
        if molecule is None:
            raise ValueError("Expected molecule is not None.")
        molecule = self.chem.MolFromInchi(self.chem.MolToInchi(molecule))
        if molecule is None:
            raise ValueError("Expected molecule is not None.")
        canonical = self.chem.MolToSmiles(self.tautomers.Canonicalize(molecule))
        molecule = self.chem.MolFromSmiles(canonical)
        if molecule is None:
            raise ValueError("Expected molecule is not None.")
        formula = CalcMolFormula(molecule)
        self.chem.Kekulize(molecule, True)
        atoms = tuple(molecule.GetAtoms())
        properties = atom_properties(
            atoms,
            formula=formula,
            elements=self.elements,
            masses=self.element_masses,
            periodic_table=self.table,
        )
        hydrogen_counts = from_plain(
            cast(object, properties.hydrogens.tolist()),
            list[int],
        )
        features = [
            [
                *np.eye(len(self.elements))[element],
                *np.eye(8)[self.groups[element]],
                *np.eye(6)[count],
            ]
            for element, count in zip(
                from_plain(cast(object, properties.elements.tolist()), list[int]),
                hydrogen_counts,
                strict=True,
            )
        ]
        bonds = tuple(molecule.GetBonds())
        row.update(
            smiles=canonical,
            valid=True,
            features=from_plain(features, list[list[float]]),
            bonds=[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds],
            bond_orders=[
                {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 2}[
                    str(bond.GetBondType())
                ]
                for bond in bonds
            ],
            formula=from_plain(cast(object, properties.formula.tolist()), list[float]),
            hydrogens=hydrogen_counts,
            atom_masses=from_plain(
                cast(object, properties.masses.tolist()),
                list[float],
            ),
            hydrogenated_masses=from_plain(
                cast(object, properties.hydrogenated_masses.tolist()),
                list[float],
            ),
            mass=properties.mass,
        )
        return row
