"""Prepare complete GLACIER graph and physical data in isolated RDKit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.custom_types import AtomicRow
from priml.baselines.casmi2026.isolated import SIMULATOR_RDKIT, SIMULATOR_SYMBOLS
from priml.baselines.casmi2026.scripts.atomic_data import prepare_graphs
from priml.baselines.casmi2026.scripts.graph_properties import (
    atom_properties,
)
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
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import GlacierRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")
    CalcMolFormula = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcMolFormula")


_THIS: Final = Path(__file__).resolve()


def prepare_glacier_graphs(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Generate complete graph, formula, hydrogen and isotope-aware mass assets.

    Args:
      input_path: Parquet table containing original SMILES strings.
      output: New external dataset directory.
      version: Required isolated RDKit release.
      batch_rows: Maximum generated graph rows retained before writing.

    """
    if version != SIMULATOR_RDKIT:
        raise ValueError(f"GLACIER graph preparation requires RDKit {SIMULATOR_RDKIT}.")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    prepare_graphs(
        input_path,
        output=destination / "graphs",
        version=version,
        batch_rows=batch_rows,
    )
    chemistry = GlacierChemistry(version)
    schema = pa.schema(
        [
            ("input_smiles", pa.string()),
            ("smiles", pa.string()),
            ("valid", pa.bool_()),
            ("formula", pa.list_(pa.float64())),
            ("atom_formulas", pa.list_(pa.list_(pa.float64()))),
            ("hydrogens", pa.list_(pa.int64())),
            ("atom_masses", pa.list_(pa.float64())),
            ("hydrogenated_masses", pa.list_(pa.float64())),
            ("total_hydrogens", pa.int64()),
        ],
    )
    source = parquet.ParquetFile(destination / "graphs" / "graphs.parquet")
    total, valid = 0, 0
    writer = parquet.ParquetWriter(destination / "physical.parquet", schema)
    try:
        for group in range(source.num_row_groups):
            graphs = from_plain(
                source.read_row_group(group).to_pylist(),
                list[AtomicRow],
            )
            rows = [chemistry.row(graph) for graph in graphs]
            writer.write_table(Table.from_pylist(rows, schema=schema))
            total += len(rows)
            valid += sum(row["valid"] for row in rows)
    finally:
        writer.close()
    np.save(destination / "element_masses.npy", chemistry.element_masses[:, None])
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": total,
        "valid_rows": valid,
        "inputs": input_digests([input_path]),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("graph_properties.py"),
                _THIS.with_name("rdkit_backend.py"),
            ),
        ),
        "graph_manifest_sha256": hashlib.sha256(
            (destination / "graphs" / "manifest.json").read_bytes(),
        ).hexdigest(),
    }
    write_manifest(destination, manifest)


class GlacierChemistry:
    """Extract physical records after the separate GLACIER graph canonicalization."""

    def __init__(self, version: str) -> None:
        context = MolecularContext(version)
        self.chem = context.chem
        self.table = self.chem.GetPeriodicTable()
        self.elements = SIMULATOR_SYMBOLS
        self.element_masses: NDArray[np.float64] = np.array(
            [self.table.GetMostCommonIsotopeMass(element) for element in self.elements],
        )

    def row(self, graph: AtomicRow) -> GlacierRow:
        """Return the complete ordered physical record, including rejected inputs.

        Args:
          graph: Complete prepared atomic row with GLACIER admission and atom order.

        Returns:
          row: Every physical field used during collation and simulation.

        """
        row: GlacierRow = {
            "input_smiles": graph["input_smiles"],
            "smiles": graph["smiles"],
            "valid": graph["valid"],
            "formula": [],
            "atom_formulas": [],
            "hydrogens": [],
            "atom_masses": [],
            "hydrogenated_masses": [],
            "total_hydrogens": 0,
        }
        if not graph["valid"]:
            return row
        molecule = self.chem.MolFromSmiles(graph["smiles"])
        if molecule is None:
            raise ValueError("Expected molecule is not None.")
        formula = CalcMolFormula(molecule)
        self.chem.Kekulize(molecule, True)
        properties = atom_properties(
            tuple(molecule.GetAtoms()),
            formula=formula,
            elements=self.elements,
            masses=self.element_masses,
            periodic_table=self.table,
        )
        row.update(
            formula=from_plain(cast(object, properties.formula.tolist()), list[float]),
            atom_formulas=from_plain(
                cast(object, properties.atom_formulas.tolist()),
                list[list[float]],
            ),
            hydrogens=from_plain(
                cast(object, properties.hydrogens.tolist()),
                list[int],
            ),
            atom_masses=from_plain(
                cast(object, properties.masses.tolist()),
                list[float],
            ),
            hydrogenated_masses=from_plain(
                cast(object, properties.hydrogenated_masses.tolist()),
                list[float],
            ),
            total_hydrogens=properties.total_hydrogens,
        )
        return row
