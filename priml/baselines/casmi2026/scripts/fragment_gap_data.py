"""Prepare distinct fixed/fallback-mass gap graphs in an isolated SDK process."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import logging

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.custom_types import GapGraph
from priml.baselines.casmi2026.isolated import ELEMENT_MASSES, IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import MolecularContext
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from pyarrow import Table, parquet

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import GapGraphRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    pa = lazy_import("pyarrow")


logger = logging.getLogger(__name__)

_THIS: Final = Path(__file__).resolve()


def prepare_fragment_gap(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Write every complete original gap graph and scientific provenance.

    Args:
      input_path: Parquet SMILES requests, retaining duplicates and invalid rows.
      output: New external dataset directory containing complete numeric graphs.
      version: Required isolated scientific-library release.
      batch_rows: Maximum complete rows in each Arrow group.

    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    chemistry = GapChemistry(version)
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("valid", pa.bool_()),
            ("weights", pa.list_(pa.float64())),
            ("bonds", pa.list_(pa.list_(pa.int64()))),
            ("pieces", pa.int64()),
        ],
    )
    count = 0
    with parquet.ParquetWriter(destination / "graphs.parquet", schema) as writer:
        for batch in parquet.ParquetFile(input_path).iter_batches(
            batch_size=batch_rows,
            columns=["smiles"],
        ):
            strings = from_plain(cast(object, batch.column(0).to_pylist()), list[str])
            rows: list[GapGraphRow] = []
            for smiles in strings:
                graph = chemistry.graph(smiles)
                rows.append(
                    {
                        "smiles": smiles,
                        "valid": graph is not None,
                        "weights": from_plain(
                            cast(object, graph.weights.tolist()),
                            list[float],
                        )
                        if graph is not None
                        else [],
                        "bonds": from_plain(
                            cast(object, graph.bonds.tolist()),
                            list[list[int]],
                        )
                        if graph is not None
                        else [],
                        "pieces": graph.pieces if graph is not None else 0,
                    },
                )
            writer.write_table(Table.from_pylist(rows, schema=schema))
            count += len(rows)
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": count,
        "inputs": input_digests([input_path]),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "chemistry.py",
            ),
        ),
    }
    write_manifest(destination, manifest)


class GapChemistry:
    """Prepare ordered graph weights with fifteen fixed masses and SDK fallback."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Gap graphs require RDKit {IDENTITY_RDKIT}.")
        self.chem = MolecularContext(version).chem
        masses = ELEMENT_MASSES
        self.masses = {
            element: masses[element]
            for element in (
                "C",
                "H",
                "N",
                "O",
                "P",
                "S",
                "F",
                "Cl",
                "Br",
                "I",
                "Na",
                "K",
                "Si",
                "B",
                "Se",
            )
        }

    def graph(self, smiles: str) -> GapGraph | None:
        """Return the original graph or explicit empty/invalid-input rejection.

        Args:
          smiles: Original string without standardization or isotope-mass adjustment.

        Returns:
          graph: Every ordered atom weight, bond and component count, or None.

        """
        try:
            molecule = self.chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None or molecule.GetNumAtoms() == 0:
                return None
            periodic = self.chem.GetPeriodicTable()
            weights = np.empty(molecule.GetNumAtoms(), dtype=np.float64)
            for atom in molecule.GetAtoms():
                element = atom.GetSymbol()
                mass = self.masses.get(element)
                if mass is None:
                    mass = periodic.GetMostCommonIsotopeMass(element)
                weights[atom.GetIdx()] = mass + atom.GetTotalNumHs() * self.masses["H"]
            bonds = np.array(
                [
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    for bond in molecule.GetBonds()
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            return GapGraph(
                weights=weights,
                bonds=bonds,
                pieces=len(self.chem.GetMolFrags(molecule)),
            )
        except Exception:
            logger.debug("Gap graph preparation failed.", exc_info=True)
            return None
