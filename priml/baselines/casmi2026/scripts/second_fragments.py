"""Generate the distinct engine2 unrounded neutral-fragment data independently."""

from __future__ import annotations

from itertools import chain, combinations
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import tempfile

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.isolated import (
    ELEMENT_MASSES,
    FORMULA_ELEMENTS,
    IDENTITY_RDKIT,
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
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pyarrow import Table, parquet

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import FragmentMassRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")


_THIS: Final = Path(__file__).resolve()


def prepare_second_fragments(input_path: Path, *, output: Path, version: str) -> None:
    """Retain ordered pool rows and every default unrounded fragment mass.

    Args:
      input_path: Pool Parquet SMILES in the exact stable, mass-sorted pool order.
      output: New external dataset directory containing the two ragged arrays.
      version: Required isolated RDKit release, independent from ICE and GLACIER.

    """
    chemistry = FragmentChemistry(version)
    source = parquet.ParquetFile(input_path)
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    offsets = np.zeros(source.metadata.num_rows + 1, dtype=np.int64)
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("valid", pa.bool_()),
            ("weights", pa.list_(pa.float64())),
            ("bonds", pa.list_(pa.list_(pa.int64()))),
        ],
    )
    position = 0
    with (
        tempfile.TemporaryFile(dir=destination) as raw,
        parquet.ParquetWriter(destination / "graphs.parquet", schema) as writer,
    ):
        for batch in source.iter_batches(batch_size=256, columns=["smiles"]):
            strings = from_plain(cast(object, batch.column(0).to_pylist()), list[str])
            rows: list[FragmentMassRow] = []
            for smiles in strings:
                graph = chemistry.graph(smiles)
                weights, bonds = (
                    graph
                    if graph is not None
                    else (np.zeros(0), np.empty((0, 2), dtype=np.int64))
                )
                masses = (
                    fragment_masses(weights, bonds=bonds)
                    if graph is not None
                    else np.zeros(0)
                )
                masses.tofile(raw)
                position += 1
                offsets[position] = offsets[position - 1] + len(masses)
                rows.append(
                    {
                        "smiles": smiles,
                        "valid": graph is not None,
                        "weights": from_plain(
                            cast(object, weights.tolist()),
                            list[float],
                        ),
                        "bonds": from_plain(
                            cast(object, bonds.tolist()),
                            list[list[int]],
                        ),
                    },
                )
            writer.write_table(Table.from_pylist(rows, schema=schema))
        save_stream(raw, destination / "frag_mass.npy", dtype=np.float64)
    np.save(destination / "frag_off.npy", offsets, allow_pickle=False)
    total = int(cast(np.int64, offsets[-1]))
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": position,
        "masses": total,
        "max_breaks": 2,
        "max_bonds": 34,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("rdkit_backend.py"))),
    }
    write_manifest(destination, manifest)


def fragment_masses(
    weights: NDArray[np.float64],
    *,
    bonds: NDArray[np.int64],
    max_breaks: int = 2,
    max_bonds: int = 34,
) -> NDArray[np.float64]:
    """Enumerate parent and connected-component masses with exact traversal order.

    Args:
      weights: Complete fixed-mass atom weights, including attached hydrogens.
      bonds: Every undirected bond in original scientific-library order.
      max_breaks: Add two-bond cuts at values at least two; one-bond cuts always run.
      max_bonds: Use the parent-only fallback above this bond count.

    Returns:
      masses: Sorted exact-float union of parent and every cut component mass.

    """
    parent = float(weights.sum())
    if len(bonds) == 0 or len(bonds) > max_bonds:
        return np.array([parent], dtype=np.float64)
    edges = from_plain(cast(object, bonds.tolist()), list[tuple[int, int]])
    result = {parent}
    cuts = chain(
        combinations(range(len(edges)), 1),
        combinations(range(len(edges)), 2) if max_breaks >= 2 else (),
    )
    for removed in cuts:
        for component in _components(len(weights), edges=edges, removed=removed):
            result.add(float(weights[component].sum()))
    return np.array(sorted(result), dtype=np.float64)


class FragmentChemistry:
    """Use isolated molecular primitives with the second engine's fixed atomic masses."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Second-engine fragments require RDKit {IDENTITY_RDKIT}.")
        self.chem = MolecularContext(version).chem
        self.masses = {
            element: ELEMENT_MASSES[element]
            for element in (*FORMULA_ELEMENTS, "Na", "K", "Si", "B", "Se")
        }

    def graph(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
        """Return complete atom weights and bonds, ignoring isotope labels.

        Args:
          smiles: Exact pool string without simulator-family canonicalization.

        Returns:
          graph: All atom weights and ordered bonds, or explicit source rejection.

        """
        molecule = self.chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        weights = np.zeros(molecule.GetNumAtoms())
        for atom in molecule.GetAtoms():
            weights[atom.GetIdx()] = (
                self.masses.get(atom.GetSymbol(), 0.0)
                + atom.GetTotalNumHs() * self.masses["H"]
            )
        if np.any(cast("NDArray[np.bool_]", weights == 0)):
            return None
        bonds = np.array(
            [
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                for bond in molecule.GetBonds()
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        return weights, bonds


def _components(
    count: int,
    *,
    edges: Sequence[tuple[int, int]],
    removed: tuple[int, ...],
) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(count)]
    for index, (left, right) in enumerate(edges):
        if index not in removed:
            neighbors[left].append(right)
            neighbors[right].append(left)
    components: list[list[int]] = []
    seen: set[int] = set()
    for root in range(count):
        if root in seen:
            continue
        frontier, members = [root], [root]
        seen.add(root)
        while frontier:
            atom = frontier.pop()
            for neighbor in neighbors[atom]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
                    members.append(neighbor)
        components.append(members)
    return components
