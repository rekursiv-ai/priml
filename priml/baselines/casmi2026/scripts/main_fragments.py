"""Prepare the first engine's rounded generated-candidate fragment pair."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import tempfile

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.isolated import ELEMENT_MASSES, IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.ragged import save_stream
from priml.baselines.casmi2026.scripts.rdkit_backend import MolecularContext
from priml.paths import validated_output_path


if TYPE_CHECKING:
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


def prepare_main_fragments(input_path: Path, *, output: Path, version: str) -> None:
    """Write every requested rounded fragment pair and complete graph provenance.

    Args:
      input_path: Ordered Parquet SMILES, including duplicates and invalid strings.
      output: New external dataset directory for complete rows and provenance.
      version: Required isolated scientific release, independent of the simulators.

    """
    chemistry = FragmentChemistry(version)
    source = parquet.ParquetFile(input_path)
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    offsets_two = np.zeros(source.metadata.num_rows + 1, dtype=np.int64)
    offsets_one = np.zeros_like(offsets_two)
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
        tempfile.TemporaryFile(dir=destination) as raw_two,
        tempfile.TemporaryFile(dir=destination) as raw_one,
        tempfile.TemporaryFile(dir=destination) as raw_base,
        parquet.ParquetWriter(destination / "molecules.parquet", schema) as writer,
    ):
        for batch in source.iter_batches(batch_size=256, columns=["smiles"]):
            strings = cast(Sequence[str], batch.column(0).to_pylist())
            rows: list[FragmentMassRow] = []
            for smiles in strings:
                graph = chemistry.graph(smiles)
                base, depth_two, depth_one = fragment_rows(graph)
                depth_two.tofile(raw_two)
                depth_one.tofile(raw_one)
                base.tofile(raw_base)
                position += 1
                offsets_two[position] = offsets_two[position - 1] + len(depth_two)
                offsets_one[position] = offsets_one[position - 1] + len(depth_one)
                weights, bonds = (
                    graph
                    if graph is not None
                    else (
                        np.zeros(0),
                        np.empty((0, 2), dtype=np.int32),
                    )
                )
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
        for name, raw, offsets in (
            ("frag", raw_two, offsets_two),
            ("f1", raw_one, offsets_one),
        ):
            save_stream(raw, destination / f"{name}_mass.npy", dtype=np.float32)
            np.save(destination / f"{name}_off.npy", offsets, allow_pickle=False)
        save_stream(raw_base, destination / "base_mass.npy", dtype=np.float64)
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": position,
        "depth_two_masses": int(cast(np.int64, offsets_two[-1])),
        "depth_one_masses": int(cast(np.int64, offsets_one[-1])),
        "base_masses": int(cast(np.int64, offsets_two[-1])),
        "capacity": 200_000,
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


class FragmentChemistry:
    """Read complete fixed-mass graphs in the isolated preparation environment."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(
                f"Main fragment preparation requires RDKit {IDENTITY_RDKIT}.",
            )
        self.chem = MolecularContext(version).chem
        self.masses = ELEMENT_MASSES

    def graph(
        self,
        smiles: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]] | None:
        """Return the ordered graph or an explicit chemistry rejection.

        Args:
          smiles: Original molecular string without standardization or isotope mass.

        Returns:
          graph: Complete fixed atom weights and ordered bonds, or rejected input.

        """
        molecule = self.chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        weights = np.zeros(molecule.GetNumAtoms())
        for atom in molecule.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in self.masses:
                return None
            weights[atom.GetIdx()] = (
                self.masses[symbol] + atom.GetTotalNumHs() * self.masses["H"]
            )
        edges = [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in molecule.GetBonds()
        ]
        bonds = np.array(edges, dtype=np.int32).reshape(len(edges), 2)
        return weights, bonds


def fragment_masses(
    weights: NDArray[np.float64],
    *,
    bonds: NDArray[np.int32],
    depth: int = 2,
    capacity: int = 200_000,
) -> NDArray[np.float64]:
    """Enumerate the ordered, capped and rounded complete fragment set.

    Args:
      weights: Complete fixed atom masses, including attached hydrogens.
      bonds: Undirected atom-index pairs in scientific-library encounter order.
      depth: Include two-bond cuts at values at least two; always include one cut.
      capacity: Limit encountered masses before rounding, sorting and deduplication.

    Returns:
      masses: Sorted unique masses rounded to one thousandth of a dalton.

    """
    atom_masses = from_plain(cast(object, weights.tolist()), list[float])
    edges = from_plain(cast(object, bonds.tolist()), list[tuple[int, int]])
    parent = 0.0
    for mass in atom_masses:
        parent += mass
    result = [parent]
    for first in range(len(edges)):
        removed: list[tuple[int, ...]] = [(first,)]
        if depth >= 2:
            removed.extend((first, second) for second in range(first + 1, len(edges)))
        for cut in removed:
            components = _component_masses(atom_masses, edges=edges, removed=cut)
            if len(components) > 1:
                result.extend(components[: max(0, capacity - len(result))])
    rounded = np.floor(np.array(result) * 1000.0 + 0.5) / 1000.0
    return np.unique(rounded)


def fragment_rows(
    graph: tuple[NDArray[np.float64], NDArray[np.int32]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
    """Return the generated base vector and its depth-two and depth-one rows.

    Args:
      graph: Complete admitted weights and bonds, or explicit chemistry rejection.

    Returns:
      base: Every rounded fragment before the float32 cast.
      depth_two: Every rounded depth-two fragment stored as float32.
      depth_one: Every rounded depth-one fragment stored as float32.

    """
    base = base_fragment_masses(graph)
    depth_two = base.astype(np.float32)
    if graph is None or not len(graph[1]):
        return base, depth_two, depth_two.copy()
    weights, bonds = graph
    depth_one = fragment_masses(weights, bonds=bonds, depth=1).astype(np.float32)
    return base, depth_two, depth_one


def base_fragment_masses(
    graph: tuple[NDArray[np.float64], NDArray[np.int32]] | None,
) -> NDArray[np.float64]:
    """Return the complete pre-cast float64 fragment vector for generated MAIN94.

    Args:
      graph: Complete fixed-mass graph, or an explicit rejected structure.

    Returns:
      masses: Complete rounded float64 fragments; no-bond parents remain unrounded.

    """
    if graph is None:
        return np.zeros(0, dtype=np.float64)
    weights, bonds = graph
    return (
        fragment_masses(weights, bonds=bonds)
        if len(bonds)
        else np.array([weights.sum()])
    )


def _component_masses(
    weights: Sequence[float],
    *,
    edges: Sequence[tuple[int, int]],
    removed: tuple[int, ...],
) -> list[float]:
    roots = list(range(len(weights)))
    for index, edge in enumerate(edges):
        if index in removed:
            continue
        left, right = edge
        while left != roots[left]:
            left = roots[left]
        while right != roots[right]:
            right = roots[right]
        roots[max(left, right)] = min(left, right)
    components: dict[int, float] = {}
    for index, mass in enumerate(weights):
        root = index
        while root != roots[root]:
            root = roots[root]
        components[root] = components.get(root, 0.0) + mass
    return list(components.values())
