"""Generate intrinsic element graphs and complete analog structural data independently."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import tempfile

from treekle import from_plain

from priml.baselines.casmi2026.isolated import (
    IDENTITY_RDKIT,
    AnalogRecord,
    ElementGraph,
    formula_vector,
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
    from rdkit import Chem

    import numpy as np
    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import AnalogMetadata
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")
    np = lazy_import("numpy")


_THIS: Final = Path(__file__).resolve()


def prepare_analog(input_path: Path, *, output: Path, version: str) -> None:
    """Generate ordered graph hashes, formulas and full/core numeric graphs.

    Args:
      input_path: Parquet with smiles and nullable molecular formula columns.
      output: New external dataset directory.
      version: Required isolated scientific-library release.

    """
    chemistry = AnalogChemistry(version)
    source = parquet.ParquetFile(input_path)
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True)
    count = source.metadata.num_rows
    fixed = {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(count, width),
        )
        for name, dtype, width in (
            ("hashes", np.int64, 8),
            ("numbers", np.int16, 8),
            ("formula", np.int16, 10),
        )
    }
    offsets = {
        name: np.zeros(count + 1, dtype=np.int64)
        for name in ("graph_atom", "graph_bond", "core_atom", "core_bond")
    }
    layouts = (
        ("graph_elements", np.int16, 1),
        ("graph_bonds", np.int32, 2),
        ("core_elements", np.int16, 1),
        ("core_bonds", np.int32, 2),
        ("core_mask", np.bool_, 1),
    )
    schema = pa.schema(
        [
            (name, pa.string())
            for name in (
                "smiles",
                "formula_input",
                "core_smiles",
                "element_smiles",
                "query_smiles",
            )
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
        for batch in source.iter_batches(batch_size=256, columns=["smiles", "formula"]):
            smiles = cast(Sequence[str], batch.column(0).to_pylist())
            formulas = cast(Sequence[str | None], batch.column(1).to_pylist())
            metadata: list[AnalogMetadata] = []
            for index, value in enumerate(smiles):
                row = chemistry.record(value, formula=formulas[index])
                for name, values in (
                    ("hashes", row.hashes),
                    ("numbers", row.numbers),
                    ("formula", row.formula),
                ):
                    fixed[name][row_index] = values
                for prefix, graph in (("graph", row.graph), ("core", row.core)):
                    elements = (
                        graph.elements
                        if graph is not None
                        else np.empty(0, dtype=np.int16)
                    )
                    bonds = (
                        graph.bonds
                        if graph is not None
                        else np.empty((0, 2), dtype=np.int32)
                    )
                    elements.tofile(raw[f"{prefix}_elements"])
                    bonds.tofile(raw[f"{prefix}_bonds"])
                    for kind, size in (("atom", len(elements)), ("bond", len(bonds))):
                        selected = offsets[f"{prefix}_{kind}"]
                        selected[row_index + 1] = selected[row_index] + size
                row.core_mask.tofile(raw["core_mask"])
                metadata.append(
                    {
                        "smiles": value,
                        "formula_input": formulas[index],
                        "core_smiles": row.core_smiles,
                        "element_smiles": row.element_smiles,
                        "query_smiles": row.query_smiles,
                    },
                )
                row_index += 1
            writer.write_table(Table.from_pylist(metadata, schema=schema))
        for name, dtype, width in layouts:
            save_stream(
                raw[name],
                destination / f"{name}.npy",
                dtype=dtype,
                width=width,
            )
    for name, values in offsets.items():
        np.save(destination / f"{name}_off.npy", values)
    for values in fixed.values():
        values.flush()
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
            ),
        ),
    }
    write_manifest(destination, manifest)


class AnalogChemistry:
    """Use standard molecular I/O and independently assembled element-only graphs."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Analog structural data requires RDKit {IDENTITY_RDKIT}.")
        self.chem = MolecularContext(version).chem

    def record(self, smiles: str, *, formula: str | None) -> AnalogRecord:
        """Build every graph identity, formula vector and containment input.

        Args:
          smiles: Exact candidate or training structure before representation changes.
          formula: Supplied candidate formula; invalid strings retain a negative row.

        Returns:
          record: Complete fields, including explicit empty records for rejected graphs.

        """
        hashes = np.zeros(8, dtype=np.int64)
        numbers = np.zeros(8, dtype=np.int16)
        core_smiles, element_smiles, query_smiles = "", "", ""
        graph: ElementGraph | None = None
        core_graph: ElementGraph | None = None
        mask = np.zeros(0, dtype=np.bool_)
        molecule = self.chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
        if molecule is not None:
            members = [
                atom.GetIdx()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() != 1
            ]
            element = self._standalone(molecule, members=members)
            if element.GetNumAtoms():
                neighbors = _neighbors(element)
                core = _core(neighbors)
                rings = _neighbors(element, ring_only=True)
                systems = _components(
                    rings,
                    active={index for index, values in enumerate(rings) if values},
                )
                core_smiles = self._canonical(element, members=core)
                if core:
                    hashes[0] = _identity("E:" + core_smiles)
                    hashes[1] = _identity(
                        "G:" + self._canonical(element, members=core, generic=True),
                    )
                ring_types = sorted(
                    {
                        (
                            -len(system),
                            _identity("R:" + self._canonical(element, members=system)),
                        )
                        for system in systems
                    },
                )
                unique = list(dict.fromkeys(value for _, value in ring_types))
                hashes[2 : 2 + min(6, len(unique))] = unique[:6]
                ring_elements = [
                    element.GetAtomWithIdx(index).GetAtomicNum()
                    for index, values in enumerate(rings)
                    if values
                ]
                numbers[:] = [
                    element.GetNumBonds()
                    - element.GetNumAtoms()
                    + len(_components(neighbors, active=set(range(len(neighbors))))),
                    len(systems),
                    len(core),
                    ring_elements.count(7),
                    ring_elements.count(8),
                    ring_elements.count(16),
                    max((len(system) for system in systems), default=0),
                    element.GetNumAtoms(),
                ]
                element_smiles = self.chem.MolToSmiles(
                    element,
                    canonical=True,
                    isomericSmiles=False,
                )
                query_smiles = core_smiles
                graph = self._numeric(element_smiles)
                core_graph = self._numeric(query_smiles) if query_smiles else None
                if graph is not None:
                    canonical_neighbors: list[list[int]] = [
                        [] for _ in range(len(graph.elements))
                    ]
                    for left, right in from_plain(
                        cast(object, graph.bonds.tolist()),
                        list[tuple[int, int]],
                    ):
                        canonical_neighbors[left].append(right)
                        canonical_neighbors[right].append(left)
                    mask = np.zeros(len(graph.elements), dtype=np.bool_)
                    mask[_core(canonical_neighbors)] = True
        return AnalogRecord(
            smiles=smiles,
            formula_input=formula,
            hashes=hashes,
            numbers=numbers,
            formula=formula_vector(formula),
            core_smiles=core_smiles,
            element_smiles=element_smiles,
            query_smiles=query_smiles,
            graph=graph,
            core=core_graph,
            core_mask=mask,
        )

    def _standalone(
        self,
        molecule: Chem.Mol,
        *,
        members: Sequence[int],
        generic: bool = False,
    ) -> Chem.Mol:
        editable = self.chem.RWMol()
        indices: dict[int, int] = {}
        for index in sorted(members):
            atom = self.chem.Atom(
                6 if generic else molecule.GetAtomWithIdx(index).GetAtomicNum(),
            )
            atom.SetNoImplicit(True)
            indices[index] = editable.AddAtom(atom)
        edges: set[tuple[int, int]] = set()
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left in indices and right in indices:
                edge = (
                    min(indices[left], indices[right]),
                    max(indices[left], indices[right]),
                )
                if edge[0] != edge[1] and edge not in edges:
                    editable.AddBond(*edge, self.chem.BondType.SINGLE)
                    edges.add(edge)
        result = editable.GetMol()
        result.UpdatePropertyCache(strict=False)
        self.chem.FastFindRings(result)
        return result

    def _canonical(
        self,
        molecule: Chem.Mol,
        *,
        members: Sequence[int],
        generic: bool = False,
    ) -> str:
        if not members:
            return ""
        return self.chem.MolToSmiles(
            self._standalone(molecule, members=members, generic=generic),
            canonical=True,
            isomericSmiles=False,
        )

    def _numeric(self, smiles: str) -> ElementGraph | None:
        molecule = self.chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            return None
        return ElementGraph(
            elements=np.array(
                [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                dtype=np.int16,
            ),
            bonds=np.array(
                [
                    [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                    for bond in molecule.GetBonds()
                ],
                dtype=np.int32,
            ).reshape(-1, 2),
        )


def _identity(text: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
        & ((1 << 62) - 1)
    ) or 1


def _neighbors(molecule: Chem.Mol, *, ring_only: bool = False) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        if ring_only and not bond.IsInRing():
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbors[left].append(right)
        neighbors[right].append(left)
    return neighbors


def _core(neighbors: Sequence[Sequence[int]]) -> list[int]:
    degrees = [len(values) for values in neighbors]
    removed: set[int] = set()
    leaves = deque(index for index, degree in enumerate(degrees) if degree <= 1)
    while leaves:
        index = leaves.popleft()
        if index in removed:
            continue
        removed.add(index)
        for neighbor in neighbors[index]:
            degrees[neighbor] -= 1
            if neighbor not in removed and degrees[neighbor] <= 1:
                leaves.append(neighbor)
    return [index for index in range(len(neighbors)) if index not in removed]


def _components(
    neighbors: Sequence[Sequence[int]],
    *,
    active: set[int],
) -> list[list[int]]:
    remaining = set(active)
    result: list[list[int]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: list[int] = []
        remaining.remove(root)
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in neighbors[index]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result
