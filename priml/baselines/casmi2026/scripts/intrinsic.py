"""Generate intrinsic chemistry and loss capabilities using isolated primitives."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib

from treekle import from_plain

import numpy as np

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT, record_fields
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
    from rdkit import Chem
    from rdkit.Chem.Crippen import MolLogP
    from rdkit.Chem.rdMolDescriptors import (
        CalcFractionCSP3,
        CalcNumAromaticRings,
        CalcNumHBA,
        CalcNumHBD,
        CalcNumRotatableBonds,
        CalcTPSA,
    )
    from rdkit.Chem.Scaffolds.MurckoScaffold import (
        GetScaffoldForMol,
        MakeScaffoldGeneric,
    )
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    MolLogP = lazy_import("rdkit.Chem.Crippen", "MolLogP")
    CalcFractionCSP3 = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcFractionCSP3")
    CalcNumAromaticRings = lazy_import(
        "rdkit.Chem.rdMolDescriptors",
        "CalcNumAromaticRings",
    )
    CalcNumHBA = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcNumHBA")
    CalcNumHBD = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcNumHBD")
    CalcNumRotatableBonds = lazy_import(
        "rdkit.Chem.rdMolDescriptors",
        "CalcNumRotatableBonds",
    )
    CalcTPSA = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcTPSA")
    GetScaffoldForMol = lazy_import(
        "rdkit.Chem.Scaffolds.MurckoScaffold",
        "GetScaffoldForMol",
    )
    MakeScaffoldGeneric = lazy_import(
        "rdkit.Chem.Scaffolds.MurckoScaffold",
        "MakeScaffoldGeneric",
    )


_THIS: Final = Path(__file__).resolve()


def prepare_intrinsic(input_path: Path, *, output: Path, version: str) -> None:
    """Write every intrinsic molecular descriptor and diagnostic-loss capability.

    Args:
      input_path: Parquet input with a SMILES column.
      output: New external dataset directory.
      version: Required isolated RDKit release.

    """
    chemistry = IntrinsicChemistry(version)
    values = from_plain(
        cast(object, parquet.read_table(input_path).column("smiles").to_pylist()),
        list[str | None],
    )
    smiles = [value for value in dict.fromkeys(values) if value is not None]
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    records = np.lib.format.open_memmap(
        destination / "records.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(smiles), len(record_fields())),
    )
    capabilities = np.lib.format.open_memmap(
        destination / "capabilities.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(smiles), len(capability_rules())),
    )
    for index, value in enumerate(smiles):
        records[index] = chemistry.descriptors(value)
        capabilities[index] = chemistry.capabilities(value)
    parquet.write_table(
        Table.from_pydict({"smiles": smiles}),
        destination / "molecules.parquet",
    )
    _ = (destination / "record_names.txt").write_text("\n".join(record_fields()) + "\n")
    _ = (destination / "capability_names.txt").write_text(
        "\n".join(name for name, _ in capability_rules()) + "\n",
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": chemistry.version,
        "numpy": np.__version__,
        "rows": len(smiles),
        "record_fields": list(record_fields()),
        "capability_fields": [name for name, _ in capability_rules()],
        "inputs": input_digests([input_path]),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("rdkit_backend.py"),
                _THIS.parent.parent / "fragnet.py",
            ),
        ),
    }
    write_manifest(destination, manifest)


class IntrinsicChemistry:
    """Own the standard descriptors, scaffolds and substructure query primitives."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(
                f"Intrinsic chemistry preparation requires RDKit {IDENTITY_RDKIT}.",
            )
        context = MolecularContext(version)
        self.version = context.version
        self.chem = context.chem
        fields = record_fields()
        self.patterns = tuple(
            (fields.index(name), self.chem.MolFromSmarts(smarts))
            for name, smarts in descriptor_rules()
        )
        self.loss_patterns = tuple(
            tuple(self.chem.MolFromSmarts(value) for value in smarts.split("|"))
            for _, smarts in capability_rules()
        )

    def descriptors(self, smiles: str) -> NDArray[np.float32]:
        """Compute the complete 34-channel molecular record with finite defaults.

        Args:
          smiles: Exact candidate structure string, without additional normalization.

        Returns:
          record: All declared descriptors; zero if parsing or descriptor evaluation fails.

        """
        values = np.zeros(len(record_fields()), dtype=np.float32)
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None:
                return values
            scaffold, generic, scaffold_size = self._scaffold(molecule)
            values[:4] = [*_hash_halves("S:" + scaffold), *_hash_halves("G:" + generic)]
            heavy = molecule.GetNumHeavyAtoms()
            values[4:10] = [
                molecule.GetRingInfo().NumRings(),
                CalcNumAromaticRings(molecule),
                scaffold_size,
                scaffold_size / max(heavy, 1),
                sum(max(atom.GetFormalCharge(), 0) for atom in molecule.GetAtoms()),
                -sum(min(atom.GetFormalCharge(), 0) for atom in molecule.GetAtoms()),
            ]
            for index, pattern in self.patterns:
                values[index] = len(molecule.GetSubstructMatches(pattern))
            elements = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
            values[19:30] = [
                *(elements[element] for element in ("N", "O", "S", "P")),
                sum(elements[element] for element in ("F", "Cl", "Br", "I")),
                MolLogP(molecule),
                CalcTPSA(molecule),
                CalcNumHBD(molecule),
                CalcNumHBA(molecule),
                CalcNumRotatableBonds(molecule),
                CalcFractionCSP3(molecule),
            ]
            values[33] = heavy
        except Exception:  # noqa: BLE001 -- Scientific descriptor failures return the recipe's complete zero record.
            return np.zeros(len(record_fields()), dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def capabilities(self, smiles: str) -> NDArray[np.uint8]:
        """Count every declared diagnostic-loss capability with the 255-match cap.

        Args:
          smiles: Exact candidate structure string.

        Returns:
          counts: Complete ordered capability counts, including summed alternatives.

        """
        counts = np.zeros(len(self.loss_patterns), dtype=np.uint8)
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None:
                return counts
            for index, patterns in enumerate(self.loss_patterns):
                counts[index] = min(
                    255,
                    sum(
                        len(molecule.GetSubstructMatches(pattern, maxMatches=255))
                        for pattern in patterns
                    ),
                )
        except Exception:  # noqa: BLE001 -- Scientific matching failures return the complete zero capability record.
            return np.zeros(len(self.loss_patterns), dtype=np.uint8)
        return counts

    def _scaffold(self, molecule: Chem.Mol) -> tuple[str, str, int]:
        try:
            scaffold = GetScaffoldForMol(molecule)
            if scaffold is None:
                return "", "", 0
            generic = (
                self.chem.MolToSmiles(MakeScaffoldGeneric(scaffold))
                if scaffold.GetNumAtoms()
                else ""
            )
            return self.chem.MolToSmiles(scaffold), generic, scaffold.GetNumHeavyAtoms()
        except Exception:  # noqa: BLE001 -- Scaffold failures retain the recipe's independent empty-scaffold hashes.
            return "", "", 0


def descriptor_rules() -> tuple[tuple[str, str], ...]:
    """Declare all intrinsic functional-group query inputs.

    Returns:
      queries: Ordered descriptor names and standard SMARTS inputs.

    """
    return (
        ("quat_n", "[NX4+;!$([N+][O-])]"),
        (
            "basic_amine",
            "[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(N#*);!$(N=*);!$([N]-[N,O])]",
        ),
        ("amide_n", "[NX3][CX3]=[OX1]"),
        ("arom_n", "[n]"),
        ("cooh", "[CX3](=O)[OX2H1,OX1-]"),
        ("so3h", "[SX4](=O)(=O)[OX2H1,OX1-]"),
        ("po3h", "[PX4](=O)([OX2H1,OX1-])"),
        ("phenol_oh", "[OX2H1]c"),
        ("alc_oh", "[OX2H1][CX4]"),
        ("glycoside", "[OX2;R][CX4;R]([OX2,NX3,#6;!R])[CX4;R][CX4;R]([OX2H1,OX2])"),
        ("ester", "[#6][CX3](=O)[OX2][#6]"),
        ("methoxy", "[CH3][OX2][#6]"),
    )


def capability_rules() -> tuple[tuple[str, str], ...]:
    """Declare the complete diagnostic-loss substructure queries and alternatives.

    Returns:
      queries: Complete ordered capability names and alternative SMARTS inputs.

    """
    return (
        ("oh_aliph", "[OX2H1][CX4]"),
        ("cooh", "[CX3](=O)[OX2H1,OX1-]"),
        ("phenol", "[OX2H1]c"),
        ("nh2", "[#7;H2,H3;!$([#7]-[#7,#8]);!$([#7]=*)]"),
        ("carbonyl", "[#6][CX3](=[OX1])"),
        ("ester", "[#6][CX3](=O)[OX2][#6]"),
        ("acetyl", "[CH3][CX3](=O)[OX2,NX3]"),
        ("ome", "[CH3][OX2][#6]"),
        ("ome_ar", "[CH3][OX2]c"),
        ("nme", "[CH3][NX3,NX4+]"),
        (
            "hex",
            "[O,N,n][C;R]1[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]|[O,N,n][C;R]1([CH2][OX2])[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]1[OX2]",
        ),
        ("dhex", "[O,N,n][C;R]1[O;R][C;R]([CH3])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]"),
        (
            "pent",
            "[O,N,n][C;R]1[O;R][CH2;R][C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]|[O,N,n][CH1;R]1[O;R][C;R]([CH2][OX2])[C;R]([OX2])[C;R]1[OX2]",
        ),
        (
            "glca",
            "[O,N,n][C;R]1[O;R][C;R]([CX3](=O)[OX2,OX1-])[C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]",
        ),
        (
            "cglyc",
            "c[C;R]1[O;R][C;R][C;R]([OX2])[C;R]([OX2])[C;R]1[OX2]|c[C;R]1[O;R][C;R][C;R]([OX2])[C;R]1[OX2]",
        ),
        ("malonyl", "[OX2][CX3](=O)[CH2][CX3](=O)[OX2H1,OX1-]"),
        ("coum", "[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])[cH][cH]1"),
        ("caff", "[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])c([OX2H1])[cH]1"),
        (
            "ferul",
            "[OX2,NX3][CX3](=O)[CX3]=[CX3]c1[cH][cH]c([OX2H1])c([OX2][CH3])[cH]1",
        ),
        ("galloyl", "[OX2][CX3](=O)c1cc([OX2H1])c([OX2H1])c([OX2H1])c1"),
        ("sulf", "[SX4](=O)(=O)[OX2H1,OX1-]"),
        ("phos", "[PX4](=O)[OX2H1,OX1-]"),
    )


def _hash_halves(value: str) -> tuple[float, float]:
    hashed = int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=6).digest(),
        "little",
    )
    return float(hashed >> 24), float(hashed & 0xFF_FFFF)
