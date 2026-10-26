"""Prepare both distinct simulator admissions in the isolated chemistry process."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import hashlib

from treekle import from_plain

from priml.baselines.casmi2026.isolated import SIMULATOR_RDKIT
from priml.baselines.casmi2026.scripts.ice_data import IceChemistry
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.paths import validated_output_path


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from pyarrow import Table, parquet

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import SimulatorAdmissionRow
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    pa = lazy_import("pyarrow")
    parquet = lazy_import("pyarrow.parquet")


def prepare_admission(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Write every requested paired decision and canonical cache observation.

    Args:
      input_path: Parquet table with every requested SMILES, including duplicates.
      output: New external directory for admission records and provenance.
      version: Required isolated scientific-library release.
      batch_rows: Maximum number of complete records per Arrow row group.

    """
    if version != SIMULATOR_RDKIT:
        raise ValueError(f"Simulator admission requires RDKit {SIMULATOR_RDKIT}.")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    chemistry = AdmissionChemistry(version, directory=destination)
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("ice_accepted", pa.bool_()),
            ("ice_reason", pa.string()),
            ("glacier_accepted", pa.bool_()),
            ("glacier_reason", pa.string()),
            ("glacier_canonical", pa.string()),
            ("glacier_heavy_atoms", pa.int64()),
        ],
    )
    total = 0
    with parquet.ParquetWriter(destination / "admission.parquet", schema) as writer:
        for batch in parquet.ParquetFile(input_path).iter_batches(
            batch_size=batch_rows,
        ):
            values = from_plain(
                batch.column("smiles").to_pylist(),
                list[str],
            )
            rows = [chemistry.row(value) for value in values]
            writer.write_table(Table.from_pylist(rows, schema=schema))
            total += len(rows)
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "rows": total,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("ice_data.py"))),
        "tautomer_transforms_sha256": hashlib.sha256(
            (destination / "tautomer-transforms.txt").read_bytes(),
        ).hexdigest(),
    }
    write_manifest(destination, manifest)


class AdmissionChemistry:
    """Keep cheap admission separate from graph and physical featurization."""

    def __init__(self, version: str, *, directory: Path) -> None:
        context = IceChemistry(version, directory=directory)
        self.chem = context.chem
        self.tautomers = context.tautomers
        self.elements = frozenset(context.elements)
        self._glacier_cache: dict[str, tuple[str | None, int, str]] = {}

    def row(self, smiles: str) -> SimulatorAdmissionRow:
        """Return complete paired cheap decisions for the original input string.

        Args:
          smiles: Original string before either simulator canonicalization path.

        Returns:
          row: Both complete decisions, reasons and GLACIER canonical cache fields.

        """
        accepted, reason = self._ice(smiles)
        canonical, heavy, glacier_reason = self._glacier(smiles)
        return {
            "smiles": smiles,
            "ice_accepted": accepted,
            "ice_reason": reason,
            "glacier_accepted": canonical is not None,
            "glacier_reason": glacier_reason,
            "glacier_canonical": canonical,
            "glacier_heavy_atoms": heavy,
        }

    def _ice(self, smiles: str) -> tuple[bool, str]:
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None or molecule.GetNumAtoms() == 0:
                return False, "rdkit parse"
            unsupported = sorted(
                {atom.GetSymbol() for atom in molecule.GetAtoms()} - self.elements,
            )
            if unsupported:
                return False, f"elements {unsupported}"
            self.chem.RemoveStereochemistry(molecule)
            molecule = self.chem.MolFromSmiles(self.chem.MolToSmiles(molecule))
            if molecule is None:
                return False, "inchi round trip"
            molecule = self.chem.MolFromInchi(self.chem.MolToInchi(molecule))
            if molecule is None:
                return False, "inchi round trip"
            canonical = self.chem.MolToSmiles(self.tautomers.Canonicalize(molecule))
            if not canonical or self.chem.MolFromSmiles(canonical) is None:
                return False, "inchi round trip"
            return True, ""
        except Exception as error:  # noqa: BLE001 -- Scientific-extension failures are explicit per-input rejection records.
            return False, f"exception {error!r}"[:200]

    def _glacier(self, smiles: str) -> tuple[str | None, int, str]:
        if smiles in self._glacier_cache:
            return self._glacier_cache[smiles]
        result = self._glacier_decision(smiles)
        self._glacier_cache[smiles] = result
        return result

    def _glacier_decision(self, smiles: str) -> tuple[str | None, int, str]:
        try:
            if not smiles:
                return None, 0, "empty"
            canonical = self._glacier_canonical(smiles)
            if canonical is None:
                molecule = self.chem.MolFromSmiles(smiles)
                if molecule is None:
                    return None, 0, "rdkit parse"
                if self.chem.RemoveHs(molecule).GetNumAtoms() > 100:
                    return None, 0, "> 100 heavy atoms"
                return None, 0, "element outside VALID_ATOM_NUM"
            molecule = self.chem.MolFromSmiles(canonical)
            if molecule is None or molecule.GetNumAtoms() == 0:
                return None, 0, "canonical re-parse"
            return canonical, molecule.GetNumHeavyAtoms(), ""
        except Exception as error:  # noqa: BLE001 -- Scientific-extension failures retain the complete per-input cache record.
            return None, 0, f"exception {error!r}"[:200]

    def _glacier_canonical(self, smiles: str) -> str | None:
        try:
            molecule = self.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            if any(
                atom.GetSymbol() not in self.elements for atom in molecule.GetAtoms()
            ):
                return None
            molecule = self.chem.RemoveHs(molecule)
            if molecule.GetNumAtoms() > 100:
                return None
            return self.chem.MolToSmiles(molecule)
        except Exception:  # noqa: BLE001 -- Failed scientific canonicalization is classified by the independent admission pass.
            return None
