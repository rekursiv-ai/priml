"""Generate independent identity/product decisions in an isolated scientific SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json
import logging

from treekle import from_plain, to_plain

import numpy as np

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.pubchem_data import PubChemChemistry
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyarrow import Table, parquet
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    import pyarrow as pa

    from priml.baselines.casmi2026.custom_types import (
        IdentityRow,
        MetricIdentity,
    )
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    ExactMolWt = lazy_import("rdkit.Chem.Descriptors", "ExactMolWt")
    CalcMolFormula = lazy_import("rdkit.Chem.rdMolDescriptors", "CalcMolFormula")
    pa = lazy_import("pyarrow")


logger = logging.getLogger(__name__)

_THIS: Final = Path(__file__).resolve()


class IdentityChemistry:
    """Preserve separate standardization, property, fingerprint and identity outcomes."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"Identity preparation requires RDKit {IDENTITY_RDKIT}.")
        self.raw = PubChemChemistry(version)
        self.chemistry = self.raw.chemistry

    def record(self, smiles: str) -> tuple[IdentityRow, NDArray[np.uint8]]:
        """Prepare every raw and standardized decision without a shared validity gate.

        Args:
          smiles: Exact input string, including empty and chemically invalid requests.

        Returns:
          row: Every independent raw and standardized molecular decision.
          fingerprint: Complete packed16551-bit standardized fingerprint or zero row.

        """
        try:
            standardized = self.chemistry.standardize(smiles)
        except Exception:
            logger.debug("Identity standardization failed.", exc_info=True)
            standardized = None
        properties = self.properties(standardized) if standardized is not None else None
        fingerprint = (
            self.raw.raw_fingerprint(standardized) if standardized is not None else None
        )
        row: IdentityRow = {
            "smiles": smiles,
            "standardized": standardized,
            "key": self.raw.score_key(smiles),
            "formula": self.formula(smiles),
            "metric": json.dumps(
                to_plain(self.identity(smiles)),
                separators=(",", ":"),
            ),
            "standard_key": self.raw.score_key(standardized)
            if standardized is not None
            else None,
            "standard_metric": json.dumps(
                to_plain(
                    self.identity(standardized) if standardized is not None else None,
                ),
                separators=(",", ":"),
            ),
            "standard_formula_lookup": self.formula(standardized)
            if standardized is not None
            else "",
            "properties_valid": properties is not None,
            "standard_formula": properties[0] if properties is not None else "",
            "standard_mass": properties[1] if properties is not None else 0.0,
            "standard_heavy": properties[2] if properties is not None else 0,
            "fingerprint_valid": fingerprint is not None,
        }
        return row, fingerprint if fingerprint is not None else np.zeros(2069, np.uint8)

    def formula(self, smiles: str) -> str:
        """Prepare the final formula gate independently of exact mass/heavy count.

        Args:
          smiles: Raw or standardized input, without additional normalization.

        Returns:
          formula: Scientific formula or the exact input on known SDK failure.

        """
        try:
            molecule = self.chemistry.chem.MolFromSmiles(smiles)
            if molecule is None:
                return smiles
            return CalcMolFormula(molecule)
        except Exception:
            logger.debug("Identity formula failed.", exc_info=True)
            return smiles

    def properties(self, smiles: str) -> tuple[str, float, int] | None:
        """Compute formula, mass and heavy count independently of fingerprint/key.

        Args:
          smiles: Original or explicitly standardized input string.

        Returns:
          properties: Formula, exact mass and heavy count, or known SDK rejection.

        """
        try:
            molecule = self.chemistry.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            return (
                CalcMolFormula(molecule),
                float(ExactMolWt(molecule)),
                molecule.GetNumHeavyAtoms(),
            )
        except Exception:
            logger.debug("Identity properties failed.", exc_info=True)
            return None

    def identity(self, smiles: str) -> MetricIdentity | None:
        """Remove stereo before tautomer canonicalization, then count heavy elements.

        Args:
          smiles: Exact input; no largest-fragment or uncharging operation is applied.

        Returns:
          identity: Official key and sorted heavy-element counts, or known rejection.

        """
        try:
            molecule = self.chemistry.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            self.chemistry.chem.RemoveStereochemistry(molecule)
            key = self.chemistry.chem.MolToInchiKey(
                self.chemistry.tautomers.Canonicalize(molecule),
            )[:14]
        except (ValueError, RuntimeError, TypeError):
            return None
        if not key:
            return None
        original = self.chemistry.chem.MolFromSmiles(smiles)
        if original is None:
            return None
        composition: dict[str, int] = {}
        for atom in original.GetAtoms():
            element = atom.GetSymbol()
            if element != "H":
                composition[element] = composition.get(element, 0) + 1
        return key, tuple(sorted(composition.items()))


def prepare_identity(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Write all ordered requests, independent decisions and complete fingerprints.

    Args:
      input_path: Parquet smiles requests, preserving all original rows and duplicates.
      output: New external dataset directory; existing outputs are never replaced.
      version: Explicit isolated scientific-library release.
      batch_rows: Maximum complete metadata rows per bounded Arrow group.

    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    chemistry = IdentityChemistry(version)
    source = parquet.ParquetFile(input_path)
    fingerprints = np.lib.format.open_memmap(
        destination / "fingerprints.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(source.metadata.num_rows, 2069),
    )
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("standardized", pa.string()),
            ("key", pa.string()),
            ("formula", pa.string()),
            ("metric", pa.string()),
            ("standard_key", pa.string()),
            ("standard_metric", pa.string()),
            ("standard_formula_lookup", pa.string()),
            ("properties_valid", pa.bool_()),
            ("standard_formula", pa.string()),
            ("standard_mass", pa.float64()),
            ("standard_heavy", pa.int64()),
            ("fingerprint_valid", pa.bool_()),
        ],
    )
    index = 0
    with parquet.ParquetWriter(destination / "molecules.parquet", schema) as writer:
        for batch in source.iter_batches(batch_size=batch_rows, columns=["smiles"]):
            rows: list[dict[str, object]] = []
            for value in from_plain(
                cast(object, batch.column(0).to_pylist()),
                list[str],
            ):
                row, fingerprint = chemistry.record(value)
                rows.append(dict(row))
                fingerprints[index] = fingerprint
                index += 1
            writer.write_table(Table.from_pylist(rows, schema=schema))
    fingerprints.flush()
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "raw_bits": 16_551,
        "numpy": np.__version__,
        "rows": index,
        "inputs": input_digests([input_path]),
        "generator": source_digests(
            (
                _THIS,
                _THIS.with_name("pubchem_data.py"),
                _THIS.with_name("rdkit_backend.py"),
            ),
        ),
    }
    write_manifest(destination, manifest)
