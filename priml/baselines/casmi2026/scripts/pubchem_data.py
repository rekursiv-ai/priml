"""Prepare independent PubChem partial/raw/key primitives in an isolated SDK."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import logging

import numpy as np

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import Chemistry
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyarrow import Table, parquet
    from rdkit.Chem.MACCSkeys import GenMACCSKeys

    import pyarrow as pa
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    GenMACCSKeys = lazy_import("rdkit.Chem.MACCSkeys", "GenMACCSKeys")
    pa = lazy_import("pyarrow")


logger = logging.getLogger(__name__)

_THIS: Final = Path(__file__).resolve()


class PubChemChemistry:
    """Keep parsing, full raw fingerprint and score-key validity independent."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(f"PubChem preparation requires RDKit {IDENTITY_RDKIT}.")
        self.chemistry = Chemistry(version)

    def record(
        self,
        smiles: str,
    ) -> tuple[bool, NDArray[np.uint8], bool, NDArray[np.uint8], str | None]:
        """Return parsed/complete ECFP4/raw-valid/complete raw bits/key separately.

        Args:
          smiles: Exact original structure string without identity normalization.

        Returns:
          parsed: Whether the independent partial-fingerprint parser accepted it.
          ecfp4: Complete packed4096-bit partial fingerprint, including valid zero rows.
          raw_valid: Whether all seven raw fingerprint generators succeeded.
          raw_fp: Complete packed16551-bit raw fingerprint, or an explicit zero row.
          key: Independent source tautomer-canonical InChIKey14, or None.

        """
        molecule = self.chemistry.chem.MolFromSmiles(smiles)
        ecfp4 = (
            np.packbits(self.chemistry.generators[0].GetFingerprintAsNumPy(molecule))
            if molecule is not None
            else np.zeros(512, np.uint8)
        )
        raw = self.raw_fingerprint(smiles)
        return (
            molecule is not None,
            ecfp4,
            raw is not None,
            raw if raw is not None else np.zeros((16_551 + 7) // 8, np.uint8),
            self.score_key(smiles),
        )

    def raw_fingerprint(self, smiles: str) -> NDArray[np.uint8] | None:
        """Prepare every original fingerprint block without an identity gate.

        Args:
          smiles: Original structure string, parsed independently of the partial stage.

        Returns:
          fingerprint: Complete packed raw bits, or None for the source failure boundary.

        """
        try:
            molecule = self.chemistry.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            blocks = [
                generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
                for generator in self.chemistry.generators
            ]
            blocks.append(
                np.array(list(GenMACCSKeys(molecule)), np.uint8),
            )
            return np.packbits(np.concatenate(blocks))
        except Exception:
            logger.debug("PubChem full fingerprint failed.", exc_info=True)
            return None

    def score_key(self, smiles: str) -> str | None:
        """Prepare the source key independently of fingerprint validity.

        Args:
          smiles: Unmodified input structure, including wildcard and empty strings.

        Returns:
          key: First14 canonical-tautomer InChIKey characters, or source rejection.

        """
        try:
            molecule = self.chemistry.chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            key = self.chemistry.chem.MolToInchiKey(
                self.chemistry.tautomers.Canonicalize(molecule),
            )
            return key[:14] if key else None
        except Exception:
            logger.debug("PubChem score key failed.", exc_info=True)
            return None


def prepare_pubchem(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Preserve every requested row and each independently observed validity field.

    Args:
      input_path: Parquet smiles or normalized_smiles requests in original row order.
      output: New external prepared-data directory, retaining every duplicate row.
      version: Exact isolated scientific-library version.
      batch_rows: Maximum complete rows per Arrow group.

    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    chemistry = PubChemChemistry(version)
    source = parquet.ParquetFile(input_path)
    column = "smiles" if "smiles" in source.schema_arrow.names else "normalized_smiles"
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True)
    count = source.metadata.num_rows
    ecfp4 = np.lib.format.open_memmap(
        destination / "ecfp4.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, 512),
    )
    raw_fp = np.lib.format.open_memmap(
        destination / "raw_fp.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, (16_551 + 7) // 8),
    )
    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("parsed", pa.bool_()),
            ("raw_valid", pa.bool_()),
            ("key", pa.string()),
        ],
    )
    index = 0
    with parquet.ParquetWriter(destination / "molecules.parquet", schema) as writer:
        for batch in source.iter_batches(batch_size=batch_rows, columns=[column]):
            strings = cast(Sequence[str], batch.column(0).to_pylist())
            metadata: list[dict[str, object]] = []
            for value in strings:
                parsed, partial, valid, full, key = chemistry.record(value)
                ecfp4[index], raw_fp[index] = partial, full
                metadata.append(
                    {"smiles": value, "parsed": parsed, "raw_valid": valid, "key": key},
                )
                index += 1
            writer.write_table(Table.from_pylist(metadata, schema=schema))
    ecfp4.flush()
    raw_fp.flush()
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "numpy": np.__version__,
        "rows": index,
        "raw_bits": 16_551,
        "ecfp4_bits": 4096,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("rdkit_backend.py"))),
    }
    write_manifest(destination, manifest)
