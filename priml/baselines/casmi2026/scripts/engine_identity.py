"""Generate the second engine's independent identity table in the isolated SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import logging

from treekle import from_plain

from priml.baselines.casmi2026.isolated import IDENTITY_RDKIT
from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    source_digests,
    write_manifest,
)
from priml.baselines.casmi2026.scripts.rdkit_backend import Chemistry
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from pyarrow import Table, parquet

    import pyarrow as pa
else:
    from wrapt import lazy_import

    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    pa = lazy_import("pyarrow")


logger = logging.getLogger(__name__)
_THIS: Final = Path(__file__).resolve()


class EngineKeyChemistry:
    """Preserve the public raw canonical key's empty-string and failure distinction."""

    def __init__(self, version: str) -> None:
        if version != IDENTITY_RDKIT:
            raise ValueError(
                f"Engine identity preparation requires RDKit {IDENTITY_RDKIT}.",
            )
        self.context = Chemistry(version)

    def key(self, smiles: str) -> str | None:
        """Return tautomer-canonical InChIKey14, retaining a valid empty string.

        Args:
          smiles: Exact raw input, without stereo or charge normalization.

        Returns:
          key: Full first14 characters, including empty, or an explicit SDK failure.

        """
        try:
            molecule = self.context.chem.MolFromSmiles(smiles)
            return (
                self.context.chem.MolToInchiKey(
                    self.context.tautomers.Canonicalize(molecule),
                )[:14]
                if molecule is not None
                else None
            )
        except Exception:
            logger.debug("Engine identity calculation failed.", exc_info=True)
            return None


def prepare_engine_identity(
    input_path: Path,
    *,
    output: Path,
    version: str,
    batch_rows: int = 256,
) -> None:
    """Publish every ordered input, duplicate and independent engine-key decision.

    Args:
      input_path: Complete raw candidate SMILES requests in encounter order.
      output: New external dataset directory, never replacing existing assets.
      version: Exact separately provisioned scientific-library release.
      batch_rows: Maximum complete rows per Arrow group.

    """
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    destination = validated_output_path(output, protected=[input_path])
    destination.mkdir(parents=True, exist_ok=False)
    chemistry = EngineKeyChemistry(version)
    schema = pa.schema([("smiles", pa.string()), ("key", pa.string())])
    count = 0
    with parquet.ParquetWriter(destination / "keys.parquet", schema) as writer:
        for batch in parquet.ParquetFile(input_path).iter_batches(
            batch_size=batch_rows,
            columns=["smiles"],
        ):
            strings = from_plain(cast(object, batch.column(0).to_pylist()), list[str])
            writer.write_table(
                Table.from_pylist(
                    [
                        {"smiles": value, "key": chemistry.key(value)}
                        for value in strings
                    ],
                    schema=schema,
                ),
            )
            count += len(strings)
    manifest: dict[str, object] = {
        "format_version": 1,
        "rdkit": version,
        "rows": count,
        "inputs": input_digests([input_path]),
        "generator": source_digests((_THIS, _THIS.with_name("rdkit_backend.py"))),
    }
    write_manifest(destination, manifest)
