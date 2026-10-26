"""Verify complete transferred pixels and bind their acquisition evidence to a root."""

from __future__ import annotations

from pathlib import Path
from typing import Self, override

import hashlib
import shutil

from configgle import Fig
from treekle import from_plain, json_dumps

from priml.baselines.rsna_knee.train_loop import (
    ORIGINAL_RECEIPT,
    RECEIPT,
    SELECTED_FILES,
    SERIES_FILES,
    TRANSFER,
    VERIFIED_FILES,
    verified_inventory,
    verify_members,
)
from priml.paths import resolve_working_dir, validated_output_path


class ReceiptAdoption:
    """Verify every transferred member before binding its new canonical root."""

    class Config(Fig["ReceiptAdoption"]):
        base_dir: Path | None = Path("/opt/scratch")
        """Resource root beneath which every logical directory below resolves."""

        input_dir: Path = Path("/datasets/rsna-knee/raw")
        """Transferred official input tree, including every acquired DICOM member."""

        source_dir: Path = Path("/artifacts/rsna-knee/acquisition-transferred")
        """Unchanged original acquisition receipt and complete manifests."""

        working_dir: Path = Path("/artifacts/rsna-knee/acquisition")
        """New evidence directory preserving the original receipt and transfer proof."""

        @override
        def finalize(self) -> Self:
            self.input_dir = resolve_working_dir(self.base_dir, self.input_dir)
            self.source_dir = resolve_working_dir(self.base_dir, self.source_dir)
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Verify transferred acquisition evidence and publish a root-bound receipt.

        Args:
          *args: Passthrough arguments are rejected.

        """
        if args:
            raise ValueError("Receipt adoption accepts configuration fields only.")
        config = self.config
        output = validated_output_path(
            config.working_dir,
            protected=[config.input_dir, config.source_dir],
        )
        if output.exists():
            raise FileExistsError(output)
        original = (config.source_dir / RECEIPT).read_bytes()
        receipt, members = verified_inventory(config.source_dir)
        selected = {Path(name).parent for name in members}
        total_bytes = sum(int(row["bytes"]) for row in members.values())
        if total_bytes != from_plain(receipt["bytes"], int) or len(
            selected,
        ) != from_plain(receipt["series"], int):
            raise ValueError("DICOM acquisition inventory totals differ.")
        verify_members(config.input_dir, members=members, selected=selected)
        source_digest = hashlib.sha256(original).hexdigest()
        provenance = {
            "source_receipt_sha256": source_digest,
            "source_input_dir": from_plain(receipt["input_dir"], str),
            "input_dir": str(config.input_dir.resolve()),
            "verified_files": len(members),
            "verified_bytes": total_bytes,
            "verification": "Every transferred member length, CRC32 and SHA256 verified against the original complete inventory.",
        }
        output.mkdir(parents=True)
        for name in (SELECTED_FILES, VERIFIED_FILES, SERIES_FILES):
            shutil.copyfile(config.source_dir / name, output / name)
        (output / ORIGINAL_RECEIPT).write_bytes(original)
        (output / TRANSFER).write_text(
            json_dumps(provenance, indent=2) + "\n",
        )
        receipt["input_dir"] = str(config.input_dir.resolve())
        receipt["transfer_source_receipt_sha256"] = source_digest
        (output / RECEIPT).write_text(
            json_dumps(receipt, indent=2) + "\n",
        )


def adopt_receipt() -> ReceiptAdoption.Config:
    """Build the credential-free transferred acquisition verification job."""
    return ReceiptAdoption.Config()
