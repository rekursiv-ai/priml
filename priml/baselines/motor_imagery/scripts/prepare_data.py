#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage raw EEG data without training.

Examples:
    uv --quiet run --frozen python -m priml.baselines.motor_imagery.scripts.prepare_data --print-config
    uv --quiet run --frozen python -m priml.baselines.motor_imagery.scripts.prepare_data --archive /opt/scratch/datasets/motor-imagery.zip

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import hashlib
import shutil
import subprocess
import zipfile

from treekle import json_dumps

from priml.baselines.motor_imagery.experiments import exp000
from priml.paths import validated_output_path


class Flags(Protocol):
    """Parsed staging flags."""

    directory: Path
    download_directory: Path
    archive: Path | None
    print_config: bool


def main() -> int:
    """Stage only explicitly requested inputs; return the process exit status.

    Returns:
      status: Zero after successful staging or configuration inspection.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    if flags.print_config:
        exp000().pprint(hide_default_values=False)
        return 0
    directory = validated_output_path(flags.directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to merge an existing EEG stage: {directory}.")
    archive = flags.archive
    if archive is None:
        download_directory = validated_output_path(flags.download_directory)
        download_directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603 -- Fixed Kaggle argv, no shell or credentials in arguments.
            [
                shutil.which("kaggle") or "kaggle",
                "competitions",
                "download",
                "-c",
                "low-cost-motor-imagery-decoding-for-rehab-cross-subject",
                "-p",
                str(download_directory),
            ],
            check=True,
        )
        archive = (
            download_directory
            / "low-cost-motor-imagery-decoding-for-rehab-cross-subject.zip"
        )
    stage_archive(archive, directory=directory)
    return 0


def stage_archive(archive: Path, *, directory: Path) -> None:
    """Extract real inputs and record whole-file SHA-256 provenance.

    Args:
      archive: Competition archive already downloaded with authorized access.
      directory: Raw-data destination; filtering is deferred to the timed run.

    Raises:
      ValueError: Archive paths escape the destination, or required data is absent.

    """
    destination = validated_output_path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if (
                not (destination / member.filename)
                .resolve()
                .is_relative_to(destination.resolve())
            ):
                raise ValueError(
                    f"Archive member escapes staging destination: {member.filename}.",
                )
        source.extractall(destination)
    for name in ("Training set", "Cross participant test set"):
        if not list((destination / name).glob("*.csv")):
            raise ValueError(f"Competition archive has no CSV recordings in {name}.")
    files = sorted(destination.glob("*/*.csv"))
    manifest: dict[str, str] = {}
    for path in files:
        with path.open("rb") as stream:
            manifest[str(path.relative_to(destination))] = hashlib.file_digest(
                stream,
                "sha256",
            ).hexdigest()
    (destination / "manifest.json").write_text(
        json_dumps(manifest, indent=2) + "\n",
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--download-directory",
        type=Path,
        default=Path("/opt/scratch/caches/kaggle/motor-imagery"),
        help="Reusable archive cache, separate from the fresh extraction directory.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/opt/scratch/datasets/motor-imagery"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Extract an existing competition archive.",
    )
    parser.add_argument("--print-config", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
