#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage the real ROGII CSV release and rebuild its geographic split map.

The source must contain train/, test/, and sample_submission.csv. It may be an
extracted Kaggle download or the data/ directory of the pinned winner repository.
The destination must not already exist.
'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import csv
import io
import logging
import shutil

from priml.baselines.wellbore_geology.data import (
    discover_wells,
    read_numeric_csv,
)
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.paths import validated_output_path


if TYPE_CHECKING:
    import numpy as np
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class StagedData:
    """Counts and paths of a staged real competition release."""

    directory: Path
    train_wells: tuple[str, ...]
    test_wells: tuple[str, ...]


def geographic_map(train_directory: Path) -> bytes:
    """Return the source's six-decimal full-sequence X/Y centroid CSV.

    Args:
      train_directory: Paired training CSVs from the real release.

    Returns:
      centroids: CSV bytes consumed by geographic stratified fold assignment.

    """
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("well_id", "horizontal_avg_X", "horizontal_avg_Y"))
    for well_id in discover_wells(train_directory):
        arrays = read_numeric_csv(
            train_directory / f"{well_id}__horizontal_well.csv",
            columns=("X", "Y"),
        )
        if len(arrays["X"]) == 0:
            raise ValueError(f"Training horizontal CSV is empty: {well_id}.")
        if any(np.count_nonzero(~np.isnan(values)) == 0 for values in arrays.values()):
            raise ValueError(f"Training X/Y centroid is missing: {well_id}.")
        writer.writerow(
            (well_id, *(f"{np.nanmean(values):.6f}" for values in arrays.values())),
        )
    return output.getvalue().encode()


def stage_data(source: Path, *, destination: Path) -> StagedData:
    """Copy real CSV pairs into a new directory and rebuild the geographic map.

    Args:
      source: Extracted Kaggle release or the winner's bundled data directory.
      destination: New staging directory outside the checkout.

    Returns:
      staged: Validated paired well identifiers and output directory.

    Raises:
      FileNotFoundError: A required input CSV is missing.
      FileExistsError: The destination already exists.
      ValueError: Source and destination overlap.

    """
    source = source.resolve()
    destination = validated_output_path(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and staging destination must not overlap.")
    train_wells = discover_wells(source / "train")
    test_wells = discover_wells(source / "test")
    sample = (source / "sample_submission.csv").read_bytes()
    centroids = geographic_map(source / "train")
    destination.mkdir(parents=True, exist_ok=False)
    for split, ids in (("train", train_wells), ("test", test_wells)):
        (destination / split).mkdir()
        for well_id in ids:
            for suffix in ("horizontal_well", "typewell"):
                name = f"{well_id}__{suffix}.csv"
                shutil.copyfile(source / split / name, destination / split / name)
    (destination / "sample_submission.csv").write_bytes(sample)
    (destination / "train_png_typewell_map.csv").write_bytes(centroids)
    return StagedData(
        directory=destination,
        train_wells=train_wells,
        test_wells=test_wells,
    )


def main() -> int:
    """Stage the source directory and return zero on success.

    Returns:
      code: Zero after copying the release and rebuilding the centroid map.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    staged = stage_data(flags.source, destination=flags.directory)
    logger.info(
        "Staged %d training wells and %d visible test wells at %s.",
        len(staged.train_wells),
        len(staged.test_wells),
        staged.directory,
    )
    return 0


class Flags(Protocol):
    """Parsed preparation flags."""

    source: Path
    directory: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--directory",
        type=Path,
        default=WellboreStudy.Config().dataset_directory,
        help="New destination directory for the competition CSVs.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
