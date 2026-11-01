#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Download Maze-Hard and cache it as the flat arrays training reads.

Run once before the first experiment. Two trees are written beside each other,
sharing every input and the whole test split:

  hard-1k            each training maze in its eight dihedral views, the stored
                     label moved along with the view (exp000).
  hard-1k-canonical  the same views, each relabeled by the stored labels' own
                     search rule in its own frame (exp001 onward).

The test split is the 1,000 test mazes as given: never transformed, never
relabeled, so every recipe is scored on the same grids. The build is
deterministic -- no randomness at all -- so the same source bytes produce
byte-identical arrays. Idempotent: a split already built from the same source
is left alone; one built differently is refused rather than silently reused.
The source CSVs are fetched through the Hugging Face cache (``priml[hub]``).

Examples:
  prepare_data.py
  prepare_data.py --directory /opt/scratch/datasets/my-maze

'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast

import argparse
import csv
import hashlib
import logging

from huggingface_hub import hf_hub_download
from treekle import json_dumps, json_loads

import numpy as np

from priml.baselines.maze.data import (
    CANONICAL,
    DIHEDRAL_VIEWS,
    ORIGINAL,
    VOCAB_SIZE,
    canonical_labels,
    tokenize,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


DEFAULT_DIRECTORY: Final = "/opt/scratch/datasets/maze"
"""Where both trees are written unless ``--directory`` says otherwise."""

SOURCE_REPO: Final = "sapientinc/maze-30x30-hard-1k"
"""Hugging Face dataset holding the source CSVs."""

SOURCE_REVISION: Final = "549754de3d67eebe57b6b22c7d226b9412d55b07"
"""Immutable revision pin; a dataset that moves silently changes every result."""

SOURCE_SHA256: Final = {
    "train.csv": "25f9e39b5aee16967da41ba63b2d1d75cf608c2360d8a1419d827f9859dbafa0",
    "test.csv": "5120d52088fb3f8036e62e90103808a79ba58847d8601f9451f40991b0903a62",
}
"""Required digests at :data:`SOURCE_REVISION`, verified before parsing."""


def main() -> int:
    """Prepare both trees; return the process exit code.

    Returns:
      result: Process exit code (0 on success).

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(flags.directory)
    return 0


def prepare(
    directory: Path | str = DEFAULT_DIRECTORY,
    *,
    csv_directory: Path | str | None = None,
) -> Path:
    """Build :data:`ORIGINAL` and :data:`CANONICAL` under ``directory``.

    Args:
      directory: Destination root.
      csv_directory: Local ``train.csv`` / ``test.csv`` to read instead of
        fetching and verifying the pinned ones. Lets a test build the
        pipeline hermetically.

    Returns:
      directory: Where the trees were written.

    """
    out = validated_output_path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        views = len(DIHEDRAL_VIEWS) if split == "train" else 1
        builds = {
            name: _Build(
                vocab_size=VOCAB_SIZE,
                revision=SOURCE_REVISION if csv_directory is None else "local",
                views=views,
                canonical=name == CANONICAL and split == "train",
            )
            for name in (ORIGINAL, CANONICAL)
        }
        markers = {name: out / name / split / "dataset.json" for name in builds}
        if all(marker.is_file() for marker in markers.values()):
            for name, marker in markers.items():
                _check_existing(marker, build=builds[name])
            logger.info("maze %r already prepared; skipping", split)
            continue
        source = (
            _fetch(f"{split}.csv")
            if csv_directory is None
            else Path(csv_directory) / f"{split}.csv"
        )
        inputs, labels = _read_split(source, views=views)
        groups = np.arange(0, len(inputs) + 1, views, dtype=np.int32)
        for name, build in builds.items():
            _write_split(
                out / name / split,
                inputs=inputs,
                labels=canonical_labels(inputs) if build.canonical else labels,
                groups=groups,
                build=build,
            )
        logger.info("maze %r: %d mazes -> %d rows", split, len(groups) - 1, len(inputs))
    logger.info("maze data ready at %s", out)
    return out


def _read_split(
    source: Path,
    *,
    views: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return ``[N * views, side * side]`` inputs and labels, views consecutive."""
    inputs: list[NDArray[np.uint8]] = []
    labels: list[NDArray[np.uint8]] = []
    for question, answer in _rows(source):
        side = round(len(question) ** 0.5)
        grid, solved = tokenize(question, side=side), tokenize(answer, side=side)
        for view in DIHEDRAL_VIEWS[:views]:
            inputs.append(view(grid).reshape(-1))
            labels.append(view(solved).reshape(-1))
    return np.stack(inputs), np.stack(labels)


def _rows(csv_path: Path) -> list[tuple[str, str]]:
    """Return each ``(question, answer)`` pair of the source CSV."""
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # Header.
        rows: list[tuple[str, str]] = []
        for row in reader:
            if len(row) != 4:
                raise ValueError(
                    f"{csv_path}:{reader.line_num}: expected 4 fields (source, "
                    f"question, answer, rating); got {len(row)}.",
                )
            rows.append((row[1], row[2]))
    return rows


def _write_split(
    destination: Path,
    *,
    inputs: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    groups: NDArray[np.int32],
    build: _Build,
) -> None:
    """Write one split's arrays, then the marker that says it is complete."""
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "all__inputs.npy", inputs)
    np.save(destination / "all__labels.npy", labels)
    np.save(destination / "all__group_indices.npy", groups)
    # Written last: its presence is what marks the split complete.
    marker = _Marker.of(build, seq_len=inputs.shape[-1])
    (destination / "dataset.json").write_text(
        json_dumps(marker, target=_Marker, sort=False),
    )


def _check_existing(marker: Path, *, build: _Build) -> None:
    """Refuse a split whose recorded build differs from ``build``."""
    recorded = json_loads(marker.read_text(), _Marker)
    if recorded != _Marker.of(build, seq_len=recorded.seq_len):
        raise ValueError(
            f"{marker.parent} was built with {recorded}, not the requested "
            f"{build}; delete it to rebuild.",
        )


def _fetch(filename: str) -> Path:
    """Return the pinned source CSV from the hub cache, its digest verified."""
    path = Path(
        hf_hub_download(
            SOURCE_REPO,
            filename,
            repo_type="dataset",
            revision=SOURCE_REVISION,
        ),
    )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = SOURCE_SHA256[filename]
    if actual != expected:
        raise RuntimeError(
            f"{filename} at revision {SOURCE_REVISION} has digest {actual}, "
            f"expected {expected}; the cached copy is corrupt or the source "
            "changed. Delete it from the Hugging Face cache and rerun.",
        )
    return path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        default=DEFAULT_DIRECTORY,
        help=f"destination root (default: {DEFAULT_DIRECTORY})",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    directory: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Build:
    """What a split was built from; recorded in its ``dataset.json``.

    The marker adds ``seq_len``; it and ``vocab_size`` are what the loader
    reads. The rest exists so a rerun from a different source is refused
    instead of reusing stale arrays.
    """

    vocab_size: int
    revision: str
    views: int
    canonical: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class _Marker(_Build):
    """A split's ``dataset.json``: its build, and the length of its rows."""

    seq_len: int

    @classmethod
    def of(cls, build: _Build, *, seq_len: int) -> Self:
        """Return the marker of a split built as ``build`` with rows of ``seq_len``.

        Args:
          build: What the split was built from.
          seq_len: The length of each of its rows.

        Returns:
          marker: The split's ``dataset.json``.

        """
        return cls(
            vocab_size=build.vocab_size,
            revision=build.revision,
            views=build.views,
            canonical=build.canonical,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
