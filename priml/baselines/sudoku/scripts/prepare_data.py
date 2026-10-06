#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Download Sudoku-Extreme and cache it as the flat arrays training reads.

Run once before the first experiment. Idempotent: a split already built with the
same flags is left alone, so re-running costs nothing; one built with different
flags is refused rather than silently reused.

The training split is subsampled to 1,000 puzzles by default and then expanded
with many validity-preserving transformations of each. That is deliberate: the
benchmark's difficulty is in generalizing from few distinct puzzles, and
holding the puzzle count low while raising the copy count separates learning
the RULES from memorizing instances. The test split is written verbatim --
never subsampled, never transformed -- because a transformed test puzzle would
not be a held-out puzzle.

The build is deterministic: one seeded generator is consumed train-then-test in
a pinned draw order, so the same flags produce byte-identical arrays.

The default destination matches the one ``SudokuData.Config`` resolves under a
default ``TrainLoop``, so preparing and training agree without either naming a
path.

Examples:
  prepare_data.py
  prepare_data.py --directory /datasets/my-sudoku

'''
# fmt: on

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast
from urllib import request

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile

import numpy as np

from priml.baselines.sudoku.data import SudokuData
from priml.lib.custom_json import parse, to_builtins
from priml.paths import validated_output_path
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.sudoku.puzzle_spec import SudokuSpec


logger = logging.getLogger(__name__)


SOURCE_URL: Final = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve"
"""Base URL of the source CSVs.

Fetched over plain HTTP rather than through a Hugging Face client: one file per
split, at a pinned revision, verified by digest -- nothing a dependency would
add. Keeping it stdlib means preparing data needs no optional extra."""

SOURCE_REVISION: Final = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
"""Immutable revision pin.

A dataset that moves under you silently changes every result measured against
it, so the revision is pinned and the downloaded bytes are digest-checked."""


SOURCE_SHA256: Final = {
    "train.csv": "64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e",
    "test.csv": "a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8",
}
"""Required digests at :data:`SOURCE_REVISION`, verified before parsing."""


def main() -> int:
    """Prepare the dataset; return the process exit code.

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
    prepare(
        flags.directory,
        num_puzzles=flags.num_puzzles,
        copies_per_puzzle=flags.copies_per_puzzle,
        seed=flags.seed,
    )
    return 0


def default_directory() -> Path:
    """Return the dataset directory a default ``TrainLoop`` would resolve.

    Returns:
      result: The Path.

    """
    config = SudokuData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def prepare(
    directory: Path | str | None = None,
    *,
    num_puzzles: int = 1_000,
    copies_per_puzzle: int = 1_000,
    seed: int = 42,
    csv_directory: Path | str | None = None,
    spec: SudokuSpec | None = None,
) -> Path:
    """Build both splits under ``directory`` if they are not already there.

    Args:
      directory: Destination; ``None`` uses :func:`default_directory`.
      num_puzzles: Training puzzles kept from the source split.
      copies_per_puzzle: Transformed copies written per kept puzzle, on top of
        the original.
      seed: Seeds the one generator driving the whole build.
      csv_directory: Local ``train.csv`` / ``test.csv`` to read instead of
        downloading. Lets a test build the pipeline hermetically.
      spec: Geometry and vocabulary written to ``dataset.json``; ``None`` is
        the one :class:`SudokuData` reads by default.

    Returns:
      directory: Where the splits were written.

    Raises:
      ValueError: A count is out of range, the destination is unusable, or a
        split already there was built with different parameters.

    """
    if num_puzzles < 1:
        raise ValueError(f"--num-puzzles must be >= 1; got {num_puzzles}.")
    if copies_per_puzzle < 0:
        raise ValueError(
            f"--copies-per-puzzle must be >= 0; got {copies_per_puzzle}.",
        )
    spec = spec if spec is not None else SudokuData.Config().spec
    out = validated_output_path(
        directory if directory is not None else default_directory(),
    )
    out.mkdir(parents=True, exist_ok=True)
    # One generator consumed train-then-test in a pinned order: reordering the
    # splits or reseeding between them would change every array.
    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        source = (
            Path(csv_directory) / f"{split}.csv" if csv_directory is not None else None
        )
        _build_split(
            split,
            out=out,
            build=_Build(
                vocab_size=spec.vocab_size,
                seq_len=math.prod(spec.grid_shape),
                revision=SOURCE_REVISION,
                seed=seed,
                num_puzzles=num_puzzles if split == "train" else None,
                copies_per_puzzle=copies_per_puzzle if split == "train" else 0,
            ),
            spec=spec,
            rng=rng,
            csv_path=source,
        )
    logger.info("sudoku data ready at %s", out)
    return out


def _build_split(
    split: str,
    *,
    out: Path,
    build: _Build,
    spec: SudokuSpec,
    rng: np.random.Generator,
    csv_path: Path | None,
) -> None:
    """Convert one source CSV into the flat arrays training reads."""
    destination = out / split
    marker = destination / "dataset.json"
    if marker.is_file():
        _check_existing(marker, build=build)
        logger.info("sudoku %r already prepared; skipping", split)
        return
    downloaded: Path | None = None
    try:
        if csv_path is None:
            downloaded = csv_path = _download(f"{split}.csv", into=out)
            _verify(csv_path, filename=f"{split}.csv")
        puzzles, solutions = _read_csv(csv_path, spec=spec)
        inputs, labels, bounds = _expand(
            puzzles,
            solutions,
            build=build,
            spec=spec,
            rng=rng,
        )
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "all__inputs.npy", _tokenize(inputs, spec=spec))
        np.save(destination / "all__labels.npy", _tokenize(labels, spec=spec))
        np.save(
            destination / "all__group_indices.npy",
            np.array(bounds, dtype=np.int32),
        )
        # Written last: its presence is what marks the split complete.
        marker.write_text(json.dumps(to_builtins(asdict(build))))
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)
    logger.info(
        "sudoku %r: %d puzzles -> %d rows",
        split,
        len(bounds) - 1,
        bounds[-1],
    )


def _check_existing(marker: Path, *, build: _Build) -> None:
    """Refuse a split whose recorded build parameters differ from ``build``."""
    recorded = parse(marker.read_text(), dict[str, object])
    expected = cast(dict[str, object], to_builtins(asdict(build)))
    if recorded != expected:
        raise ValueError(
            f"{marker.parent} was built with {recorded}, not the requested "
            f"{expected}; delete it to rebuild.",
        )


def _expand(
    puzzles: list[NDArray[np.int64]],
    solutions: list[NDArray[np.int64]],
    *,
    build: _Build,
    spec: SudokuSpec,
    rng: np.random.Generator,
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[int]]:
    """Subsample, then add each kept puzzle's transformed copies after it."""
    if build.num_puzzles is not None and build.num_puzzles < len(puzzles):
        selected: NDArray[np.int64] = rng.choice(
            len(puzzles),
            size=build.num_puzzles,
            replace=False,
        )
        keep = cast(list[int], selected.tolist())
        puzzles = [puzzles[i] for i in keep]
        solutions = [solutions[i] for i in keep]
    inputs: list[NDArray[np.int64]] = []
    labels: list[NDArray[np.int64]] = []
    bounds = [0]
    for puzzle, solution in zip(puzzles, solutions, strict=True):
        inputs.append(puzzle)
        labels.append(solution)
        for _ in range(build.copies_per_puzzle):
            grid, answer = _transform(puzzle, solution=solution, spec=spec, rng=rng)
            inputs.append(grid)
            labels.append(answer)
        bounds.append(len(inputs))
    return inputs, labels, bounds


def _download(filename: str, *, into: Path) -> Path:
    """Fetch one source CSV to a temporary file beside the dataset."""
    url = f"{SOURCE_URL}/{SOURCE_REVISION}/{filename}"
    into.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=into, prefix=f".{filename}.", suffix=".part")
    os.close(handle)
    path = Path(staged)
    logger.info("downloading %s", url)
    try:
        # Stream rather than read whole: the training CSV is hundreds of MB.
        response = cast(
            _Readable,
            request.urlopen(url, timeout=120),  # noqa: S310 -- The URL is a fixed HTTPS dataset endpoint.
        )
        with response, path.open("wb") as out:
            shutil.copyfileobj(response, out)
    except BaseException:
        path.unlink()
        raise
    return path


def _read_csv(
    csv_path: Path,
    *,
    spec: SudokuSpec,
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]]]:
    """Parse the source CSV into digit grids, empty cells as zero."""
    puzzles: list[NDArray[np.int64]] = []
    solutions: list[NDArray[np.int64]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # Header.
        for row in reader:
            where = f"{csv_path}:{reader.line_num}"
            if len(row) != 4:
                raise ValueError(
                    f"{where}: expected 4 fields (source, question, answer, "
                    f"rating); got {len(row)}.",
                )
            _source, question, answer, _rating = row
            puzzles.append(_grid(question.replace(".", "0"), spec=spec, where=where))
            solutions.append(_grid(answer, spec=spec, where=where))
    return puzzles, solutions


def _grid(text: str, *, spec: SudokuSpec, where: str = "") -> NDArray[np.int64]:
    """Return one row's digits as a ``spec.grid_shape`` array."""
    cells = math.prod(spec.grid_shape)
    if len(text) != cells:
        raise ValueError(f"{where}: a grid needs {cells} cells; got {len(text)}.")
    return np.asarray(
        np.frombuffer(text.encode(), dtype=np.uint8).reshape(spec.grid_shape)
        - ord("0"),
        dtype=np.int64,
    )


# Digits arrive as 0-9 with 0 meaning empty; the model's vocabulary reserves 0 for
# padding, so everything shifts up by one: 0 pad, 1 empty, 2-10 digits.
def _tokenize(
    grids: list[NDArray[np.int64]],
    *,
    spec: SudokuSpec,
) -> NDArray[np.int64]:
    """Stack digit grids and shift into the token vocabulary."""
    stacked = np.stack(grids).reshape(len(grids), -1)
    digits = spec.vocab_size - 2
    if not np.all((stacked >= 0) & (stacked <= digits)):
        raise ValueError(f"Every cell must be a digit 0-{digits}, 0 meaning empty.")
    return stacked + 1


# Sudoku is invariant under relabeling the digits, transposing the grid, and permuting
# bands of rows or stacks of columns (and rows within a band, or columns within a
# stack). None of those can turn a valid grid invalid, because each maps every
# constraint group onto another constraint group.
#
# The draw order is pinned -- digits, transpose, bands, rows, stacks, columns -- because
# the whole build's byte-identity depends on it.
def _permutation(rng: np.random.Generator, size: int) -> list[int]:
    values: NDArray[np.int64] = rng.permutation(size)
    return cast(list[int], values.tolist())


def _transform(
    puzzle: NDArray[np.int64],
    *,
    solution: NDArray[np.int64],
    spec: SudokuSpec,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return a different valid puzzle with the correspondingly moved solution."""
    side, _ = spec.grid_shape
    box, _ = spec.box_shape
    digits = np.concatenate(
        ([0], np.fromiter(_permutation(rng, side), dtype=np.int64) + 1),
    )
    transpose = rng.random() < 0.5
    bands = _permutation(rng, side // box)
    rows = np.concatenate([b * box + np.array(_permutation(rng, box)) for b in bands])
    stacks = _permutation(rng, side // box)
    columns = np.concatenate(
        [s * box + np.array(_permutation(rng, box)) for s in stacks],
    )
    mapping = np.array(
        [rows[i // side] * side + columns[i % side] for i in range(side * side)],
    )

    return (
        _permuted(puzzle, mapping=mapping, digits=digits, transpose=transpose),
        _permuted(solution, mapping=mapping, digits=digits, transpose=transpose),
    )


def _permuted(
    grid: NDArray[np.int64],
    *,
    mapping: NDArray[np.int64],
    digits: NDArray[np.int64],
    transpose: bool,
) -> NDArray[np.int64]:
    """Apply one cell permutation and digit relabeling to ``grid``."""
    if transpose:
        grid = grid.T
    return digits[grid.flatten()[mapping].reshape(grid.shape)]


def _verify(csv_path: Path, *, filename: str) -> None:
    """Reject a download whose bytes are not the pinned ones."""
    digest = hashlib.sha256()
    with csv_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = SOURCE_SHA256[filename]
    if actual != expected:
        raise RuntimeError(
            f"{filename} at revision {SOURCE_REVISION} has digest "
            f"{actual}, expected {expected}; the download is corrupt or the "
            "source changed. Rerun to download it again.",
        )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        help=f"destination (default: {default_directory()})",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=1_000,
        help="training puzzles kept from the source split",
    )
    parser.add_argument(
        "--copies-per-puzzle",
        type=int,
        default=1_000,
        help="transformed copies written per kept training puzzle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeds the whole build",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    directory: str | None
    num_puzzles: int
    copies_per_puzzle: int
    seed: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Build:
    """What a split was built from; recorded in its ``dataset.json``.

    ``vocab_size`` and ``seq_len`` are what the loaders read. The rest exists so
    a rerun with different flags is refused instead of reusing stale arrays.
    """

    vocab_size: int
    seq_len: int
    revision: str
    seed: int
    num_puzzles: int | None
    copies_per_puzzle: int


class _Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
