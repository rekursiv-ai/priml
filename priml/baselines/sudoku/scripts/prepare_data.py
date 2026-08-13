"""Download Sudoku-Extreme and cache it as the flat arrays training reads.

Run once before the first experiment. Idempotent: a split already present is
left alone, so re-running costs nothing.

The training split is subsampled to a few hundred puzzles and then expanded
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
  uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.prepare_data
  uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.prepare_data --directory /datasets/my-sudoku

"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request

import numpy as np

from priml.baselines.sudoku.data import SudokuData
from priml.train.train_loop import TrainLoop


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

GRID: Final = 9
"""Side of the puzzle grid."""

BOX: Final = 3
"""Side of one constraint box."""


def main() -> int:
    """Prepare the dataset; return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(
        args.directory,
        num_puzzles=args.num_puzzles,
        copies_per_puzzle=args.copies_per_puzzle,
        seed=args.seed,
    )
    return 0


def default_directory() -> Path:
    """Return the dataset directory a default ``TrainLoop`` would resolve."""
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

    Returns:
      directory: Where the splits were written.

    """
    out = Path(directory) if directory is not None else default_directory()
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
            num_puzzles=num_puzzles if split == "train" else None,
            copies_per_puzzle=copies_per_puzzle if split == "train" else 0,
            rng=rng,
            csv_path=source,
        )
    logger.info("sudoku data ready at %s", out)
    return out


def _build_split(
    split: str,
    *,
    out: Path,
    num_puzzles: int | None,
    copies_per_puzzle: int,
    rng: np.random.Generator,
    csv_path: Path | None,
) -> None:
    """Convert one source CSV into the flat arrays training reads."""
    destination = out / split
    if (destination / "dataset.json").is_file():
        logger.info("sudoku %r already prepared; skipping", split)
        return
    downloaded: Path | None = None
    if csv_path is None:
        downloaded = csv_path = _download(f"{split}.csv", into=out)
        _verify(csv_path, filename=f"{split}.csv")

    puzzles, solutions = _read_csv(csv_path)
    if num_puzzles is not None and num_puzzles < len(puzzles):
        # ``choice`` returns an untyped array; naming the index list keeps the
        # comprehensions' element type from widening to Unknown.
        keep: list[int] = [
            int(i) for i in rng.choice(len(puzzles), size=num_puzzles, replace=False)
        ]
        puzzles = [puzzles[i] for i in keep]
        solutions = [solutions[i] for i in keep]

    all_inputs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    bounds = [0]
    written = 0
    for puzzle, solution in zip(puzzles, solutions, strict=True):
        for copy_index in range(1 + copies_per_puzzle):
            if copy_index == 0:
                grid, answer = puzzle, solution
            else:
                grid, answer = _transform(puzzle, solution=solution, rng=rng)
            all_inputs.append(grid)
            all_labels.append(answer)
            written += 1
        bounds.append(written)

    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "all__inputs.npy", _tokenize(all_inputs))
    np.save(destination / "all__labels.npy", _tokenize(all_labels))
    np.save(
        destination / "all__group_indices.npy",
        np.array(bounds, dtype=np.int32),
    )
    (destination / "dataset.json").write_text(
        json.dumps({"vocab_size": 11, "seq_len": GRID * GRID}),
    )
    if downloaded is not None:
        downloaded.unlink(missing_ok=True)
    logger.info("sudoku %r: %d puzzles -> %d rows", split, len(puzzles), written)


def _download(filename: str, *, into: Path) -> Path:
    """Fetch one source CSV to a temporary file beside the dataset.

    Args:
      filename: Source file to fetch.
      into: Directory to stage under, so the partial file shares a filesystem
        with its destination and a full disk fails here rather than midway
        through the build.

    Returns:
      path: The downloaded file. The caller deletes it once the split is built.

    """
    url = f"{SOURCE_URL}/{SOURCE_REVISION}/{filename}"
    into.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=into, prefix=f".{filename}.", suffix=".part")
    os.close(handle)
    path = Path(staged)
    logger.info("downloading %s", url)
    # Stream rather than read whole: the training CSV is hundreds of MB.
    with (
        urllib.request.urlopen(url) as response,  # noqa: S310 -- fixed https URL from pinned constants
        path.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    return path


def _read_csv(csv_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Parse the source CSV into digit grids, empty cells as zero."""
    puzzles: list[np.ndarray] = []
    solutions: list[np.ndarray] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for _source, question, answer, _rating in reader:
            puzzles.append(_grid(question.replace(".", "0")))
            solutions.append(_grid(answer))
    return puzzles, solutions


def _grid(text: str) -> np.ndarray:
    """An 81-character row as a ``[9, 9]`` digit array."""
    return np.frombuffer(text.encode(), dtype=np.uint8).reshape(GRID, GRID) - ord("0")


def _tokenize(grids: list[np.ndarray]) -> np.ndarray:
    """Stack digit grids and shift into the token vocabulary.

    Digits arrive as 0-9 with 0 meaning empty; the model's vocabulary reserves
    0 for padding, so everything shifts up by one: 0 pad, 1 empty, 2-10 digits.
    """
    stacked = np.concatenate(grids).reshape(len(grids), -1)
    assert np.all((stacked >= 0) & (stacked <= 9))
    return stacked + 1


def _transform(
    puzzle: np.ndarray,
    *,
    solution: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a different valid puzzle with the correspondingly moved solution.

    Sudoku is invariant under relabeling the digits, transposing the grid, and
    permuting bands of rows or stacks of columns (and rows within a band, or
    columns within a stack). None of those can turn a valid grid invalid,
    because each maps every constraint group onto another constraint group.

    The draw order is pinned -- digits, transpose, bands, rows, stacks, columns
    -- because the whole build's byte-identity depends on it.
    """
    digits = np.pad(rng.permutation(np.arange(1, GRID + 1)), (1, 0))
    transpose = rng.random() < 0.5
    bands = rng.permutation(BOX)
    rows = np.concatenate([b * BOX + rng.permutation(BOX) for b in bands])
    stacks = rng.permutation(BOX)
    columns = np.concatenate([s * BOX + rng.permutation(BOX) for s in stacks])
    mapping = np.array(
        [rows[i // GRID] * GRID + columns[i % GRID] for i in range(GRID * GRID)],
    )

    def apply(grid: np.ndarray) -> np.ndarray:
        if transpose:
            grid = grid.T
        return digits[grid.flatten()[mapping].reshape(GRID, GRID).copy()]

    return apply(puzzle), apply(solution)


def _verify(csv_path: Path, *, filename: str) -> None:
    """Reject a download whose bytes are not the pinned ones.

    Raises:
      RuntimeError: The digest differs. The revision is pinned, so upstream
        cannot have changed: the download is corrupt or the cache was
        modified. Delete the cached file and retry.

    """
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
            f"cache was modified. Delete {csv_path} and retry.",
        )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        default=None,
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


if __name__ == "__main__":
    raise SystemExit(main())
