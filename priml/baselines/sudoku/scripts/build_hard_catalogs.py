#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Assemble five public hardest-sudoku catalogs into one provenance-tagged dataset.

The output is one evaluation dataset in the ``sapientinc/sudoku-extreme`` CSV
schema plus two provenance columns:

  source,question,answer,rating,source_name,source_url

next to a staged npy copy in the training layout (uint8 ``[N, 81]`` token
arrays, 1 blank, 2..10 digits, vocab 11) that ``PuzzleDataset`` reads
directly, a dataset card (README.md), and an integrity manifest
(manifest.json).

Labels are classical data preparation only: an exhaustive enumerator derives
each row's solution and certifies it unique, and an independent validator
re-checks it. Neither is part of any evaluation path.

Exact-duplicate puzzles across catalogs keep one row (the first catalog in
priority order) and semicolon-join every source into the provenance columns.

Examples:
  build_hard_catalogs.py --workers 32
  build_hard_catalogs.py --dataset-dir /opt/scratch/datasets/x --skip-extreme-overlap

'''
# fmt: on

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast
from urllib import request

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import zipfile

from huggingface_hub import hf_hub_download

import numpy as np

from priml.baselines.sudoku.scripts.prepare_data import (
    SOURCE_REVISION,
    SOURCE_SHA256,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence
    from http.client import HTTPResponse


logger = logging.getLogger(__name__)

DATASET_NAME: Final = "sudoku-hard-catalogs"

CSV_HEADER: Final = (
    "source",
    "question",
    "answer",
    "rating",
    "source_name",
    "source_url",
)

TDOKU_DATA_URL: Final = (
    "https://github.com/t-dillon/tdoku/raw/"
    "a05701253ed175e2859063dec3cf01a7f5b9c6a7/data.zip"
)
"""The tdoku data archive (BSD-2-Clause) at the commit that last changed it;
its digest is pinned below."""

TDOKU_DATA_SHA256: Final = (
    "9be0601c721ac4e702e3fe097576f025fcb99b216aabfe9dbea37cac43e6bc4f"
)

NAMED_HARDEST_ROWS: Final = (
    # AI Escargot (Inkala 2006).
    (
        "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
        "https://www.sudokuwiki.org/Escargot",
    ),
    # Inkala 2010.
    (
        "..53.....8......2..7..1.5..4....53...1..7...6..32...8..6.5....9..4....3......97..",
        "https://norvig.com/sudoku.html",
    ),
    # Inkala 2012 ("world's hardest").
    (
        "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
        "https://www.sudokuwiki.org/Arto_Inkala_Sudoku",
    ),
    # Golden Nugget (tarek, enjoysudoku forum; SE 11.9).
    (
        "000000039000010005003005800008009006070020000100400000009008050020000600400700000",
        "https://sudokusnake.com/goldennugget.php",
    ),
)
"""The celebrity hardest puzzles, each with its canonical page."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogSpec:
    """One source catalog: where it comes from and how it is read."""

    source_name: str
    """Short provenance id (the ``source_name`` CSV column)."""

    source: str
    """sudoku-extreme-style source id (tdoku member basename, or the catalog
    id for inline sets); the ``source`` CSV column."""

    source_url: str
    """Public URL the catalog comes from; empty when every inline row carries
    its own URL."""

    expected_count: int
    """Unique-puzzle count the catalog must parse to."""

    zip_member: str = ""
    """tdoku archive member path; empty for inline ``rows`` catalogs."""

    rows: tuple[tuple[str, str], ...] = ()
    """Inline (81-character puzzle, canonical URL) rows for small named sets."""


CATALOGS: Final = (
    CatalogSpec(
        source_name="named_hardest",
        source="named_hardest",
        source_url="",
        expected_count=4,
        rows=NAMED_HARDEST_ROWS,
    ),
    CatalogSpec(
        source_name="magictour1465",
        source="puzzles3_magictour_top1465",
        source_url=TDOKU_DATA_URL,
        expected_count=1_465,
        zip_member="data/puzzles3_magictour_top1465",
    ),
    CatalogSpec(
        source_name="backtrack_hardest",
        source="puzzles6_forum_hardest_1106",
        source_url=TDOKU_DATA_URL,
        expected_count=375,
        zip_member="data/puzzles6_forum_hardest_1106",
    ),
    CatalogSpec(
        source_name="se11plus",
        source="puzzles5_forum_hardest_1905_11+",
        source_url=TDOKU_DATA_URL,
        expected_count=48_766,
        zip_member="data/puzzles5_forum_hardest_1905_11+",
    ),
    CatalogSpec(
        source_name="royle17",
        source="puzzles2_17_clue",
        source_url=TDOKU_DATA_URL,
        expected_count=49_158,
        zip_member="data/puzzles2_17_clue",
    ),
)
"""The five catalogs in dedup priority order: an exact-duplicate puzzle keeps
the row of the EARLIEST catalog here and appends the later catalog's
provenance to the semicolon-joined columns."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogReport:
    """Per-catalog build accounting."""

    source_name: str
    raw_count: int
    """Unique puzzles parsed from the catalog."""

    new_rows: int
    """Rows this catalog contributed first (kept rows)."""

    duplicate_rows: int
    """Rows already contributed by an earlier catalog (provenance merged)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OverlapReport:
    """One catalog's overlap with the sudoku-extreme splits.

    sudoku-extreme stores validity-preserving shuffles (transpose, band/row
    and stack/column permutations, digit relabeling) of its source rows, so
    the exact-string join counts LITERAL row sharing while ``derived_*``
    count extreme rows whose ``source`` tag names this catalog (transformed
    variants).
    """

    source_name: str
    total: int
    in_extreme_train: int
    in_extreme_test: int
    in_neither: int
    derived_in_extreme_train: int
    derived_in_extreme_test: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildReport:
    """Whole-dataset build accounting."""

    dataset_dir: Path
    row_count: int
    catalogs: tuple[CatalogReport, ...]
    duplicate_count: int
    duplicate_pairs: dict[str, int]
    """Histogram of merge events keyed ``kept_source+dup_source``."""

    overlap: tuple[OverlapReport, ...] = ()
    """Per-catalog sudoku-extreme overlap; empty when the join was skipped."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Enumeration:
    """Completions of one puzzle, enumerated in a fixed order, capped at two."""

    status: Literal["proper", "improper", "unsatisfiable"]
    """``proper``: exactly one completion; ``improper``: at least two;
    ``unsatisfiable``: none."""

    grid: tuple[int, ...] | None
    """The first completion in enumeration order; None when unsatisfiable."""


def build_dataset(
    dataset_dir: Path,
    *,
    archive: zipfile.ZipFile | None,
    catalogs: tuple[CatalogSpec, ...] = CATALOGS,
    workers: int = 8,
    extreme_train_csv: Path | None = None,
    extreme_test_csv: Path | None = None,
) -> BuildReport:
    """Read, label, dedup, and publish the combined hard-catalog dataset.

    Publication is atomic: everything builds in a sibling directory and one
    rename publishes it, so a concurrent reader never observes a truncated
    dataset.

    Args:
      dataset_dir: Dataset root to publish.
      archive: The tdoku data archive the zip catalogs read; None when every
        catalog is inline.
      catalogs: Source catalogs in dedup priority order.
      workers: Enumeration processes.
      extreme_train_csv: sudoku-extreme ``train.csv`` for the per-catalog
        overlap join; None skips the overlap section.
      extreme_test_csv: sudoku-extreme ``test.csv``; None skips it.

    Returns:
      report: Row counts, per-catalog accounting, dedup stats, and (when the
        extreme CSVs are given) the per-catalog overlap table.

    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    if (extreme_train_csv is None) != (extreme_test_csv is None):
        raise ValueError("extreme_train_csv and extreme_test_csv come together.")
    rows: dict[tuple[int, ...], _AssemblyRow] = {}
    catalog_reports: list[CatalogReport] = []
    duplicate_pairs: dict[str, int] = {}
    catalog_puzzles: dict[str, tuple[tuple[int, ...], ...]] = {}
    for spec in catalogs:
        puzzles, urls = _read_catalog(spec, archive)
        catalog_puzzles[spec.source_name] = puzzles
        solutions = _labels(spec, puzzles, workers=workers)
        new_rows = 0
        for puzzle, solution, url in zip(puzzles, solutions, urls, strict=True):
            if not is_valid_completion(solution, puzzle):
                raise ValueError(
                    f"catalog {spec.source_name}: label fails the independent "
                    f"validator for puzzle {question_text(puzzle)}.",
                )
            existing = rows.get(puzzle)
            if existing is None:
                rows[puzzle] = _AssemblyRow(
                    puzzle=puzzle,
                    solution=solution,
                    sources=[spec.source],
                    source_names=[spec.source_name],
                    source_urls=[url],
                )
                new_rows += 1
                continue
            if existing.solution != solution:
                raise ValueError(
                    f"conflicting labels for duplicate puzzle {question_text(puzzle)}.",
                )
            existing.sources.append(spec.source)
            existing.source_names.append(spec.source_name)
            existing.source_urls.append(url)
            pair = f"{existing.source_names[0]}+{spec.source_name}"
            duplicate_pairs[pair] = duplicate_pairs.get(pair, 0) + 1
        catalog_reports.append(
            CatalogReport(
                source_name=spec.source_name,
                raw_count=len(puzzles),
                new_rows=new_rows,
                duplicate_rows=len(puzzles) - new_rows,
            ),
        )
        logger.info(
            "catalog %s: %d puzzles (%d new, %d duplicate)",
            spec.source_name,
            len(puzzles),
            new_rows,
            len(puzzles) - new_rows,
        )
    overlap: tuple[OverlapReport, ...] = ()
    if extreme_train_csv is not None and extreme_test_csv is not None:
        overlap = extreme_overlap(
            catalogs,
            catalog_puzzles,
            train_csv=extreme_train_csv,
            test_csv=extreme_test_csv,
        )
    report = BuildReport(
        dataset_dir=dataset_dir,
        row_count=len(rows),
        catalogs=tuple(catalog_reports),
        duplicate_count=sum(duplicate_pairs.values()),
        duplicate_pairs=duplicate_pairs,
        overlap=overlap,
    )
    _publish(dataset_dir, rows=tuple(rows.values()), report=report)
    logger.info(
        "published %d rows (%d cross-catalog duplicates merged) to %s",
        report.row_count,
        report.duplicate_count,
        dataset_dir,
    )
    return report


def extreme_overlap(
    catalogs: tuple[CatalogSpec, ...],
    catalog_puzzles: dict[str, tuple[tuple[int, ...], ...]],
    *,
    train_csv: Path,
    test_csv: Path,
) -> tuple[OverlapReport, ...]:
    """Join each catalog against the extreme splits: exact rows and source tags.

    Args:
      catalogs: Source catalog specs in build order.
      catalog_puzzles: Parsed puzzle rows keyed by ``source_name``.
      train_csv: sudoku-extreme ``train.csv`` path.
      test_csv: sudoku-extreme ``test.csv`` path.

    Returns:
      reports: One :class:`OverlapReport` per catalog, in input order.

    """
    train_questions, train_tags = _questions_and_tags(train_csv)
    test_questions, test_tags = _questions_and_tags(test_csv)
    reports: list[OverlapReport] = []
    for spec in catalogs:
        questions = [question_text(p) for p in catalog_puzzles[spec.source_name]]
        in_either = sum(
            question in train_questions or question in test_questions
            for question in questions
        )
        reports.append(
            OverlapReport(
                source_name=spec.source_name,
                total=len(questions),
                in_extreme_train=sum(q in train_questions for q in questions),
                in_extreme_test=sum(q in test_questions for q in questions),
                in_neither=len(questions) - in_either,
                derived_in_extreme_train=train_tags.get(spec.source, 0),
                derived_in_extreme_test=test_tags.get(spec.source, 0),
            ),
        )
    return tuple(reports)


def parse_puzzle_line(line: str, *, blank_token: int = 1) -> tuple[int, ...]:
    """Parse one 81-character puzzle row (``.`` or ``0`` = blank).

    Args:
      line: Candidate row; surrounding whitespace is ignored.
      blank_token: Token value of blank cells.

    Returns:
      tokens: 81 tokens, ``blank_token`` blank and ``2..10`` digit givens.

    Raises:
      ValueError: The row is not 81 valid puzzle characters.

    """
    text = line.strip()
    if len(text) != 81:
        raise ValueError(f"puzzle row must have 81 characters, got {len(text)}.")
    tokens: list[int] = []
    for character in text:
        if character in ".0":
            tokens.append(blank_token)
        elif "1" <= character <= "9":
            tokens.append(int(character) + 1)
        else:
            raise ValueError(f"invalid puzzle character {character!r}.")
    return tuple(tokens)


def parse_catalog_text(text: str) -> tuple[tuple[int, ...], ...]:
    """Parse one puzzle per non-empty line, in order, dropping exact repeats.

    ``#`` comment lines (the tdoku data-file convention) are skipped, and a
    repeated row keeps its first occurrence, so a mirror with duplicated lines
    cannot inflate the population.

    Args:
      text: Catalog text.

    Returns:
      puzzles: Unique parsed puzzle rows in source order.

    """
    puzzles: dict[tuple[int, ...], None] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            puzzles.setdefault(parse_puzzle_line(stripped))
    if not puzzles:
        raise ValueError("catalog text contains no puzzle rows.")
    return tuple(puzzles)


def question_text(tokens: Sequence[int], *, blank_token: int = 1) -> str:
    """Render puzzle tokens as the 81-character CSV question string."""
    return "".join("." if t == blank_token else str(t - 1) for t in tokens)


def answer_text(tokens: Sequence[int]) -> str:
    """Render solution tokens as the 81-digit CSV answer string."""
    return "".join(str(t - 1) for t in tokens)


def enumerate_completions(puzzle: tuple[int, ...]) -> Enumeration:
    """Enumerate a puzzle's completions by complete branching, capped at two.

    Depth-first search over bit masks, always branching on the empty cell
    with the fewest candidates and trying digits in ascending order, so the
    first completion is deterministic. Every completion is re-checked by
    :func:`is_valid_completion`, which shares no code with the search.

    Args:
      puzzle: 81 tokens; ``0``/``1`` empty, ``2..10`` digit givens.

    Returns:
      enumeration: Whether the puzzle is proper, and its first completion.

    Raises:
      ValueError: The puzzle is not 81 integer tokens in ``0..10``.

    """
    if len(puzzle) != 81 or any(
        type(token) is not int or token < 0 or token > 10 for token in puzzle
    ):
        raise ValueError("malformed puzzle: expected 81 integer tokens in 0..10.")
    solutions = _Enumerator(puzzle).run()
    for solution in solutions:
        if not is_valid_completion(solution, puzzle):
            raise AssertionError("an enumerated completion failed the validator.")
    grid = next(iter(solutions), None)
    if grid is None:
        return Enumeration(status="unsatisfiable", grid=None)
    return Enumeration(
        status="improper" if len(solutions) > 1 else "proper",
        grid=grid,
    )


def is_valid_completion(grid: Sequence[int], puzzle: Sequence[int]) -> bool:
    """Check a completed token grid against Sudoku rules and the givens.

    Direct set checks over rows, columns, and boxes; deliberately independent
    of :func:`enumerate_completions`.

    Args:
      grid: Candidate 81-cell completion; every token must be in ``2..10``.
      puzzle: Original 81-cell puzzle; ``0`` and ``1`` are empty, ``2..10``
        are givens.

    Returns:
      valid: Whether ``grid`` is a valid completion extending ``puzzle``.

    """
    if len(grid) != 81 or len(puzzle) != 81:
        return False
    if any(token < 2 or token > 10 for token in grid):
        return False
    if any(token < 0 or token > 10 for token in puzzle):
        return False
    if any(
        given >= 2 and value != given for value, given in zip(grid, puzzle, strict=True)
    ):
        return False
    expected = set(range(2, 11))
    rows = (set(grid[row * 9 : (row + 1) * 9]) for row in range(9))
    columns = ({grid[row * 9 + column] for row in range(9)} for column in range(9))
    boxes = (
        {
            grid[(top + row) * 9 + left + column]
            for row in range(3)
            for column in range(3)
        }
        for top in range(0, 9, 3)
        for left in range(0, 9, 3)
    )
    return all(group == expected for group in (*rows, *columns, *boxes))


def write_split(split_dir: Path, *, inputs: np.ndarray, labels: np.ndarray) -> None:
    """Write one dataset split in the training npy layout.

    Args:
      split_dir: Split directory (``test/`` or ``train/``), created if absent.
      inputs: ``[N, 81]`` uint8 puzzle token rows.
      labels: ``[N, 81]`` uint8 solution token rows.

    """
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "all__inputs.npy", inputs)
    np.save(split_dir / "all__labels.npy", labels)
    np.save(
        split_dir / "all__group_indices.npy",
        np.arange(len(inputs) + 1, dtype=np.int32),
    )
    (split_dir / "dataset.json").write_text(
        json.dumps({"vocab_size": 11, "seq_len": 81}) + "\n",
    )


def download_tdoku_archive() -> zipfile.ZipFile:
    """Download the tdoku data archive, pinning its digest.

    Returns:
      archive: The in-memory zip archive.

    Raises:
      ValueError: The payload does not match the pinned digest.

    """
    with cast(
        "HTTPResponse",
        request.urlopen(
            "https://github.com/t-dillon/tdoku/raw/"
            "a05701253ed175e2859063dec3cf01a7f5b9c6a7/data.zip",
            timeout=300,
        ),
    ) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    logger.info("downloaded %s: %d bytes", TDOKU_DATA_URL, len(payload))
    if digest != TDOKU_DATA_SHA256:
        raise ValueError(
            f"archive digest mismatch for {TDOKU_DATA_URL}: "
            f"{digest} != {TDOKU_DATA_SHA256}.",
        )
    return zipfile.ZipFile(io.BytesIO(payload))


def extreme_csv(name: str) -> Path:
    """Download one Sudoku-Extreme CSV at the pinned revision; verify its digest.

    Args:
      name: ``train.csv`` or ``test.csv``.

    Returns:
      path: The cached, digest-checked file.

    Raises:
      ValueError: The file does not match the pinned digest.

    """
    path = Path(
        str(
            hf_hub_download(
                "sapientinc/sudoku-extreme",
                name,
                repo_type="dataset",
                revision=SOURCE_REVISION,
            ),
        ),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256[name]:
        raise ValueError(
            f"{name} at revision {SOURCE_REVISION} has digest {digest}, "
            f"not {SOURCE_SHA256[name]}.",
        )
    return path


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      exit_code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    dataset_dir = validated_output_path(flags.dataset_dir)
    extreme_train_csv: Path | None = None
    extreme_test_csv: Path | None = None
    if not flags.skip_extreme_overlap:
        extreme_train_csv, extreme_test_csv = (
            extreme_csv(name) for name in ("train.csv", "test.csv")
        )
    build_dataset(
        dataset_dir,
        archive=download_tdoku_archive(),
        workers=flags.workers,
        extreme_train_csv=extreme_train_csv,
        extreme_test_csv=extreme_test_csv,
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    dataset_dir: Path
    workers: int
    skip_extreme_overlap: bool


@dataclass(slots=True, kw_only=True)
class _AssemblyRow:
    """One deduped dataset row accumulating provenance across catalogs."""

    puzzle: tuple[int, ...]
    solution: tuple[int, ...]
    sources: list[str]
    source_names: list[str]
    source_urls: list[str]


class _Enumerator:
    """Mutable bit-mask search state, independent of the set-based validator."""

    def __init__(self, puzzle: tuple[int, ...]) -> None:
        self.board = [-1] * 81
        self.row_masks = [0] * 9
        self.column_masks = [0] * 9
        self.box_masks = [0] * 9
        self.empties: list[int] = []
        self.solutions: list[tuple[int, ...]] = []
        self.valid = self._load(puzzle)

    def run(self) -> tuple[tuple[int, ...], ...]:
        """Return up to two completions in deterministic order."""
        if self.valid:
            self._search(0)
        return tuple(self.solutions)

    def _load(self, puzzle: tuple[int, ...]) -> bool:
        """Load givens into masks, rejecting contradictory boards."""
        for cell, token in enumerate(puzzle):
            if token < 2:
                self.empties.append(cell)
                continue
            bit = 1 << (token - 2)
            row, column = divmod(cell, 9)
            box = (row // 3) * 3 + column // 3
            if (
                self.row_masks[row] & bit
                or self.column_masks[column] & bit
                or self.box_masks[box] & bit
            ):
                return False
            self._place(cell, bit)
        return True

    def _search(self, position: int) -> None:
        """Depth-first fewest-candidates search with ascending digits."""
        if len(self.solutions) >= 2:
            return
        if position == len(self.empties):
            self.solutions.append(tuple(digit + 2 for digit in self.board))
            return
        best = self._best_empty(position)
        if best is None:
            return
        best_position, candidates = best
        empties = self.empties
        empties[position], empties[best_position] = (
            empties[best_position],
            empties[position],
        )
        cell = empties[position]
        while candidates and len(self.solutions) < 2:
            bit = candidates & -candidates
            candidates ^= bit
            self._place(cell, bit)
            self._search(position + 1)
            self._remove(cell, bit)
        empties[position], empties[best_position] = (
            empties[best_position],
            empties[position],
        )

    def _best_empty(self, position: int) -> tuple[int, int] | None:
        """Return the remaining cell with the fewest candidates, or None if stuck."""
        best_position = position
        best_candidates = 0
        best_count = 10
        for candidate_position in range(position, len(self.empties)):
            row, column = divmod(self.empties[candidate_position], 9)
            box = (row // 3) * 3 + column // 3
            candidates = 0x1FF & ~(
                self.row_masks[row] | self.column_masks[column] | self.box_masks[box]
            )
            count = candidates.bit_count()
            if count == 0:
                return None
            if count < best_count:
                best_position = candidate_position
                best_candidates = candidates
                best_count = count
                if count == 1:
                    break
        return best_position, best_candidates

    def _place(self, cell: int, bit: int) -> None:
        """Place one digit bit into a cell and its three masks."""
        row, column = divmod(cell, 9)
        box = (row // 3) * 3 + column // 3
        self.board[cell] = bit.bit_length() - 1
        self.row_masks[row] |= bit
        self.column_masks[column] |= bit
        self.box_masks[box] |= bit

    def _remove(self, cell: int, bit: int) -> None:
        """Undo one digit placement."""
        row, column = divmod(cell, 9)
        box = (row // 3) * 3 + column // 3
        self.board[cell] = -1
        self.row_masks[row] ^= bit
        self.column_masks[column] ^= bit
        self.box_masks[box] ^= bit


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(f"/opt/scratch/datasets/{DATASET_NAME}"),
        help="Dataset root to publish (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Enumeration processes for label derivation (default: 8).",
    )
    parser.add_argument(
        "--skip-extreme-overlap",
        action="store_true",
        help="Skip the sudoku-extreme overlap join (offline builds).",
    )


def _read_catalog(
    spec: CatalogSpec,
    archive: zipfile.ZipFile | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Parse one catalog; return its puzzles and per-row URLs."""
    if bool(spec.rows) == bool(spec.zip_member):
        raise ValueError(
            f"catalog {spec.source_name}: exactly one of rows or zip_member "
            "must be set.",
        )
    if spec.rows:
        puzzles = tuple(parse_puzzle_line(row) for row, _ in spec.rows)
        urls = tuple(url for _, url in spec.rows)
        # Inline rows only: a zip catalog cannot repeat a row because
        # ``parse_catalog_text`` drops repeated lines.
        if len(set(puzzles)) != len(puzzles):
            raise ValueError(f"catalog {spec.source_name} has duplicate inline rows.")
    else:
        if archive is None:
            raise ValueError(f"catalog {spec.source_name} needs the tdoku archive.")
        puzzles = parse_catalog_text(archive.read(spec.zip_member).decode())
        urls = (spec.source_url,) * len(puzzles)
    if len(puzzles) != spec.expected_count:
        raise ValueError(
            f"catalog {spec.source_name} parsed {len(puzzles)} unique puzzles "
            f"(exact-duplicate lines drop at parse); "
            f"expected {spec.expected_count}.",
        )
    return puzzles, urls


def _labels(
    spec: CatalogSpec,
    puzzles: tuple[tuple[int, ...], ...],
    *,
    workers: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every puzzle's unique solution; refuse a non-proper row."""
    logger.info(
        "catalog %s: enumerating %d labels with %d workers",
        spec.source_name,
        len(puzzles),
        workers,
    )
    if workers == 1:
        outcomes = [enumerate_completions(puzzle) for puzzle in puzzles]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = list(
                executor.map(enumerate_completions, puzzles, chunksize=64),
            )
    non_proper = [
        (row, outcome.status)
        for row, outcome in enumerate(outcomes)
        if outcome.status != "proper"
    ]
    if non_proper:
        raise ValueError(
            f"catalog {spec.source_name} has non-proper rows "
            f"(index, status): {non_proper[:20]}.",
        )
    return tuple(cast("tuple[int, ...]", outcome.grid) for outcome in outcomes)


def _questions_and_tags(csv_path: Path) -> tuple[set[str], dict[str, int]]:
    """Load a sudoku-extreme CSV's question set and source-tag histogram."""
    questions: set[str] = set()
    tags: dict[str, int] = {}
    with csv_path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        question_column = header.index("question")
        source_column = header.index("source")
        for row in reader:
            questions.add(row[question_column])
            tags[row[source_column]] = tags.get(row[source_column], 0) + 1
    return questions, tags


def _publish(
    dataset_dir: Path,
    *,
    rows: tuple[_AssemblyRow, ...],
    report: BuildReport,
) -> None:
    """Write the CSV, npy splits, card, and manifest; publish atomically."""
    staging = dataset_dir.with_name(f"{dataset_dir.name}.tmp.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    _write_csv(staging / "test.csv", rows)
    inputs = np.array([row.puzzle for row in rows], dtype=np.uint8).reshape(-1, 81)
    labels = np.array([row.solution for row in rows], dtype=np.uint8).reshape(-1, 81)
    write_split(staging / "test", inputs=inputs, labels=labels)
    write_split(staging / "train", inputs=inputs[:1], labels=labels[:1])
    (staging / "README.md").write_text(_dataset_card(report))
    manifest = {
        "dataset": DATASET_NAME,
        "build_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": report.row_count,
        "catalogs": [
            {
                "source_name": catalog.source_name,
                "raw_count": catalog.raw_count,
                "new_rows": catalog.new_rows,
                "duplicate_rows": catalog.duplicate_rows,
            }
            for catalog in report.catalogs
        ],
        "duplicate_count": report.duplicate_count,
        "duplicate_pairs": report.duplicate_pairs,
        "extreme_overlap": [
            {
                "source_name": overlap.source_name,
                "total": overlap.total,
                "in_extreme_train": overlap.in_extreme_train,
                "in_extreme_test": overlap.in_extreme_test,
                "in_neither": overlap.in_neither,
                "derived_in_extreme_train": overlap.derived_in_extreme_train,
                "derived_in_extreme_test": overlap.derived_in_extreme_test,
            }
            for overlap in report.overlap
        ],
        "files": {
            str(member.relative_to(staging)): hashlib.sha256(
                member.read_bytes(),
            ).hexdigest()
            for member in sorted(staging.rglob("*"))
            if member.is_file()
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not dataset_dir.exists():
        staging.replace(dataset_dir)
        return
    # Rebuild: retire the live dataset only after the replacement is fully
    # written; the rename gap is the unavoidable minimum.
    retired = dataset_dir.with_name(f"{dataset_dir.name}.old.{os.getpid()}")
    dataset_dir.rename(retired)
    try:
        staging.replace(dataset_dir)
    except BaseException:
        retired.replace(dataset_dir)
        raise
    shutil.rmtree(retired)


def _write_csv(path: Path, rows: tuple[_AssemblyRow, ...]) -> None:
    """Write the provenance-tagged evaluation CSV (sudoku-extreme schema + 2)."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(
                (
                    ";".join(row.sources),
                    question_text(row.puzzle),
                    answer_text(row.solution),
                    "0",
                    ";".join(row.source_names),
                    ";".join(dict.fromkeys(row.source_urls)),
                ),
            )


def _dataset_card(report: BuildReport) -> str:
    """Render the dataset card (README.md) from the build accounting."""
    url_by_name = {
        spec.source_name: spec.source_url or "per-row URLs" for spec in CATALOGS
    }
    lines = [
        f"# {DATASET_NAME}",
        "",
        "Five public hardest-sudoku catalogs assembled into one evaluation",
        "dataset with per-row provenance.",
        "",
        (
            f"Rows: {report.row_count} unique puzzles. Built (UTC): "
            f"{datetime.now(UTC).date().isoformat()}."
        ),
        "",
        "## Schema",
        "",
        "`test.csv` uses the `sapientinc/sudoku-extreme` evaluation-CSV schema",
        "plus two provenance columns:",
        "",
        "| column | meaning |",
        "|---|---|",
        "| `source` | upstream file id (tdoku archive member basename, or |",
        "| | `named_hardest` for the inline celebrity set) |",
        "| `question` | 81 chars, `.` = blank, `1`-`9` = givens |",
        "| `answer` | 81 digits, the puzzle's unique solution |",
        "| `rating` | `0` placeholder -- the sudoku-extreme difficulty |",
        "| | rating is not reproduced here; do not compare |",
        "| `source_name` | catalog id (see table below) |",
        "| `source_url` | public URL the catalog came from |",
        "",
        "Duplicate puzzles across catalogs keep ONE row and semicolon-join",
        "`source`, `source_name`, and `source_url` (URLs deduplicated, order",
        "preserved). Row order: catalog priority order below, source order",
        "within each catalog.",
        "",
        "The `test/` and `train/` directories hold the same rows in the npy",
        "training layout (uint8 `[N, 81]` tokens: 1 blank, 2-10 digits, vocab",
        "11); `train/` is a single-row stub (this is an evaluation-only",
        "dataset). `manifest.json` pins per-file sha256 digests.",
        "",
        "## Catalogs",
        "",
        "| source_name | puzzles | new | dup | fetched from |",
        "|---|---|---|---|---|",
        *(
            f"| {catalog.source_name} | {catalog.raw_count} | "
            f"{catalog.new_rows} | {catalog.duplicate_rows} | "
            f"{url_by_name.get(catalog.source_name, '')} |"
            for catalog in report.catalogs
        ),
        "",
        "- `named_hardest`: AI Escargot (Inkala 2006), Inkala 2010, Inkala",
        "  2012, Golden Nugget; each row's `source_url` is its own canonical",
        "  page (sudokuwiki.org, norvig.com, sudokusnake.com).",
        "- `magictour1465`: the magictour top1465 hardest list, via the",
        "  tdoku data archive member `data/puzzles3_magictour_top1465`.",
        "- `backtrack_hardest`: the enjoysudoku-forum HardestDatabase110626",
        "  list (hardest for classical backtracking), member",
        "  `data/puzzles6_forum_hardest_1106`.",
        "- `se11plus`: every public puzzle rated SE >= 11.0 (1905 list),",
        "  member `data/puzzles5_forum_hardest_1905_11+`.",
        "- `royle17`: Gordon Royle's completed 17-clue collection (CC-BY-2.5",
        "  per the collection's mirrors), member `data/puzzles2_17_clue`.",
        "",
        "The tdoku data archive (BSD-2-Clause, github.com/t-dillon/tdoku) is",
        f"digest-pinned: sha256 `{TDOKU_DATA_SHA256}`.",
        "",
        "## Duplicates",
        "",
        f"{report.duplicate_count} cross-catalog exact-duplicate rows were",
        "merged (kept-catalog + duplicate-catalog histogram):",
        "",
        *(
            [
                f"- `{pair}`: {count}"
                for pair, count in sorted(report.duplicate_pairs.items())
            ]
            or ["- none"]
        ),
    ]
    if report.overlap:
        lines += [
            "",
            "## Overlap with sudoku-extreme",
            "",
            "Two joins against the `sapientinc/sudoku-extreme` train/test",
            "splits. `exact` hash-joins canonical question strings (literal row",
            "sharing). `derived` counts extreme rows whose `source` tag names",
            "the catalog: extreme stores validity-preserving shuffles of its",
            "source rows, so exact overlap can be 0 while a whole catalog is",
            "present as transformed variants.",
            "",
            (
                "| source_name | total | exact train | exact test | exact neither"
                " | derived train | derived test |"
            ),
            "|---|---|---|---|---|---|---|",
            *(
                f"| {overlap.source_name} | {overlap.total} | "
                f"{overlap.in_extreme_train} | {overlap.in_extreme_test} | "
                f"{overlap.in_neither} | {overlap.derived_in_extreme_train} | "
                f"{overlap.derived_in_extreme_test} |"
                for overlap in report.overlap
            ),
        ]
    lines += [
        "",
        "## Labels",
        "",
        "Every `answer` is the puzzle's unique solution, derived by classical",
        "code as data preparation only (never part of any evaluation path):",
        "an exhaustive enumerator certifies each row has exactly one",
        "completion, and an independent completion validator re-checks it.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
