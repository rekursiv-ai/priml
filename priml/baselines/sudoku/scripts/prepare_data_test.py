"""Tests for the sudoku data preparer.

Hermetic: every test feeds local CSV text, so nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override
from urllib import request

import argparse
import csv
import hashlib
import inspect
import io
import logging

import numpy as np
import pytest

from priml.baselines.sudoku.eval import Harvest, VerifierData
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.baselines.sudoku.scripts import prepare_data
from priml.baselines.sudoku.scripts.prepare_data import (
    default_directory,
)
from priml.lib.custom_json import parse
from priml.lib.testing.cli import assert_help_without_docstring
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from numpy.typing import NDArray


def _load_array(path: Path) -> NDArray[np.int64]:
    return cast("NDArray[np.int64]", np.load(path))


def _build(*, num_puzzles: int | None, copies_per_puzzle: int) -> prepare_data._Build:
    return prepare_data._Build(
        vocab_size=SPEC.vocab_size,
        seq_len=81,
        revision=prepare_data.SOURCE_REVISION,
        seed=0,
        num_puzzles=num_puzzles,
        copies_per_puzzle=copies_per_puzzle,
    )


class ParsedFlags(Protocol):
    directory: str | None
    num_puzzles: int
    copies_per_puzzle: int
    seed: int


SPEC: Final = SudokuSpec()
"""The standard 9x9 geometry every test builds against."""

SOLUTION: Final = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)
"""One valid solved grid, used to synthesize puzzles."""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """Write train/test CSVs holding four puzzles apiece."""
    rows = ["source,question,answer,rating"]
    for i in range(4):
        cells = list(SOLUTION)
        for j in range(0, 81, 6):
            cells[(j + i) % 81] = "."
        rows.append(f"synthetic,{''.join(cells)},{SOLUTION},1.0")
    for split in ("train", "test"):
        (tmp_path / f"{split}.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_training_split_expands_and_test_split_does_not(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """Copies raise the training rows; the test split stays verbatim.

    A transformed test puzzle would not be a held-out puzzle, so only the
    training side is expanded.
    """
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    train = _load_array(out / "train" / "all__inputs.npy")
    test = _load_array(out / "test" / "all__inputs.npy")
    assert train.shape == (3 * 3, 81)  # 3 puzzles x (1 original + 2 copies)
    assert test.shape == (4, 81)  # Every source puzzle, untouched.


def test_group_indices_bound_each_puzzles_copies(tmp_path: Path, csv_dir: Path) -> None:
    """The loader shuffles within a puzzle, so the boundaries must be right."""
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    bounds = _load_array(out / "train" / "all__group_indices.npy")
    assert bounds.tolist() == [0, 3, 6, 9]


def test_tokens_land_in_the_documented_vocabulary(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """0 is pad, 1 is an empty cell, 2-10 are the digits."""
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    inputs = _load_array(out / "train" / "all__inputs.npy")
    labels = _load_array(out / "train" / "all__labels.npy")
    assert inputs.min() >= 1  # No padding in stored rows.
    assert inputs.max() <= 10
    unique_labels = cast(list[int], np.unique(labels).tolist())
    assert set(unique_labels) <= set(range(2, 11))  # Solved: no empties.
    metadata = parse((out / "train" / "dataset.json").read_text(), dict[str, object])
    assert (metadata["vocab_size"], metadata["seq_len"]) == (11, 81)


def test_transformations_keep_the_solution_valid(tmp_path: Path, csv_dir: Path) -> None:
    """Every generated copy must still be a solvable puzzle.

    A transformation that broke sudoku's constraints would teach the model
    contradictions, so this checks the property rather than the mechanics:
    each label grid has all nine digits in every row, column, and box.
    """
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=4,
        csv_directory=csv_dir,
    )
    labels = _load_array(out / "train" / "all__labels.npy")
    label_rows = cast(list[list[int]], labels.tolist())
    for row in label_rows:
        grid = [row[index : index + 9] for index in range(0, 81, 9)]
        expected = set(range(2, 11))
        assert all(set(line) == expected for line in grid)
        assert all(set(column) == expected for column in zip(*grid, strict=True))
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = [line[c : c + 3] for line in grid[r : r + 3]]
                assert {value for line in box for value in line} == expected


def test_the_clues_survive_transformation(tmp_path: Path, csv_dir: Path) -> None:
    """A transformed puzzle's clues must agree with its transformed solution."""
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=4,
        csv_directory=csv_dir,
    )
    inputs = _load_array(out / "train" / "all__inputs.npy")
    labels = _load_array(out / "train" / "all__labels.npy")
    given = inputs > 1  # Token 1 is an empty cell.
    assert np.array_equal(inputs[given], labels[given])


def test_the_build_is_deterministic(tmp_path: Path, csv_dir: Path) -> None:
    """One seed, one dataset -- otherwise a result cannot be reproduced."""
    first = prepare_data.prepare(
        tmp_path / "a",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    second = prepare_data.prepare(
        tmp_path / "b",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    assert np.array_equal(
        _load_array(first / "train" / "all__inputs.npy"),
        _load_array(second / "train" / "all__inputs.npy"),
    )


def test_a_different_seed_builds_different_data(tmp_path: Path, csv_dir: Path) -> None:
    first = prepare_data.prepare(
        tmp_path / "a",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    second = prepare_data.prepare(
        tmp_path / "b",
        num_puzzles=3,
        copies_per_puzzle=2,
        seed=1,
        csv_directory=csv_dir,
    )
    assert not np.array_equal(
        _load_array(first / "train" / "all__inputs.npy"),
        _load_array(second / "train" / "all__inputs.npy"),
    )


def test_rerunning_leaves_a_prepared_split_alone(tmp_path: Path, csv_dir: Path) -> None:
    """Idempotent, so re-running the preparer costs nothing."""
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    marker = out / "train" / "all__inputs.npy"
    stamp = marker.stat().st_mtime_ns
    prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    assert marker.stat().st_mtime_ns == stamp


def test_an_explicit_destination_is_validated_before_anything_is_written(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="normalized"):
        prepare_data.prepare(f"{tmp_path}/a/../data", csv_directory=csv_dir)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize(
    ("num_puzzles", "copies_per_puzzle", "flag"),
    [
        (0, 1, "--num-puzzles"),
        (-1, 1, "--num-puzzles"),
        (2, -1, "--copies-per-puzzle"),
    ],
)
def test_counts_are_validated_before_the_destination_is_created(
    tmp_path: Path,
    csv_dir: Path,
    num_puzzles: int,
    copies_per_puzzle: int,
    flag: str,
) -> None:
    with pytest.raises(ValueError, match=flag):
        prepare_data.prepare(
            tmp_path / "data",
            num_puzzles=num_puzzles,
            copies_per_puzzle=copies_per_puzzle,
            csv_directory=csv_dir,
        )
    assert not (tmp_path / "data").exists()


def test_a_malformed_row_is_reported_by_line(tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    csv_path.write_text(
        "source,question,answer,rating\n"
        f"s,{SOLUTION},{SOLUTION},1\n"
        f"s,{SOLUTION[:80]},{SOLUTION},1\n",
    )
    with pytest.raises(ValueError, match=r"source\.csv:3: .*81 cells.*got 80"):
        prepare_data._read_csv(csv_path, spec=SPEC)
    csv_path.write_text(
        f"source,question,answer,rating\ns,{SOLUTION},{SOLUTION[:79]},1\n",
    )
    with pytest.raises(ValueError, match=r"source\.csv:2: .*81 cells.*got 79"):
        prepare_data._read_csv(csv_path, spec=SPEC)
    csv_path.write_text(f"source,question,answer,rating\ns,{SOLUTION},1\n")
    with pytest.raises(ValueError, match=r"source\.csv:2: expected 4 fields"):
        prepare_data._read_csv(csv_path, spec=SPEC)


def test_a_rebuild_with_different_parameters_is_refused(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """A split built under other flags is not silently reused."""
    prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    for num_puzzles, copies_per_puzzle, seed in ((3, 1, 42), (2, 2, 42), (2, 1, 7)):
        with pytest.raises(ValueError, match="was built with"):
            prepare_data.prepare(
                tmp_path / "data",
                num_puzzles=num_puzzles,
                copies_per_puzzle=copies_per_puzzle,
                seed=seed,
                csv_directory=csv_dir,
            )
    bounds = _load_array(tmp_path / "data" / "train" / "all__group_indices.npy")
    assert bounds.tolist() == [0, 2, 4]


def test_a_non_default_spec_is_what_the_build_writes(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    spec = SudokuSpec()
    spec.vocab_size = 12
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
        spec=spec,
    )
    metadata = parse(
        (out / "train" / "dataset.json").read_text(),
        dict[str, object],
    )
    assert metadata["vocab_size"] == 12
    assert metadata["seq_len"] == 81


def test_a_failed_download_leaves_no_staged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged CSV is hundreds of MB; every failure must remove it."""
    out = tmp_path / "output"
    out.mkdir()
    timeouts: list[float] = []

    class Truncated(io.BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            if self.tell():
                raise OSError("connection reset")
            return super().read(size)

    def open_source(url: str, *, timeout: float) -> io.BytesIO:
        del url
        timeouts.append(timeout)
        return Truncated(b"x" * 64)

    monkeypatch.setattr(request, "urlopen", open_source)
    with pytest.raises(OSError, match="connection reset"):
        prepare_data._download("train.csv", into=out)
    assert timeouts == [120]
    assert list(out.glob(".*.part")) == []

    def reject(path: Path, *, filename: str) -> None:
        del path, filename
        raise RuntimeError("digest mismatch")

    def open_complete(url: str, *, timeout: float) -> io.BytesIO:
        del url, timeout
        return io.BytesIO(b"x")

    monkeypatch.setattr(request, "urlopen", open_complete)
    monkeypatch.setattr(prepare_data, "_verify", reject)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        prepare_data.prepare(out, num_puzzles=1, copies_per_puzzle=0)
    assert list(out.glob(".*.part")) == []


def test_default_directory_matches_the_loaders(tmp_path: Path) -> None:
    """Preparer and training agree without either naming a path."""
    del tmp_path
    assert default_directory().name == "sudoku-extreme"
    trm_loaders = (
        PuzzleDataset.Config(),
        VerifierData.Config(),
        Harvest.Config(),
    )
    for config in trm_loaders:
        assert resolve_working_dir("/opt/scratch", config.working_dir) == (
            default_directory()
        )
    assert PuzzleDataset.Config().finalize().working_dir == default_directory()


def test_grid_and_csv_parsing_preserve_digits_and_empty_cells(tmp_path: Path) -> None:
    grid = prepare_data._grid("0" + SOLUTION[1:], spec=SPEC)
    assert grid.shape == (9, 9)
    assert grid.dtype == np.int64
    assert grid[0, 0] == 0
    assert grid[0, 1] == 3

    csv_path = tmp_path / "source.csv"
    csv_path.write_text(
        f"source,question,answer,rating\ns,{'.' + SOLUTION[1:]},{SOLUTION},1\n",
    )
    puzzles, solutions = prepare_data._read_csv(csv_path, spec=SPEC)
    assert len(puzzles) == len(solutions) == 1
    assert puzzles[0][0, 0] == 0
    assert np.array_equal(solutions[0], prepare_data._grid(SOLUTION, spec=SPEC))


def test_csv_reader_preserves_embedded_crlf_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "source.csv"
    csv_path.write_bytes(
        (
            "source,question,answer,rating\r\n"
            f'"first\r\nsecond",{SOLUTION},{SOLUTION},1\r\n'
        ).encode(),
    )
    opened_with: list[str | None] = []
    original_open = Path.open

    def track_open(path: Path, *, newline: str | None = None) -> object:
        if path == csv_path:
            opened_with.append(newline)
        return original_open(path, newline=newline)

    monkeypatch.setattr(Path, "open", track_open)
    puzzles, solutions = prepare_data._read_csv(csv_path, spec=SPEC)
    assert opened_with == [""]
    assert len(puzzles) == len(solutions) == 1
    assert np.array_equal(solutions[0], prepare_data._grid(SOLUTION, spec=SPEC))


def test_tokenize_shifts_digits_and_rejects_out_of_range_values() -> None:
    grid = prepare_data._grid(SOLUTION, spec=SPEC)
    tokens = prepare_data._tokenize([grid], spec=SPEC)
    assert tokens.shape == (1, 81)
    assert tokens.dtype == np.int64
    assert tokens.tolist() == [[int(digit) + 1 for digit in SOLUTION]]
    invalid = grid.copy()
    invalid[0, 0] = 10
    with pytest.raises(ValueError, match=r"^Every cell must be a digit 0-9"):
        prepare_data._tokenize([invalid], spec=SPEC)


@pytest.mark.parametrize(
    ("draw", "transpose"),
    [(0.25, True), (0.5, False), (0.75, False)],
)
def test_transform_applies_the_seeded_permutations(
    draw: float,
    transpose: bool,
) -> None:
    class ReversedPermutation:
        def permutation(self, size: int) -> np.ndarray:
            return np.arange(size)[::-1]

        def random(self) -> float:
            return draw

    puzzle = prepare_data._grid("0" + SOLUTION[1:], spec=SPEC)
    solution = prepare_data._grid(SOLUTION, spec=SPEC)
    actual_puzzle, actual_solution = prepare_data._transform(
        puzzle,
        solution=solution,
        spec=SPEC,
        rng=cast(np.random.Generator, ReversedPermutation()),
    )
    expected_puzzle = puzzle.T if transpose else puzzle
    expected_puzzle = expected_puzzle[::-1, ::-1]
    is_empty = cast("NDArray[np.bool_]", expected_puzzle == 0)
    expected_puzzle = cast(
        "NDArray[np.int64]",
        np.where(is_empty, 0, 10 - expected_puzzle),
    )
    expected_solution = solution.T if transpose else solution
    expected_solution = 10 - expected_solution[::-1, ::-1]
    assert actual_puzzle.dtype == actual_solution.dtype == np.int64
    assert np.array_equal(actual_puzzle, expected_puzzle)
    assert np.array_equal(actual_solution, expected_solution)


def test_split_creation_handles_nested_output_and_keeps_original(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        f"source,question,answer,rating\ns,{'.' + SOLUTION[1:]},{SOLUTION},1\n",
    )
    out = tmp_path / "nested" / "data"
    prepare_data._build_split(
        "train",
        out=out,
        build=_build(num_puzzles=None, copies_per_puzzle=0),
        spec=SPEC,
        rng=np.random.default_rng(3),
        csv_path=csv_path,
    )
    (out / "train" / "dataset.json").unlink()
    prepare_data._build_split(
        "train",
        out=out,
        build=_build(num_puzzles=None, copies_per_puzzle=0),
        spec=SPEC,
        rng=np.random.default_rng(3),
        csv_path=csv_path,
    )
    inputs = _load_array(out / "train" / "all__inputs.npy")
    labels = _load_array(out / "train" / "all__labels.npy")
    bounds = _load_array(out / "train" / "all__group_indices.npy")
    assert inputs.shape == labels.shape == (1, 81)
    expected_inputs = prepare_data._tokenize(
        [prepare_data._grid("0" + SOLUTION[1:], spec=SPEC)],
        spec=SPEC,
    )
    expected_labels = prepare_data._tokenize(
        [prepare_data._grid(SOLUTION, spec=SPEC)],
        spec=SPEC,
    )
    assert np.array_equal(inputs, expected_inputs)
    assert np.array_equal(labels, expected_labels)
    assert bounds.tolist() == [0, 1]


def test_build_split_rejects_mismatched_puzzle_and_solution_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = prepare_data._grid(SOLUTION, spec=SPEC)

    def mismatched_csv(
        path: Path,
        *,
        spec: SudokuSpec,
    ) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]]]:
        del path, spec
        return [grid], []

    monkeypatch.setattr(prepare_data, "_read_csv", mismatched_csv)
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        prepare_data._build_split(
            "train",
            out=tmp_path,
            build=_build(num_puzzles=None, copies_per_puzzle=0),
            spec=SPEC,
            rng=np.random.default_rng(0),
            csv_path=tmp_path / "unused.csv",
        )


def test_equal_puzzle_limit_preserves_source_order(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    source = csv_dir / "train.csv"
    with source.open(newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    expected_inputs = np.stack(
        [prepare_data._grid(row[1].replace(".", "0"), spec=SPEC) + 1 for row in rows],
    ).reshape(len(rows), -1)
    expected_labels = np.stack(
        [prepare_data._grid(row[2], spec=SPEC) + 1 for row in rows],
    ).reshape(len(rows), -1)
    out = prepare_data.prepare(
        tmp_path / "data",
        num_puzzles=len(rows),
        copies_per_puzzle=0,
        csv_directory=csv_dir,
    )
    assert np.array_equal(
        _load_array(out / "train" / "all__inputs.npy"),
        expected_inputs,
    )
    assert np.array_equal(
        _load_array(out / "train" / "all__labels.npy"),
        expected_labels,
    )


def test_subsampling_disables_replacement(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    calls: list[tuple[int, int, bool]] = []

    class CheckingRng:
        def choice(
            self,
            population: int,
            *,
            size: int,
            replace: bool = True,
        ) -> NDArray[np.int64]:
            calls.append((population, size, replace))
            assert replace is False
            return np.array([0, 1, 2], dtype=np.int64)

    prepare_data._build_split(
        "train",
        out=tmp_path,
        build=_build(num_puzzles=3, copies_per_puzzle=0),
        spec=SPEC,
        rng=cast(np.random.Generator, CheckingRng()),
        csv_path=csv_dir / "train.csv",
    )
    assert calls == [(4, 3, False)]


def test_subsampling_never_duplicates_source_puzzles(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    for seed in range(12):
        out = prepare_data.prepare(
            tmp_path / str(seed),
            num_puzzles=3,
            copies_per_puzzle=0,
            seed=seed,
            csv_directory=csv_dir,
        )
        inputs = _load_array(out / "train" / "all__inputs.npy")
        assert len(np.unique(inputs, axis=0)) == 3


def test_prepare_defaults_and_uses_default_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "nested" / "default"
    monkeypatch.setattr(prepare_data, "default_directory", lambda: destination)
    calls: list[
        tuple[str, Path, int | None, int, Path | None, np.random.Generator]
    ] = []

    def record_split(
        split: str,
        *,
        out: Path,
        build: prepare_data._Build,
        spec: SudokuSpec,
        rng: np.random.Generator,
        csv_path: Path | None,
    ) -> None:
        assert spec == SPEC
        assert (build.seed, build.revision) == (42, prepare_data.SOURCE_REVISION)
        calls.append(
            (split, out, build.num_puzzles, build.copies_per_puzzle, csv_path, rng),
        )

    monkeypatch.setattr(prepare_data, "_build_split", record_split)
    result = prepare_data.prepare()
    assert result == destination
    assert destination.is_dir()
    assert [call[:5] for call in calls] == [
        ("train", destination, 1_000, 1_000, None),
        ("test", destination, None, 0, None),
    ]
    assert calls[0][5] is calls[1][5]
    assert calls[0][5].integers(0, 1_000) == np.random.default_rng(42).integers(
        0,
        1_000,
    )


def test_prepared_splits_log_exact_counts_and_skip_existing(
    tmp_path: Path,
    csv_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        out = prepare_data.prepare(
            tmp_path / "data",
            num_puzzles=2,
            copies_per_puzzle=1,
            csv_directory=csv_dir,
        )
    assert [record.getMessage() for record in caplog.records] == [
        "sudoku 'train': 2 puzzles -> 4 rows",
        "sudoku 'test': 4 puzzles -> 4 rows",
        f"sudoku data ready at {out}",
    ]
    train_bounds = _load_array(out / "train" / "all__group_indices.npy")
    assert train_bounds.dtype == np.int32
    assert np.array_equal(train_bounds, np.array([0, 2, 4], dtype=np.int32))

    caplog.clear()
    with caplog.at_level(logging.INFO):
        prepare_data.prepare(
            tmp_path / "data",
            num_puzzles=2,
            copies_per_puzzle=1,
            csv_directory=csv_dir,
        )
    assert [record.getMessage() for record in caplog.records] == [
        "sudoku 'train' already prepared; skipping",
        "sudoku 'test' already prepared; skipping",
        f"sudoku data ready at {out}",
    ]


def test_defaults_and_cli_flags_are_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parameters = inspect.signature(prepare_data.prepare).parameters
    assert cast(object, parameters["num_puzzles"].default) == 1_000
    assert cast(object, parameters["copies_per_puzzle"].default) == 1_000
    assert cast(object, parameters["seed"].default) == 42
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    flags = cast(ParsedFlags, parser.parse_args([]))
    assert (
        flags.directory,
        flags.num_puzzles,
        flags.copies_per_puzzle,
        flags.seed,
    ) == (
        None,
        1_000,
        1_000,
        42,
    )
    help_text = parser.format_help()
    assert "--directory DIRECTORY" in help_text
    assert "--num-puzzles NUM_PUZZLES" in help_text
    assert "--copies-per-puzzle COPIES_PER_PUZZLE" in help_text
    assert "--seed SEED" in help_text
    assert parser._option_string_actions["--directory"].help == (
        f"destination (default: {default_directory()})"
    )
    assert parser._option_string_actions["--num-puzzles"].help == (
        "training puzzles kept from the source split"
    )
    assert parser._option_string_actions["--copies-per-puzzle"].help == (
        "transformed copies written per kept training puzzle"
    )
    assert parser._option_string_actions["--seed"].help == "seeds the whole build"

    basic_config: list[tuple[int, str]] = []

    def record_basic_config(*, level: int, format: str) -> None:
        basic_config.append((level, format))

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)
    received: list[tuple[object, int, int, int]] = []

    def record_prepare(*args: object, **kwargs: int) -> None:
        received.append(
            (
                args[0],
                kwargs["num_puzzles"],
                kwargs["copies_per_puzzle"],
                kwargs["seed"],
            ),
        )

    monkeypatch.setattr(prepare_data, "prepare", record_prepare)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(tmp_path / "sudoku"),
            "--num-puzzles",
            "7",
            "--copies-per-puzzle",
            "3",
            "--seed",
            "9",
        ],
    )
    assert prepare_data.main() == 0
    assert received == [(str(tmp_path / "sudoku"), 7, 3, 9)]
    assert basic_config == [(logging.INFO, "%(message)s")]

    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        prepare_data.main()
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: prepare_data.py")
    assert "Download Sudoku-Extreme" in help_text
    assert "Run once before the first experiment." in help_text
    assert "exec uv" not in help_text


def test_downloaded_split_verifies_and_removes_the_staged_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "output"
    downloaded = out / "train.csv"
    downloaded_paths: list[tuple[str, Path]] = []
    verified: list[tuple[Path, str]] = []
    out.mkdir()
    downloaded.write_text(
        f"source,question,answer,rating\ns,{'.' + SOLUTION[1:]},{SOLUTION},1\n",
    )

    def download(filename: str, *, into: Path) -> Path:
        downloaded_paths.append((filename, into))
        return downloaded

    def verify(path: Path, *, filename: str) -> None:
        verified.append((path, filename))

    unlink_calls: list[tuple[Path, bool]] = []
    original_unlink = Path.unlink

    def record_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_calls.append((path, missing_ok))
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(prepare_data, "_download", download)
    monkeypatch.setattr(prepare_data, "_verify", verify)
    monkeypatch.setattr(Path, "unlink", record_unlink)
    prepare_data._build_split(
        "train",
        out=out,
        build=_build(num_puzzles=None, copies_per_puzzle=0),
        spec=SPEC,
        rng=np.random.default_rng(0),
        csv_path=None,
    )

    assert downloaded_paths == [("train.csv", out)]
    assert verified == [(downloaded, "train.csv")]
    assert unlink_calls == [(downloaded, True)]
    assert not downloaded.exists()
    assert _load_array(out / "train" / "all__inputs.npy").shape == (1, 81)


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


def test_download_is_streamed_and_digest_verification_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = b"source,question,answer,rating\\ns,q,a,1\\n"

    def open_source(url: str, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        assert url == (
            f"{prepare_data.SOURCE_URL}/{prepare_data.SOURCE_REVISION}/train.csv"
        )
        return io.BytesIO(payload)

    monkeypatch.setattr(request, "urlopen", open_source)
    download_dir = tmp_path / "nested" / "cache"
    with caplog.at_level(logging.INFO):
        downloaded = prepare_data._download("train.csv", into=download_dir)
        second_download = prepare_data._download("train.csv", into=download_dir)
    assert downloaded.parent == download_dir
    assert downloaded.name.startswith(".train.csv.")
    assert downloaded.name.endswith(".part")
    assert downloaded.read_bytes() == payload
    assert second_download.read_bytes() == payload
    assert [record.getMessage() for record in caplog.records] == [
        (
            f"downloading {prepare_data.SOURCE_URL}/"
            f"{prepare_data.SOURCE_REVISION}/train.csv"
        ),
    ] * 2

    monkeypatch.setattr(
        prepare_data,
        "SOURCE_SHA256",
        {"train.csv": hashlib.sha256(payload).hexdigest()},
    )
    read_sizes: list[int | None] = []

    class RecordingReader(io.BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    original_open = Path.open

    def open_download(path: Path, mode: str = "r") -> object:
        if path == downloaded and mode == "rb":
            return RecordingReader(payload)
        return original_open(path, mode)

    monkeypatch.setattr(Path, "open", open_download)
    prepare_data._verify(downloaded, filename="train.csv")
    assert read_sizes == [1 << 20, 1 << 20]
    monkeypatch.setattr(prepare_data, "SOURCE_SHA256", {"train.csv": "0" * 64})
    with pytest.raises(RuntimeError, match=r"train\.csv at revision") as exc_info:
        prepare_data._verify(downloaded, filename="train.csv")
    assert hashlib.sha256(payload).hexdigest() in str(exc_info.value)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
