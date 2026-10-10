"""Tests for the hard-catalog dataset build."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast
from urllib import request

import csv
import hashlib
import io
import zipfile

import numpy as np
import pytest

from priml.baselines.sudoku.scripts import build_hard_catalogs, prepare_data
from priml.baselines.sudoku.scripts.build_hard_catalogs import (
    CSV_HEADER,
    BuildReport,
    CatalogSpec,
    OverlapReport,
    answer_text,
    build_dataset,
    enumerate_completions,
    is_valid_completion,
    parse_catalog_text,
    parse_puzzle_line,
    question_text,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from numpy.typing import NDArray


SOLVED: Final = (
    "123456789456789123789123456234567891567891234891234567345678912678912345912345678"
)


def _blank(cells: tuple[int, ...]) -> str:
    """SOLVED with the given cells blanked -- each blank is forced."""
    return "".join("." if i in cells else c for i, c in enumerate(SOLVED))


PUZZLE_A: Final = _blank((0, 10, 20))
PUZZLE_B: Final = _blank((5, 15, 25))
PUZZLE_C: Final = _blank((40, 50, 60))


def _spec(
    name: str,
    rows: tuple[str, ...],
    *,
    zip_member: str = "",
    expected_count: int = 0,
) -> CatalogSpec:
    return CatalogSpec(
        source_name=name,
        source=name,
        source_url=f"https://example.com/{name}.zip" if zip_member else "",
        expected_count=expected_count or len(rows),
        zip_member=zip_member,
        rows=tuple((row, f"https://example.com/{name}") for row in rows),
    )


def _array(path: Path) -> NDArray[np.integer]:
    """Load one npy array."""
    return cast("NDArray[np.integer]", np.load(path))


def _manifest(dataset_dir: Path) -> dict[str, object]:
    """Load the published manifest."""
    return from_plain(
        loads((dataset_dir / "manifest.json").read_text()),
        dict[str, object],
    )


def _archive(text: str) -> zipfile.ZipFile:
    """Return a one-member in-memory archive holding ``text`` at ``data/tiny``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("data/tiny", text)
    return zipfile.ZipFile(buffer)


def test_question_and_answer_text_round_trip() -> None:
    tokens = parse_puzzle_line(PUZZLE_A)
    assert question_text(tokens) == PUZZLE_A
    assert answer_text(parse_puzzle_line(SOLVED)) == SOLVED
    with pytest.raises(ValueError, match="81 characters"):
        parse_puzzle_line(SOLVED[:-1])
    with pytest.raises(ValueError, match="invalid puzzle character"):
        parse_puzzle_line("x" + SOLVED[1:])


def test_catalog_text_skips_comments_and_repeats() -> None:
    puzzles = parse_catalog_text(f"# comment\n\n{PUZZLE_A}\n{PUZZLE_B}\n{PUZZLE_A}\n")
    assert puzzles == (parse_puzzle_line(PUZZLE_A), parse_puzzle_line(PUZZLE_B))
    with pytest.raises(ValueError, match="no puzzle rows"):
        parse_catalog_text("# only a comment\n")


def test_enumeration_classifies_proper_improper_and_unsatisfiable() -> None:
    proper = enumerate_completions(parse_puzzle_line(PUZZLE_A))
    assert (proper.status, proper.grid) == ("proper", parse_puzzle_line(SOLVED))
    # Blanking every 1 and 2 admits the 1<->2 digit swap: two completions.
    improper = parse_puzzle_line("".join("." if c in "12" else c for c in SOLVED))
    assert enumerate_completions(improper).status == "improper"
    contradiction = parse_puzzle_line("11" + "." * 79)
    assert enumerate_completions(contradiction) == build_hard_catalogs.Enumeration(
        status="unsatisfiable",
        grid=None,
    )
    with pytest.raises(ValueError, match="malformed"):
        enumerate_completions((1,) * 80)


def test_validator_rejects_each_broken_rule() -> None:
    solution = parse_puzzle_line(SOLVED)
    puzzle = parse_puzzle_line(PUZZLE_A)
    assert is_valid_completion(solution, puzzle)
    swapped = list(solution)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert not is_valid_completion(swapped, (1,) * 81)
    assert not is_valid_completion(solution[:-1], puzzle)
    assert not is_valid_completion((1, *solution[1:]), puzzle)
    other_given = (solution[1], *puzzle[1:])
    assert not is_valid_completion(solution, other_given)


def test_build_dataset_end_to_end(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    report = build_dataset(
        dataset_dir,
        archive=None,
        catalogs=(
            _spec("cat_a", (PUZZLE_A, PUZZLE_B)),
            _spec("cat_b", (PUZZLE_B, PUZZLE_C)),
        ),
        workers=1,
    )
    assert report.row_count == 3
    assert report.duplicate_count == 1
    assert report.duplicate_pairs == {"cat_a+cat_b": 1}
    assert [c.new_rows for c in report.catalogs] == [2, 1]
    assert [c.duplicate_rows for c in report.catalogs] == [0, 1]

    with (dataset_dir / "test.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == CSV_HEADER
    assert len(rows) == 4
    by_question = {row[1]: row for row in rows[1:]}
    assert by_question[PUZZLE_A][2] == SOLVED
    assert by_question[PUZZLE_A][3] == "0"
    duplicate = by_question[PUZZLE_B]
    assert duplicate[0] == "cat_a;cat_b"
    assert duplicate[4] == "cat_a;cat_b"
    assert duplicate[5] == "https://example.com/cat_a;https://example.com/cat_b"
    assert by_question[PUZZLE_C][4] == "cat_b"

    inputs = _array(dataset_dir / "test" / "all__inputs.npy")
    labels = _array(dataset_dir / "test" / "all__labels.npy")
    groups = _array(dataset_dir / "test" / "all__group_indices.npy")
    assert inputs.shape == labels.shape == (3, 81)
    assert inputs.dtype == labels.dtype == np.uint8
    assert [int(groups.item(index)) for index in range(4)] == [0, 1, 2, 3]
    for row in range(3):
        assert is_valid_completion(
            [int(labels.item(row, cell)) for cell in range(81)],
            [int(inputs.item(row, cell)) for cell in range(81)],
        )
    metadata = loads((dataset_dir / "test" / "dataset.json").read_text())
    assert metadata == {"vocab_size": 11, "seq_len": 81}
    assert _array(dataset_dir / "train" / "all__inputs.npy").shape == (1, 81)

    card = (dataset_dir / "README.md").read_text()
    assert "Rows: 3 unique puzzles" in card
    assert "cat_a+cat_b" in card

    manifest = _manifest(dataset_dir)
    assert manifest["n_rows"] == 3
    for relative, digest in from_plain(manifest["files"], dict[str, str]).items():
        payload = (dataset_dir / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest


def _write_extreme_csv(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a minimal sudoku-extreme-schema CSV of (source, question) rows."""
    lines = ["source,question,answer,rating"]
    lines += [f"{source},{question},{SOLVED},1" for source, question in rows]
    path.write_text("\n".join(lines) + "\n")


def test_extreme_overlap_join(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_extreme_csv(train_csv, (("cat_a", PUZZLE_A),))
    _write_extreme_csv(test_csv, (("cat_a", PUZZLE_B), ("other", PUZZLE_B)))
    dataset_dir = tmp_path / "dataset"
    report = build_dataset(
        dataset_dir,
        archive=None,
        catalogs=(_spec("cat_a", (PUZZLE_A, PUZZLE_B, PUZZLE_C)),),
        workers=1,
        extreme_train_csv=train_csv,
        extreme_test_csv=test_csv,
    )
    assert report.overlap == (
        OverlapReport(
            source_name="cat_a",
            total=3,
            in_extreme_train=1,
            in_extreme_test=1,
            in_neither=1,
            derived_in_extreme_train=1,
            derived_in_extreme_test=1,
        ),
    )
    card = (dataset_dir / "README.md").read_text()
    assert "## Overlap with sudoku-extreme" in card
    assert "| cat_a | 3 | 1 | 1 | 1 | 1 | 1 |" in card
    overlap = from_plain(
        _manifest(dataset_dir)["extreme_overlap"],
        list[dict[str, object]],
    )
    assert overlap[0]["in_neither"] == 1
    assert overlap[0]["derived_in_extreme_test"] == 1
    with pytest.raises(ValueError, match="come together"):
        build_dataset(
            dataset_dir,
            archive=None,
            catalogs=(_spec("cat_a", (PUZZLE_A,)),),
            extreme_train_csv=train_csv,
        )


def test_rebuild_replaces_existing_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for catalog in ("cat_a", "cat_b"):
        report = build_dataset(
            dataset_dir,
            archive=None,
            catalogs=(_spec(catalog, (PUZZLE_A,)),),
            workers=1,
        )
        assert report.row_count == 1
    catalogs = from_plain(_manifest(dataset_dir)["catalogs"], list[dict[str, object]])
    assert catalogs[0]["source_name"] == "cat_b"
    assert not list(tmp_path.glob("dataset.old.*"))


def test_rebuild_restores_live_dataset_when_publish_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    build_dataset(
        dataset_dir,
        archive=None,
        catalogs=(_spec("cat_a", (PUZZLE_A,)),),
        workers=1,
    )
    original_manifest = (dataset_dir / "manifest.json").read_bytes()
    replace = Path.replace

    def fail_replacement(path: Path, target: Path) -> Path:
        if path.name.startswith("dataset.tmp."):
            raise OSError("injected replacement failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replacement)
    with pytest.raises(OSError, match="injected replacement"):
        build_hard_catalogs._publish(
            dataset_dir,
            rows=(),
            report=BuildReport(
                dataset_dir=dataset_dir,
                row_count=0,
                catalogs=(),
                duplicate_count=0,
                duplicate_pairs={},
            ),
        )
    assert (dataset_dir / "manifest.json").read_bytes() == original_manifest


def test_zip_catalog_reads_the_archive(tmp_path: Path) -> None:
    report = build_dataset(
        tmp_path / "dataset",
        archive=_archive(f"# comment\n{PUZZLE_A}\n{PUZZLE_B}\n"),
        catalogs=(_spec("tiny", (), zip_member="data/tiny", expected_count=2),),
        workers=1,
    )
    assert report.row_count == 2
    with pytest.raises(ValueError, match="needs the tdoku archive"):
        build_dataset(
            tmp_path / "other",
            archive=None,
            catalogs=(_spec("tiny", (), zip_member="data/tiny", expected_count=2),),
        )


def test_invalid_catalogs_raise(tmp_path: Path) -> None:
    improper = "".join("." if c in "12" else c for c in SOLVED)
    with pytest.raises(ValueError, match="non-proper"):
        build_dataset(
            tmp_path / "dataset",
            archive=None,
            catalogs=(_spec("bad", (improper,)),),
            workers=1,
        )
    with pytest.raises(ValueError, match="expected 5"):
        build_dataset(
            tmp_path / "dataset",
            archive=None,
            catalogs=(_spec("cat_a", (PUZZLE_A,), expected_count=5),),
            workers=1,
        )
    with pytest.raises(ValueError, match="duplicate inline rows"):
        build_dataset(
            tmp_path / "dataset",
            archive=None,
            catalogs=(_spec("cat_a", (PUZZLE_A, PUZZLE_A)),),
            workers=1,
        )
    with pytest.raises(ValueError, match="exactly one of rows or zip_member"):
        build_dataset(
            tmp_path / "dataset",
            archive=None,
            catalogs=(_spec("cat_a", (PUZZLE_A,), zip_member="data/tiny"),),
            workers=1,
        )
    with pytest.raises(ValueError, match="workers"):
        build_dataset(tmp_path / "dataset", archive=None, workers=0)


def test_duplicated_zip_line_surfaces_as_the_unique_count_error(
    tmp_path: Path,
) -> None:
    # A duplicated mirror line drops at parse, so the count check fires; its
    # message must name the dedup instead of implying the source shrank.
    with pytest.raises(ValueError, match="parsed 2 unique puzzles"):
        build_dataset(
            tmp_path / "dataset",
            archive=_archive(f"{PUZZLE_A}\n{PUZZLE_A}\n{PUZZLE_B}\n"),
            catalogs=(_spec("tiny", (), zip_member="data/tiny", expected_count=3),),
        )


def test_download_pins_the_archive_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("data/tiny", f"{PUZZLE_A}\n")
    payload = buffer.getvalue()

    class _Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return payload

    opened: list[str] = []

    def urlopen(url: str, *, timeout: float) -> _Response:
        del timeout
        opened.append(url)
        return _Response()

    monkeypatch.setattr(request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="digest mismatch"):
        build_hard_catalogs.download_tdoku_archive()
    monkeypatch.setattr(
        build_hard_catalogs,
        "TDOKU_DATA_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    archive = build_hard_catalogs.download_tdoku_archive()
    assert archive.read("data/tiny").decode() == f"{PUZZLE_A}\n"
    # Pinned to a commit, never a moving branch.
    assert opened == [build_hard_catalogs.TDOKU_DATA_URL] * 2
    assert "/a05701253ed175e2859063dec3cf01a7f5b9c6a7/" in opened[0]


def test_extreme_csv_pins_the_revision_and_the_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("source,question,answer,rating\n")
    asked: list[str | None] = []

    def download(repo_id: str, name: str, *, repo_type: str, revision: str) -> str:
        del repo_id, name, repo_type
        asked.append(revision)
        return str(csv_path)

    monkeypatch.setattr(build_hard_catalogs, "hf_hub_download", download)
    with pytest.raises(ValueError, match="digest"):
        build_hard_catalogs.extreme_csv("test.csv")
    monkeypatch.setitem(
        prepare_data.SOURCE_SHA256,
        "test.csv",
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )
    assert build_hard_catalogs.extreme_csv("test.csv") == csv_path
    assert asked == [prepare_data.SOURCE_REVISION] * 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
