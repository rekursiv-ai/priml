"""Tests for staging admitted ARC2 datasets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import argparse
import sys
import tempfile

import numpy as np
import pytest

from priml.baselines.arcagi2.scripts import prepare_data
from priml.lib.testing.cli import assert_help_without_docstring


def test_cli_arguments_and_main_dispatch() -> None:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    defaults = parser.parse_args(["source"])
    assert vars(defaults) == {
        "source": Path("source"),
        "destination": Path("/opt/scratch/datasets/arc2concept-aug-1000"),
    }
    flags = parser.parse_args(
        ["source-tree", "--destination", "prepared-tree"],
    )
    assert vars(flags) == {
        "source": Path("source-tree"),
        "destination": Path("prepared-tree"),
    }

    with (
        patch.object(
            sys,
            "argv",
            ["prepare_data.py", "source-tree", "--destination", "prepared-tree"],
        ),
        patch.object(prepare_data, "prepare") as stage,
    ):
        assert prepare_data.main() == 0
    stage.assert_called_once_with(
        Path("source-tree"),
        destination=Path("prepared-tree"),
    )


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


def test_main_help_preserves_the_executable_description() -> None:
    descriptions: list[str | None] = []

    def create_parser(*, description: str | None) -> argparse.ArgumentParser:
        descriptions.append(description)
        return argparse.ArgumentParser(description=description)

    with (
        patch.object(
            prepare_data,
            "argparse",
            SimpleNamespace(ArgumentParser=create_parser),
        ),
        patch.object(sys, "argv", ["prepare_data.py", "--help"]),
        pytest.raises(SystemExit) as raised,
    ):
        prepare_data.main()

    assert raised.value.code == 0
    assert descriptions == [(prepare_data.__doc__ or "").split("\n", 2)[2]]


def _source_tree(source: Path) -> list[Path]:
    source.mkdir()
    (source / "_build_params.json").write_text(
        '{"subsets": ["training2", "evaluation2", "concept"], '
        '"test_set_name": "evaluation2"}',
    )
    (source / "identifiers.json").write_text('["<blank>", "puzzle"]')
    (source / "test_puzzles.json").write_text('{"puzzle": []}')
    expected: list[Path] = [
        Path("identifiers.json"),
        Path("test_puzzles.json"),
    ]
    for split in ("train", "test"):
        directory = source / split
        directory.mkdir()
        (directory / "dataset.json").write_text('{"split": "arc2"}')
        expected.append(Path(split) / "dataset.json")
        for name in (
            "inputs",
            "labels",
            "puzzle_indices",
            "group_indices",
            "puzzle_identifiers",
        ):
            np.save(directory / f"all__{name}.npy", np.array([2, 3]))
            expected.append(Path(split) / f"all__{name}.npy")
    expected.append(Path("_build_params.json"))
    return expected


def test_files_are_exact_arc2_contract_and_ignore_extras(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _source_tree(source)
    (source / "unadmitted.txt").write_text("ignored")

    assert prepare_data._files(source) == expected


def test_files_reject_wrong_subset_and_missing_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _source_tree(source)
    params = source / "_build_params.json"
    params.write_text(
        '{"subsets": ["training2", "evaluation2", "other"], '
        '"test_set_name": "evaluation2"}',
    )
    with pytest.raises(
        ValueError,
        match=r"^Expected an ARC2 training2/evaluation2/concept build$",
    ):
        prepare_data._files(source)

    params.write_text(
        '{"subsets": ["training2", "evaluation2", "concept"], '
        '"test_set_name": "evaluation2"}',
    )
    (source / expected[2]).unlink()
    with pytest.raises(ValueError, match="Missing or linked prepared ARC2 file"):
        prepare_data._files(source)


def test_prepare_rejects_a_changed_source_while_staging(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_tree(source)
    destination = tmp_path / "prepared"

    with (
        patch.object(prepare_data, "_digest", side_effect=[b"before", b"after"]),
        pytest.raises(
            ValueError,
            match=r"^ARC2 source changed while staging identifiers\.json$",
        ),
    ):
        prepare_data.prepare(source, destination=destination)

    assert not destination.exists()


def test_prepare_copies_only_contract_files_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _source_tree(source)
    (source / "extra").write_text("not admitted")
    parent = tmp_path / "prepared"
    parent.mkdir()
    destination = parent / "arc2"

    with patch.object(
        tempfile,
        "TemporaryDirectory",
        wraps=tempfile.TemporaryDirectory,
    ) as temporary_directory:
        prepare_data.prepare(source, destination=destination)
    assert temporary_directory.call_args.kwargs == {
        "prefix": "prepare-data-",
        "dir": parent,
    }

    assert sorted(
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    ) == sorted(expected)
    for name in expected:
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert not (destination / "extra").exists()
    with pytest.raises(FileExistsError, match=str(destination)):
        prepare_data.prepare(source, destination=destination)
    with pytest.raises(
        ValueError,
        match=r"^output path aliases protected input artifact:",
    ):
        prepare_data.prepare(source, destination=source)


def test_prepare_creates_missing_destination_ancestors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_tree(source)
    destination = tmp_path / "new" / "levels" / "deep" / "arc2"

    prepare_data.prepare(source, destination=destination)

    assert (destination / "_build_params.json").is_file()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
