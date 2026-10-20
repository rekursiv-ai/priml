"""Real staging CLI boundaries, archive safety, and raw-data provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import subprocess
import sys
import zipfile

from treekle import json_loads

import pytest

from priml.baselines.motor_imagery.scripts.prepare_data import main, stage_archive


if TYPE_CHECKING:
    from pathlib import Path


def write_archive(path: Path) -> None:
    """Write both required raw recording directories for staging tests."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Training set/S002-001_eeg.csv", "Fz,Marker_val\n0,rest\n")
        archive.writestr(
            "Cross participant test set/S008_test_001_eeg.csv",
            "Fz,Marker_val\n0,cue_start\n",
        )


def test_staging_records_raw_files_without_preprocessing(tmp_path: Path) -> None:
    archive = tmp_path / "input.zip"
    write_archive(archive)
    output = tmp_path / "raw"
    stage_archive(archive, directory=output)
    manifest = json_loads((output / "manifest.json").read_text(), dict[str, str])
    assert len(manifest) == 2
    assert all(len(digest) == 64 for digest in manifest.values())
    assert (
        output / "Training set/S002-001_eeg.csv"
    ).read_text() == "Fz,Marker_val\n0,rest\n"


def test_archive_rejects_path_escape_and_missing_recordings(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("../escape.csv", "invalid")
    with pytest.raises(ValueError, match="escapes staging destination"):
        stage_archive(archive, directory=tmp_path / "raw")
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("unrelated.csv", "invalid")
    with pytest.raises(ValueError, match="no CSV recordings"):
        stage_archive(archive, directory=tmp_path / "missing")


@pytest.mark.parametrize("complete", [True, False])
def test_staging_refuses_to_merge_or_reuse_an_existing_population(
    tmp_path: Path,
    complete: bool,
) -> None:
    first = tmp_path / "first.zip"
    write_archive(first)
    output = tmp_path / "raw"
    stage_archive(first, directory=output)
    second = tmp_path / "second.zip"
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("Training set/S001-001_eeg.csv", "new population")
        if complete:
            archive.writestr("Cross participant test set/S013_test.csv", "test")
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        stage_archive(second, directory=output)
    assert (output / "manifest.json").read_bytes() == before
    assert not (output / "Training set/S001-001_eeg.csv").exists()


def test_cli_prints_full_config_without_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--print-config"])
    assert main() == 0
    assert "max_epochs=18" in capsys.readouterr().out


def test_cli_extracts_explicit_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "input.zip"
    write_archive(archive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--archive",
            str(archive),
            "--directory",
            str(tmp_path / "raw"),
        ],
    )
    assert main() == 0


def test_cli_refuses_an_existing_stage_before_any_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raw"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--directory", str(output)],
    )
    with pytest.raises(FileExistsError, match="Refusing to merge"):
        main()


def test_cli_download_uses_existing_auth_without_accepting_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raw"
    cache = tmp_path / "cache"

    def download(argv: list[str], *, check: bool) -> None:
        assert check
        assert argv[1:4] == ["competitions", "download", "-c"]
        assert "auth" not in argv
        assert "accept" not in argv
        write_archive(
            cache / "low-cost-motor-imagery-decoding-for-rehab-cross-subject.zip",
        )

    monkeypatch.setattr(subprocess, "run", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--directory",
            str(output),
            "--download-directory",
            str(cache),
        ],
    )
    assert main() == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
