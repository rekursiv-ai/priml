"""Tests for preparing the checksum-pinned ETTh1 CSV."""

from __future__ import annotations

from pathlib import Path
from urllib import request

import hashlib
import io
import sys

import pytest

from priml.baselines.etth1.experiments import exp000
from priml.baselines.etth1.scripts import data


PAYLOAD = b"date,a,b\n2000-01-01,1,2\n"


@pytest.fixture(autouse=True)
def pin_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(PAYLOAD).hexdigest())


def test_existing_verified_csv_is_kept(tmp_path: Path) -> None:
    (tmp_path / "ETTh1.csv").write_bytes(PAYLOAD)
    assert (
        data.prepare(tmp_path, source=tmp_path / "absent.csv") == tmp_path / "ETTh1.csv"
    )
    assert (tmp_path / "ETTh1.csv").read_bytes() == PAYLOAD


def test_existing_unverified_csv_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "ETTh1.csv").write_bytes(PAYLOAD + b"tampered\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.prepare(tmp_path)


def test_local_source_is_installed_beside_no_staging(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(PAYLOAD)
    directory = tmp_path / "nested" / "etth1"
    path = data.prepare(directory, source=source)
    assert path == directory / "ETTh1.csv"
    assert path.read_bytes() == PAYLOAD
    assert sorted(directory.iterdir()) == [path]


def test_unverified_source_installs_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"wrong\n")
    directory = tmp_path / "etth1"
    with pytest.raises(ValueError, match="refusing"):
        data.prepare(directory, source=source)
    assert not directory.exists()


def test_download_reads_only_the_pinned_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(request, "urlopen", urlopen)
    path = data.prepare(tmp_path)
    assert calls == [(data.DATASET_URL, 60)]
    assert data.DATASET_COMMIT in data.DATASET_URL
    assert path.read_bytes() == PAYLOAD


def test_main_prints_the_prepared_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(PAYLOAD)
    directory = tmp_path / "etth1"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--directory", str(directory), "--source", str(source)],
    )
    assert data.main() == 0
    assert capsys.readouterr().out == f"{directory / 'ETTh1.csv'}\n"
    assert (directory / "ETTh1.csv").read_bytes() == PAYLOAD


def test_main_defaults_to_the_experiment_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Path, Path | None]] = []

    def prepare(directory: Path, *, source: Path | None = None) -> Path:
        calls.append((directory, source))
        return directory / "ETTh1.csv"

    monkeypatch.setattr(data, "prepare", prepare)
    monkeypatch.setattr(sys, "argv", ["prepare_data"])
    assert data.main() == 0
    expected = Path(exp000().copy_tree().finalize().dataset.working_dir)
    assert calls == [(expected, None)]
    assert capsys.readouterr().out == f"{expected / 'ETTh1.csv'}\n"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
