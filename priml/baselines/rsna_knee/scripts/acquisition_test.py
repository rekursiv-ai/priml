"""Require complete series acquisition and secret-free transport errors."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, partialmethod
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, override
from zipfile import ZipFile, ZipInfo

import csv
import fcntl
import logging
import subprocess
import sys

from treekle import json_loads

import httpx
import pytest

from priml.baselines.rsna_knee.scripts import acquisition
from priml.baselines.rsna_knee.train_loop import verify_inventory


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from contextlib import AbstractContextManager


def test_import_does_not_initialize_kaggle_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Kaggle creates its user configuration directory on import, before test
    # fixtures can isolate it; concurrent first imports race in xdist workers.
    for name in tuple(sys.modules):
        if name in ("kaggle", "kagglesdk") or name.startswith(
            ("kaggle.", "kagglesdk."),
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "kaggle", None)
    monkeypatch.setitem(sys.modules, "kagglesdk", None)
    spec = spec_from_file_location(
        f"{acquisition.__name__}_isolated",
        acquisition.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert "Acquisition" in vars(module)


def test_acquisition_uses_installed_cli_token_and_streams_only_archive_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[httpx.Request] = []
    original = httpx.Client
    monkeypatch.setattr(subprocess, "run", _token)
    monkeypatch.setattr(
        original,
        "stream",
        partialmethod(_cli_stream, original=original.stream),
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        partial(_cli_client, original=original, calls=calls),
    )
    assert acquisition._archive_url() == "https://archive.test/secret-url"
    assert len(calls) == 1


class _UnreadablePayload(httpx.SyncByteStream):
    @override
    def __iter__(self) -> Iterator[bytes]:
        raise AssertionError("Archive payload must be acquired only by bounded ranges.")


@pytest.fixture
def _isolated_cli_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", _token)
    original = httpx.Client

    def authorize(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-token"
        return httpx.Response(302, headers={"Location": "https://example.test/secret"})

    monkeypatch.setattr(
        httpx,
        "Client",
        partial(original, transport=httpx.MockTransport(authorize)),
    )


def _token(
    command: list[str],
    *,
    capture_output: bool,
    check: bool,
    text: bool,
    timeout: int,
    authenticated: bool = True,
) -> subprocess.CompletedProcess[str]:
    assert command == ["/usr/bin/env", "kaggle", "auth", "print-access-token"]
    assert capture_output
    assert check
    assert text
    assert timeout == 60
    if not authenticated:
        raise subprocess.CalledProcessError(
            1,
            cmd=command,
            output="secret",
            stderr="secret",
        )
    return subprocess.CompletedProcess(
        command,
        returncode=0,
        stdout="secret-token\n",
        stderr="",
    )


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "os",
        "timeout",
        "empty",
        "multiline",
        "http",
        "transport",
        "redirect",
        "insecure",
        "loop",
        "no_location",
    ],
)
def test_acquisition_authentication_failures_never_expose_token_or_cli_output(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    original = httpx.Client
    monkeypatch.setattr(subprocess, "run", partial(_failure_run, mode=mode))
    monkeypatch.setattr(
        httpx,
        "Client",
        partial(original, transport=httpx.MockTransport(partial(_denied, mode=mode))),
    )
    with pytest.raises(
        RuntimeError,
        match="Authenticated archive request failed",
    ) as caught:
        acquisition._archive_url()
    assert "secret" not in str(caught.value)
    suffix = (
        " HTTP 403."
        if mode == "http"
        else " HTTP 302."
        if mode == "no_location"
        else " ConnectError."
        if mode == "transport"
        else ""
    )
    assert (
        str(caught.value) == "Authenticated archive request failed. "
        "Use an installed, authenticated Kaggle CLI." + suffix
    )
    assert caught.value.__suppress_context__ or mode in {
        "missing",
        "empty",
        "multiline",
        "redirect",
        "insecure",
        "loop",
    }


@pytest.mark.parametrize("authenticated", [True, False])
@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("suffix", [".dcm", ".DCM", ".dCm"])
@pytest.mark.usefixtures("_isolated_cli_authentication")
def test_full_acquisition_verifies_series_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *,
    authenticated: bool,
    relative: bool,
    suffix: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="httpx")
    monkeypatch.setattr(
        subprocess,
        "run",
        partial(_token, authenticated=authenticated),
    )
    raw = BytesIO()
    with ZipFile(raw, "w") as writer:
        writer.writestr(f"train_images/train/series/a{suffix}", b"pixel-a")
        writer.writestr("train_images/train/series/b.dcm", b"pixel-b")
        writer.writestr("test_images/test/other/c.dcm", b"pixel-c")
        writer.writestr("train_images/unused/unused/d.dcm", b"unused")
    monkeypatch.setattr(
        httpx,
        "stream",
        _ArchiveStream(raw=raw.getvalue(), lock=tmp_path / "staging.lock").stream,
    )
    config = acquisition.acquire()
    config.base_dir = None
    config.working_dir = tmp_path / "receipts"
    config.input_dir = Path("raw") if relative else tmp_path / "raw"
    config.staging_dir = tmp_path / "staging"
    config.selected_series = tmp_path / "selected.csv"
    config.workers = 1
    config.input_dir.mkdir()
    config.selected_series.write_text(
        "StudyInstanceUID,SeriesInstanceUID\ntrain,series\n",
    )
    (config.input_dir / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\ntest,other\n",
    )
    if authenticated:
        config.make().run()
        receipt = json_loads(
            (config.working_dir / "acquisition.json").read_text(),
            dict[str, object],
        )
        assert receipt["complete"] is True
        assert receipt["files"] == 3
        assert (config.staging_dir / ".ensure_complete").is_file()
        assert receipt["bytes"] == 21
        assert (
            config.input_dir / f"train_images/train/series/a{suffix}"
        ).read_bytes() == b"pixel-a"
        assert not (config.input_dir / "train_images/unused").exists()
        with (config.working_dir / "files.csv").open() as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == 3
        assert all(len(record["sha256"]) == 64 for record in records)
        verify_inventory(
            config.input_dir,
            acquisition_dir=config.working_dir,
            selected={
                Path("train_images/train/series"),
                Path("test_images/test/other"),
            },
        )
        config.make().run()
    else:
        with pytest.raises(RuntimeError, match="Authenticated") as caught:
            config.make().run()
        assert "secret" not in str(caught.value)
    assert "secret" not in caplog.text
    with pytest.raises(ValueError, match="configuration fields"):
        config.make().run("unexpected")


@pytest.mark.parametrize("suffix", [".dcm", ".DCM", ".dCm"])
@pytest.mark.usefixtures("_isolated_cli_authentication")
def test_acquisition_rejects_obsolete_published_pixels_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    raw = BytesIO()
    with ZipFile(raw, "w") as writer:
        writer.writestr("train_images/train/series/a.dcm", b"new pixels")
    monkeypatch.setattr(httpx, "stream", _ArchiveStream(raw=raw.getvalue()).stream)
    config = acquisition.acquire()
    config.base_dir = None
    config.working_dir = tmp_path / "receipts"
    config.input_dir = tmp_path / "raw"
    config.staging_dir = tmp_path / "staging"
    config.selected_series = tmp_path / "selected.csv"
    published = config.input_dir / "train_images/train/series"
    published.mkdir(parents=True)
    stale = published / f"stale{suffix}"
    stale.write_bytes(b"old pixels")
    config.selected_series.write_text(
        "StudyInstanceUID,SeriesInstanceUID\ntrain,series\n",
    )
    (config.input_dir / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\n",
    )
    with pytest.raises(ValueError, match="Obsolete published DICOM"):
        config.make().run()
    assert stale.read_bytes() == b"old pixels"
    assert not (published / "a.dcm").exists()
    assert (
        json_loads(
            (config.working_dir / "acquisition.json").read_text(),
            dict[str, object],
        )["complete"]
        is False
    )


@pytest.mark.parametrize(
    ("link_name", "target_name"),
    [
        ("train_images", "train_images"),
        ("train_images/study", "train_images/study"),
        ("train_images/study/series", "train_images/study/series"),
        ("train_images/study/series/a.dcm", "train_images/study/series/a.dcm"),
        ("train_images/study/series/b.dcm.part", "train_images/study/series/b.dcm"),
    ],
)
@pytest.mark.usefixtures("_isolated_cli_authentication")
def test_recovery_does_not_publish_any_member_until_every_member_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_name: str,
    target_name: str,
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("train_images/study/series/a.dcm", b"existing")
        archive.writestr("train_images/study/series/b.dcm", b"new pixels")
        archive.writestr("train_images/study/series/c.dcm", b"bad pixels")
    raw = buffer.getvalue().replace(b"bad pixels", b"bad other!")
    monkeypatch.setattr(httpx, "stream", _ArchiveStream(raw=raw).stream)
    config = acquisition.acquire()
    config.base_dir = None
    config.input_dir = tmp_path / "raw"
    config.staging_dir = tmp_path / "staging"
    config.working_dir = tmp_path / "receipts"
    config.selected_series = tmp_path / "selected.csv"
    config.workers = 1
    published = config.input_dir / "train_images/study/series"
    published.mkdir(parents=True)
    (published / "a.dcm").write_bytes(b"existing")
    config.staging_dir.mkdir()
    link = config.staging_dir / link_name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(
        config.input_dir / target_name,
        target_is_directory=(config.input_dir / target_name).is_dir(),
    )
    config.selected_series.write_text(
        "StudyInstanceUID,SeriesInstanceUID\nstudy,series\n",
    )
    (config.input_dir / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\n",
    )
    with pytest.raises(ValueError, match="CRC"):
        config.make().run()
    assert (published / "a.dcm").read_bytes() == b"existing"
    assert not (published / "b.dcm").exists()
    assert not (config.staging_dir / ".ensure_complete").exists()
    assert not (config.staging_dir / "train_images/study/series/a.dcm").samefile(
        published / "a.dcm",
    )
    assert (
        config.staging_dir / "train_images/study/series/b.dcm"
    ).read_bytes() == b"new pixels"
    assert (
        json_loads(
            (config.working_dir / "acquisition.json").read_text(),
            dict[str, object],
        )["complete"]
        is False
    )


def _tiny_acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = BytesIO()
    with ZipFile(raw, "w") as writer:
        writer.writestr("train_images/train/series/a.dcm", b"pixel-a")
    monkeypatch.setattr(httpx, "stream", _ArchiveStream(raw=raw.getvalue()).stream)
    config = acquisition.acquire()
    config.base_dir = None
    config.working_dir = tmp_path / "receipts"
    config.input_dir = tmp_path / "raw"
    config.staging_dir = tmp_path / "staging"
    config.selected_series = tmp_path / "selected.csv"
    config.workers = 1
    config.input_dir.mkdir()
    config.selected_series.write_text(
        "StudyInstanceUID,SeriesInstanceUID\ntrain,series\n",
    )
    (config.input_dir / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\n",
    )
    return config


@pytest.mark.usefixtures("_isolated_cli_authentication")
def test_staged_member_is_verified_and_published_without_refetching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_acquisition(tmp_path, monkeypatch)
    staged = config.staging_dir / "train_images/train/series/a.dcm"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"pixel-a")
    config.make().run()
    assert (config.input_dir / "train_images/train/series/a.dcm").read_bytes() == (
        b"pixel-a"
    )


@pytest.mark.parametrize("nested", ["staging", "parent"])
@pytest.mark.usefixtures("_isolated_cli_authentication")
def test_input_and_staging_directories_must_be_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: str,
) -> None:
    config = _tiny_acquisition(tmp_path, monkeypatch)
    config.staging_dir = (
        config.input_dir / "staging" if nested == "staging" else tmp_path
    )
    with pytest.raises(ValueError, match="must be separate"):
        config.make()


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArchiveStream:
    raw: bytes
    lock: Path | None = None

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
        follow_redirects: bool,
    ) -> Generator[httpx.Response]:
        del method, timeout, follow_redirects
        logging.getLogger("httpx").info("HTTP request to %s", url)
        first, last = headers["Range"].removeprefix("bytes=").split("-")
        start, end = (
            (int(first), int(last))
            if first
            else (max(0, len(self.raw) - int(last)), len(self.raw) - 1)
        )
        if (
            first
            and self.lock is not None
            and self.raw[start : start + 4] == b"PK\x03\x04"
        ):
            with self.lock.open("a") as lock, pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield httpx.Response(
            206,
            content=self.raw[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(self.raw)}"},
        )


@pytest.mark.parametrize("workers", [0, 9])
def test_acquisition_bounds_concurrency(workers: int) -> None:
    config = acquisition.acquire()
    config.base_dir = None
    config.workers = workers
    with pytest.raises(ValueError, match="workers"):
        config.make()


@pytest.mark.parametrize(
    "name",
    ["acquisition.json", "selected-files.csv", "files.csv", "series.csv"],
)
@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_manifests_cannot_overwrite_selection_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    alias: str,
) -> None:
    raw = BytesIO()
    with ZipFile(raw, "w") as writer:
        writer.writestr("train_images/train/series/a.dcm", b"pixels")
    monkeypatch.setattr(httpx, "stream", _ArchiveStream(raw=raw.getvalue()).stream)
    config = acquisition.acquire()
    config.base_dir = None
    config.working_dir = tmp_path / "receipts"
    config.input_dir = tmp_path / "raw"
    config.staging_dir = tmp_path / "staging"
    config.working_dir.mkdir()
    config.input_dir.mkdir()
    destination = config.working_dir / name
    config.selected_series = (
        destination if alias == "same" else tmp_path / "selected.csv"
    )
    original = b"StudyInstanceUID,SeriesInstanceUID\ntrain,series\n"
    config.selected_series.write_bytes(original)
    if alias == "symlink":
        destination.symlink_to(config.selected_series)
    elif alias == "hardlink":
        destination.hardlink_to(config.selected_series)
    (config.input_dir / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID\n",
    )
    with pytest.raises(ValueError, match="protected input"):
        config.make().run()
    assert config.selected_series.read_bytes() == original
    assert not (config.input_dir / "train_images").exists()


@pytest.mark.parametrize(
    "names",
    [[], ["train_images/a/b/c.dcm", "train_images/a/b/c.dcm"]],
)
def test_acquisition_refuses_missing_or_duplicate_members(
    tmp_path: Path,
    names: list[str],
) -> None:
    table = tmp_path / "series.csv"
    table.write_text("StudyInstanceUID,SeriesInstanceUID\na,b\n")
    with pytest.raises(ValueError, match=r"selected|Duplicate") as caught:
        acquisition.select_members(
            [ZipInfo(name) for name in names],
            training_csv=table,
            test_csv=table,
        )
    assert str(caught.value) == (
        "Duplicate or unsafe selected ZIP filename."
        if names
        else "The complete archive is missing selected DICOM series."
    )


@pytest.mark.parametrize("training_root", ["train_images", "train_series"])
@pytest.mark.parametrize("test_root", ["test_images", "test_series"])
def test_selection_keeps_both_splits_and_skips_interleaved_unselected_files(
    tmp_path: Path,
    training_root: str,
    test_root: str,
) -> None:
    table = tmp_path / "series.csv"
    table.write_text("StudyInstanceUID,SeriesInstanceUID\nstudy,series\n")
    selected = [
        ZipInfo(f"{training_root}/study/series/z.dcm"),
        ZipInfo(f"{test_root}/study/series/a.DCM"),
    ]
    members = [
        ZipInfo("metadata.csv"),
        ZipInfo(f"{training_root}/study/series/notes.txt"),
        ZipInfo("unknown/study/series/ignored.dcm"),
        selected[0],
        ZipInfo(f"{training_root}/other/series/ignored.dcm"),
        selected[1],
    ]
    result = acquisition.select_members(members, training_csv=table, test_csv=table)
    assert result == sorted(selected, key=lambda member: member.filename)


@pytest.mark.parametrize("unsafe", [True, False])
def test_selection_rejects_traversal_and_incomplete_split_inventories(
    tmp_path: Path,
    unsafe: bool,
) -> None:
    table = tmp_path / "series.csv"
    study = ".." if unsafe else "study"
    table.write_text(f"StudyInstanceUID,SeriesInstanceUID\n{study},series\n")
    with pytest.raises(ValueError, match=r"selected|Duplicate") as caught:
        acquisition.select_members(
            [ZipInfo(f"train_images/{study}/series/a.dcm")],
            training_csv=table,
            test_csv=table,
        )
    assert str(caught.value) == (
        "Duplicate or unsafe selected ZIP filename."
        if unsafe
        else "The complete archive is missing selected DICOM series."
    )


@pytest.mark.parametrize(
    "filename",
    ["train_images/study/series//a.dcm", "train_images/study/./series/a.dcm"],
)
def test_selection_rejects_noncanonical_archive_names(
    tmp_path: Path,
    filename: str,
) -> None:
    training = tmp_path / "training.csv"
    testing = tmp_path / "testing.csv"
    training.write_text("StudyInstanceUID,SeriesInstanceUID\nstudy,series\n")
    testing.write_text("StudyInstanceUID,SeriesInstanceUID\n")
    with pytest.raises(ValueError, match="unsafe selected ZIP filename"):
        acquisition.select_members(
            [ZipInfo(filename)],
            training_csv=training,
            test_csv=testing,
        )


def _cli_client(
    *,
    follow_redirects: bool,
    timeout: int,
    original: type[httpx.Client],
    calls: list[httpx.Request],
) -> httpx.Client:
    assert follow_redirects is False
    assert timeout == 60
    return original(
        follow_redirects=follow_redirects,
        timeout=timeout,
        transport=httpx.MockTransport(partial(_archive_headers, calls=calls)),
    )


@contextmanager
def _cli_stream(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: object,
    original: Callable[..., AbstractContextManager[httpx.Response]],
) -> Generator[httpx.Response]:
    assert method == "POST"
    assert list(headers) == ["Authorization"]
    with original(
        client,
        method=method,
        url=url,
        headers=headers,
        json=json,
    ) as response:
        yield response


def _archive_headers(
    request: httpx.Request,
    *,
    calls: list[httpx.Request],
) -> httpx.Response:
    if request.url.host != "api.kaggle.com":
        raise AssertionError("The signed archive is contacted only by bounded ranges.")
    calls.append(request)
    assert request.method == "POST"
    assert str(request.url) == (
        "https://api.kaggle.com/v1/competitions.CompetitionApiService/DownloadDataFiles"
    )
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert json_loads(request.content.decode(), dict[str, str]) == {
        "competitionName": "rsna-knee-abnormality-detection",
    }
    return httpx.Response(
        302,
        headers={"Location": "https://archive.test/secret-url"},
        stream=_UnreadablePayload(),
    )


def _failure_run(
    command: list[str],
    *,
    mode: str,
    **options: object,
) -> subprocess.CompletedProcess[str]:
    del options
    if mode == "missing":
        raise subprocess.CalledProcessError(127, cmd=command, output="secret-token")
    if mode == "os":
        raise OSError("secret-token")
    if mode == "timeout":
        raise subprocess.TimeoutExpired(command, timeout=60, output="secret-token")
    return subprocess.CompletedProcess(
        command,
        returncode=0,
        stdout=""
        if mode == "empty"
        else "secret-token\nsecret-extra"
        if mode == "multiline"
        else "secret-token",
    )


def _denied(request: httpx.Request, *, mode: str) -> httpx.Response:
    if mode == "transport":
        raise httpx.ConnectError("secret-token", request=request)
    if mode == "redirect":
        return httpx.Response(200)
    if mode == "insecure":
        return httpx.Response(
            302,
            headers={"Location": "http://example.test/secret"},
        )
    if mode == "loop":
        return httpx.Response(302, headers={"Location": str(request.url)})
    if mode == "no_location":
        return httpx.Response(302)
    return httpx.Response(403, text="secret-token")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
