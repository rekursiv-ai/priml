"""Tests for the resumable, idempotent data-ensure primitive.

All tests are hermetic: the only "remote" is a temporary source directory
plus a fetch closure that copies from it, or a localhost ``http.server``
bound to ``127.0.0.1`` for the HTTP-resume path. No real network access.
"""

from __future__ import annotations

from http import client, server
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final, override
from urllib import request

import datetime
import fcntl
import functools
import hashlib
import io
import shutil
import sys
import threading

import pytest

from priml.data import ensure
from priml.data.ensure import (
    DataSpec,
    EnsureResult,
    Fetch,
    FileSpec,
    ensure_data,
    resumable_http_download,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


_MARKER: Final = ".ensure_complete"


class _Response(client.HTTPResponse):
    """Minimal typed response carrying only headers for content-length tests."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = client.HTTPMessage()
        for name, value in headers.items():
            self.headers[name] = value


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _source_tree(root: Path, files: dict[str, bytes]) -> None:
    """Populate a fake upstream source directory."""
    for rel, data in files.items():
        _write(root / rel, data)


# The closure records every requested path in ``calls`` so a test can assert whether a
# fetch was triggered at all.
def _copy_fetch(source: Path, calls: list[str]) -> Fetch:
    """Build a fetch closure that copies ``rel_path`` out of ``source``."""

    def fetch(*, rel_path: str, dest: Path) -> None:
        calls.append(rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel_path, dest)

    return fetch


_FILES: Final[dict[str, bytes]] = {
    "a.json": b'{"hello": "world"}',
    "sub/b.bin": bytes(range(256)) * 4,
}


def _manifest(files: dict[str, bytes]) -> list[FileSpec]:
    return [
        FileSpec(rel_path=rel, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        for rel, data in files.items()
    ]


def _spec(target: Path, source: Path, calls: list[str]) -> DataSpec:
    return DataSpec(
        target_dir=target,
        manifest=_manifest(_FILES),
        fetch=_copy_fetch(source, calls),
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _source_tree(root, _FILES)
    return root


def test_present_skips_fetch(tmp_path: Path, source: Path):
    """All files present and matching -> PRESENT, fetch never called."""
    target = tmp_path / "target"
    _source_tree(target, _FILES)
    calls: list[str] = []

    result = ensure_data(_spec(target, source, calls))

    assert result is EnsureResult.PRESENT
    assert calls == []


def test_missing_file_downloads(
    tmp_path: Path,
    source: Path,
    caplog: pytest.LogCaptureFixture,
):
    """A missing manifest file is fetched; result is DOWNLOADED."""
    target = tmp_path / "target"
    calls: list[str] = []
    caplog.set_level("INFO", logger=ensure.__name__)

    result = ensure_data(_spec(target, source, calls))

    assert result is EnsureResult.DOWNLOADED
    assert sorted(calls) == sorted(_FILES)
    for rel, data in _FILES.items():
        assert (target / rel).read_bytes() == data
    assert [record.getMessage() for record in caplog.records] == [
        f"ensure_data: fetching 2/2 files into {target}",
        "ensure_data: fetch a.json",
        "ensure_data: fetch sub/b.bin",
        f"ensure_data: {target} complete",
    ]
    assert caplog.records[0].args == (2, 2, target)
    assert caplog.records[1].args == ("a.json",)
    assert caplog.records[2].args == ("sub/b.bin",)
    assert caplog.records[3].args == (target,)


def test_partial_wrong_size_recovers(tmp_path: Path, source: Path):
    """A truncated file (wrong size) is repaired -> complete."""
    target = tmp_path / "target"
    _source_tree(target, _FILES)
    _write(target / "a.json", b'{"hello": "wor')  # Truncated.
    calls: list[str] = []

    result = ensure_data(_spec(target, source, calls))

    assert result is EnsureResult.DOWNLOADED
    assert calls == ["a.json"]
    assert (target / "a.json").read_bytes() == _FILES["a.json"]


def test_corrupt_file_is_archived(tmp_path: Path, source: Path):
    """A right-size wrong-sha file is archived to .corrupt.<ts>, refetched."""
    target = tmp_path / "target"
    _source_tree(target, _FILES)
    corrupt = bytes([0]) + _FILES["a.json"][1:]  # Same length, different bytes.
    assert len(corrupt) == len(_FILES["a.json"])
    _write(target / "a.json", corrupt)
    calls: list[str] = []

    result = ensure_data(_spec(target, source, calls))

    assert result is EnsureResult.DOWNLOADED
    assert "a.json" in calls
    archived = list(target.parent.glob("target.corrupt.*"))
    assert len(archived) == 1
    assert (archived[0] / "a.json").read_bytes() == corrupt
    assert (target / "a.json").read_bytes() == _FILES["a.json"]


def test_fetch_failure_then_rerun_recovers(tmp_path: Path, source: Path):
    """A fetch that raises midway is recovered on the next ensure_data call."""
    target = tmp_path / "target"
    calls: list[str] = []
    boom = {"armed": True}

    def flaky_fetch(*, rel_path: str, dest: Path) -> None:
        calls.append(rel_path)
        if boom["armed"]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"partially-written")  # Leaves debris.
            boom["armed"] = False
            raise OSError("simulated mid-download failure")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel_path, dest)

    spec = DataSpec(
        target_dir=target,
        manifest=_manifest(_FILES),
        fetch=flaky_fetch,
    )

    with pytest.raises(OSError, match="simulated"):
        ensure_data(spec)

    result = ensure_data(spec)

    assert result is EnsureResult.DOWNLOADED
    for rel, data in _FILES.items():
        assert (target / rel).read_bytes() == data


def test_idempotent_second_call(tmp_path: Path, source: Path):
    """Two calls in a row: the second is a no-op (PRESENT, no fetch)."""
    target = tmp_path / "target"

    first = ensure_data(_spec(target, source, []))
    second_calls: list[str] = []
    second = ensure_data(_spec(target, source, second_calls))

    assert first is EnsureResult.DOWNLOADED
    assert second is EnsureResult.PRESENT
    assert second_calls == []


def test_size_only_manifest(tmp_path: Path, source: Path):
    """A manifest entry with size but no sha matches on size alone."""
    target = tmp_path / "target"
    _source_tree(target, _FILES)
    calls: list[str] = []
    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path="a.json", size=len(_FILES["a.json"]))],
        fetch=_copy_fetch(source, calls),
    )

    assert ensure_data(spec) is EnsureResult.PRESENT
    assert calls == []


def test_existence_only_manifest(tmp_path: Path, source: Path):
    """A manifest entry with neither size nor sha matches on existence."""
    target = tmp_path / "target"
    _write(target / "a.json", b"any content at all")
    calls: list[str] = []
    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path="a.json")],
        fetch=_copy_fetch(source, calls),
    )

    assert ensure_data(spec) is EnsureResult.PRESENT
    assert calls == []


def test_completion_marker_written_after_build(tmp_path: Path, source: Path):
    """A successful build stamps the completion marker under the target."""
    target = tmp_path / "target"

    assert ensure_data(_spec(target, source, [])) is EnsureResult.DOWNLOADED
    assert (target / _MARKER).is_file()


def test_partial_tree_without_marker_not_present(tmp_path: Path, source: Path):
    """An existence-only manifest over a partial tree must not read PRESENT.

    The files exist (so the manifest "verifies") but no marker was written,
    so the data is rebuilt rather than accepted as complete.
    """
    target = tmp_path / "target"
    # Seed a partial-but-existing tree with no completion marker, mimicking
    # another job's in-progress build under an existence-only manifest.
    _write(target / "a.json", b"partial")
    _write(target / "sub/b.bin", b"partial")
    calls: list[str] = []
    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path=rel) for rel in _FILES],  # existence-only.
        fetch=_copy_fetch(source, calls),
    )

    # No marker -> not accepted as present despite both files existing.
    assert not (target / _MARKER).exists()
    result = ensure_data(spec)

    assert result is EnsureResult.PRESENT  # Adopted under lock + marker written.
    assert (target / _MARKER).is_file()
    # A later call is now a true fast-path no-op.
    assert ensure_data(spec) is EnsureResult.PRESENT


def test_concurrent_ensure_builds_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent ensures serialize on the lock; the builder runs once.

    The second caller blocks on the build lock, then re-checks under the
    lock and finds the dataset already complete -- it does not re-run the
    slow builder nor observe a partial tree as present.
    """
    target = tmp_path / "target"
    builds = {"n": 0}
    in_build = threading.Event()
    release_build = threading.Event()
    second_lock_attempted = threading.Event()
    real_flock = fcntl.flock

    def observe_flock(handle: IO[str], operation: int) -> None:
        if threading.current_thread().name == "second" and operation == fcntl.LOCK_EX:
            second_lock_attempted.set()
        real_flock(handle, operation)

    monkeypatch.setattr(fcntl, "flock", observe_flock)

    def make_fetch() -> Fetch:
        done = {"v": False}  # One fetch writes the whole tree (cf. ARC build)

        def whole_tree_fetch(*, rel_path: str, dest: Path) -> None:
            del rel_path, dest
            if done["v"]:
                return
            builds["n"] += 1
            in_build.set()
            assert release_build.wait(timeout=2)
            for rel, data in _FILES.items():
                _write(target / rel, data)
            done["v"] = True

        return whole_tree_fetch

    def spec() -> DataSpec:
        return DataSpec(
            target_dir=target,
            manifest=[FileSpec(rel_path=rel) for rel in _FILES],  # existence-only.
            fetch=make_fetch(),
        )

    results: dict[str, EnsureResult] = {}

    def run(name: str) -> None:
        results[name] = ensure_data(spec())

    first = threading.Thread(target=run, args=("first",), name="first")
    first.start()
    in_build.wait(timeout=2)  # `first` thread holds the lock and is building.
    second = threading.Thread(target=run, args=("second",), name="second")
    second.start()
    assert second_lock_attempted.wait(timeout=2)
    release_build.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert builds["n"] == 1  # `second` caller did not re-run the builder.
    assert results["first"] is EnsureResult.DOWNLOADED
    assert results["second"] is EnsureResult.PRESENT
    assert (target / _MARKER).is_file()


# --- HTTP resume path (localhost only) -------------------------------------


class _RangeHandler(server.SimpleHTTPRequestHandler):
    """Serves a single in-memory blob with HTTP Range support."""

    blob: bytes = b""

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args  # Silence per-request stderr logging in tests.

    @override
    def do_GET(self) -> None:
        blob = type(self).blob
        rng = self.headers.get("Range")
        if rng is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        start = int(rng.removeprefix("bytes=").split("-", 1)[0])
        chunk = blob[start:]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def http_blob() -> Iterator[tuple[str, bytes]]:
    blob = bytes(range(256)) * 64  # 16 KiB.
    handler = functools.partial(_RangeHandler)
    _RangeHandler.blob = blob
    http_server = server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    # Poll at 10ms (not the 0.5s default): http_server.shutdown() blocks until
    # serve_forever notices the stop flag on its next poll, so the default
    # interval added ~0.5s of pure teardown to every test using this fixture.
    thread = threading.Thread(
        target=functools.partial(http_server.serve_forever, poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    host, port = http_server.server_address[:2]
    try:
        yield f"http://{host}:{port}/blob", blob
    finally:
        http_server.shutdown()
        http_server.server_close()


def test_http_download_full(tmp_path: Path, http_blob: tuple[str, bytes]):
    """resumable_http_download fetches a full blob into dest atomically."""
    url, blob = http_blob
    dest = tmp_path / "out.bin"

    resumable_http_download(url=url, dest=dest)

    assert dest.read_bytes() == blob
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_http_download_resumes_from_part(tmp_path: Path, http_blob: tuple[str, bytes]):
    """A pre-existing .part prefix is resumed via an HTTP Range request."""
    url, blob = http_blob
    dest = tmp_path / "out.bin"
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(blob[:5000])  # Simulate an interrupted prior transfer.

    resumable_http_download(url=url, dest=dest)

    assert dest.read_bytes() == blob
    assert not part.exists()


@pytest.mark.parametrize(
    "case",
    [
        (206, b"x", b"yz", 3, 1),
        (200, b"old", b"abc", 3, 0),
        (200, b"", b"abc", 3, 0),
    ],
)
def test_http_download_protocol_and_progress_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[int, bytes, bytes, int, int],
):
    status, prefix, body, expected_total, expected_initial = case
    dest = tmp_path / "nested" / "deeper" / "output.bin"
    part = dest.with_suffix(".bin.part")
    if prefix:
        part.parent.mkdir(parents=True)
        part.write_bytes(prefix)
    urlopen_requests: list[request.Request] = []
    added_headers: list[tuple[str, str]] = []
    progress_kwargs: list[dict[str, object]] = []
    read_sizes: list[int | None] = []
    updates: list[int] = []

    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(body)
            self.status = status
            self.headers = {"Content-Length": str(len(body))}

        @override
        def read(self, size: int | None = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    response = Response()

    def urlopen(sent: request.Request) -> Response:
        urlopen_requests.append(sent)
        return response

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def update(self, amount: int) -> None:
            updates.append(amount)

    def make_progress(**kwargs: object) -> Progress:
        progress_kwargs.append(kwargs)
        return Progress()

    real_add_header = request.Request.add_header

    def add_header_spy(
        sent: request.Request,
        name: str,
        value: str,
    ) -> None:
        added_headers.append((name, value))
        real_add_header(sent, name, value)

    monkeypatch.setattr(request.Request, "add_header", add_header_spy)
    monkeypatch.setattr(request, "urlopen", urlopen)
    monkeypatch.setattr(ensure, "tqdm", make_progress)

    resumable_http_download(url="https://example.test/data", dest=dest)

    if prefix:
        assert added_headers == [("Range", f"bytes={len(prefix)}-")]
        assert urlopen_requests[0].headers == {"Range": f"bytes={len(prefix)}-"}
    else:
        assert added_headers == []
        assert urlopen_requests[0].headers == {}
    assert dest.read_bytes() == (prefix + body if status == 206 else body)
    assert not part.exists()
    assert progress_kwargs == [
        {
            "total": expected_total,
            "initial": expected_initial,
            "desc": dest.name,
            "unit": "B",
            "unit_scale": True,
            "disable": not sys.stdout.isatty(),
        },
    ]
    assert read_sizes == [ensure._SETTINGS.chunk_bytes] * 2
    assert updates == [len(body)]


def test_internal_manifest_and_content_length_contracts(tmp_path: Path):
    """Pin manifest corruption filtering and progress-size accounting."""
    target = tmp_path / "target"
    files = [
        FileSpec(rel_path="no-digest", size=4),
        FileSpec(
            rel_path="partial",
            size=4,
            sha256=hashlib.sha256(b"good").hexdigest(),
        ),
        FileSpec(
            rel_path="corrupt",
            size=4,
            sha256=hashlib.sha256(b"good").hexdigest(),
        ),
    ]
    _write(target / "partial", b"bad")
    _write(target / "corrupt", b"evil")
    _write(target / "no-digest", b"evil")
    spec = DataSpec(
        target_dir=target,
        manifest=files,
        fetch=_copy_fetch(tmp_path, []),
    )

    assert ensure._corrupt_files(spec) == [files[2]]
    assert (
        ensure._content_length(
            _Response({"Content-Length": "17"}),
            already=5,
            resumed=True,
        )
        == 22
    )
    assert (
        ensure._content_length(
            _Response({"Content-Length": "17"}),
            already=5,
            resumed=False,
        )
        == 17
    )
    assert ensure._content_length(_Response({}), already=5, resumed=True) is None
    assert ensure._is_complete(spec) is False


def test_sha256_reads_configured_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "payload"
    contents = b"x" * (ensure._SETTINGS.chunk_bytes + 3)
    path.write_bytes(contents)
    read_sizes: list[int | None] = []

    class ReadSpy(io.BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    def open_spy(
        path: Path,
        mode: str = "r",
        *_args: object,
        **_kwargs: object,
    ) -> ReadSpy:
        del path, mode
        return ReadSpy(contents)

    monkeypatch.setattr(Path, "open", open_spy)

    assert ensure._sha256(path) == hashlib.sha256(contents).hexdigest()
    assert read_sizes == [ensure._SETTINGS.chunk_bytes] * 3


def test_archive_name_and_corrupt_archive_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "dataset"
    _write(target / "payload", b"bad")
    real_datetime = datetime.datetime
    zones: list[datetime.tzinfo | None] = []

    class DateTimeSpy:
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> datetime.datetime:
            zones.append(tz)
            return real_datetime.now(tz)

    monkeypatch.setattr(
        "priml.data.ensure.datetime.datetime",
        DateTimeSpy,
    )
    ensure._archive_dir(target)

    archives = list(tmp_path.glob("dataset.corrupt.*"))
    assert zones == [datetime.UTC]
    assert len(archives) == 1
    assert archives[0].name.startswith("dataset.corrupt.20")
    assert archives[0].name.endswith("Z")
    assert (archives[0] / "payload").read_bytes() == b"bad"
    assert caplog.records[-1].getMessage() == (
        f"ensure_data: archiving corrupt {target} -> {archives[0]}"
    )


def test_marker_has_utc_timestamp_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    marker_dir = tmp_path / "nested" / "dataset"
    real_datetime = datetime.datetime
    zones: list[datetime.tzinfo | None] = []

    class DateTimeSpy:
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> datetime.datetime:
            zones.append(tz)
            return real_datetime.now(tz)

    monkeypatch.setattr(
        "priml.data.ensure.datetime.datetime",
        DateTimeSpy,
    )
    ensure._write_marker(marker_dir)

    marker = marker_dir / _MARKER
    assert zones == [datetime.UTC]
    assert marker.read_text().count("\n") == 1
    assert len(marker.read_text().strip()) == 22
    assert marker.read_text().endswith("Z\n")


def test_ensure_logs_fast_path_and_adoption(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    target = tmp_path / "target"
    _write(target / "a.json", _FILES["a.json"])
    calls: list[str] = []
    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path="a.json", size=len(_FILES["a.json"]))],
        fetch=_copy_fetch(tmp_path, calls),
    )

    caplog.set_level("INFO", logger=ensure.__name__)
    assert ensure_data(spec) is EnsureResult.PRESENT
    assert caplog.records[
        -1
    ].getMessage() == "ensure_data: adopting present tree (1 files)".replace(
        "adopting present tree",
        f"{target} adopting present tree",
    )
    assert caplog.records[-1].args == (target, 1)
    caplog.clear()
    assert ensure_data(spec) is EnsureResult.PRESENT
    assert caplog.records[-1].getMessage() == f"ensure_data: {target} present (1 files)"
    assert calls == []


def test_locked_complete_tree_logs_manifest_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    target = tmp_path / "target"
    _source_tree(target, {"payload": b"abc"})
    _write(target / _MARKER, b"complete\n")
    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path="payload", size=3)],
        fetch=_copy_fetch(tmp_path, []),
    )
    caplog.set_level("INFO", logger=ensure.__name__)

    assert ensure._ensure_locked(spec) is EnsureResult.PRESENT

    assert caplog.records[-1].getMessage() == f"ensure_data: {target} present (1 files)"
    assert caplog.records[-1].args == (target, 1)


def test_failed_fetch_reports_exact_unsatisfied_paths(tmp_path: Path):
    target = tmp_path / "target"

    def wrong_fetch(*, rel_path: str, dest: Path) -> None:
        del rel_path
        _write(dest, b"no")

    spec = DataSpec(
        target_dir=target,
        manifest=[FileSpec(rel_path="nested/missing.bin", size=3)],
        fetch=wrong_fetch,
    )

    with pytest.raises(
        RuntimeError,
        match=r"ensure_data: manifest still unsatisfied after fetch: \['nested/missing.bin'\]",
    ):
        ensure_data(spec)
    assert not (target / _MARKER).exists()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
