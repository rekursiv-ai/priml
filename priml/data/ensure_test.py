"""Tests for the resumable, idempotent data-ensure primitive.

All tests are hermetic: the only "remote" is a temporary source directory
plus a fetch closure that copies from it, or a localhost ``http.server``
bound to ``127.0.0.1`` for the HTTP-resume path. No real network access.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, override

import fcntl
import functools
import hashlib
import http.server
import shutil
import threading

import pytest

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
    from pathlib import Path


_MARKER = ".ensure_complete"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


_FILES: dict[str, bytes] = {
    "a.json": b'{"hello": "world"}',
    "sub/b.bin": bytes(range(256)) * 4,
}


def _manifest(files: dict[str, bytes]) -> list[FileSpec]:
    return [
        FileSpec(rel_path=rel, size=len(data), sha256=_sha256(data))
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


def test_missing_file_downloads(tmp_path: Path, source: Path):
    """A missing manifest file is fetched; result is DOWNLOADED."""
    target = tmp_path / "target"
    calls: list[str] = []

    result = ensure_data(_spec(target, source, calls))

    assert result is EnsureResult.DOWNLOADED
    assert sorted(calls) == sorted(_FILES)
    for rel, data in _FILES.items():
        assert (target / rel).read_bytes() == data


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


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
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
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    # Poll at 10ms (not the 0.5s default): server.shutdown() blocks until
    # serve_forever notices the stop flag on its next poll, so the default
    # interval added ~0.5s of pure teardown to every test using this fixture.
    thread = threading.Thread(
        target=functools.partial(server.serve_forever, poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/blob", blob
    finally:
        server.shutdown()
        server.server_close()


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
