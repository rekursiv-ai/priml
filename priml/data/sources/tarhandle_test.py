"""Contract and backend-specific tests for :mod:`tarhandle`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import io
import os
import tarfile
import threading

import pytest

from priml.data.sources.tarhandle import TarFileHandle


if TYPE_CHECKING:
    from pathlib import Path


def _write_archive(path: Path, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name=name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize("use_mmap", [True, False], ids=["mmap", "standard"])
def test_shared_reader_contract(tmp_path: Path, use_mmap: bool) -> None:
    path = tmp_path / "samples.tar"
    payloads = {
        "sample_0.bin": b"zero",
        "sample_1.bin": b"one",
        "sample_2.bin": b"two",
    }
    _write_archive(path, payloads)

    with TarFileHandle(path, use_mmap=use_mmap) as handle:
        assert handle.path == path
        assert handle.name == str(path)
        assert handle.use_mmap is use_mmap
        assert f"mode={'mmap' if use_mmap else 'standard'}" in repr(handle)
        for name, payload in payloads.items():
            member = handle.getmember(name)
            assert member.name == name
            extracted = handle.extractfile(member)
            assert extracted is not None
            assert extracted.read() == payload
            by_name = handle.extractfile(name)
            assert by_name is not None
            assert by_name.read() == payload

    with pytest.raises(ValueError, match="closed"):
        _ = handle.getmember("sample_0.bin")


@pytest.mark.parametrize("use_mmap", [True, False], ids=["mmap", "standard"])
def test_missing_members_raise_key_error(tmp_path: Path, use_mmap: bool) -> None:
    path = tmp_path / "samples.tar"
    _write_archive(path, {"present.bin": b"present"})

    with TarFileHandle(path, use_mmap=use_mmap) as handle:
        with pytest.raises(KeyError):
            _ = handle.getmember("missing.bin")
        with pytest.raises(KeyError):
            _ = handle.extractfile("missing.bin")


@pytest.mark.parametrize(
    ("archive_format", "prefix"),
    [(tarfile.PAX_FORMAT, "pax"), (tarfile.GNU_FORMAT, "gnu")],
    ids=["pax", "gnu"],
)
def test_mmap_indexes_extended_names(
    tmp_path: Path,
    archive_format: int,
    prefix: str,
) -> None:
    path = tmp_path / f"{prefix}.tar"
    name = f"{prefix}/" + prefix * 60 + ".bin"
    payload = f"{prefix}-payload".encode()
    with tarfile.open(path, "w", format=archive_format) as archive:
        member = tarfile.TarInfo(name=name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with TarFileHandle(path) as handle:
        assert handle.getmember(name).name == name
        extracted = handle.extractfile(name)
        assert extracted is not None
        assert extracted.read() == payload


def test_mmap_non_regular_members_have_no_payload(tmp_path: Path) -> None:
    path = tmp_path / "mixed.tar"
    payload = b"payload"
    with tarfile.open(path, "w") as archive:
        regular = tarfile.TarInfo(name="regular.bin")
        regular.size = len(payload)
        archive.addfile(regular, io.BytesIO(payload))
        link = tarfile.TarInfo(name="link.bin")
        link.type = tarfile.SYMTYPE
        link.linkname = "regular.bin"
        archive.addfile(link)
        directory = tarfile.TarInfo(name="subdir")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)

    with TarFileHandle(path) as handle:
        assert handle.getmember("link.bin").name == "link.bin"
        assert handle.getmember("subdir").name == "subdir"
        assert handle.extractfile("link.bin") is None
        assert handle.extractfile("subdir") is None


def test_mmap_extraction_survives_handle_close(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_archive(path, {"sample.bin": b"payload"})
    handle = TarFileHandle(path)
    extracted = handle.extractfile("sample.bin")
    assert extracted is not None

    handle.close()

    assert extracted.read() == b"payload"


def test_mmap_rejects_compressed_archives(tmp_path: Path) -> None:
    path = tmp_path / "compressed.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        payload = b"compressed"
        member = tarfile.TarInfo(name="sample.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="compressed"):
        _ = TarFileHandle(path, use_mmap=True)


def test_standard_mode_reads_compressed_archives(tmp_path: Path) -> None:
    path = tmp_path / "compressed.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        payload = b"compressed"
        member = tarfile.TarInfo(name="sample.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with TarFileHandle(path, use_mmap=False) as handle:
        extracted = handle.extractfile("sample.bin")
        assert extracted is not None
        assert extracted.read() == payload


def test_standard_close_releases_every_thread_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "threads.tar"
    _write_archive(path, {"sample.bin": b"payload"})
    real_close = tarfile.TarFile.close
    closed: set[int] = set()

    def tracked_close(archive: tarfile.TarFile) -> None:
        closed.add(id(archive))
        real_close(archive)

    monkeypatch.setattr(tarfile.TarFile, "close", tracked_close)
    handle = TarFileHandle(path, use_mmap=False)
    barrier = threading.Barrier(4)

    def read_once() -> None:
        barrier.wait()
        extracted = handle.extractfile("sample.bin")
        assert extracted is not None
        assert extracted.read() == b"payload"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(read_once) for _ in range(4)]
        for future in futures:
            future.result()

    handle.close()

    assert len(closed) == 4


def test_standard_mode_reopens_after_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "samples.tar"
    _write_archive(path, {"sample.bin": b"payload"})
    real_close = tarfile.TarFile.close
    closed: set[int] = set()

    def tracked_close(archive: tarfile.TarFile) -> None:
        closed.add(id(archive))
        real_close(archive)

    monkeypatch.setattr(tarfile.TarFile, "close", tracked_close)
    handle = TarFileHandle(path, use_mmap=False)
    monkeypatch.setattr(os, "getpid", lambda: 1_000)
    first = handle._get_tarfile()

    monkeypatch.setattr(os, "getpid", lambda: 2_000)
    second = handle._get_tarfile()

    assert first is not second
    assert id(first) in closed
    handle.close()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
