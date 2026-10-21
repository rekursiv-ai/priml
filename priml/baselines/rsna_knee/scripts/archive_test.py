"""Exercise range extraction against actual standard-library ZIP archives."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import codecs
import hashlib
import struct
import zlib

import httpx
import pytest

from priml.baselines.rsna_knee.scripts.archive import (
    directory_location,
    fetch_member,
    index_archive,
    read_range,
    verify_member,
)


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.mark.parametrize("compression", [ZIP_DEFLATED, ZIP_STORED])
def test_complete_directory_and_members_resume_with_crc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: int,
) -> None:
    archive = BytesIO()
    with ZipFile(archive, "w", compression=compression) as writer:
        writer.writestr("train_images/study/series/a.dcm", b"pixels" * 200)
        writer.writestr("test_images/study/series/b.dcm", b"other" * 100)
    raw = archive.getvalue()
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=raw))
    members, size = index_archive("https://example.test/archive", working_dir=tmp_path)
    assert size == len(raw)
    assert len(members) == 2
    (tmp_path / "pixels" / members[0].filename).parent.mkdir(parents=True)
    result = fetch_member(
        members[0],
        url="https://example.test/archive",
        archive_size=size,
        staging_dir=tmp_path / "pixels",
    )
    assert result.size == 1200
    assert result.sha256 == hashlib.sha256(b"pixels" * 200).hexdigest()
    assert (tmp_path / "pixels" / result.name).read_bytes() == b"pixels" * 200
    assert (
        fetch_member(
            members[0],
            url="https://example.test/archive",
            archive_size=size,
            staging_dir=tmp_path / "pixels",
        )
        == result
    )
    again, _ = index_archive("https://example.test/archive", working_dir=tmp_path)
    assert [item.filename for item in again] == [item.filename for item in members]
    index = next(tmp_path.glob("index-*.zip"))
    metadata = index.read_bytes()
    partial_path = index.with_suffix(".part")
    index.rename(partial_path)
    partial_path.write_bytes(metadata[:11])
    resumed, _ = index_archive("https://example.test/archive", working_dir=tmp_path)
    assert [item.filename for item in resumed] == [item.filename for item in members]
    index.rename(partial_path)
    partial_path.write_bytes(metadata + b"invalid-partial")
    with pytest.raises(ValueError, match="Partial ZIP"):
        index_archive("https://example.test/archive", working_dir=tmp_path)


def test_directory_rejects_missing_end_record() -> None:
    with pytest.raises(ValueError, match=r"^Invalid ZIP end record\.$"):
        directory_location(b"not an archive")


@pytest.mark.parametrize("encoding", ["utf-8", "cp437"])
@pytest.mark.parametrize("extra_size", [258, 65_535])
def test_members_preserve_names_and_maximal_local_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    extra_size: int,
) -> None:
    payload = hashlib.shake_256(b"ZIP payload").digest(70_000)
    name = "study/série/汉.dcm" if encoding == "utf-8" else "café.dcm"
    member = ZipInfo(name if encoding == "utf-8" else "cafe.dcm")
    member.compress_type = ZIP_DEFLATED
    member.extra = struct.pack("<HH", 0xCAFE, extra_size - 4) + bytes(extra_size - 4)
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
        archive.writestr("padding", bytes(100_000))
    raw = buffer.getvalue()
    if encoding == "cp437":
        raw = raw.replace(b"cafe.dcm", name.encode(encoding))
    with ZipFile(BytesIO(raw)) as archive:
        member = archive.infolist()[0]
        assert member.filename == name
        assert len(member.extra) == extra_size
        assert archive.read(member) == payload
    requests: list[str] = []
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=raw, requests=requests))
    result = fetch_member(
        member,
        url="https://example.test/archive",
        archive_size=len(raw),
        staging_dir=tmp_path,
    )
    assert (tmp_path / name).read_bytes() == payload
    assert result.name == name
    assert result.size == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    # The HTTP allowance includes the format's largest possible local extra field.
    maximum = 30 + len(name.encode(encoding)) + 65_535 + member.compress_size
    assert requests == [f"bytes=0-{maximum - 1}"]


@pytest.mark.parametrize("encoding", ["utf-8", "cp437"])
def test_codec_case_aliases_preserve_decoders(encoding: str) -> None:
    assert codecs.lookup(encoding) is codecs.lookup(encoding.upper())
    raw = bytes(range(256)) if encoding == "cp437" else "série/汉.dcm".encode()
    assert raw.decode(encoding) == raw.decode(encoding.upper())


def test_member_publication_failure_preserves_complete_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = BytesIO()
    payload = b"pixels" * 20
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("series/a.dcm", payload)
    raw = buffer.getvalue()
    with ZipFile(BytesIO(raw)) as archive:
        member = archive.infolist()[0]
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=raw))
    with monkeypatch.context() as interrupted:
        interrupted.setattr(Path, "replace", _failed_replace)
        with pytest.raises(OSError, match="Interrupted publication"):
            fetch_member(
                member,
                url="https://example.test/archive",
                archive_size=len(raw),
                staging_dir=tmp_path,
            )
    assert not (tmp_path / "series/a.dcm").exists()
    assert (tmp_path / "series/a.dcm.part").read_bytes() == payload
    result = fetch_member(
        member,
        url="https://example.test/archive",
        archive_size=len(raw),
        staging_dir=tmp_path,
    )
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "series/a.dcm").read_bytes() == payload
    assert not (tmp_path / "series/a.dcm.part").exists()


@pytest.mark.parametrize("comment", [b"PK\x05\x06", b"PK\x05\x06" + bytes(18)])
def test_index_accepts_signature_bearing_zip_comments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comment: bytes,
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("a.dcm", b"pixels")
        archive.comment = comment
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=buffer.getvalue()))
    members, _ = index_archive("https://example.test/archive", working_dir=tmp_path)
    assert [member.filename for member in members] == ["a.dcm"]


def test_directory_publication_failure_keeps_resumable_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("a.dcm", b"pixels")
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=buffer.getvalue()))
    with monkeypatch.context() as interrupted:
        interrupted.setattr(Path, "replace", _failed_replace)
        with pytest.raises(OSError, match="Interrupted publication"):
            index_archive("https://example.test/archive", working_dir=tmp_path)
    members, _ = index_archive("https://example.test/archive", working_dir=tmp_path)
    assert [member.filename for member in members] == ["a.dcm"]


def test_suffix_must_index_the_actual_complete_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("old.dcm", b"old pixels")
    earlier = buffer.getvalue()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("current.dcm", b"current pixels")
    raw = buffer.getvalue()
    with ZipFile(BytesIO(raw)) as archive:
        assert archive.namelist() == ["current.dcm"]
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=raw))
    members, _ = index_archive(
        "https://example.test/archive",
        working_dir=tmp_path / "good",
    )
    assert [member.filename for member in members] == ["current.dcm"]
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=raw, suffix=earlier))
    with pytest.raises(RuntimeError, match="range"):
        index_archive("https://example.test/archive", working_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@contextmanager
def _stream(
    method: str,
    url: str,
    *,
    raw: bytes,
    headers: dict[str, str],
    timeout: int,
    follow_redirects: bool,
    suffix: bytes | None = None,
    requests: list[str] | None = None,
) -> Generator[httpx.Response]:
    del timeout, follow_redirects
    assert method == "GET"
    assert url == "https://example.test/archive"
    if requests is not None:
        requests.append(headers["Range"])
    first, last = headers["Range"].removeprefix("bytes=").split("-")
    start, end = (
        (int(first), int(last))
        if first
        else (max(0, len(raw) - int(last)), len(raw) - 1)
    )
    payload = raw[start : end + 1]
    if suffix is not None and not first:
        start, end, payload = 0, len(suffix) - 1, suffix
    yield httpx.Response(
        206,
        content=payload,
        headers={"Content-Range": f"bytes {start}-{end}/{len(raw)}"},
    )


@pytest.mark.parametrize("sentinel_field", ["none", "offset", "size", "count"])
@pytest.mark.parametrize("prefix", [b"", b"PK\x05\x06prefix"])
def test_directory_end_records_preserve_all_integer_fields(
    sentinel_field: str,
    prefix: bytes,
) -> None:
    offset, size, count = (1 << 63) + 7, (1 << 63) + 5, (1 << 63) + 3
    if sentinel_field != "none":
        record64 = struct.pack(
            "<4sQ2H2I4Q",
            b"PK\x06\x06",
            44,
            45,
            45,
            0,
            0,
            count,
            count,
            size,
            offset,
        )
        sentinel = struct.pack(
            "<4s4H2LH",
            b"PK\x05\x06",
            0,
            0,
            1,
            65_535 if sentinel_field == "count" else 1,
            0xFFFFFFFF if sentinel_field == "size" else 1,
            0xFFFFFFFF if sentinel_field == "offset" else 1,
            3,
        )
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(prefix), 1)
        tail = prefix + record64 + locator + sentinel + b"end"
    else:
        offset, size, count = 1_234_567, 234_567, 1234
        tail = (
            prefix
            + struct.pack(
                "<4s4H2LH",
                b"PK\x05\x06",
                0,
                0,
                count,
                count,
                size,
                offset,
                3,
            )
            + b"end"
        )
    assert directory_location(tail) == (offset, size, count)


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (b"PK\x05\x06", r"^Invalid ZIP end record\.$"),
        (
            b"PK\x05\x06" + bytes(18) + b"extra",
            r"^Invalid ZIP end record comment length\.$",
        ),
        (
            b"PK\x05\x06" + bytes(6) + b"\xff\xff" + bytes(10),
            r"^Missing ZIP64 end record\.$",
        ),
        (
            struct.pack("<4s4H2LH", b"PK\x05\x06", 1, 0, 0, 0, 0, 0, 0),
            r"^Unsupported multi-disk ZIP directory\.$",
        ),
    ],
)
def test_invalid_end_records_fail_closed(tail: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        directory_location(tail)


@pytest.mark.parametrize("name", ["../escape.dcm", "/absolute.dcm", "directory/"])
def test_members_cannot_escape_staging(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=r"^Unsafe ZIP member path\.$"):
        fetch_member(ZipInfo(name), url="unused", archive_size=0, staging_dir=tmp_path)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("crc", r"^ZIP member size or CRC verification failed\.$"),
        ("header", r"^ZIP local header differs from its central directory\.$"),
        ("size", r"^ZIP compressed payload is incomplete\.$"),
        ("compression", r"^Unsupported ZIP compression method\.$"),
        ("deflate", r"^ZIP deflate stream is incomplete\.$"),
        ("inflation", r"^ZIP deflate stream is incomplete\.$"),
    ],
)
def test_members_reject_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str,
) -> None:
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as writer:
        writer.writestr("a.dcm", b"pixels" * 20)
    raw = bytearray(archive.getvalue())
    with ZipFile(BytesIO(raw)) as source:
        member = source.infolist()[0]
    if fault == "crc":
        member.CRC += 1
    elif fault == "header":
        raw[0] = 0
    elif fault == "size":
        member.compress_size = len(raw) + 1
    elif fault == "compression":
        member.compress_type = 99
    elif fault == "inflation":
        member.file_size -= 2
    else:
        member.compress_size -= 1
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=bytes(raw)))
    with pytest.raises(ValueError, match=message):
        fetch_member(
            member,
            url="https://example.test/archive",
            archive_size=len(raw),
            staging_dir=tmp_path,
        )
    assert not (tmp_path / "a.dcm").exists()


@pytest.mark.parametrize("fault", ["size", "count"])
def test_index_rejects_impossible_or_incomplete_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    archive = BytesIO()
    with ZipFile(archive, "w") as writer:
        writer.writestr("a.dcm", b"pixels")
    raw = bytearray(archive.getvalue())
    if fault == "size":
        raw[-10:-6] = (2 << 30).to_bytes(4, "little")
    else:
        raw[-12:-10] = (2).to_bytes(2, "little")
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=bytes(raw)))
    with pytest.raises(ValueError, match="ZIP directory"):
        index_archive("https://example.test/archive", working_dir=tmp_path)


@pytest.mark.parametrize("field", ["signature", "size", "locator"])
def test_zip64_rejects_unrecognized_record_layout(field: str) -> None:
    record = struct.pack(
        "<4sQ2H2I4Q",
        b"BAD!" if field == "signature" else b"PK\x06\x06",
        45 if field == "size" else 44,
        45,
        45,
        0,
        0,
        2,
        2,
        100,
        200,
    )
    locator = struct.pack(
        "<4sIQI",
        b"BAD!" if field == "locator" else b"PK\x06\x07",
        0,
        0,
        1,
    )
    end = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        65_535,
        65_535,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    with pytest.raises(ValueError, match=r"^Unsupported ZIP64 end record\.$"):
        directory_location(record + locator + end)


@pytest.mark.parametrize("status", [200, 206, 403])
def test_archive_range_requires_partial_response(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    monkeypatch.setattr(httpx, "stream", partial(_range_stream, status=status))
    if status == 206:
        assert read_range("https://example.test/secret", byte_range="bytes=-3") == (
            b"abc",
            "bytes 7-9/10",
        )
    else:
        with pytest.raises(RuntimeError, match="range"):
            read_range("https://example.test/secret", byte_range="bytes=-3")


@pytest.mark.parametrize(
    "content_range",
    ["", "bytes 0-9/10", "bytes 8-9/10", "bytes 0-2/10"],
)
def test_archive_range_rejects_invalid_length(
    monkeypatch: pytest.MonkeyPatch,
    content_range: str,
) -> None:
    monkeypatch.setattr(
        httpx,
        "stream",
        partial(_range_stream, content_range=content_range),
    )
    with pytest.raises(RuntimeError, match="range"):
        read_range("https://example.test/secret", byte_range="bytes=-3")


def test_archive_range_rejects_a_short_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "stream", partial(_range_stream, content=b"ab"))
    with pytest.raises(RuntimeError, match=r"^Archive range body is incomplete\.$"):
        read_range("https://example.test/secret", byte_range="bytes=-3")


def test_directory_must_list_every_declared_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("a.dcm", b"first")
        archive.writestr("b.dcm", b"second")
    raw = bytearray(buffer.getvalue())
    end = raw.rfind(b"PK\x05\x06")
    raw[end + 8 : end + 12] = (3).to_bytes(2, "little") * 2
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=bytes(raw)))
    with pytest.raises(ValueError, match=r"^ZIP directory inventory is incomplete\.$"):
        index_archive("https://example.test/archive", working_dir=tmp_path)


def test_resumed_member_is_verified_against_its_directory_entry(
    tmp_path: Path,
) -> None:
    payload = b"resumed pixels"
    member = ZipInfo("a.dcm")
    member.file_size, member.CRC = len(payload), zlib.crc32(payload)
    path = tmp_path / "a.dcm"
    path.write_bytes(payload)
    verified = verify_member(member, path=path)
    assert (verified.name, verified.size, verified.crc32) == (
        "a.dcm",
        len(payload),
        member.CRC,
    )
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()
    path.write_bytes(b"corrupt pixels")
    with pytest.raises(ValueError, match="size or CRC"):
        verify_member(member, path=path)


def test_archive_transport_does_not_expose_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "stream", partial(_range_stream, status=0))
    with pytest.raises(RuntimeError, match="range") as caught:
        read_range("https://example.test/secret", byte_range="bytes=-3")
    assert "secret" not in str(caught.value)


@contextmanager
def _range_stream(
    *args: object,
    status: int = 206,
    content_range: str = "bytes 7-9/10",
    content: bytes = b"abc",
    **kwargs: object,
) -> Generator[httpx.Response]:
    del args, kwargs
    if status == 0:
        raise httpx.ConnectError("secret")
    yield httpx.Response(
        status,
        content=content,
        headers={"Content-Range": content_range},
    )


def _failed_replace(path: Path, target: str | Path) -> Path:
    del path, target
    raise OSError("Interrupted publication")


@pytest.mark.parametrize("encoding", ["utf-8", "cp437"])
def test_encrypted_member_headers_are_rejected_before_plaintext_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    name = "série/汉.dcm" if encoding == "utf-8" else "café.dcm"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(name if encoding == "utf-8" else "cafe.dcm", b"pixels")
    raw = buffer.getvalue()
    if encoding == "cp437":
        raw = raw.replace(b"cafe.dcm", name.encode(encoding))
    flagged = bytearray(raw)
    flagged[6] |= 1
    central = flagged.index(b"PK\x01\x02")
    flagged[central + 8] |= 1
    with ZipFile(BytesIO(flagged)) as archive:
        member = archive.infolist()[0]
        with pytest.raises(RuntimeError, match="encrypted"):
            archive.read(member)
    monkeypatch.setattr(httpx, "stream", partial(_stream, raw=bytes(flagged)))
    with pytest.raises(ValueError, match=r"^Unsupported encrypted ZIP member\.$"):
        fetch_member(
            member,
            url="https://example.test/archive",
            archive_size=len(flagged),
            staging_dir=tmp_path,
        )
    assert not (tmp_path / name).exists()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
