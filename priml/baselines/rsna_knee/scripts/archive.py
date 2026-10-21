"""Read a ZIP central directory and selected members using bounded HTTP ranges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import hashlib
import logging
import re
import shutil
import struct
import zlib

from treekle import from_plain


if TYPE_CHECKING:
    import httpx
else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedFile:
    """A complete locally verified archive member."""

    name: str
    size: int
    crc32: int
    sha256: str


def directory_location(tail: bytes) -> tuple[int, int, int]:
    """Read directory offset, size and entry count from ZIP or ZIP64 tail records.

    Args:
      tail: Final bytes of the complete remote ZIP.

    Returns:
      offset: First central-directory byte.
      size: Complete central-directory byte count.
      count: Complete archive entry count.

    """
    cursor = len(tail)
    location: tuple[int, int, int] | None = None
    error = ValueError("Invalid ZIP end record.")
    while (end := tail.rfind(b"PK\x05\x06", 0, cursor)) >= 0:
        cursor = end
        try:
            location = _directory_candidate(tail, end=end)
        except ValueError as invalid:
            error = invalid
    if location is None:
        raise error
    return location


def index_archive(url: str, *, working_dir: Path) -> tuple[list[ZipInfo], int]:
    """Acquire a resumable complete directory without acquiring archive payloads.

    Args:
      url: Ephemeral signed archive address.
      working_dir: Persistent directory for the complete metadata catalog.

    Returns:
      members: Complete archive inventory with original payload offsets.
      archive_size: Complete remote archive size.

    """
    # A maximal comment can follow both ZIP64 records and the classic footer.
    tail, content_range = read_range(url, byte_range=f"bytes=-{65_535 + 56 + 20 + 22}")
    archive_size = int(content_range.rsplit("/", 1)[1])
    offset, size, count = directory_location(tail)
    if size > 1 << 30 or offset + size > archive_size:
        raise ValueError("ZIP directory exceeds the bounded metadata allocation.")
    identity = hashlib.sha256(urlsplit(url).path.encode() + tail).hexdigest()
    working_dir.mkdir(parents=True, exist_ok=True)
    index = working_dir / f"index-{identity}.zip"
    partial = index.with_suffix(".part")
    if not index.is_file():
        completed = partial.stat().st_size if partial.is_file() else 0
        if completed > size:
            raise ValueError("Partial ZIP directory is longer than its source.")
        with partial.open("ab") as handle:
            while completed < size:
                last = min(completed + 16 * 1024 * 1024, size) - 1
                block, _ = read_range(
                    url,
                    byte_range=f"bytes={offset + completed}-{offset + last}",
                )
                handle.write(block)
                handle.flush()
                completed += len(block)
                logging.getLogger(__name__).info(
                    "Archive directory: %d/%d bytes",
                    completed,
                    size,
                )
        publishing = index.with_suffix(".publishing")
        with partial.open("rb") as source, publishing.open("wb") as handle:
            shutil.copyfileobj(source, handle)
            footer = b"PK\x05\x06" + bytes(4)
            footer += min(count, 65_535).to_bytes(2, "little") * 2
            footer += size.to_bytes(4, "little") + bytes(6)
            handle.write(footer)
        publishing.replace(index)
    with ZipFile(index) as archive:
        members = archive.infolist()
    if len(members) != count:
        raise ValueError("ZIP directory inventory is incomplete.")
    return members, archive_size


def fetch_member(
    member: ZipInfo,
    *,
    url: str,
    archive_size: int,
    staging_dir: Path,
) -> VerifiedFile:
    """Resume whole verified files, checking length and CRC before publication.

    Args:
      member: Entry from the complete central directory.
      url: Ephemeral signed archive address.
      archive_size: Complete archive byte count.
      staging_dir: Directory invisible to training until acquisition finishes.

    Returns:
      verified: Exact name, size, CRC and SHA-256 of the finished file.

    """
    relative = Path(member.filename)
    if relative.is_absolute() or ".." in relative.parts or member.is_dir():
        raise ValueError("Unsafe ZIP member path.")
    if member.flag_bits & 1:
        raise ValueError("Unsupported encrypted ZIP member.")
    destination = staging_dir / relative
    if destination.is_file():
        payload = destination.read_bytes()
    else:
        name = member.filename.encode("utf-8" if member.flag_bits & 0x800 else "cp437")
        last = (
            min(
                archive_size,
                member.header_offset + 30 + len(name) + 65_535 + member.compress_size,
            )
            - 1
        )
        block, _ = read_range(url, byte_range=f"bytes={member.header_offset}-{last}")
        name_size = int.from_bytes(block[26:28], "little")
        extra_size = int.from_bytes(block[28:30], "little")
        if block[:4] != b"PK\x03\x04" or block[30 : 30 + name_size] != name:
            raise ValueError("ZIP local header differs from its central directory.")
        first = 30 + name_size + extra_size
        compressed = block[first : first + member.compress_size]
        if len(compressed) != member.compress_size:
            raise ValueError("ZIP compressed payload is incomplete.")
        if member.compress_type == ZIP_DEFLATED:
            decoder = zlib.decompressobj(-15)
            payload = decoder.decompress(compressed, member.file_size + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError("ZIP deflate stream is incomplete.")
        elif member.compress_type == ZIP_STORED:
            payload = compressed
        else:
            raise ValueError("Unsupported ZIP compression method.")
    verified = _verified_payload(member, payload=payload)
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(payload)
        partial.replace(destination)
    return verified


def read_range(url: str, *, byte_range: str) -> tuple[bytes, str]:
    """Read only an honored bounded range, without exposing the signed address.

    Args:
      url: Ephemeral authenticated address, retained only in memory.
      byte_range: HTTP byte-range expression.

    Returns:
      payload: Complete response bytes.
      content_range: Public response position and archive size.

    """
    first, last = byte_range.removeprefix("bytes=").split("-")
    maximum = int(last) - int(first) + 1 if first else int(last)
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"Range": byte_range},
            timeout=60,
            follow_redirects=True,
        ) as response:
            content_range = cast(str, response.headers.get("Content-Range", ""))
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if response.status_code != 206 or match is None:
                raise RuntimeError("Archive server did not honor the bounded range.")
            start, end, total = (int(match[i]) for i in (1, 2, 3))
            if (
                end < start
                or end >= total
                or end - start + 1 > maximum
                or (first and (start != int(first) or end != int(last)))
                or (
                    not first and (end != total - 1 or start != max(0, total - maximum))
                )
            ):
                raise RuntimeError("Archive range exceeds the requested bound.")
            payload = response.read()
            if len(payload) != end - start + 1:
                raise RuntimeError("Archive range body is incomplete.")
            return payload, content_range
    except httpx.HTTPError:
        raise RuntimeError("Archive range transport failed.") from None


def verify_member(member: ZipInfo, *, path: Path) -> VerifiedFile:
    """Verify a resumed local member against its complete archive identity.

    Args:
      member: Authoritative central-directory size and CRC32.
      path: Complete staged or published local member.

    Returns:
      verified: Checked size, CRC32 and complete SHA256 identity.

    """
    return _verified_payload(member, payload=path.read_bytes())


def _verified_payload(member: ZipInfo, *, payload: bytes) -> VerifiedFile:
    if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
        raise ValueError("ZIP member size or CRC verification failed.")
    return VerifiedFile(
        name=member.filename,
        size=len(payload),
        crc32=member.CRC,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _directory_candidate(tail: bytes, *, end: int) -> tuple[int, int, int]:
    if end + 22 > len(tail):
        raise ValueError("Invalid ZIP end record.")
    _, disk, directory_disk, disk_count, count, size, offset, comment = from_plain(
        struct.unpack_from("<4s4H2LH", tail, end),
        tuple[bytes, int, int, int, int, int, int, int],
    )
    if end + 22 + comment != len(tail):
        raise ValueError("Invalid ZIP end record comment length.")
    if disk or directory_disk:
        raise ValueError("Unsupported multi-disk ZIP directory.")
    if offset == 0xFFFFFFFF or size == 0xFFFFFFFF or count == 0xFFFF:
        if end < 76:
            raise ValueError("Missing ZIP64 end record.")
        (
            signature,
            record_bytes,
            _,
            _,
            disk,
            directory_disk,
            disk_count,
            count,
            size,
            offset,
        ) = from_plain(
            struct.unpack_from("<4sQ2H2I4Q", tail, end - 76),
            tuple[bytes, int, int, int, int, int, int, int, int, int],
        )
        if (
            signature != b"PK\x06\x06"
            or record_bytes != 44
            or tail[end - 20 : end - 16] != b"PK\x06\x07"
            or tail[end - 16 : end - 12] != bytes(4)
            or int.from_bytes(tail[end - 4 : end], "little") != 1
            or disk
            or directory_disk
            or disk_count != count
        ):
            raise ValueError("Unsupported ZIP64 end record.")
    elif disk_count != count:
        raise ValueError("ZIP directory entry counts differ.")
    return offset, size, count
