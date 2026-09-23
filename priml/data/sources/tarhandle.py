"""Tar file handle with optional memory-mapping for thread-safe access."""

from __future__ import annotations

from contextlib import ExitStack, suppress
from io import BytesIO
from pathlib import Path
from typing import IO, Protocol, Self, override

import mmap
import os
import tarfile
import threading


__all__ = ["TarFileHandle", "TarFileProtocol"]


class TarFileProtocol(Protocol):
    """Protocol for tar file handles that support getmember and extractfile.

    This protocol is satisfied by both tarfile.TarFile and TarFileHandle,
    allowing processors to work with either implementation.
    """

    # Filesystem path of the archive, or None when it has none. Spelled
    # ``name`` because that is what ``tarfile.TarFile`` calls it and this
    # protocol claims conformance to it; ``TarFileHandle`` mirrors the spelling
    # rather than the reverse. None is reachable -- a TarFile opened from a file
    # object has no path -- so a caller reopening the archive to seek by offset
    # must handle its absence.
    name: str | None

    def getmember(self, name: str) -> tarfile.TarInfo:
        """Get member info by name.

        Args:
            name: Name of the tar member to retrieve.

        Returns:
            member: TarInfo object for the requested member.

        Raises:
            KeyError: If member not found.

        """
        ...

    def extractfile(self, member: str | tarfile.TarInfo) -> IO[bytes] | None:
        """Extract a member as a file-like object.

        Args:
            member: Either a member name (str) or TarInfo object.

        Returns:
            data: File-like object (IO[bytes]) with the member's data, or None
                if the member is not a regular file.

        """
        ...


class TarFileHandle:
    """Tar file handle with optional memory-mapping for thread-safe random access.

    Supports two modes:
    - mmap mode (default): Builds a member index once via tarfile (correctly
      resolving PAX/GNU long names) and serves each read by copying the member
      out of the mapped pages into a ``BytesIO``. Thread-safe by design - the
      mmap lets multiple threads read concurrently without seeking a shared
      file handle or issuing per-read syscalls.
    - standard mode: Uses standard tarfile.open() with thread-local handles.

    Benefits of mmap mode:
    - No per-read syscall/seek: members are read straight from mapped pages,
      not via ``seek``/``read`` on a shared handle (the source of fork/thread
      offset races)
    - Fast random access via member index
    - OS-managed page cache (members faulted in lazily, shared across readers)
    - Thread-safe without locks

    Example:
        # With mmap (default)
        handle = TarFileHandle(Path("/data/shard_00.tar"))
        member = handle.getmember("image_001.jpg")
        data = handle.extractfile(member).read()

        # Without mmap
        handle = TarFileHandle(Path("/data/shard_00.tar"), use_mmap=False)
        member = handle.getmember("image_001.jpg")
        data = handle.extractfile(member).read()

    """

    def __init__(self, path: Path, use_mmap: bool = True):
        """Initialize tar file handle.

        Args:
            path: Path to tar file to open.
            use_mmap: If True, use memory-mapped I/O. If False, use standard tarfile.

        """
        self.path = path
        self.use_mmap = use_mmap
        self.name = str(path)
        self._closed = False

        if use_mmap:
            _reject_compressed(path)
            self._resources = ExitStack()
            # The descriptor is closed here because ``mmap`` dups it: the
            # mapping stays valid for this object's whole life without holding
            # an unscoped handle open alongside it.
            with Path(path).open("rb") as source:
                self._mmap = self._resources.enter_context(
                    mmap.mmap(source.fileno(), 0, prot=mmap.PROT_READ),
                )

            # Build index of tar members: name -> (offset, size, TarInfo)
            # This is done once at init and shared across all threads.
            self._index: dict[str, tuple[int, int, tarfile.TarInfo]] = {}
            self._build_index()
        else:
            # Standard mode: use thread-local tarfile handles, tagged with the
            # PID that opened them so an inherited handle reopened after fork
            # does not share a file offset with the parent across processes.
            self._local = threading.local()
            self._opened = ExitStack()
            self._opened_lock = threading.Lock()
            self._index = {}

    def getmember(self, name: str) -> tarfile.TarInfo:
        """Get member info by name.

        Args:
            name: Name of the tar member to retrieve.

        Returns:
            member: TarInfo object for the requested member.

        Raises:
            KeyError: If member not found.
            ValueError: If the handle is closed.

        """
        self._reject_if_closed()
        if self.use_mmap:
            if name not in self._index:
                raise KeyError(f"Member {name} not found in tar archive")
            _, _, info = self._index[name]
            return info
        return self._get_tarfile().getmember(name)

    def extractfile(self, member: str | tarfile.TarInfo) -> IO[bytes] | None:
        """Extract a member as a file-like object.

        Args:
            member: Either a member name (str) or TarInfo object.

        Returns:
            data: File-like object (IO[bytes]) with the member's data, or None
                if the member is not a regular file.

        Raises:
            KeyError: If member not found.
            ValueError: If the handle is closed.

        """
        self._reject_if_closed()
        if self.use_mmap:
            name = member if isinstance(member, str) else member.name

            # Absent is an error, not None: None is reserved for a member that
            # EXISTS without a payload, and standard mode raises here too.
            if name not in self._index:
                raise KeyError(f"Member {name} not found in tar archive")

            offset, size, info = self._index[name]
            # A directory or link has no payload to map; the protocol says
            # such a member reads as None rather than as empty bytes.
            if not info.isfile():
                return None

            # ``BytesIO`` copies the slice into its own buffer at construction
            # (it does not hold a live view of the mmap), so the read is one
            # member-sized copy faulted straight from the mapped pages -- no
            # seek/read syscall on a shared handle. The copy also means the
            # returned object does not pin the mmap (safe to close on __del__).
            return BytesIO(memoryview(self._mmap)[offset : offset + size])
        return self._get_tarfile().extractfile(member)

    def close(self) -> None:
        """Release the mapping and every archive handle any thread opened.

        Idempotent, so ``__del__`` may call it after an explicit close. In
        standard mode each thread opens its own handle and registers it, which
        is what lets this close the ones belonging to threads that have since
        exited -- a thread's locals are unreachable from here otherwise.
        """
        self._closed = True
        if self.use_mmap:
            self._resources.close()
            return
        with self._opened_lock:
            self._opened.close()
        if hasattr(self._local, "tarfile"):
            del self._local.tarfile

    def __enter__(self) -> Self:
        """Return this handle so callers can scope it with ``with``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on scope exit, whether or not the body raised."""
        del exc_info
        self.close()

    def __del__(self) -> None:
        """Best-effort release for a handle nobody closed.

        Every failure is suppressed: a destructor runs during interpreter
        shutdown, when the modules needed to report one may already be gone.
        """
        with suppress(Exception):
            self.close()

    @override
    def __repr__(self) -> str:
        mode = "mmap" if self.use_mmap else "standard"
        if self.use_mmap:
            return (
                f"TarFileHandle({self.path}, {len(self._index)} members, mode={mode})"
            )
        return f"TarFileHandle({self.path}, mode={mode})"

    # ``ExitStack`` is reusable and ``threading.local`` is empty after close, so
    # ``_get_tarfile`` would otherwise reopen the archive and hand back a working reader
    # from an object the caller had already released.
    def _reject_if_closed(self) -> None:
        """Raise once the handle is closed."""
        if self._closed:
            raise ValueError(f"{self.path} handle is closed.")

    # Enumeration goes through ``tarfile.TarFile`` so PAX (``x``) and GNU (``L``) long-
    # name extension headers are resolved correctly; a hand-rolled 512-byte header
    # parser only reads the legacy 100-byte name field and mis-parses those.
    # ``member.offset_data`` and ``member.size`` from tarfile then drive the mmap slice
    # reads in ``extractfile``.
    #
    # Directories and links are indexed alongside regular files so ``getmember`` answers
    # for them as ``tarfile.TarFile`` does; dropping them turned a member that exists
    # into a KeyError, which reads as a missing archive entry. ``extractfile`` still
    # returns None for them, which is what the protocol documents.
    def _build_index(self) -> None:
        """Build a name -> (data_offset, size, TarInfo) index over every member."""
        with tarfile.open(self.path, "r") as tar:
            for member in tar.getmembers():
                self._index[member.name] = (
                    member.offset_data,
                    member.size,
                    member,
                )

    def _get_tarfile(self) -> tarfile.TarFile:
        """Get thread-local tarfile handle (standard mode only)."""
        # Reopen if this handle was inherited across a fork: the parent's
        # TarFile fd would otherwise be shared and its seek offset raced.
        current_pid = os.getpid()
        if getattr(self._local, "pid", None) != current_pid:
            stale = getattr(self._local, "tarfile", None)
            # The inherited handle is useless in this process but still holds a
            # descriptor; dropping the reference alone would leak it.
            if isinstance(stale, tarfile.TarFile):
                with suppress(OSError):
                    stale.close()
            # Opened into a local stack and then handed to the shared one:
            # ownership must outlive this call (``threading`` exposes no way to
            # enumerate another thread's locals, so ``close`` could not reach
            # the handle otherwise), while anything that raises before the
            # transfer completes must still close it.
            with ExitStack() as stack:
                opened = stack.enter_context(tarfile.open(self.path, "r"))
                with self._opened_lock:
                    self._opened.push(stack.pop_all())
            self._local.tarfile = opened
            self._local.pid = current_pid
        # threading.local attributes are untyped; the assignment above fixes it.
        handle = self._local.tarfile
        assert isinstance(handle, tarfile.TarFile)
        return handle


# ``mmap`` mode reads members by ``member.offset_data``, which tarfile reports as an
# offset into the DECOMPRESSED stream; the mapping holds the file's real bytes. For a
# compressed archive those are different coordinate spaces, so the slice returns
# unrelated content -- measured as a silent empty read rather than an error. Standard
# mode decompresses through tarfile and is unaffected, so it remains the supported path
# for these.
def _reject_compressed(
    path: Path,
    *,
    magic: tuple[bytes, ...] = (
        b"\x1f\x8b",  # Gzip.
        b"BZh",  # bzip2.
        b"\xfd7zXZ",  # Xz.
        b"\x04\x22\x4d\x18",  # lz4.
        b"\x28\xb5\x2f\xfd",  # Zstd.
    ),
) -> None:
    """Raise when ``path`` starts with a compressed container's magic bytes."""
    with path.open("rb") as probe:
        head = probe.read(8)
    if any(head.startswith(prefix) for prefix in magic):
        raise ValueError(
            f"{path} is a compressed archive; mmap mode reads members by "
            "uncompressed offset and cannot address it. Pass use_mmap=False.",
        )
