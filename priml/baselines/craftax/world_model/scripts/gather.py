#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Copy the published shards of chosen capture workers to another archive root.

SOURCE and DESTINATION are archive roots as rsync paths: a local directory or
[USER@]HOST:DIRECTORY. rsync copies between a local and a remote root, never
between two remote ones, so shards move from node to node through a local root:
gather from the first node into it, then from it to the second. A remote root is
reached through ssh, or through RSYNC_RSH, which must run the command it is
given through a shell, as ssh does; a remote destination root must exist. Each
WORKER, such as arm1/w0, names both of a capture worker's directories,
train/arm1/w0 and val/arm1/w0.

Only the shards named in the source's MANIFEST.jsonl move, so unpublished .tmp
files never do. Nothing is copied when the source root holds HALT.json, or when
a destination manifest's published lines are not a prefix of the source's:
that directory holds another capture of the same worker, and the two would
repeat world seeds. A torn tail a crashed writer left is copied with its
manifest, and ignored, as every reader ignores it.
New shard files are copied by content (rsync --checksum), and then checked
against their manifest lines' SHA-256s on whichever root is local; a remote
destination's copy is rsync's checked copy of that verified local source. Only
then is each destination manifest replaced by the source's: after an fsync of
the shard files on a local destination, after a sync on a remote one. So an
interrupted gather publishes nothing new, rerunning it finishes the copy, and
a third run copies nothing.

Examples:
  priml/baselines/craftax/world_model/scripts/gather.py node-d:/opt/scratch/datasets/craftax/world-model/archive-v1 /opt/scratch/datasets/craftax/world-model/archive-v1 arm1/w0 arm2/w0
  priml/baselines/craftax/world_model/scripts/gather.py /opt/scratch/datasets/craftax/world-model/archive-v1 node-c:/opt/scratch/datasets/craftax/world-model/archive-v1 arm1/w0 arm2/w0

'''
# fmt: on

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import hashlib
import os
import re
import subprocess
import tempfile

from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_manifest,
)
from priml.baselines.craftax.world_model.capture.control import (
    check_halt,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once every new shard is published.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    shards = gather(flags.source, flags.destination, workers=flags.workers)
    for shard in shards:
        print(shard)
    print(f"Published {len(shards)} shards at {flags.destination}.")
    return 0


def gather(
    source: str,
    destination: str,
    *,
    workers: Sequence[tuple[int, int]],
) -> list[str]:
    """Publish at ``destination`` every shard of ``workers`` published at ``source``.

    Args:
      source: Archive root to copy from, a local directory or ``[user@]host:dir``.
      destination: Archive root to copy to, likewise.
      workers: ``(arm, worker)`` pairs; each names its training and validation
        directories.

    Returns:
      shards: The newly published shards, relative to the roots.

    Raises:
      ValueError: Both roots are remote, a destination manifest is not a prefix
        of the source's, or a new file's SHA-256 differs from its manifest line.
      CaptureHaltedError: The source root holds ``HALT.json``.

    """
    if _is_remote(source) and _is_remote(destination):
        raise ValueError(
            "At most one archive root may be remote; gather via a local one.",
        )
    directories = [
        str(shard_directory(Path(), split=split, arm=arm, worker=worker))
        for arm, worker in workers
        for split in (TRAIN, VALIDATION)
    ]
    with tempfile.TemporaryDirectory(prefix="world_model_gather_") as scratch:
        theirs, ours = Path(scratch) / "source", Path(scratch) / "destination"
        _fetch_manifests(source, into=theirs, directories=directories)
        check_halt(theirs)
        if not _is_remote(destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
        _fetch_manifests(destination, into=ours, directories=directories)
        new = {d: _unpublished(d, theirs=theirs, ours=ours) for d in directories}
        new = {d: lines for d, lines in new.items() if lines}
        if new:
            _publish(new, source=source, destination=destination, scratch=Path(scratch))
    return [f"{d}/{line.shard}" for d, lines in new.items() for line in lines]


class Flags(Protocol):
    """Parsed command-line flags."""

    source: str
    destination: str
    workers: list[tuple[int, int]]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("source", help="Archive root to copy from.")
    parser.add_argument("destination", help="Archive root to copy to.")
    parser.add_argument(
        "workers",
        nargs="+",
        type=_worker,
        metavar="WORKER",
        help="Capture worker as armA/wW, e.g. arm1/w0.",
    )


def _worker(text: str) -> tuple[int, int]:
    """Parse ``armA/wW`` into ``(A, W)``."""
    match = re.fullmatch(r"arm([0-3])/w([0-3])", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"Expected armA/wW, A and W in 0-3: {text!r}.")
    return int(match[1]), int(match[2])


def _is_remote(root: str) -> bool:
    """Return whether rsync reads ``root`` as remote: a colon before any slash."""
    return ":" in root.split("/", 1)[0]


def _fetch_manifests(root: str, *, into: Path, directories: Sequence[str]) -> None:
    """Copy ``HALT.json`` and each directory's manifest, where ``root`` has them."""
    patterns = ["/HALT.json"]
    for directory in directories:
        parts = directory.split("/")
        patterns += ["/" + "/".join(parts[: i + 1]) + "/" for i in range(len(parts))]
        patterns.append(f"/{directory}/MANIFEST.jsonl")
    filters = [f"--include={p}" for p in dict.fromkeys(patterns)]
    _rsync("--recursive", *filters, "--exclude=*", f"{root}/", f"{into}/")


def _unpublished(directory: str, *, theirs: Path, ours: Path) -> list[ManifestLine]:
    """Return the source's manifest lines of ``directory`` after the destination's."""
    paths = [root / directory / "MANIFEST.jsonl" for root in (theirs, ours)]
    # Published lines alone, as ``read_manifest`` reads them: the source's next
    # append drops a torn tail, so a copied one, compared as a line, would
    # refuse every later gather.
    source, destination = (
        p.read_text().split("\n")[:-1] if p.exists() else [] for p in paths
    )
    if source[: len(destination)] != destination:
        raise ValueError(
            f"The destination's {directory} manifest is not a prefix of the "
            "source's: it holds another capture of that worker, and the two "
            "would repeat world seeds.",
        )
    return read_manifest(theirs / directory)[len(destination) :]


def _publish(
    new: dict[str, list[ManifestLine]],
    *,
    source: str,
    destination: str,
    scratch: Path,
) -> None:
    """Copy and check the new shards, then publish the source's manifests."""
    remote = _is_remote(destination)
    files = [
        f"{d}/{name}"
        for d, lines in new.items()
        for line in lines
        for name in _shard_files(line).values()
    ]
    _copy(files, source, destination, listing=scratch / "files")
    _check(new, root=Path(source if remote else destination), fsync=not remote)
    manifests = [f"{d}/MANIFEST.jsonl" for d in new]
    # A remote root is flushed before its manifests are replaced, so no manifest
    # line can survive a crash that loses the shard files it names.
    sync = ["--rsync-path=sync && rsync"] if remote else []
    theirs = str(scratch / "source")
    _copy(manifests, theirs, destination, listing=scratch / "manifests", options=sync)
    if not remote:
        for name in [*manifests, *new]:
            _fsync(Path(destination) / name)


def _shard_files(line: ManifestLine) -> dict[str, str]:
    """Return a shard's file names, keyed like the SHA-256s of its manifest line."""
    return {
        suffix: f"{line.shard}.{suffix}.jsonl"
        if suffix == "meta"
        else f"{line.shard}.{suffix}.zst"
        for suffix in line.sha256
    }


def _copy(
    names: Sequence[str],
    source: str,
    destination: str,
    *,
    listing: Path,
    options: Sequence[str] = (),
) -> None:
    """Copy the files ``names`` between two roots, creating their directories."""
    listing.write_text("".join(f"{name}\n" for name in names))
    _rsync(*options, f"--files-from={listing}", f"{source}/", f"{destination}/")


def _check(new: dict[str, list[ManifestLine]], *, root: Path, fsync: bool) -> None:
    """Check the new shard files under ``root`` against their lines; fsync if asked."""
    for directory, lines in new.items():
        for line in lines:
            for key, name in _shard_files(line).items():
                path = root / directory / name
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != line.sha256[key]:
                    raise ValueError(f"SHA-256 mismatch for {path}.")
                if fsync:
                    _fsync(path)
        if fsync:
            _fsync(root / directory)


def _rsync(*args: str) -> None:
    """Run rsync, comparing files by content and keeping interrupted copies aside."""
    command = ["rsync", "--checksum", "--partial-dir=.rsync-partial", *args]
    subprocess.run(command, check=True)  # noqa: S603 -- Fixed rsync argv; no shell.


def _fsync(path: Path) -> None:
    """Persist a file's data or a directory's entries."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
