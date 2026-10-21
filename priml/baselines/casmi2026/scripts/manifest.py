"""Digest inputs and generators, and publish the manifest of a prepared dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import json

from treekle import to_plain


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Args:
      path: File to digest.

    Returns:
      digest: Lowercase hexadecimal digest.

    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, hashlib.sha256).hexdigest()


def input_digests(paths: Iterable[Path]) -> dict[str, str]:
    """Map the absolute path of every input to its digest.

    Args:
      paths: Input files the dataset was prepared from.

    Returns:
      digests: Resolved path to digest, in the given order.

    """
    return {str(path.resolve()): sha256_file(path) for path in paths}


def source_digests(paths: Iterable[Path]) -> dict[str, str]:
    """Map the file name of every generator source to its digest.

    Args:
      paths: Source files that produced the dataset.

    Returns:
      digests: File name to digest, in the given order.

    """
    return {path.name: sha256_file(path) for path in paths}


def write_manifest(destination: Path, manifest: Mapping[str, object]) -> None:
    """Write ``manifest.json`` into a prepared dataset directory.

    Args:
      destination: Directory holding the prepared dataset.
      manifest: Plain header fields, including input and generator digests.

    """
    text = json.dumps(to_plain(manifest), indent=2) + "\n"
    _ = (destination / "manifest.json").write_text(text)
