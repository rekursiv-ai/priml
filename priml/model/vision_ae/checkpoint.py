"""Weights files named by where they live and what their bytes hash to.

A checkpoint is a node in the config tree, not a path string, so a run's
weights print, diff, and land in a corpus receipt. Every published default pins
a revision AND a SHA-256: the revision says which upload, the digest says the
bytes did not change underneath it. Only ``path()`` resolves -- never
``finalize`` or ``make()`` -- so building a config never touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib import parse

import hashlib
import re

from configgle import Fig

from priml.data.ensure import DataSpec, FileSpec, ensure_data, resumable_http_download
from priml.hub import get_cache_dir


if TYPE_CHECKING:
    import huggingface_hub
else:
    from wrapt import lazy_import

    huggingface_hub = lazy_import("huggingface_hub")


class HubFile:
    """A file in a Hugging Face Hub repository, pinned to one revision."""

    class Config(Fig["HubFile"]):
        """Repository, file, revision, and expected digest."""

        repo_id: str = ""
        """Hub repository, ``owner/name``."""

        filename: str = ""
        """Path of the file within the repository."""

        revision: str = ""
        """Full lowercase commit SHA; a branch name would let the bytes move under a run."""

        sha256: str = ""
        """Expected lowercase hex digest; empty skips verification."""

    def __init__(self, config: Config) -> None:
        if not config.repo_id or not config.filename:
            raise ValueError(
                "HubFile needs repo_id and filename; got "
                f"{config.repo_id!r}, {config.filename!r}.",
            )
        if re.fullmatch(r"[0-9a-f]{40}", config.revision) is None:
            raise ValueError(
                "HubFile needs a pinned revision, a 40-digit lowercase commit SHA; "
                f"got {config.revision!r}.",
            )
        _require_digest("HubFile", digest=config.sha256)
        self.config = config

    def path(self) -> Path:
        """Download into the Hugging Face cache if absent, then verify.

        Returns:
          path: Local file.

        Raises:
          RuntimeError: The file does not hash to ``sha256``.

        """
        path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.filename,
                revision=self.config.revision,
            ),
        )
        verify_sha256(path, expected=self.config.sha256)
        return path

    def identity(self) -> dict[str, str]:
        """Return the repository coordinates and digest.

        Returns:
          identity: ``repo_id``, ``filename``, ``revision``, ``sha256``.

        """
        return {
            "repo_id": self.config.repo_id,
            "filename": self.config.filename,
            "revision": self.config.revision,
            "sha256": self.config.sha256,
        }


class UrlFile:
    """A file fetched over HTTPS once and cached under the model cache."""

    class Config(Fig["UrlFile"]):
        """Source URL and expected digest."""

        url: str = ""
        """HTTPS URL naming immutable bytes, e.g. a raw file at a commit SHA."""

        sha256: str = ""
        """Expected lowercase hex digest; required, since a URL alone pins nothing."""

    def __init__(self, config: Config) -> None:
        if not config.url.startswith("https://") or not config.sha256:
            raise ValueError(
                f"UrlFile needs an https URL and a sha256; got {config.url!r}.",
            )
        _require_digest("UrlFile", digest=config.sha256)
        # The cached copy is named for the URL path's last segment alone: a query
        # string can carry a token, or outrun the file-name limit.
        self.filename = parse.urlsplit(config.url).path.rsplit("/", 1)[-1]
        if self.filename in {"", ".", ".."}:
            raise ValueError(f"UrlFile needs a URL naming a file; got {config.url!r}.")
        self.config = config

    def path(self) -> Path:
        """Download into the cache, keyed by digest, if absent; then verify.

        Returns:
          path: Local file.

        Raises:
          RuntimeError: The downloaded bytes do not hash to ``sha256``.

        """
        target = get_cache_dir() / "url" / self.config.sha256
        _ = ensure_data(
            DataSpec(
                target_dir=target,
                manifest=[FileSpec(rel_path=self.filename, sha256=self.config.sha256)],
                fetch=self._fetch,
            ),
        )
        return target / self.filename

    def identity(self) -> dict[str, str]:
        """Return the URL and digest.

        Returns:
          identity: ``url`` and ``sha256``.

        """
        return {"url": self.config.url, "sha256": self.config.sha256}

    def _fetch(self, *, rel_path: str, dest: Path) -> None:
        """Download the configured URL to ``dest``; ``rel_path`` is its name."""
        del rel_path
        resumable_http_download(url=self.config.url, dest=dest)


class LocalFile:
    """A file already on this machine."""

    class Config(Fig["LocalFile"]):
        """Path and expected digest."""

        path: Path | str = ""
        """Local file."""

        sha256: str = ""
        """Expected lowercase hex digest; empty skips verification."""

    def __init__(self, config: Config) -> None:
        if not str(config.path):
            raise ValueError("LocalFile needs a path.")
        _require_digest("LocalFile", digest=config.sha256)
        self.config = config

    def path(self) -> Path:
        """Return the verified path.

        Returns:
          path: Local file.

        Raises:
          FileNotFoundError: The file does not exist.
          RuntimeError: The file does not hash to ``sha256``.

        """
        path = Path(self.config.path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")
        verify_sha256(path, expected=self.config.sha256)
        return path

    def identity(self) -> dict[str, str]:
        """Return the digest, which names the file across machines.

        Returns:
          identity: ``sha256``, computed when not configured.

        """
        return {"sha256": self.config.sha256 or sha256_file(self.path())}


def sha256_file(path: Path) -> str:
    """Hash a file without holding it in memory.

    Args:
      path: File to hash.

    Returns:
      digest: Lowercase hex SHA-256.

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise unless ``path`` hashes to ``expected``; an empty digest skips the check.

    Args:
      path: File to verify.
      expected: Lowercase hex SHA-256, or empty.

    Raises:
      RuntimeError: The digest differs, as ``ensure_data`` raises for a
        downloaded file.

    """
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{path} hashes to {actual}, not the configured {expected}; the "
            "file changed or is truncated.",
        )


def _require_digest(source: str, digest: str) -> None:
    """Raise unless ``digest`` is empty or 64 lowercase hex digits."""
    # At construction: ``hexdigest`` is lowercase, so another spelling would fail
    # only after a full hash, and ``UrlFile`` makes the digest a directory name.
    if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            f"{source} sha256 must be 64 lowercase hex digits; got {digest!r}.",
        )
