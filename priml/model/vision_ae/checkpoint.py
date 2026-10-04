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

import hashlib

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
        """Commit SHA; a branch name would let the bytes move under a run."""

        sha256: str | None = None
        """Expected hex digest; ``None`` skips verification."""

    def __init__(self, config: Config) -> None:
        if not config.repo_id or not config.filename or not config.revision:
            raise ValueError(
                "HubFile needs repo_id, filename, and a pinned revision; got "
                f"{config.repo_id!r}, {config.filename!r}, {config.revision!r}.",
            )
        self.config = config

    def path(self) -> Path:
        """Download into the Hugging Face cache if absent, then verify.

        Returns:
          path: Local file.

        """
        path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.filename,
                revision=self.config.revision,
            ),
        )
        verify_sha256(path, self.config.sha256)
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
            "sha256": self.config.sha256 or "",
        }


class UrlFile:
    """A file fetched over HTTPS once and cached under the model cache."""

    class Config(Fig["UrlFile"]):
        """Source URL and expected digest."""

        url: str = ""
        """HTTPS URL naming immutable bytes, e.g. a raw file at a commit SHA."""

        sha256: str = ""
        """Expected hex digest; required, since a URL alone pins nothing."""

    def __init__(self, config: Config) -> None:
        if not config.url.startswith("https://") or not config.sha256:
            raise ValueError(
                f"UrlFile needs an https URL and a sha256; got {config.url!r}.",
            )
        self.config = config

    def path(self) -> Path:
        """Download into the cache, keyed by digest, if absent; then verify.

        Returns:
          path: Local file.

        Raises:
          RuntimeError: The downloaded bytes do not hash to ``sha256``.

        """
        name = self.config.url.rsplit("/", 1)[-1]
        target = get_cache_dir() / "url" / self.config.sha256
        _ = ensure_data(
            DataSpec(
                target_dir=target,
                manifest=[FileSpec(rel_path=name, sha256=self.config.sha256)],
                fetch=self._fetch,
            ),
        )
        return target / name

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

        sha256: str | None = None
        """Expected hex digest; ``None`` skips verification."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def path(self) -> Path:
        """Return the verified path.

        Returns:
          path: Local file.

        Raises:
          FileNotFoundError: The file does not exist.

        """
        path = Path(self.config.path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")
        verify_sha256(path, self.config.sha256)
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


def verify_sha256(path: Path, expected: str | None) -> None:
    """Raise unless ``path`` hashes to ``expected``; ``None`` skips the check.

    Args:
      path: File to verify.
      expected: Lowercase hex SHA-256, or ``None``.

    Raises:
      ValueError: The digest differs.

    """
    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{path} hashes to {actual}, not the configured {expected}; the "
            "file changed or is truncated.",
        )
