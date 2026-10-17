"""Weights files as configs: a Hub file, a URL, or a local path, each with a digest.

``path()`` downloads if needed and verifies the digest; nothing else touches the
network.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from urllib import parse

import functools
import hashlib
import re

from configgle import Fig, Makeable
from treekle import json_loads

from priml.data.ensure import (
    DataSpec,
    FileSpec,
    ensure_data,
    resumable_http_download,
)
from priml.hub import get_cache_dir


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    import huggingface_hub
    import safetensors
    import torch
else:
    from wrapt import lazy_import

    huggingface_hub = lazy_import("huggingface_hub")
    safetensors = lazy_import("safetensors")
    torch = lazy_import("torch")


@runtime_checkable
class CheckpointFile(Protocol):
    """A weights file that resolves to a local path, downloading if it must."""

    def path(self) -> Path:
        """Return the local path, verified when a digest is configured.

        Returns:
          path: Local file.

        Raises:
          RuntimeError: The bytes do not hash to the configured digest.

        """
        ...

    def identity(self) -> dict[str, str]:
        """Return what names this file across machines, for a corpus receipt."""
        ...


def read_state(path: Path, *, prefixes: Sequence[str] = ("",)) -> dict[str, Tensor]:
    """Read the tensors whose keys start with any of ``prefixes``, on the CPU.

    Unselected tensors are never read from disk.

    Args:
      path: A ``.safetensors`` file, or a ``torch.save``-d state dict.
      prefixes: Key prefixes to keep; the default keeps every key.

    Returns:
      state: Selected tensors by their stored names.

    """
    keep = tuple(prefixes)
    if path.suffix == ".safetensors":
        with safetensors.safe_open(path, framework="pt") as stream:
            return {
                name: stream.get_tensor(name)
                for name in stream.keys()  # noqa: SIM118 -- safe_open is not a Mapping; keys() is its only listing.
                if name.startswith(keep)
            }
    state = cast(
        "dict[str, Tensor]",
        torch.load(path, map_location="cpu", weights_only=True, mmap=True),
    )
    return {name: tensor for name, tensor in state.items() if name.startswith(keep)}


def hub_or_local(
    path_or_repo: Path | str,
    filename: str,
    *,
    revision: str,
) -> Makeable[CheckpointFile]:
    """Name one file of a local directory, or of a Hub repository at ``revision``.

    Args:
      path_or_repo: An existing local directory, or a Hub ``owner/name``.
      filename: File within it.
      revision: Full commit SHA, used for a Hub repository only.

    Returns:
      source: A :class:`LocalFile` or :class:`HubFile` config.

    """
    path = Path(path_or_repo)
    if path.is_dir():
        return LocalFile.Config(path=path / filename)
    return HubFile.Config(
        repo_id=str(path_or_repo),
        filename=filename,
        revision=revision,
    )


def read_json(source: Makeable[CheckpointFile]) -> dict[str, object]:
    """Read a JSON object, e.g. a ``config.json``, from a checkpoint file.

    Args:
      source: Where the file lives.

    Returns:
      config: The parsed object.

    """
    return json_loads(source.make().path().read_text(), dict[str, object])


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
      RuntimeError: The digest differs.

    """
    if not expected:
        return
    stat = path.stat()
    _verify_sha256(
        path.resolve(),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        expected=expected,
    )


# Both halves of an autoencoder read one weights file; without the cache, every build
# hashed that file once per half. The size and mtime in the key re-verify a file
# rewritten in place. A mismatch raises, and ``cache`` does not store exceptions.
@functools.cache
def _verify_sha256(path: Path, *, size: int, mtime_ns: int, expected: str) -> None:
    """Raise unless ``path`` hashes to ``expected``; remembered per file version."""
    del size, mtime_ns
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
