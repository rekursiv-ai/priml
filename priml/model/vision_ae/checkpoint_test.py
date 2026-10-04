"""Checkpoint files resolve, verify their digests, and state their identity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import hashlib
import io

import pytest

from priml.model.vision_ae.checkpoint import HubFile, LocalFile, UrlFile, sha256_file
from priml.model.vision_ae.custom_types import CheckpointFile


if TYPE_CHECKING:
    from pathlib import Path

    import urllib.request


def _file(tmp_path: Path, payload: bytes = b"weights") -> tuple[Path, str]:
    path = tmp_path / "weights.pt"
    _ = path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_every_source_satisfies_the_protocol(tmp_path: Path) -> None:
    path, digest = _file(tmp_path)
    assert isinstance(LocalFile.Config(path=path).make(), CheckpointFile)
    assert isinstance(
        UrlFile.Config(url="https://example.com/a.pt", sha256=digest).make(),
        CheckpointFile,
    )
    assert isinstance(
        HubFile.Config(repo_id="o/r", filename="f", revision="abc").make(),
        CheckpointFile,
    )


def test_local_file_verifies_its_digest(tmp_path: Path) -> None:
    path, digest = _file(tmp_path)
    assert LocalFile.Config(path=path, sha256=digest).make().path() == path
    with pytest.raises(ValueError, match="hashes to"):
        _ = LocalFile.Config(path=path, sha256="0" * 64).make().path()


def test_local_file_identity_is_its_digest(tmp_path: Path) -> None:
    path, digest = _file(tmp_path)
    assert LocalFile.Config(path=path).make().identity() == {"sha256": digest}


def test_a_missing_local_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _ = LocalFile.Config(path=tmp_path / "absent.pt").make().path()


def test_hub_file_needs_a_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        _ = HubFile.Config(repo_id="o/r", filename="f").make()


def test_hub_file_downloads_the_pinned_revision_and_verifies_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _file(tmp_path)
    calls: list[dict[str, str]] = []

    def download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(path)

    # The module's lazy proxy is replaced whole: patching the real package would
    # import it, ~100 ms this test does not need.
    monkeypatch.setattr(
        "priml.model.vision_ae.checkpoint.huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    source = HubFile.Config(repo_id="o/r", filename="f", revision="abc", sha256=digest)
    assert source.make().path() == path
    assert calls == [{"repo_id": "o/r", "filename": "f", "revision": "abc"}]
    assert source.make().identity()["revision"] == "abc"


def test_url_file_needs_https_and_a_digest() -> None:
    with pytest.raises(ValueError, match="https"):
        _ = UrlFile.Config(url="http://example.com/a.pt", sha256="0" * 64).make()
    with pytest.raises(ValueError, match="sha256"):
        _ = UrlFile.Config(url="https://example.com/a.pt").make()


def test_url_file_reuses_a_cached_copy_keyed_by_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached file with the right digest is returned without any download."""
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache"))
    payload = b"statistics"
    digest = hashlib.sha256(payload).hexdigest()
    cached = tmp_path / "cache" / "url" / digest / "stats.pt"
    cached.parent.mkdir(parents=True)
    _ = cached.write_bytes(payload)

    def refuse(*args: object) -> object:
        raise AssertionError(f"downloaded {args}")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    source = UrlFile.Config(url="https://example.com/x/stats.pt", sha256=digest)
    assert source.make().path() == cached


def test_url_file_downloads_once_and_refuses_the_wrong_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache"))
    payload = b"statistics"
    digest = hashlib.sha256(payload).hexdigest()
    urls: list[str] = []

    def serve(request: urllib.request.Request) -> _Response:
        urls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", serve)
    source = UrlFile.Config(url="https://example.com/x/stats.pt", sha256=digest)
    path = source.make().path()
    assert path.read_bytes() == payload
    assert source.make().path() == path
    assert urls == ["https://example.com/x/stats.pt"]
    wrong = UrlFile.Config(url="https://example.com/y/stats.pt", sha256="0" * 64)
    with pytest.raises(RuntimeError, match="unsatisfied"):
        _ = wrong.make().path()


class _Response(io.BytesIO):
    """The part of ``HTTPResponse`` a whole, unranged download reads."""

    status = 200
    headers: ClassVar[dict[str, str]] = {}


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path, digest = _file(tmp_path, b"x" * 100_000)
    assert sha256_file(path) == digest


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
