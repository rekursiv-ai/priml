"""Tests for the sprite cache."""

from __future__ import annotations

from pathlib import Path

import urllib.error
import urllib.request

import pytest

from priml.baselines.craftax.game.render import assets, sprites


def _png(directory: Path, name: str, payload: bytes = b"\x89PNG-stub") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def test_the_revision_defaults_to_a_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A branch would let a texture change under a cache that has no reason to
    # re-fetch it, silently altering every frame.
    seen: list[str] = []

    def capture(url: str, **kwargs: object) -> object:
        del kwargs
        seen.append(url)
        raise urllib.error.URLError("stop here")

    monkeypatch.setattr(urllib.request, "urlopen", capture)
    with pytest.raises(RuntimeError):
        assets.fetch("zombie.png", directory=tmp_path)
    assert "/v1.6.1/" in seen[0]


def test_a_cached_sprite_is_not_downloaded_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _png(tmp_path, "zombie.png")

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("a cached sprite must not be re-fetched")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert assets.fetch("zombie.png", directory=tmp_path).exists()


def test_a_download_failure_is_reported_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match=r"zombie\.png"):
        assets.fetch("zombie.png", directory=tmp_path)


def test_an_interrupted_download_leaves_no_cached_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A truncated PNG would be treated as cached forever, so every later run
    # would fail to decode it and none would re-fetch it.
    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError):
        assets.fetch("zombie.png", directory=tmp_path)
    assert not (tmp_path / "zombie.png").exists()


def test_the_digest_changes_when_a_sprite_changes(tmp_path: Path) -> None:
    # The point of the digest: an upstream texture swapped under the same name
    # changes every frame, and nothing else would notice.
    _png(tmp_path, "zombie.png", b"one")
    before = assets.digest(tmp_path)
    _png(tmp_path, "zombie.png", b"two")
    assert assets.digest(tmp_path) != before


def test_the_digest_ignores_file_order(tmp_path: Path) -> None:
    _png(tmp_path, "b.png", b"b")
    _png(tmp_path, "a.png", b"a")
    first = assets.digest(tmp_path)
    (tmp_path / "a.png").unlink()
    _png(tmp_path, "a.png", b"a")
    assert assets.digest(tmp_path) == first


def test_every_sprite_is_named_once() -> None:
    names = sprites.every_sprite()
    assert len(names) == len(set(names))
    assert "zombie.png" in names
    # The two flat-colour blocks contribute no file.
    assert "" not in names


def test_the_sprite_tables_cover_every_enum_value() -> None:
    from priml.baselines.craftax.game.constants import (  # noqa: PLC0415
        BlockType,
        ItemType,
        ProjectileType,
    )

    assert len(sprites.BLOCK_SPRITES) == len(BlockType)
    assert len(sprites.ITEM_SPRITES) == len(ItemType)
    assert len(sprites.PROJECTILE_SPRITES) == len(ProjectileType)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
