"""Checks for the acquire_assets module."""

from __future__ import annotations

from pathlib import Path

import hashlib

import pytest

from priml.baselines.casmi2026.scripts.acquire_assets import (
    Asset,
    AssetAcquisition,
    acquire_assets,
    read_assets,
)


def _pins(tmp_path: Path, *contents: bytes) -> Path:
    path = tmp_path / "assets.tsv"
    path.write_text(
        "".join(
            f"owner{index}/slug{index}\t{index + 1}\t{hashlib.sha256(data).hexdigest()}\n"
            for index, data in enumerate(contents)
        ),
    )
    return path


def _config(tmp_path: Path, pins: Path) -> AssetAcquisition.Config:
    config = AssetAcquisition.Config()
    config.base_dir = tmp_path / "root"
    config.assets = pins
    return config


def test_the_shipped_pin_table_names_seventeen_archives_with_full_digests() -> None:
    assets = read_assets(AssetAcquisition.Config().assets)
    assert len(assets) == 17
    assert all(len(asset.sha256) == 64 and asset.version >= 1 for asset in assets)
    assert assets[0] == Asset(
        owner="ahmedberatozer",
        slug="casmi26-fpnet-full1",
        version=1,
        sha256="c310e8d445b23c706a2b8db3bd92f71f429258a9d0ca4d080bc14363e9ee5c0f",
    )
    assert assets[0].archive == "casmi26-fpnet-full1.zip"


def test_placed_archives_are_verified_and_stamped(tmp_path: Path) -> None:
    pins = _pins(tmp_path, b"first", b"second")
    config = _config(tmp_path, pins)
    resolved = config.copy_tree().finalize()
    assert resolved.working_dir == tmp_path / "root/datasets/casmi2026/archives"
    for index, data in enumerate((b"first", b"second")):
        folder = Path(resolved.working_dir) / f"owner{index}" / f"slug{index}"
        folder.mkdir(parents=True)
        (folder / f"slug{index}.zip").write_bytes(data)
    config.make().run()
    for index in range(2):
        folder = Path(resolved.working_dir) / f"owner{index}" / f"slug{index}"
        assert (folder / ".ensure_complete").is_file()


def test_an_absent_archive_is_named_for_download(tmp_path: Path) -> None:
    config = _config(tmp_path, _pins(tmp_path, b"first"))
    with pytest.raises(
        FileNotFoundError,
        match=r"Download version 1 of owner0/slug0 to .*slug0\.zip",
    ):
        config.make().run()


def test_a_corrupt_archive_is_moved_aside_and_requested_again(tmp_path: Path) -> None:
    config = _config(tmp_path, _pins(tmp_path, b"first"))
    folder = tmp_path / "root/datasets/casmi2026/archives/owner0/slug0"
    folder.mkdir(parents=True)
    (folder / "slug0.zip").write_bytes(b"damaged")
    with pytest.raises(FileNotFoundError, match="owner0/slug0"):
        config.make().run()
    moved = list(folder.parent.glob("slug0.corrupt.*"))
    assert len(moved) == 1
    assert (moved[0] / "slug0.zip").read_bytes() == b"damaged"


def test_arguments_are_refused_and_the_launcher_declares_the_job(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, _pins(tmp_path, b"first"))
    with pytest.raises(
        ValueError,
        match=r"^AssetAcquisition accepts configuration fields only\.$",
    ):
        config.make().run("--force")
    assert isinstance(acquire_assets(), AssetAcquisition.Config)
    assert acquire_assets().assets.name == "assets.tsv"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
