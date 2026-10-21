"""Check pinned asset preparation without downloading model weights."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import hashlib
import runpy
import sys

from configgle import traverse
from treekle import json_loads

import pytest

from priml.baselines.oscillator.experiments import exp040
from priml.baselines.oscillator.nonlinear import HistoryAblation
from priml.baselines.oscillator.pretrained import AssetManifest, ModelFiles
from priml.baselines.oscillator.pretrained_test import _assets
from priml.baselines.oscillator.scripts import asset_preparation


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


def test_existing_assets_are_verified_without_a_download(tmp_path: Path) -> None:
    cfg = _assets(tmp_path)
    original = (tmp_path / "source.json").read_bytes()
    asset_preparation.prepare_model(cfg)
    assert (tmp_path / "source.json").read_bytes() == original
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        asset_preparation.prepare_model(cfg)


@pytest.mark.parametrize("revision", ["main", "g" * 40])
def test_preparation_requires_an_immutable_revision(
    tmp_path: Path,
    revision: str,
) -> None:
    cfg = ModelFiles.Config()
    cfg.directory = tmp_path / "weights"
    cfg.revision = revision
    with pytest.raises(ValueError, match="revision"):
        asset_preparation.prepare_model(cfg)
    assert not cfg.directory.exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_preparation_is_atomic_and_hashes_the_downloaded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
) -> None:
    source = tmp_path / "cached-download"
    source.write_bytes(b"downloaded content")
    downloads = _Downloads(source=source, interrupted=interrupted)
    monkeypatch.setattr(asset_preparation, "hf_hub_download", downloads.download)
    cfg = ModelFiles.Config()
    cfg.directory = tmp_path / "weights"
    if interrupted:
        with pytest.raises(OSError, match="interrupted"):
            asset_preparation.prepare_model(cfg)
        assert not cfg.directory.exists()
    else:
        asset_preparation.prepare_model(cfg)
        assert cfg.make().directory == cfg.directory
        manifest = json_loads(
            (cfg.directory / "source.json").read_text(),
            AssetManifest,
        )
        assert manifest == {
            "model_id": cfg.model_id,
            "revision": cfg.revision,
            "files": {
                name: hashlib.sha256(b"downloaded content").hexdigest()
                for name in ("config.json", "model.safetensors")
            },
        }
        assert (cfg.directory / "model.safetensors").read_bytes() == source.read_bytes()
    assert downloads.calls == [
        (cfg.model_id, name, cfg.revision)
        for name in ("config.json", "model.safetensors")
    ]
    assert not list(tmp_path.glob(".weights-*"))


@pytest.mark.parametrize("wrapped", [False, True])
def test_preparation_discovers_every_pinned_asset_and_deduplicates_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
) -> None:
    cfg = exp040()
    cfg.recipes = [*cfg.recipes[-2:], cfg.recipes[-2].copy_tree()]
    for recipe in cfg.recipes:
        if wrapped:
            method = HistoryAblation.Config()
            method.method = recipe.study.method
            recipe.study.method = method
    assets = [match.config for match in traverse(cfg, ModelFiles.Config)]
    assert len(assets) == 3
    for files in assets:
        files.directory = tmp_path / files.model_id.rsplit("/", 1)[1]
    source = tmp_path / "cached-download"
    source.write_bytes(b"downloaded content")
    downloads = _Downloads(source=source)
    monkeypatch.setattr(asset_preparation, "hf_hub_download", downloads.download)
    monkeypatch.setattr(asset_preparation, "exp040", lambda: cfg)
    assert asset_preparation.main([]) == 0
    assert len(downloads.calls) == 4
    assert {model for model, _, _ in downloads.calls} == {
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-2",
    }
    for files in assets:
        assert files.make().directory == files.directory


def test_missing_hub_extra_explains_the_preparation_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"priml\[hub\]"):
        runpy.run_path(asset_preparation.__file__)


def test_asset_preparation_entry_exposes_help(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _CWD / "prepare_assets.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(str(script), run_name="__main__")
    assert status.value.code == 0


@dataclass(slots=True, kw_only=True)
class _Downloads:
    source: Path
    interrupted: bool = False
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def download(self, repo_id: str, filename: str, *, revision: str) -> str:
        self.calls.append((repo_id, filename, revision))
        if self.interrupted and filename == "model.safetensors":
            raise OSError("Download interrupted")
        return str(self.source)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
