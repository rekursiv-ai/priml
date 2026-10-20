"""Preparation copies authentic CSV bytes and records their digests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib

from treekle import json_loads

import pytest

from priml.baselines.name_that_llm.scripts.prepare_data import Preparation


if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_validates_then_copies_inputs_and_records_digests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "train.csv").write_text(
        "id,prompt_id,text,model\n1,p1,Text,a\n2,p2,Text,b\n",
    )
    (source / "test.csv").write_text("id,text\n3,Test\n")
    config = Preparation.Config()
    config.base_dir = tmp_path
    config.dataset.directory = "/raw"
    config.dataset.expected_rows = config.dataset.expected_prompts = 2
    config.dataset.labels = ["a", "b"]
    config.directory = "/prepared"
    config.make().run()
    prepared = tmp_path / "prepared"
    assert (prepared / "train.csv").read_bytes() == (source / "train.csv").read_bytes()
    manifest = json_loads(
        (prepared / "manifest.json").read_text(),
        target=dict[str, object],
    )
    assert manifest["rows"] == 2
    assert manifest["prompts"] == 2
    assert manifest["ids"] == ["1", "2"]
    assert manifest["sha256"] == {
        name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
        for name in ["train.csv", "test.csv"]
    }


def test_prepare_requires_real_input_files(tmp_path: Path) -> None:
    config = Preparation.Config()
    config.base_dir = tmp_path
    config.dataset.directory = "/raw"
    config.directory = "/prepared"
    with pytest.raises(FileNotFoundError):
        config.make().run()
    assert not (tmp_path / "prepared").exists()


def test_prepare_rejects_untracked_flags() -> None:
    with pytest.raises(ValueError, match="configuration fields only"):
        Preparation.Config().make().run("--unknown")


def test_prepare_rejects_smoke_caps_before_io(tmp_path: Path) -> None:
    config = Preparation.Config()
    config.base_dir = tmp_path
    config.dataset.directory = "/missing"
    config.dataset.max_prompts = 2
    config.directory = "/prepared"
    with pytest.raises(ValueError, match="complete input population"):
        config.make().run()
    assert not (tmp_path / "prepared").exists()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
