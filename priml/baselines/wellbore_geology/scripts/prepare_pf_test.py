"""Require the preparation command to stage exactly the experiment's particle caches."""

from pathlib import Path

import sys

import pytest

from priml.baselines.wellbore_geology.particles import read_particle_cache
from priml.baselines.wellbore_geology.pipeline import WellboreStudy
from priml.baselines.wellbore_geology.scripts import prepare_pf
from priml.baselines.wellbore_geology.testing import (
    tiny_family,
    write_training_wells,
)


def _study(root: Path, *, particle: bool = True) -> WellboreStudy.Config:
    study = WellboreStudy.Config()
    study.dataset_directory = root / "data"
    study.pf_directory = root / "cache"
    study.families = [tiny_family(particle=particle), tiny_family(particle=False)]
    study.families[1].name = "plain"
    return study


def test_check_only_validates_all_split_pairs_without_creating_a_cache(
    tmp_path: Path,
) -> None:
    for split in ("train", "test"):
        write_training_wells(tmp_path / "data" / split, count=2)
    prepare_pf.prepare(_study(tmp_path), check_only=True)
    assert not (tmp_path / "cache").exists()


def test_preparation_stages_only_the_experiments_particle_families(
    tmp_path: Path,
) -> None:
    for split in ("train", "test"):
        write_training_wells(tmp_path / "data" / split, count=2)
    prepare_pf.prepare(_study(tmp_path), check_only=False)
    assert [path.name for path in (tmp_path / "cache").iterdir()] == ["0803_V2"]
    for split in ("train", "test"):
        directory = tmp_path / "cache/0803_V2" / split
        assert sorted(path.name for path in directory.iterdir()) == [
            "config.json",
            "well0.npz",
            "well1.npz",
        ]
        record = read_particle_cache(directory / "well0.npz")
        assert record["last_seen_idx"].item() == 3


def test_main_resolves_the_named_experiment_and_dispatches_its_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(tmp_path)
    requested: list[str] = []
    calls: list[tuple[WellboreStudy.Config, bool]] = []

    def resolve(name: str) -> WellboreStudy.Config:
        requested.append(name)
        return study

    def record(config: WellboreStudy.Config, *, check_only: bool) -> None:
        calls.append((config, check_only))

    monkeypatch.setattr(prepare_pf, "resolve_config", resolve)
    monkeypatch.setattr(prepare_pf, "prepare", record)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_pf", "pkg.experiments.exp007", "--check-only"],
    )
    assert prepare_pf.main() == 0
    assert requested == ["pkg.experiments.exp007"]
    assert calls == [(study, True)]


def test_particle_preparation_help_is_owned_by_the_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_pf", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_pf.main()
    output = capsys.readouterr().out
    assert "Prepare an experiment's particle-filter" in output
    assert "experiment" in output
    assert "--check-only" in output
    assert "--dataset" not in output
    assert "exec uv" not in output


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
