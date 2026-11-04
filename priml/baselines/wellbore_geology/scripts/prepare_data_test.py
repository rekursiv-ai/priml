"""Tests for the ROGII staging entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest

from priml.baselines.wellbore_geology.scripts import prepare_data
from priml.baselines.wellbore_geology.scripts.prepare_data import StagedData


if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_command_routes_explicit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []

    def record(source: Path, *, destination: Path) -> StagedData:
        calls.append((source, destination))
        return StagedData(directory=destination, train_wells=("a",), test_wells=("b",))

    monkeypatch.setattr(prepare_data, "stage_data", record)
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--source", str(source), "--directory", str(destination)],
    )
    assert prepare_data.main() == 0
    assert calls == [(source, destination)]


def test_prepare_help_contains_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    text = capsys.readouterr().out
    assert "Stage the real ROGII CSV release" in text
    assert "exec uv" not in text


def _release(root: Path, *, horizontal: str = "X,Y,MD\n1,10,0\n3,,1\n") -> Path:
    """Write a two-train-well, one-test-well release under ``root``; return it."""
    for split, wells in (("train", ("w2", "w1")), ("test", ("t1",))):
        (root / split).mkdir(parents=True)
        for well in wells:
            (root / split / f"{well}__horizontal_well.csv").write_text(horizontal)
            (root / split / f"{well}__typewell.csv").write_text("TVT,GR\n0,1\n")
    (root / "sample_submission.csv").write_text("id,tvt\n")
    return root


def test_staging_copies_each_pair_and_maps_each_training_well(tmp_path: Path) -> None:
    source = _release(tmp_path / "release")

    staged = prepare_data.stage_data(source, destination=tmp_path / "staged")

    assert (staged.train_wells, staged.test_wells) == (("w1", "w2"), ("t1",))
    assert sorted(
        path.relative_to(staged.directory).as_posix()
        for path in staged.directory.rglob("*.csv")
    ) == [
        "sample_submission.csv",
        "test/t1__horizontal_well.csv",
        "test/t1__typewell.csv",
        "train/w1__horizontal_well.csv",
        "train/w1__typewell.csv",
        "train/w2__horizontal_well.csv",
        "train/w2__typewell.csv",
        "train_png_typewell_map.csv",
    ]
    # Each centroid is the mean of the cells it has: X of 1 and 3, Y of 10 alone.
    assert (staged.directory / "train_png_typewell_map.csv").read_text() == (
        "well_id,horizontal_avg_X,horizontal_avg_Y\n"
        "w1,2.000000,10.000000\n"
        "w2,2.000000,10.000000\n"
    )


def test_staging_refuses_a_destination_inside_its_source(tmp_path: Path) -> None:
    source = _release(tmp_path / "release")

    with pytest.raises(ValueError, match="must not overlap"):
        _ = prepare_data.stage_data(source, destination=source / "staged")

    assert not (source / "staged").exists()


@pytest.mark.parametrize(
    ("horizontal", "match"),
    [("X,Y\n", "is empty: w1"), ("X,Y\n,1\n", "centroid is missing: w1")],
)
def test_the_map_refuses_a_well_with_no_position(
    tmp_path: Path,
    horizontal: str,
    match: str,
) -> None:
    source = _release(tmp_path / "release", horizontal=horizontal)

    with pytest.raises(ValueError, match=match):
        _ = prepare_data.geographic_map(source / "train")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
