"""Check the comparison CLI's outputs, refusals, and flags."""

from functools import partial
from pathlib import Path
from typing import Final, cast

import argparse
import re
import shutil
import subprocess

from treekle import from_plain, json_loads

import pytest

from priml.baselines.oscillator import experiments
from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.scripts import comparison
from priml.baselines.oscillator.train_step import RidgeCorrection


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.compute_training
def test_smoke_run_writes_configs_records_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(comparison, "_git", _fake_git)
    directory = tmp_path / "run"
    argv = ["--directory", str(directory), "--experiments", "exp_smoke", "--seeds", "7"]
    assert comparison.main(argv) == 0
    labels = [f"{case}-exp_smoke-7" for case in ("matched", "mismatched")]
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [f"{label}.{suffix}" for label in labels for suffix in ("json", "txt")]
        + ["comparison.json"],
    )
    report = json_loads(
        (directory / "comparison.json").read_text(),
        dict[str, object],
    )
    assert report["experiments"] == ["exp_smoke"]
    assert report["seeds"] == [7]
    assert isinstance(report["torch"], str)
    assert report["git_head"] == "rev-parse HEAD"
    assert report["git_status"] == "status --short"
    runs = from_plain(report["runs"], list[dict[str, object]])
    assert [run["case"] for run in runs] == ["matched", "mismatched"]
    assert runs[0] == json_loads(
        (directory / f"{labels[0]}.json").read_text(),
        dict[str, object],
    )
    hashes = from_plain(report["source_sha256"], dict[str, str])
    assert "priml/baselines/oscillator/scripts/comparison.py" in hashes
    assert "priml/baselines/etth1/model.py" in hashes
    assert capsys.readouterr().out.splitlines() == [
        *(f"Finished {label}" for label in labels),
        str(directory / "comparison.json"),
    ]


@pytest.mark.cli_git
def test_git_reports_the_checkout_revision() -> None:
    git = shutil.which("git")
    assert git is not None
    head = comparison._git(git, "rev-parse", "HEAD", cwd=_CWD)
    assert re.fullmatch(r"[0-9a-f]{40}", head)


def test_every_case_seed_and_recipe_is_run_labelled_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The study itself is stubbed; this pins the loop around it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "check_output",
        partial(_record_git, calls=calls),
    )
    monkeypatch.setattr(ForecastStudy, "run", _canned_run)
    directory = tmp_path / "run"
    argv = ["--directory", str(directory), "--experiments", "exp_smoke", "exp000"]
    assert comparison.main([*argv, "--seeds", "7", "8"]) == 0
    report = json_loads(
        (directory / "comparison.json").read_text(),
        dict[str, object],
    )
    runs = from_plain(report["runs"], list[dict[str, object]])
    assert [(run["case"], run["experiment"], run["seed"]) for run in runs] == [
        (case, name, seed)
        for case in ("matched", "mismatched")
        for seed in (7, 8)
        for name in ("exp_smoke", "exp000")
    ]
    assert all(run["reference_cubic"] == 0.0 for run in runs[:4])
    assert all(run["reference_cubic"] == 0.4 for run in runs[4:])
    assert report["git_head"] == "rev-parse HEAD"
    assert [call[1:] for call in calls] == [
        ["rev-parse", "HEAD"],
        ["status", "--short"],
    ]
    assert "seed=8," in (directory / "mismatched-exp000-8.txt").read_text()


def _record_git(
    command: list[str],
    *,
    cwd: Path,
    text: bool,
    calls: list[list[str]],
) -> str:
    assert text
    assert (cwd / "priml/baselines/oscillator/scripts/comparison.py").is_file()
    calls.append(command)
    return f" {' '.join(command[1:])}\n"


def _canned_run(study: ForecastStudy) -> dict[str, object]:
    return {
        "experiment": study.config.experiment_name,
        "seed": study.config.seed,
        "reference_cubic": study.config.data.reference.cubic,
    }


def test_cli_refuses_to_overwrite_a_run(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        comparison.main(["--directory", str(tmp_path)])


def test_cli_rejects_an_unnormalized_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="normalized"):
        comparison.main(["--directory", f"{tmp_path}/a/../run"])
    assert not (tmp_path / "run").exists()


def test_cli_requires_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", _no_executable)
    with pytest.raises(RuntimeError, match="Git"):
        comparison.main(["--directory", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_cli_rejects_a_non_study_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiments, "exp_ridge", RidgeCorrection.Config, raising=False)
    with pytest.raises(TypeError, match="oscillator study"):
        comparison.main(
            ["--directory", str(tmp_path / "run"), "--experiments", "exp_ridge"],
        )
    assert not (tmp_path / "run").exists()


def test_cli_defaults_name_the_pilot_recipes_and_seeds() -> None:
    parser = argparse.ArgumentParser()
    comparison._add_arguments(parser)
    flags = cast(comparison._Flags, parser.parse_args([]))
    assert flags.experiments == ["exp000", "exp001", "exp002"]
    assert flags.seeds == [2021, 2022, 2023]


def test_cli_help_and_unknown_flags() -> None:
    with pytest.raises(SystemExit) as help_result:
        comparison.main(["--help"])
    assert help_result.value.code == 0
    with pytest.raises(SystemExit) as bad_flag:
        comparison.main(["--unknown"])
    assert bad_flag.value.code == 2


def _fake_git(git: str, *args: str, cwd: Path) -> str:
    del git, cwd
    return " ".join(args)


def _no_executable(name: str) -> None:
    del name


def test_records_are_indented_json_with_a_final_newline_and_a_tagged_nan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    comparison._write_json(path, value={"case": "matched", "mse": float("nan")})
    assert path.read_text() == (
        '{\n  "case": "matched",\n  "mse": {\n    "py/float": "nan"\n  }\n}\n'
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
