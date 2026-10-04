"""Tests for the ARC dataset preparation entry point."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import argparse
import json
import logging

import pytest

from priml.baselines.arcagi1.scripts import prepare_data


if TYPE_CHECKING:
    from priml.baselines.arcagi1.augmentation import ArcAugmentation
    from priml.baselines.arcagi1.data import ArcData, PuzzleData


def test_default_directory_matches_exp000_dataset() -> None:
    assert prepare_data.default_directory() == Path("/opt/scratch/datasets/arcagi1")


def test_dataset_config_uses_experiment_dataset_and_base() -> None:
    config = prepare_data.dataset_config("exp_smoke").finalize()
    assert config.working_dir == Path("/opt/scratch/datasets/arcagi1")


def test_num_puzzle_identifiers_reads_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "train" / "dataset.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"num_puzzle_identifiers": 7}))
    assert prepare_data.num_puzzle_identifiers(tmp_path) == 7


def test_num_puzzle_identifiers_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_data.num_puzzle_identifiers(tmp_path)


def test_main_rejects_missing_module_docstring_with_exact_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare_data, "__doc__", None)
    with pytest.raises(
        ValueError,
        match=r"\AExpected __doc__ is not None\.\Z",
    ):
        prepare_data.main()


def test_prepare_uses_the_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "prepared"
    prefix = tmp_path / "pinned" / "arc"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None
    config.working_dir = target
    config.augmentation.num_aug = 7
    config.augmentation.seed = 9
    calls: list[tuple[Path, str, ArcAugmentation]] = []

    def build_override(
        *,
        target_dir: Path,
        input_file_prefix: str,
        augmentation: ArcAugmentation,
    ) -> None:
        calls.append((target_dir, input_file_prefix, augmentation))

    monkeypatch.setattr(prepare_data, "KaggleSource", lambda: nullcontext(prefix))
    monkeypatch.setattr(prepare_data, "build_arc_dataset", build_override)

    assert prepare_data.prepare(config) == target
    assert len(calls) == 1
    built_target, built_prefix, augmentation = calls[0]
    assert built_target == target
    assert built_prefix == str(prefix)
    assert augmentation.config.num_aug == 7
    assert augmentation.config.seed == 9


def test_main_builds_tiny_local_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    puzzle = {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[5, 4, 3], [2, 1, 0]],
            },
        ],
        "test": [{"input": [[1, 2, 3], [4, 5, 6]]}],
    }
    for subset in ("training", "evaluation", "concept"):
        (source / f"arc_{subset}_challenges.json").write_text(
            json.dumps({f"{subset}-puzzle": puzzle}),
        )
        (source / f"arc_{subset}_solutions.json").write_text(
            json.dumps({f"{subset}-puzzle": [[[6, 5, 4], [3, 2, 1]]]}),
        )

    target = tmp_path / "prepared"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None

    def dataset_config_override(
        experiment: str,
    ) -> ArcData.Config | PuzzleData.Config:
        del experiment
        return config

    monkeypatch.setattr(prepare_data, "dataset_config", dataset_config_override)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(target),
            "--experiment",
            "exp000",
            "--input-prefix",
            str(source / "arc"),
            "--num-aug",
            "0",
            "--seed",
            "3",
        ],
    )

    assert prepare_data.main() == 0
    assert sorted(path.name for path in target.iterdir()) == [
        "identifiers.json",
        "test",
        "test_puzzles.json",
        "train",
    ]
    assert prepare_data.num_puzzle_identifiers(target) == 4
    assert json.loads((target / "identifiers.json").read_text()) == [
        "<blank>",
        "training-puzzle",
        "evaluation-puzzle",
        "concept-puzzle",
    ]
    assert sorted(path.name for path in (target / "train").iterdir()) == [
        "all__group_indices.npy",
        "all__inputs.npy",
        "all__labels.npy",
        "all__puzzle_identifiers.npy",
        "all__puzzle_indices.npy",
        "dataset.json",
    ]


def test_main_help_shows_the_complete_script_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    help_text = capsys.readouterr().out
    description = help_text.split("\n\n", 1)[1].split("\noptions:\n", 1)[0]
    assert description == (
        "Prepare the pinned ARC-AGI1 ``arc1concept-aug-1000`` dataset.\n"
        "\n"
        "The default destination matches ``ArcData.Config`` under a default\n"
        "``TrainLoop``. The source revision is immutable; a local ``--input-prefix``\n"
        "keeps tests and offline rebuilds hermetic.\n"
        "\n"
        "Examples:\n"
        "  prepare_data.py\n"
        "  prepare_data.py --directory /datasets/my-arcagi1\n"
        "  prepare_data.py --experiment exp007\n"
    )


def test_main_applies_overrides_and_passes_local_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    prefix = tmp_path / "local-arc"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None
    configs: list[str] = []
    calls: list[tuple[ArcData.Config | PuzzleData.Config, Path | str | None]] = []
    logging_options: list[dict[str, object]] = []

    def dataset_config_override(
        experiment: str,
    ) -> ArcData.Config | PuzzleData.Config:
        configs.append(experiment)
        return config

    def prepare_override(
        received: ArcData.Config | PuzzleData.Config,
        *,
        input_file_prefix: Path | str | None,
    ) -> None:
        calls.append((received, input_file_prefix))

    def basic_config_override(**kwargs: object) -> None:
        logging_options.append(kwargs)

    monkeypatch.setattr(prepare_data, "dataset_config", dataset_config_override)
    monkeypatch.setattr(prepare_data, "prepare", prepare_override)
    monkeypatch.setattr(logging, "basicConfig", basic_config_override)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(destination),
            "--experiment",
            "exp000",
            "--input-prefix",
            str(prefix),
            "--num-aug",
            "4",
            "--seed",
            "3",
        ],
    )

    assert prepare_data.main() == 0
    assert configs == ["exp000"]
    assert logging_options == [{"level": logging.INFO, "format": "%(message)s"}]
    assert config.working_dir == destination
    assert config.augmentation.num_aug == 4
    assert config.augmentation.seed == 3
    assert calls == [(config, prefix)]


def test_add_arguments_defaults_and_parses_all_options() -> None:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    defaults = parser.parse_args([])
    assert vars(defaults) == {
        "directory": None,
        "experiment": "exp000",
        "input_prefix": None,
        "num_aug": None,
        "seed": None,
    }

    flags = parser.parse_args(
        [
            "--directory",
            "/scratch/data",
            "--experiment",
            "exp007",
            "--input-prefix",
            "/scratch/input",
            "--num-aug",
            "3",
            "--seed",
            "11",
        ],
    )
    assert vars(flags) == {
        "directory": Path("/scratch/data"),
        "experiment": "exp007",
        "input_prefix": Path("/scratch/input"),
        "num_aug": 3,
        "seed": 11,
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
